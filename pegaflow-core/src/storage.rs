use crossbeam::sync::ShardedLock;
// ============================================================================
// StorageEngine eviction + layout notes (mirrors the high-level summary in
// lib.rs):
// - Allocation is always attempted first; eviction only happens when the pinned
//   pool cannot satisfy a request (often due to fragmentation at different
//   utilization levels). On failure we drop a batch of LRU entries and retry.
// - Eviction is batched (RECLAIM_BATCH_OBJECTS) so a single allocation failure
//   can free multiple cached objects at once instead of thrashing.
// - Pre-eviction: A background thread can be enabled to proactively reclaim
//   blocks when free space drops below a threshold, reducing allocation stalls.
// - LRU key: BlockHash (Vec<u8> digest for one logical block across layers/TP ranks).
//   LRU value: Block (stateful set of LayerBlock slots, ordered by flat slot id
//   slot_id = layer_id * tp_size + tp_rank).
// - CPU memory picture for one hash (split K/V storage):
//     BlockHash ->
//       K range: [slot0 K data][slot1 K data][slot2 K data]...
//       V range: [slot0 V data][slot1 V data][slot2 V data]...
//   Slots saved together share one allocation per K and V, so a layer's blocks
//   sit back-to-back in those ranges. When K/V are co-located, V_ptr is None
//   and the V bytes follow K in the same allocation.
// ============================================================================
use bytesize::ByteSize;
use hashlink::LruCache;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tracing::{debug, error, info};

use crate::metrics::core_metrics;
use crate::pinned_pool::{PinnedAllocation, PinnedMemoryPool};

const RECLAIM_BATCH_OBJECTS: usize = 4096;

/// Configuration for pre-eviction monitoring thread.
#[derive(Debug, Clone)]
pub struct PreEvictConfig {
    /// Enable pre-eviction background thread
    pub enabled: bool,
    /// Start evicting when free space drops below this threshold (bytes)
    pub threshold_bytes: u64,
    /// Target free space after eviction completes (bytes)
    pub target_bytes: u64,
    /// How often to check pool usage (milliseconds)
    pub check_interval_ms: u64,
}

impl Default for PreEvictConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold_bytes: 5 * 1024 * 1024 * 1024, // 5GB
            target_bytes: 8 * 1024 * 1024 * 1024,    // 8GB
            check_interval_ms: 100,
        }
    }
}

impl PreEvictConfig {
    pub fn new(threshold_bytes: u64, target_bytes: u64, check_interval_ms: u64) -> Self {
        Self {
            enabled: true,
            threshold_bytes,
            target_bytes,
            check_interval_ms,
        }
    }
}

// A "slot" in this file refers to a specific position in the flattened logical storage,
// calculated as `layer_id * tp_size + tp_rank`.
// vLLM/Connectors report the total topology (layers * tp_size) via registration,
// and this count is immutable for the lifetime of the Instance.
// NOTE: Storage is generic and operates on flat indices (slots).

/// Key for identifying blocks in storage, including namespace for model isolation.
///
/// NOTE: Using String for namespace is simple but adds ~20-50 bytes overhead per key.
/// Future optimization: intern namespaces to u32 IDs (saves memory, faster comparison).
///
/// TODO: Optimize BlockKey to avoid deep copy on every lookup
/// Current issue: BlockKey::new creates deep copies of namespace (String) and hash (Vec<u8>)
/// on every lookup in hot paths (slot_has_block, block_is_complete).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// Namespace for model isolation (e.g., model name, or empty string for shared storage)
    pub namespace: String,
    /// Block content hash
    pub hash: Vec<u8>,
}

impl BlockKey {
    pub fn new(namespace: String, hash: Vec<u8>) -> Self {
        Self { namespace, hash }
    }
}

pub type BlockHash = Vec<u8>;
type LayerBlockSlots = Vec<Option<Arc<LayerBlock>>>;

/// State machine for a logical block (all layer/TP slots for one hash).
enum BlockState {
    /// In-flight, still accepting writes for empty slots.
    Filling(FillingBlock),
    /// Fully populated; read-only view.
    Sealed(SealedBlock),
}

/// Mutable block while we are still receiving layer slots.
struct FillingBlock {
    slots: LayerBlockSlots,
    remaining: usize,
    total_slots: usize,
    footprint: u64,
}

impl FillingBlock {
    fn new(total_slots: usize) -> Self {
        Self {
            slots: vec![None; total_slots],
            remaining: total_slots,
            total_slots,
            footprint: 0,
        }
    }

    fn slot_has_block(&self, slot_id: usize) -> bool {
        self.slots
            .get(slot_id)
            .and_then(|opt| opt.as_ref())
            .is_some()
    }

    fn get_slot(&self, slot_id: usize) -> Option<Arc<LayerBlock>> {
        self.slots.get(slot_id).and_then(|opt| opt.clone())
    }

    fn insert_slot(
        &mut self,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<bool, BlockInsertError> {
        if total_slots != self.total_slots {
            return Err(BlockInsertError::SlotCountMismatch {
                expected: self.total_slots,
                got: total_slots,
            });
        }

        if slot_id >= self.total_slots {
            return Err(BlockInsertError::SlotOutOfBounds {
                slot_id,
                total_slots: self.total_slots,
            });
        }

        if self.slots[slot_id].is_some() {
            return Err(BlockInsertError::SlotAlreadyFilled { slot_id });
        }

        self.footprint += block.memory_footprint();
        self.slots[slot_id] = Some(block);
        self.remaining = self
            .remaining
            .checked_sub(1)
            .expect("remaining should not underflow");
        Ok(self.remaining == 0)
    }
}

/// Immutable view after all slots are filled; no further writes allowed.
struct SealedBlock {
    slots: Arc<[Arc<LayerBlock>]>,
    footprint: u64,
}

impl SealedBlock {
    fn slot_has_block(&self, slot_id: usize) -> bool {
        self.slots.get(slot_id).is_some()
    }

    fn get_slot(&self, slot_id: usize) -> Option<Arc<LayerBlock>> {
        self.slots.get(slot_id).cloned()
    }
}

/// State machine for a logical block (all layer/TP slots for one hash).
impl BlockState {
    fn insert_slot(
        &mut self,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<(), BlockInsertError> {
        match self {
            BlockState::Filling(filling) => {
                let completed = filling.insert_slot(slot_id, block, total_slots)?;
                if completed {
                    let sealed_slots: Vec<Arc<LayerBlock>> = filling
                        .slots
                        .iter()
                        .map(|opt| opt.as_ref().expect("all slots filled").clone())
                        .collect();
                    *self = BlockState::Sealed(SealedBlock {
                        slots: sealed_slots.into(),
                        footprint: filling.footprint,
                    });
                }
                Ok(())
            }
            BlockState::Sealed(_) => Err(BlockInsertError::Sealed),
        }
    }

    fn slot_has_block(&self, slot_id: usize) -> bool {
        match self {
            BlockState::Filling(filling) => filling.slot_has_block(slot_id),
            BlockState::Sealed(sealed) => sealed.slot_has_block(slot_id),
        }
    }

    fn get_slot(&self, slot_id: usize) -> Option<Arc<LayerBlock>> {
        match self {
            BlockState::Filling(filling) => filling.get_slot(slot_id),
            BlockState::Sealed(sealed) => sealed.get_slot(slot_id),
        }
    }

    fn is_complete(&self) -> bool {
        matches!(self, BlockState::Sealed(_))
    }

    /// Total pinned memory occupied by all filled slots (O(1), cached on insert).
    fn memory_footprint(&self) -> u64 {
        match self {
            BlockState::Filling(f) => f.footprint,
            BlockState::Sealed(s) => s.footprint,
        }
    }
}

/// Wrapper for per-slot layer blocks with a fixed weight for cache eviction.
pub struct Block {
    inner: ShardedLock<BlockState>,
}

impl Block {
    pub fn new(total_slots: usize) -> Self {
        Self {
            inner: ShardedLock::new(BlockState::Filling(FillingBlock::new(total_slots))),
        }
    }

    pub fn insert_slot(
        &self,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<(), BlockInsertError> {
        let mut state = self.inner.write().expect("block entry write lock poisoned");
        state.insert_slot(slot_id, block, total_slots)
    }

    pub fn slot_has_block(&self, slot_id: usize) -> bool {
        let state = self.inner.read().expect("block entry lock poisoned");
        state.slot_has_block(slot_id)
    }

    pub fn get_slot(&self, slot_id: usize) -> Option<Arc<LayerBlock>> {
        let state = self.inner.read().expect("block entry lock poisoned");
        state.get_slot(slot_id)
    }

    pub fn is_complete(&self) -> bool {
        let state = self.inner.read().expect("block entry lock poisoned");
        state.is_complete()
    }

    /// Total pinned memory occupied by this block (sum of all filled slots).
    pub fn memory_footprint(&self) -> u64 {
        self.inner
            .read()
            .expect("block lock poisoned")
            .memory_footprint()
    }
}

/// CPU block data stored in pinned memory for a single layer/TP slot.
pub struct LayerBlock {
    /// Pointer to K segment (or combined data if contiguous)
    k_ptr: std::ptr::NonNull<u8>,
    /// Pointer to V segment (if stored separately)
    v_ptr: Option<std::ptr::NonNull<u8>>,
    size: usize,
    /// Shared RAII allocation handle for K memory (automatically freed when last reference drops)
    #[allow(dead_code)]
    k_allocation: Arc<PinnedAllocation>,
    /// Shared RAII allocation handle for V memory (if separate from K)
    #[allow(dead_code)]
    v_allocation: Option<Arc<PinnedAllocation>>,
}

impl LayerBlock {
    pub fn new_contiguous(ptr: *mut u8, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        let k_ptr =
            std::ptr::NonNull::new(ptr).expect("contiguous block K pointer must be non-null");
        Self {
            k_ptr,
            v_ptr: None,
            size,
            k_allocation: allocation,
            v_allocation: None,
        }
    }

    pub fn new_split(
        k_ptr: *mut u8,
        v_ptr: *mut u8,
        size: usize,
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
    ) -> Self {
        let k_ptr = std::ptr::NonNull::new(k_ptr).expect("split block K pointer must be non-null");
        let v_ptr = std::ptr::NonNull::new(v_ptr).expect("split block V pointer must be non-null");
        Self {
            k_ptr,
            v_ptr: Some(v_ptr),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
        }
    }

    pub fn k_ptr(&self) -> *const u8 {
        self.k_ptr.as_ptr()
    }

    pub fn v_ptr(&self) -> Option<*const u8> {
        self.v_ptr.map(|ptr| ptr.as_ptr() as *const u8)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Total pinned memory occupied by this layer block (K + V if split).
    pub fn memory_footprint(&self) -> u64 {
        let k_bytes = self.size as u64;
        if self.v_allocation.is_some() {
            k_bytes * 2 // Split storage: K and V each occupy `size` bytes
        } else {
            k_bytes // Contiguous: single allocation
        }
    }
}

// Safety: pinned memory ownership is tracked by Arc counters on the allocations.
unsafe impl Send for LayerBlock {}
unsafe impl Sync for LayerBlock {}

pub struct StorageEngine {
    kv_storage: Arc<Mutex<LruCache<BlockKey, Arc<Block>>>>,
    pinned_pool: Arc<PinnedMemoryPool>,
    pre_evict_stop: Arc<AtomicBool>,
    pre_evict_handle: Option<JoinHandle<()>>,
}

impl StorageEngine {
    pub fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: PreEvictConfig,
    ) -> Self {
        let pinned_pool = Arc::new(PinnedMemoryPool::new(capacity_bytes, use_hugepages));
        let kv_storage = Arc::new(Mutex::new(LruCache::new_unbounded()));
        let pre_evict_stop = Arc::new(AtomicBool::new(false));

        let pre_evict_handle = if config.enabled {
            info!(
                "Starting pre-eviction monitor: threshold={}, target={}, interval={}ms",
                ByteSize(config.threshold_bytes),
                ByteSize(config.target_bytes),
                config.check_interval_ms
            );

            let pool = Arc::clone(&pinned_pool);
            let cache = Arc::clone(&kv_storage);
            let stop = Arc::clone(&pre_evict_stop);

            Some(std::thread::spawn(move || {
                Self::pre_evict_monitor(pool, cache, config, stop);
            }))
        } else {
            None
        };

        Self {
            kv_storage,
            pinned_pool,
            pre_evict_stop,
            pre_evict_handle,
        }
    }

    pub fn allocate(&self, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
        loop {
            if let Some(allocation) = self.pinned_pool.allocate(size) {
                return Some(Arc::new(allocation));
            }

            let reclaimed = self.reclaim(RECLAIM_BATCH_OBJECTS);
            if reclaimed > 0 {
                continue;
            } else {
                let (used, total) = self.pinned_pool.usage();
                error!(
                    "Pinned memory pool exhausted! Requested: {}, Used: {}, Total: {}, Cache empty",
                    ByteSize(size.get()),
                    ByteSize(used),
                    ByteSize(total)
                );
                core_metrics().pool_alloc_failures.add(1, &[]);
                return None;
            }
        }
    }

    fn reclaim(&self, target_objects: usize) -> usize {
        Self::reclaim_from_cache_by_count(&self.kv_storage, target_objects)
    }

    /// Reclaim blocks from the cache by count. Returns the number of blocks evicted.
    fn reclaim_from_cache_by_count(
        cache: &Mutex<LruCache<BlockKey, Arc<Block>>>,
        target_objects: usize,
    ) -> usize {
        if target_objects == 0 {
            return 0;
        }

        let mut freed_entries = 0;
        let mut cache_lock = cache.lock().unwrap();

        while freed_entries < target_objects {
            let Some((_hash, _layer_blocks)) = cache_lock.remove_lru() else {
                break;
            };
            freed_entries += 1;
        }
        drop(cache_lock);

        if freed_entries > 0 {
            debug!(freed_entries, "Reclaimed blocks from cache");
            core_metrics()
                .cache_block_evictions
                .add(freed_entries as u64, &[]);
        }

        freed_entries
    }

    /// Reclaim blocks from the cache by target bytes. Returns (blocks_freed, bytes_freed).
    fn reclaim_from_cache_by_bytes(
        cache: &Mutex<LruCache<BlockKey, Arc<Block>>>,
        target_bytes: u64,
    ) -> (usize, u64) {
        if target_bytes == 0 {
            return (0, 0);
        }

        let mut freed_blocks = 0;
        let mut freed_bytes = 0u64;
        let mut cache_lock = cache.lock().unwrap();

        while freed_bytes < target_bytes {
            let Some((_hash, block)) = cache_lock.remove_lru() else {
                break;
            };

            freed_bytes += block.memory_footprint();
            freed_blocks += 1;
        }
        drop(cache_lock);

        if freed_blocks > 0 {
            debug!(
                freed_blocks,
                freed_bytes = ByteSize(freed_bytes).to_string(),
                "Reclaimed blocks from cache"
            );
            core_metrics()
                .cache_block_evictions
                .add(freed_blocks as u64, &[]);
        }

        (freed_blocks, freed_bytes)
    }

    fn pre_evict_monitor(
        pool: Arc<PinnedMemoryPool>,
        cache: Arc<Mutex<LruCache<BlockKey, Arc<Block>>>>,
        config: PreEvictConfig,
        stop: Arc<AtomicBool>,
    ) {
        let interval = Duration::from_millis(config.check_interval_ms);

        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(interval);

            let (used, total) = pool.usage();
            let free = total.saturating_sub(used);

            if free < config.threshold_bytes {
                let target_free = config.target_bytes;
                let need_free = target_free.saturating_sub(free);

                debug!(
                    free = ByteSize(free).to_string(),
                    threshold = ByteSize(config.threshold_bytes).to_string(),
                    target = ByteSize(target_free).to_string(),
                    need_free = ByteSize(need_free).to_string(),
                    "Pre-eviction triggered"
                );

                let (freed_blocks, freed_bytes) =
                    Self::reclaim_from_cache_by_bytes(&cache, need_free);

                if freed_blocks > 0 {
                    info!(
                        freed_blocks,
                        freed_bytes = ByteSize(freed_bytes).to_string(),
                        "Pre-eviction completed"
                    );
                }
            }
        }

        debug!("Pre-eviction monitor thread stopped");
    }

    pub fn slot_has_block(&self, namespace: &str, block_hash: &[u8], slot_id: usize) -> bool {
        let key = BlockKey::new(namespace.to_string(), block_hash.to_vec());
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(&key)
            .map(|blocks| blocks.slot_has_block(slot_id))
            .unwrap_or(false)
    }

    pub fn block_is_complete(&self, namespace: &str, block_hash: &[u8]) -> bool {
        let key = BlockKey::new(namespace.to_string(), block_hash.to_vec());
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(&key)
            .map(|blocks| blocks.is_complete())
            .unwrap_or(false)
    }

    pub fn insert_block(
        &self,
        namespace: &str,
        block_hash: BlockHash,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<(), BlockInsertError> {
        let key = BlockKey::new(namespace.to_string(), block_hash.clone());
        let mut cache = self.kv_storage.lock().unwrap();
        let entry = cache.get(&key).cloned().unwrap_or_else(|| {
            let new_blocks = Arc::new(Block::new(total_slots));
            cache.insert(key.clone(), Arc::clone(&new_blocks));
            // Record new block insertion into cache
            core_metrics().cache_block_insertions.add(1, &[]);
            new_blocks
        });
        entry.insert_slot(slot_id, block, total_slots)
    }

    pub fn lookup_many(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<Block>>, String> {
        let mut cache = self.kv_storage.lock().unwrap();
        let mut result = Vec::with_capacity(block_hashes.len());
        for (idx, hash) in block_hashes.iter().enumerate() {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let shard_blocks = cache.get(&key).cloned().ok_or_else(|| {
                format!(
                    "missing KV block hash at index {idx} (namespace={namespace}, hash_len={})",
                    hash.len()
                )
            })?;
            result.push(shard_blocks);
        }
        Ok(result)
    }
}

impl Drop for StorageEngine {
    fn drop(&mut self) {
        // Signal the pre-eviction thread to stop
        self.pre_evict_stop.store(true, Ordering::Relaxed);

        // Wait for the thread to finish
        if let Some(handle) = self.pre_evict_handle.take() {
            debug!("Waiting for pre-eviction monitor thread to stop");
            if let Err(e) = handle.join() {
                error!("Failed to join pre-eviction monitor thread: {:?}", e);
            }
        }
    }
}

#[derive(Debug)]
pub enum BlockInsertError {
    Sealed,
    SlotOutOfBounds { slot_id: usize, total_slots: usize },
    SlotAlreadyFilled { slot_id: usize },
    SlotCountMismatch { expected: usize, got: usize },
}

impl fmt::Display for BlockInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockInsertError::Sealed => write!(f, "block is sealed and read-only"),
            BlockInsertError::SlotOutOfBounds {
                slot_id,
                total_slots,
            } => {
                write!(
                    f,
                    "slot_id {} out of bounds ({} slots)",
                    slot_id, total_slots
                )
            }
            BlockInsertError::SlotAlreadyFilled { slot_id } => {
                write!(f, "slot_id {} already has data", slot_id)
            }
            BlockInsertError::SlotCountMismatch { expected, got } => {
                write!(f, "slot count mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BlockInsertError {}
