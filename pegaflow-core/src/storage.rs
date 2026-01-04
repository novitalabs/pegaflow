// ============================================================================
// StorageEngine: Two-phase block storage with separate write and read paths.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Architecture:
// - Inflight: DashMap<BlockKey, Mutex<InflightBlock>> for concurrent writes
// - Cache: TinyLFU-admitted LRU for read-only lookup + eviction
// - Allocator: PinnedMemoryPool for pinned memory allocation
//
// Eviction only targets the cache; inflight blocks are never evicted.
// ============================================================================
use bytesize::ByteSize;
use dashmap::DashMap;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::Duration;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, info};

use crate::cache::TinyLfuCache;
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

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub pre_evict_config: PreEvictConfig,
    pub enable_lfu_admission: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            pre_evict_config: PreEvictConfig::default(),
            enable_lfu_admission: true,
        }
    }
}

impl From<PreEvictConfig> for StorageConfig {
    fn from(pre_evict_config: PreEvictConfig) -> Self {
        Self {
            pre_evict_config,
            ..Default::default()
        }
    }
}

// A "slot" in this file refers to a specific position in the flattened logical storage,
// calculated as `layer_id * tp_size + tp_rank`.
// vLLM/Connectors report the total topology (layers * tp_size) via registration,
// and this count is immutable for the lifetime of the Instance.

/// Key for identifying blocks in storage, including namespace for model isolation.
///
/// NOTE: Using String for namespace is simple but adds ~20-50 bytes overhead per key.
/// Future optimization: intern namespaces to u32 IDs (saves memory, faster comparison).
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

// ============================================================================
// Inflight Block (write path, mutable)
// ============================================================================

/// Block that is still being written. Internal to StorageEngine.
struct InflightBlock {
    slots: Vec<Option<Arc<LayerBlock>>>,
    remaining: usize,
    total_slots: usize,
    footprint: u64,
}

impl InflightBlock {
    fn new(total_slots: usize) -> Self {
        Self {
            slots: vec![None; total_slots],
            remaining: total_slots,
            total_slots,
            footprint: 0,
        }
    }

    fn slot_exists(&self, slot_id: usize) -> bool {
        self.slots
            .get(slot_id)
            .and_then(|opt| opt.as_ref())
            .is_some()
    }

    /// Insert a slot. Returns Ok(true) if block is now complete.
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
            // Already filled - this is a no-op, not an error
            return Ok(false);
        }

        self.footprint += block.memory_footprint();
        self.slots[slot_id] = Some(block);
        self.remaining = self
            .remaining
            .checked_sub(1)
            .expect("remaining should not underflow");
        Ok(self.remaining == 0)
    }

    /// Seal the block, converting to immutable SealedBlock.
    /// Panics if not all slots are filled.
    fn seal(self) -> SealedBlock {
        let slots: Vec<Arc<LayerBlock>> = self
            .slots
            .into_iter()
            .map(|opt| opt.expect("all slots must be filled before sealing"))
            .collect();
        SealedBlock {
            slots: slots.into(),
            footprint: self.footprint,
        }
    }
}

// ============================================================================
// Sealed Block (read path, immutable)
// ============================================================================

/// Immutable block after all slots are filled. Exposed to callers.
pub struct SealedBlock {
    slots: Box<[Arc<LayerBlock>]>,
    footprint: u64,
}

impl SealedBlock {
    pub fn get_slot(&self, slot_id: usize) -> Option<&Arc<LayerBlock>> {
        self.slots.get(slot_id)
    }

    pub fn memory_footprint(&self) -> u64 {
        self.footprint
    }

    /// Get all slots (for serialization)
    pub fn slots(&self) -> &[Arc<LayerBlock>] {
        &self.slots
    }

    /// Create from a vec of slots (for deserialization)
    pub fn from_slots(slots: Vec<Arc<LayerBlock>>) -> Self {
        let footprint = slots.iter().map(|s| s.memory_footprint()).sum();
        Self {
            slots: slots.into_boxed_slice(),
            footprint,
        }
    }
}

// ============================================================================
// LayerBlock (pinned memory holder)
// ============================================================================

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

    /// Total pinned memory occupied by this layer block.
    pub fn memory_footprint(&self) -> u64 {
        self.size as u64
    }
}

// Safety: pinned memory ownership is tracked by Arc counters on the allocations.
unsafe impl Send for LayerBlock {}
unsafe impl Sync for LayerBlock {}

// ============================================================================
// StorageEngine
// ============================================================================

/// Notification sent when a block is sealed (for SSD offload, etc.)
pub type SealNotification = (BlockKey, Weak<SealedBlock>);

pub struct StorageEngine {
    /// Pinned memory allocator
    pinned_pool: Arc<PinnedMemoryPool>,

    /// Write path: blocks being filled (not yet sealed)
    inflight: DashMap<BlockKey, Mutex<InflightBlock>>,

    /// Read path: sealed blocks available for lookup (TinyLFU admission + LRU eviction)
    cache: Arc<Mutex<TinyLfuCache<BlockKey, Arc<SealedBlock>>>>,

    /// Pre-eviction control
    pre_evict_stop: Arc<AtomicBool>,
    pre_evict_handle: Option<JoinHandle<()>>,

    /// Channel to notify consumers when blocks are sealed (for SSD offload)
    seal_notify_tx: Option<UnboundedSender<SealNotification>>,
}

impl StorageEngine {
    /// Create a new StorageEngine with optional seal notification channel.
    /// Returns (engine, receiver) where receiver gets notified of sealed blocks.
    pub fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: impl Into<StorageConfig>,
    ) -> (Self, UnboundedReceiver<SealNotification>) {
        let config = config.into();
        let pinned_pool = Arc::new(PinnedMemoryPool::new(capacity_bytes, use_hugepages));
        let cache = Arc::new(Mutex::new(TinyLfuCache::new_unbounded(
            capacity_bytes,
            config.enable_lfu_admission,
        )));
        let inflight = DashMap::new();
        let pre_evict_stop = Arc::new(AtomicBool::new(false));

        // Create unbounded channel for seal notifications
        let (seal_notify_tx, seal_notify_rx) = mpsc::unbounded_channel();

        let pre_evict_config = config.pre_evict_config.clone();

        let pre_evict_handle = if pre_evict_config.enabled {
            info!(
                "Starting pre-eviction monitor: threshold={}, target={}, interval={}ms",
                ByteSize(pre_evict_config.threshold_bytes),
                ByteSize(pre_evict_config.target_bytes),
                pre_evict_config.check_interval_ms
            );

            let pool = Arc::clone(&pinned_pool);
            let cache_clone = Arc::clone(&cache);
            let stop = Arc::clone(&pre_evict_stop);

            Some(std::thread::spawn(move || {
                Self::pre_evict_monitor(pool, cache_clone, pre_evict_config, stop);
            }))
        } else {
            None
        };

        (
            Self {
                pinned_pool,
                inflight,
                cache,
                pre_evict_stop,
                pre_evict_handle,
                seal_notify_tx: Some(seal_notify_tx),
            },
            seal_notify_rx,
        )
    }

    // ========================================================================
    // Allocation
    // ========================================================================

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

    // ========================================================================
    // Write path (inflight)
    // ========================================================================

    /// Check if a slot exists in inflight blocks.
    pub fn inflight_has_slot(&self, namespace: &str, block_hash: &[u8], slot_id: usize) -> bool {
        let key = BlockKey::new(namespace.to_string(), block_hash.to_vec());
        if let Some(entry) = self.inflight.get(&key) {
            let block = entry.lock().unwrap();
            block.slot_exists(slot_id)
        } else {
            false
        }
    }

    /// Insert a slot into a block. Handles:
    /// - Skip if already in cache (sealed)
    /// - Skip if slot already exists in inflight
    /// - Create inflight block if needed
    /// - Auto-seal and migrate to cache when complete
    ///
    /// Returns Ok(true) if the slot was actually inserted, Ok(false) if skipped.
    pub fn insert_slot(
        &self,
        namespace: &str,
        block_hash: BlockHash,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<bool, BlockInsertError> {
        let key = BlockKey::new(namespace.to_string(), block_hash);

        // Fast path: already sealed in cache
        {
            let cache = self.cache.lock().unwrap();
            if cache.contains_key(&key) {
                return Ok(false);
            }
        }

        // Get or create inflight block, insert slot, then release DashMap shard lock
        let completed = {
            let entry = self
                .inflight
                .entry(key.clone())
                .or_insert_with(|| Mutex::new(InflightBlock::new(total_slots)));

            let mut inflight_block = entry.lock().unwrap();

            // Check if slot already exists
            if inflight_block.slot_exists(slot_id) {
                return Ok(false);
            }

            inflight_block.insert_slot(slot_id, block, total_slots)?
        }; // entry dropped here, releasing DashMap shard lock

        if completed {
            // Seal and migrate to cache
            // Remove from inflight first, then insert to cache
            if let Some((_, mutex)) = self.inflight.remove(&key) {
                let inflight_block = mutex.into_inner().unwrap();
                let sealed = Arc::new(inflight_block.seal());

                // Notify SSD offload consumer (fire-and-forget)
                if let Some(tx) = &self.seal_notify_tx {
                    let _ = tx.send((key.clone(), Arc::downgrade(&sealed)));
                }

                let mut cache = self.cache.lock().unwrap();
                if cache.insert(key, sealed) {
                    core_metrics().cache_block_insertions.add(1, &[]);
                } else {
                    core_metrics().cache_block_admission_rejections.add(1, &[]);
                }
            }
        }

        Ok(true)
    }

    // ========================================================================
    // Read path (cache)
    // ========================================================================

    /// Check if a block exists in the cache (sealed and complete).
    pub fn cache_contains(&self, namespace: &str, block_hash: &[u8]) -> bool {
        let key = BlockKey::new(namespace.to_string(), block_hash.to_vec());
        let cache = self.cache.lock().unwrap();
        cache.contains_key(&key)
    }

    /// Lookup multiple blocks from the cache.
    /// Returns error if any block is missing.
    pub fn cache_lookup_many(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        let mut cache = self.cache.lock().unwrap();
        let mut result = Vec::with_capacity(block_hashes.len());
        for (idx, hash) in block_hashes.iter().enumerate() {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let block = cache.get(&key).ok_or_else(|| {
                format!(
                    "missing KV block hash at index {idx} (namespace={namespace}, hash_len={})",
                    hash.len()
                )
            })?;
            result.push(block);
        }
        Ok(result)
    }

    /// Insert a pre-built SealedBlock directly into the cache.
    /// Used by prefetch to insert blocks loaded from DFS.
    pub fn cache_insert(&self, namespace: &str, block_hash: Vec<u8>, block: Arc<SealedBlock>) {
        let key = BlockKey::new(namespace.to_string(), block_hash);
        let mut cache = self.cache.lock().unwrap();
        if cache.insert(key, block) {
            core_metrics().cache_block_insertions.add(1, &[]);
        } else {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }

    // ========================================================================
    // Eviction (cache only)
    // ========================================================================

    fn reclaim(&self, target_objects: usize) -> usize {
        self.reclaim_from_cache_by_count(target_objects)
    }

    fn reclaim_from_cache_by_count(&self, target_objects: usize) -> usize {
        if target_objects == 0 {
            return 0;
        }

        let mut freed_entries = 0;
        let mut cache_lock = self.cache.lock().unwrap();

        while freed_entries < target_objects {
            let Some((_key, _block)) = cache_lock.remove_lru() else {
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

    fn reclaim_from_cache_by_bytes(
        cache: &Arc<Mutex<TinyLfuCache<BlockKey, Arc<SealedBlock>>>>,
        target_bytes: u64,
    ) -> (usize, u64) {
        if target_bytes == 0 {
            return (0, 0);
        }

        let mut freed_blocks = 0;
        let mut freed_bytes = 0u64;
        let mut cache_lock = cache.lock().unwrap();

        while freed_bytes < target_bytes {
            let Some((_key, block)) = cache_lock.remove_lru() else {
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
        cache: Arc<Mutex<TinyLfuCache<BlockKey, Arc<SealedBlock>>>>,
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

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum BlockInsertError {
    SlotOutOfBounds { slot_id: usize, total_slots: usize },
    SlotCountMismatch { expected: usize, got: usize },
}

impl fmt::Display for BlockInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            BlockInsertError::SlotCountMismatch { expected, got } => {
                write!(f, "slot count mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BlockInsertError {}
