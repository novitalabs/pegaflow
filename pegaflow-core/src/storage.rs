use hashlink::LruCache;
use std::num::NonZeroU64;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard};
use tracing::info;

use crate::pinned_pool::{PinnedAllocation, PinnedMemoryPool};

const RECLAIM_BATCH_OBJECTS: usize = 256;

// A "slot" in this file refers to a specific position in the flattened logical storage,
// calculated as `layer_id * tp_size + tp_rank`.
// vLLM/Connectors report the total topology (layers * tp_size) via registration,
// and this count is immutable for the lifetime of the Instance.
// NOTE: Storage is generic and operates on flat indices (slots).

pub type BlockHash = Vec<u8>;
type ShardBlocks = Vec<Option<Arc<Block>>>;

struct ShardBlocksState {
    blocks: ShardBlocks,
    is_complete: bool,
}

impl ShardBlocksState {
    fn new(num_slots: usize) -> Self {
        Self {
            blocks: vec![None; num_slots],
            is_complete: false,
        }
    }

    fn mark_slot_ready(&mut self, slot_id: usize, block: Arc<Block>) {
        assert!(
            slot_id < self.blocks.len(),
            "slot_id {} out of bounds ({} slots)",
            slot_id,
            self.blocks.len()
        );
        self.blocks[slot_id] = Some(block);
        self.is_complete = self.blocks.iter().all(|opt| opt.is_some());
    }
}

/// Wrapper for per-slot block vectors with a fixed weight for cache eviction.
pub struct ShardBlocksWithWeight {
    inner: Mutex<ShardBlocksState>,
}

impl ShardBlocksWithWeight {
    pub fn new(num_slots: usize) -> Self {
        Self {
            inner: Mutex::new(ShardBlocksState::new(num_slots)),
        }
    }

    pub fn lock_blocks(&self) -> ShardBlocksGuard<'_> {
        ShardBlocksGuard {
            inner: self.lock_state(),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.inner
            .lock()
            .expect("layer blocks lock poisoned")
            .is_complete
    }

    fn lock_state(&self) -> MutexGuard<'_, ShardBlocksState> {
        self.inner.lock().expect("shard blocks lock poisoned")
    }
}

pub struct ShardBlocksGuard<'a> {
    inner: MutexGuard<'a, ShardBlocksState>,
}

impl<'a> Deref for ShardBlocksGuard<'a> {
    type Target = ShardBlocks;

    fn deref(&self) -> &Self::Target {
        &self.inner.blocks
    }
}

impl<'a> DerefMut for ShardBlocksGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner.blocks
    }
}

impl<'a> ShardBlocksGuard<'a> {
    pub fn mark_slot_ready(&mut self, slot_id: usize, block: Arc<Block>) {
        self.inner.mark_slot_ready(slot_id, block);
    }
}

/// CPU block data stored in pinned memory.
pub struct Block {
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

impl Block {
    pub fn new_contiguous(ptr: *mut u8, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        Self {
            k_ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
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
        Self {
            k_ptr: unsafe { std::ptr::NonNull::new_unchecked(k_ptr) },
            v_ptr: Some(unsafe { std::ptr::NonNull::new_unchecked(v_ptr) }),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
        }
    }

    pub fn k_ptr(&self) -> *mut u8 {
        self.k_ptr.as_ptr()
    }

    pub fn v_ptr(&self) -> Option<*mut u8> {
        self.v_ptr.map(|ptr| ptr.as_ptr())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

// Safety: pinned memory ownership is tracked by Arc counters on the allocations.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

pub struct StorageEngine {
    kv_storage: Mutex<LruCache<BlockHash, Arc<ShardBlocksWithWeight>>>,
    pinned_pool: Arc<PinnedMemoryPool>,
}

impl StorageEngine {
    pub fn new(capacity_bytes: usize) -> Self {
        let pinned_pool = Arc::new(PinnedMemoryPool::new(capacity_bytes));
        // Use the pool capacity as the cache weight to keep eviction proportional to bytes.
        let cache_capacity = capacity_bytes.max(1);
        let kv_storage = Mutex::new(LruCache::new(cache_capacity));

        Self {
            kv_storage,
            pinned_pool,
        }
    }

    pub fn allocate(&self, size: NonZeroU64) -> Arc<PinnedAllocation> {
        loop {
            if let Some(allocation) = self.pinned_pool.allocate(size) {
                return Arc::new(allocation);
            }

            let reclaimed = self.reclaim(RECLAIM_BATCH_OBJECTS);
            if reclaimed > 0 {
                continue;
            } else {
                let (used, total) = self.pinned_pool.usage();
                panic!(
                    "Pinned memory pool exhausted! Requested: {:.2} MB, Used: {:.2} MB, Total: {:.2} MB, Cache empty",
                    size.get() as f64 / 1e6,
                    used as f64 / 1e6,
                    total as f64 / 1e6
                );
            }
        }
    }

    fn reclaim(&self, target_objects: usize) -> usize {
        if target_objects == 0 {
            return 0;
        }

        let mut freed_entries = 0;
        let mut cache = self.kv_storage.lock().unwrap();

        while freed_entries < target_objects {
            let Some((_hash, _layer_blocks)) = cache.remove_lru() else {
                break;
            };
            freed_entries += 1;
        }

        info!("Reclaimed {} blocks from cache", freed_entries);

        freed_entries
    }

    pub fn slot_has_block(&self, block_hash: &[u8], slot_id: usize) -> bool {
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(block_hash)
            .map(|blocks| {
                let blocks = blocks.lock_blocks();
                blocks.get(slot_id).and_then(|opt| opt.as_ref()).is_some()
            })
            .unwrap_or(false)
    }

    pub fn block_is_complete(&self, block_hash: &[u8]) -> bool {
        let mut cache = self.kv_storage.lock().unwrap();
        cache
            .get(block_hash)
            .map(|blocks| blocks.is_complete())
            .unwrap_or(false)
    }

    pub fn insert_block(
        &self,
        block_hash: BlockHash,
        slot_id: usize,
        block: Arc<Block>,
        total_slots: usize,
    ) {
        let mut cache = self.kv_storage.lock().unwrap();
        let entry = cache.get(&block_hash).cloned().unwrap_or_else(|| {
            let new_blocks = Arc::new(ShardBlocksWithWeight::new(total_slots));
            cache.insert(block_hash.clone(), Arc::clone(&new_blocks));
            new_blocks
        });
        let mut blocks = entry.lock_blocks();
        blocks.mark_slot_ready(slot_id, block);
    }

    pub fn lookup_many(
        &self,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<ShardBlocksWithWeight>>, String> {
        let mut cache = self.kv_storage.lock().unwrap();
        let mut result = Vec::with_capacity(block_hashes.len());
        for hash in block_hashes {
            let shard_blocks = cache
                .get(hash)
                .cloned()
                .ok_or_else(|| "Missing KV block hash".to_string())?;
            result.push(shard_blocks);
        }
        Ok(result)
    }
}
