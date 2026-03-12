// ============================================================================
// ReadCache: sealed-block cache with pin/unpin semantics.
//
// Owns the TinyLFU cache and the pinned_for_load bookkeeping. All pin
// state is encapsulated here; callers interact through `pin_blocks`,
// `consume_pinned_blocks` (transfers Arc ownership), and `unpin_blocks`.
// ============================================================================

use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use log::error;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::core_metrics;

// ============================================================================
// ReadCache
// ============================================================================

pub(crate) struct ReadCache {
    state: Arc<ReadCacheState>,
}

struct ReadCacheState {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    /// Sealed blocks available for lookup (TinyLFU admission + LRU eviction).
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    /// Pinned blocks between query and load (prevents eviction race).
    /// Key: (instance_id, block_key), Value: (block, ref_count)
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
    /// Aggregated refcounts by block key (for attribution metrics).
    /// Value: (footprint_bytes, total_refcount)
    pinned_for_load_by_key: HashMap<BlockKey, (u64, usize)>,
}

impl ReadCache {
    pub fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            state: Arc::new(ReadCacheState {
                inner: Mutex::new(ReadCacheInner {
                    cache,
                    pinned_for_load: HashMap::new(),
                    pinned_for_load_by_key: HashMap::new(),
                }),
            }),
        }
    }

    // ====================================================================
    // Query
    // ====================================================================

    /// Batch check: returns a `Vec<bool>` parallel to `keys`, `true` if present.
    pub fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let inner = self.state.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    /// Pure memory-only prefix check. Returns `(hit, missing)`.
    pub fn check_prefix_memory_only(&self, namespace: &str, hashes: &[Vec<u8>]) -> (usize, usize) {
        let mut hit = 0usize;
        {
            let mut inner = self.state.inner.lock();
            for hash in hashes {
                let key = BlockKey::new(namespace.to_string(), hash.clone());
                if inner.cache.get(&key).is_some() {
                    hit += 1;
                } else {
                    break;
                }
            }
        }
        (hit, hashes.len() - hit)
    }

    /// Scan cache for a prefix of `keys`. Returns (hit_count, blocks_found).
    ///
    /// Stops at the first miss. Blocks are cloned (Arc) for pinning.
    pub fn get_prefix_blocks(
        &self,
        keys: &[BlockKey],
    ) -> (usize, Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut hit = 0usize;
        let mut blocks = Vec::new();
        {
            let mut inner = self.state.inner.lock();
            for key in keys {
                if let Some(block) = inner.cache.get(key) {
                    hit += 1;
                    blocks.push((key.clone(), Arc::clone(&block)));
                } else {
                    break;
                }
            }
        }
        (hit, blocks)
    }

    // ====================================================================
    // Insert
    // ====================================================================

    /// Batch-insert blocks. Emits per-block admission metrics.
    pub fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut inner = self.state.inner.lock();
        for (key, block) in blocks {
            let footprint_bytes = block.memory_footprint();
            match inner.cache.insert(key, block) {
                CacheInsertOutcome::InsertedNew => {
                    let m = core_metrics();
                    m.cache_block_insertions.add(1, &[]);
                    m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
                }
                CacheInsertOutcome::AlreadyExists => {}
                CacheInsertOutcome::Rejected => {
                    core_metrics().cache_block_admission_rejections.add(1, &[]);
                }
            }
        }
    }

    // ====================================================================
    // Pin / Unpin
    // ====================================================================

    /// Pin blocks under a single lock acquisition.
    ///
    /// `num_workers` is the number of consumers that will each call
    /// `consume_pinned_blocks` once. The pin refcount is set accordingly.
    pub fn pin_blocks(
        &self,
        instance_id: &str,
        num_workers: usize,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) {
        let mut inner = self.state.inner.lock();
        let instance_id_owned = instance_id.to_string();
        for (key, block) in blocks {
            let pin_key = (instance_id_owned.clone(), key.clone());
            let footprint = block.memory_footprint();

            match inner.pinned_for_load.entry(pin_key) {
                Entry::Occupied(mut o) => {
                    o.get_mut().1 += num_workers;
                }
                Entry::Vacant(v) => {
                    v.insert((Arc::clone(block), num_workers));
                }
            }

            match inner.pinned_for_load_by_key.entry(key.clone()) {
                Entry::Occupied(mut o) => {
                    o.get_mut().1 += num_workers;
                }
                Entry::Vacant(v) => {
                    v.insert((footprint, num_workers));
                    core_metrics()
                        .pinned_for_load_unique_bytes
                        .add(footprint as i64, &[]);
                }
            }
        }
    }

    /// Consume pinned blocks, returning owned [`Arc<SealedBlock>`] handles.
    ///
    /// Consuming a block decrements exactly one pin refcount for the
    /// `(instance_id, block_key)` pair under the same lock that clones the Arc.
    /// This keeps reservation accounting atomic while avoiding per-block
    /// drop-time locking.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "storage.consume_pinned_blocks")
    )]
    pub fn consume_pinned_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();

        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .cloned()
            .map(|key| (instance_id.to_string(), key))
            .collect();
        let mut result = Vec::with_capacity(keys.len());

        let mut inner = self.state.inner.lock();

        for (idx, (key, pin_key)) in keys.iter().zip(pin_keys.iter()).enumerate() {
            if let Some((block, _count)) = inner.pinned_for_load.get(pin_key) {
                result.push(Arc::clone(block));
            } else {
                return Err(format!(
                    "missing pinned KV block at index {} (namespace={}, hash_len={})",
                    idx,
                    key.namespace.as_str(),
                    key.hash.len()
                ));
            }
        }

        for (key, pin_key) in keys.into_iter().zip(pin_keys.into_iter()) {
            if let Some((_, count)) = inner.pinned_for_load.get_mut(&pin_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    inner.pinned_for_load.remove(&pin_key);
                }
            }

            let mut unique_bytes_to_remove: Option<u64> = None;
            if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                *total = total.saturating_sub(1);
                if *total == 0 {
                    unique_bytes_to_remove = Some(*bytes);
                }
            } else {
                error!(
                    "BUG: pinned_for_load_by_key missing key during consume: namespace={} hash_len={}",
                    key.namespace,
                    key.hash.len()
                );
            }

            if let Some(bytes) = unique_bytes_to_remove {
                inner.pinned_for_load_by_key.remove(&key);
                core_metrics()
                    .pinned_for_load_unique_bytes
                    .add(-(bytes as i64), &[]);
            }
        }

        Ok(result)
    }

    /// Unpin blocks (cancellation path, before `consume_pinned_blocks`).
    ///
    /// Decrements the refcount by 1 per hash. Returns count of
    /// entries that were successfully unpinned.
    pub fn unpin_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> usize {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .map(|key| (instance_id.to_string(), key.clone()))
            .collect();

        let mut inner = self.state.inner.lock();
        let mut unpinned = 0usize;

        for (key, pin_key) in keys.into_iter().zip(pin_keys.into_iter()) {
            if let Some((_, count)) = inner.pinned_for_load.get_mut(&pin_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    inner.pinned_for_load.remove(&pin_key);
                }
                unpinned += 1;

                // Track unique block removal
                let mut unique_bytes_to_remove: Option<u64> = None;
                if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                    *total = total.saturating_sub(1);
                    if *total == 0 {
                        unique_bytes_to_remove = Some(*bytes);
                    }
                } else {
                    error!(
                        "BUG: pinned_for_load_by_key missing key during unpin: namespace={} hash_len={}",
                        key.namespace,
                        key.hash.len()
                    );
                }

                if let Some(bytes) = unique_bytes_to_remove {
                    inner.pinned_for_load_by_key.remove(&key);
                    core_metrics()
                        .pinned_for_load_unique_bytes
                        .add(-(bytes as i64), &[]);
                }
            }
        }

        unpinned
    }

    // ====================================================================
    // Eviction
    // ====================================================================

    /// Remove up to `batch_size` LRU entries. Returns evicted blocks.
    pub fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.state.inner.lock();
        (0..batch_size)
            .map_while(|_| inner.cache.remove_lru())
            .collect()
    }

    // ====================================================================
    // Test helpers
    // ====================================================================

    #[cfg(test)]
    pub fn pin_count(&self, instance_id: &str, key: &BlockKey) -> usize {
        let pin_key = (instance_id.to_string(), key.clone());
        self.state
            .inner
            .lock()
            .pinned_for_load
            .get(&pin_key)
            .map_or(0, |(_, count)| *count)
    }
}
