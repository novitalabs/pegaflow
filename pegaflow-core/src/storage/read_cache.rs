use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use log::error;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::core_metrics;

pub(super) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    /// Key: (instance_id, block_key), Value: (block, ref_count)
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
    /// Aggregated refcounts by block key for attribution metrics.
    /// Value: (footprint_bytes, total_refcount)
    pinned_for_load_by_key: HashMap<BlockKey, (u64, usize)>,
}

impl ReadCacheInner {
    /// Decrement one pin reference for the given (instance, block) pair.
    ///
    /// Returns `true` if the pin existed and was decremented.
    fn decrement_pin(&mut self, pin_key: &(String, BlockKey)) -> bool {
        let Some((_, count)) = self.pinned_for_load.get_mut(pin_key) else {
            return false;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            self.pinned_for_load.remove(pin_key);
        }

        let block_key = &pin_key.1;
        if let Some((bytes, total)) = self.pinned_for_load_by_key.get_mut(block_key) {
            *total = total.saturating_sub(1);
            if *total == 0 {
                let bytes_val = *bytes;
                self.pinned_for_load_by_key.remove(block_key);
                core_metrics()
                    .pinned_for_load_unique_bytes
                    .add(-(bytes_val as i64), &[]);
            }
        } else {
            error!(
                "BUG: pinned_for_load_by_key missing key: namespace={} hash_len={}",
                block_key.namespace,
                block_key.hash.len()
            );
        }
        true
    }
}

impl ReadCache {
    pub(super) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            inner: Mutex::new(ReadCacheInner {
                cache,
                pinned_for_load: HashMap::new(),
                pinned_for_load_by_key: HashMap::new(),
            }),
        }
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    pub(super) fn check_prefix_memory_only(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> (usize, usize) {
        let mut hit = 0usize;
        {
            let mut inner = self.inner.lock();
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

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(super) fn get_prefix_blocks(
        &self,
        keys: &[BlockKey],
    ) -> (usize, Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut hit = 0usize;
        let mut blocks = Vec::new();
        {
            let mut inner = self.inner.lock();
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

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        let mut inner = self.inner.lock();
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

    /// Pin blocks for `num_workers` consumers that will each call `consume_pinned_blocks` once.
    pub(super) fn pin_blocks(
        &self,
        instance_id: &str,
        num_workers: usize,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) {
        let mut inner = self.inner.lock();
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
    /// Each call decrements exactly one pin refcount per block.
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "storage.consume_pinned_blocks")
    )]
    pub(super) fn consume_pinned_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        let instance_id_owned = instance_id.to_string();
        let namespace_owned = namespace.to_string();
        let pin_keys: Vec<(String, BlockKey)> = block_hashes
            .iter()
            .map(|hash| {
                (
                    instance_id_owned.clone(),
                    BlockKey::new(namespace_owned.clone(), hash.clone()),
                )
            })
            .collect();

        let mut result = Vec::with_capacity(pin_keys.len());
        let mut inner = self.inner.lock();

        // Phase 1: validate all pins exist and collect blocks
        for (idx, pin_key) in pin_keys.iter().enumerate() {
            if let Some((block, _)) = inner.pinned_for_load.get(pin_key) {
                result.push(Arc::clone(block));
            } else {
                return Err(format!(
                    "missing pinned KV block at index {} (namespace={}, hash_len={})",
                    idx,
                    pin_key.1.namespace.as_str(),
                    pin_key.1.hash.len()
                ));
            }
        }

        // Phase 2: decrement all pin refcounts
        for pin_key in &pin_keys {
            inner.decrement_pin(pin_key);
        }

        Ok(result)
    }

    /// Unpin blocks (cancellation path, before `consume_pinned_blocks`).
    ///
    /// Returns count of entries that were successfully unpinned.
    pub(super) fn unpin_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> usize {
        let instance_id_owned = instance_id.to_string();
        let namespace_owned = namespace.to_string();
        let pin_keys: Vec<(String, BlockKey)> = block_hashes
            .iter()
            .map(|hash| {
                (
                    instance_id_owned.clone(),
                    BlockKey::new(namespace_owned.clone(), hash.clone()),
                )
            })
            .collect();

        let mut inner = self.inner.lock();
        let mut unpinned = 0usize;

        for pin_key in &pin_keys {
            if inner.decrement_pin(pin_key) {
                unpinned += 1;
            }
        }

        unpinned
    }

    pub(super) fn remove_lru_batch(
        &self,
        batch_size: usize,
    ) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        (0..batch_size)
            .map_while(|_| inner.cache.remove_lru())
            .collect()
    }

    #[cfg(test)]
    pub(super) fn pin_count(&self, instance_id: &str, key: &BlockKey) -> usize {
        let pin_key = (instance_id.to_string(), key.clone());
        self.inner
            .lock()
            .pinned_for_load
            .get(&pin_key)
            .map_or(0, |(_, count)| *count)
    }
}
