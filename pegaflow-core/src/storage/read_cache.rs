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

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(super) fn get_prefix_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut blocks = Vec::new();
        {
            let mut inner = self.inner.lock();
            for key in keys {
                if let Some(block) = inner.cache.get(key) {
                    blocks.push((key.clone(), Arc::clone(&block)));
                } else {
                    break;
                }
            }
        }
        blocks
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

    /// Look up specific blocks by key without prefix-scan semantics (does not
    /// stop at first miss). Used by the serving side of cross-node transfer.
    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut found = Vec::new();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
                found.push((key.clone(), Arc::clone(&block)));
            }
        }
        found
    }

    pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> ReadCache {
        ReadCache::new(1 << 20, false, None)
    }

    fn make_block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    #[test]
    fn get_blocks_returns_existing_skips_missing() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);
        let key3 = BlockKey::new("ns".into(), vec![3]);

        cache.batch_insert(vec![
            (key1.clone(), make_block()),
            (key3.clone(), make_block()),
        ]);

        // key2 is missing — get_blocks should skip it (unlike prefix scan, no break)
        let result = cache.get_blocks(&[key1.clone(), key2.clone(), key3.clone()]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, key1);
        assert_eq!(result[1].0, key3);
    }

    #[test]
    fn get_blocks_empty_input_returns_empty() {
        let cache = make_cache();
        let result = cache.get_blocks(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn get_blocks_all_missing_returns_empty() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![10]);
        let key2 = BlockKey::new("ns".into(), vec![20]);

        let result = cache.get_blocks(&[key1, key2]);
        assert!(result.is_empty());
    }

    #[test]
    fn get_blocks_is_idempotent() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        // Call get_blocks twice; both should return the same result
        let result1 = cache.get_blocks(std::slice::from_ref(&key));
        let result2 = cache.get_blocks(std::slice::from_ref(&key));
        assert_eq!(result1.len(), 1);
        assert_eq!(result2.len(), 1);
        assert_eq!(result1[0].0, result2[0].0);

        // Also ensure pin state is unaffected (no pins created)
        assert_eq!(cache.pin_count("any-instance", &key), 0);
    }

    #[test]
    fn get_blocks_does_not_break_at_first_miss() {
        // Contrast with get_prefix_blocks which stops at first miss
        let cache = make_cache();
        let keys: Vec<BlockKey> = (0u8..5)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();

        // Insert only even-indexed keys: 0, 2, 4
        for key in keys.iter().step_by(2) {
            cache.batch_insert(vec![(key.clone(), make_block())]);
        }

        // get_blocks: returns keys 0, 2, 4 (skips 1, 3)
        let result = cache.get_blocks(&keys);
        assert_eq!(result.len(), 3);

        // get_prefix_blocks: stops at key 1 (first miss), returns only key 0
        let prefix_hit = cache.get_prefix_blocks(&keys).len();
        assert_eq!(prefix_hit, 1);
    }
}
