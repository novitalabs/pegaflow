use std::sync::Arc;

use hashlink::LruCache;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{CACHE_CLASS_COLD, CACHE_CLASS_WARM, core_metrics};

pub(super) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    cold: LruCache<BlockKey, ()>,
    warm: LruCache<BlockKey, ()>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum ResidentClass {
    Cold,
    Warm,
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
                cold: LruCache::new_unbounded(),
                warm: LruCache::new_unbounded(),
            }),
        }
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let inner = self.inner.lock();
        keys.iter().map(|k| inner.cache.contains_key(k)).collect()
    }

    /// Scan cache for a prefix of `keys`, stopping at the first miss.
    pub(super) fn get_prefix_blocks(&self, keys: &[BlockKey]) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut hit = 0usize;
        let mut blocks = Vec::with_capacity(keys.len());
        {
            let mut inner = self.inner.lock();
            for key in keys {
                if let Some(block) = inner.cache.get(key) {
                    promote_on_local_hit(&mut inner, key);
                    hit += 1;
                    blocks.push(block);
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
            insert_block(&mut inner, key, block, ResidentClass::Warm);
        }
    }

    pub(super) fn batch_insert_resident_keys(
        &self,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut resident_keys = Vec::new();
        for (key, block) in blocks {
            match insert_block(&mut inner, key.clone(), block, ResidentClass::Cold) {
                CacheInsertOutcome::InsertedNew | CacheInsertOutcome::AlreadyExists => {
                    resident_keys.push(key);
                }
                CacheInsertOutcome::Rejected => {}
            }
        }
        resident_keys
    }

    pub(super) fn batch_insert_refs(&self, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Warm,
            );
        }
    }

    /// Look up specific blocks by key without prefix-scan semantics (does not
    /// stop at first miss). Used by the serving side of cross-node transfer.
    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut found = Vec::new();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
                let block = Arc::clone(block);
                refresh_recency(&mut inner, key);
                found.push((key.clone(), block));
            }
        }
        found
    }

    pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut evicted = Vec::with_capacity(batch_size);
        while evicted.len() < batch_size {
            let candidate = inner
                .cold
                .remove_lru()
                .map(|(key, _)| (key, ResidentClass::Cold))
                .or_else(|| {
                    inner
                        .warm
                        .remove_lru()
                        .map(|(key, _)| (key, ResidentClass::Warm))
                });
            let Some((key, class)) = candidate else {
                break;
            };
            if let Some(block) = inner.cache.remove(&key) {
                let attributes = class.attributes();
                let metrics = core_metrics();
                metrics.cache_resident_blocks.add(-1, attributes);
                metrics.cache_block_evictions_by_class.add(1, attributes);
                evicted.push((key, block));
            }
        }
        evicted
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let cold_blocks = inner.cold.len() as i64;
        let warm_blocks = inner.warm.len() as i64;
        inner.cold.clear();
        inner.warm.clear();
        let metrics = core_metrics();
        metrics
            .cache_resident_blocks
            .add(-cold_blocks, &*CACHE_CLASS_COLD);
        metrics
            .cache_resident_blocks
            .add(-warm_blocks, &*CACHE_CLASS_WARM);
        inner.cache.remove_all()
    }

    pub(super) fn demote(&self, keys: &[BlockKey]) {
        let mut inner = self.inner.lock();
        for key in keys {
            if inner.cache.contains_key(key) {
                demote(&mut inner, key);
            }
        }
    }
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Cold => &*CACHE_CLASS_COLD,
            Self::Warm => &*CACHE_CLASS_WARM,
        }
    }
}

fn insert_block(
    inner: &mut ReadCacheInner,
    key: BlockKey,
    block: Arc<SealedBlock>,
    class: ResidentClass,
) -> CacheInsertOutcome {
    let footprint_bytes = block.memory_footprint();
    let outcome = inner.cache.insert(key.clone(), block);
    match outcome {
        CacheInsertOutcome::InsertedNew => {
            match class {
                ResidentClass::Cold => {
                    inner.cold.insert(key, ());
                }
                ResidentClass::Warm => {
                    inner.warm.insert(key, ());
                }
            }
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
        }
        CacheInsertOutcome::AlreadyExists => {
            promote(&mut *inner, &key);
        }
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn promote_on_local_hit(inner: &mut ReadCacheInner, key: &BlockKey) {
    if inner.cold.contains_key(key) {
        promote(inner, key);
    } else if inner.warm.contains_key(key) {
        inner.warm.get(key);
    }
}

fn refresh_recency(inner: &mut ReadCacheInner, key: &BlockKey) {
    if inner.cold.contains_key(key) {
        inner.cold.get(key);
    } else if inner.warm.contains_key(key) {
        inner.warm.get(key);
    }
}

fn promote(inner: &mut ReadCacheInner, key: &BlockKey) {
    if inner.cold.remove(key).is_some() && inner.cache.contains_key(key) {
        inner.warm.insert(key.clone(), ());
        let metrics = core_metrics();
        metrics.cache_resident_blocks.add(-1, &*CACHE_CLASS_COLD);
        metrics.cache_resident_blocks.add(1, &*CACHE_CLASS_WARM);
        metrics.cache_block_promotions.add(1, &[]);
    } else if inner.warm.contains_key(key) {
        inner.warm.get(key);
    }
}

fn demote(inner: &mut ReadCacheInner, key: &BlockKey) {
    if inner.warm.remove(key).is_some() && inner.cache.contains_key(key) {
        inner.cold.insert(key.clone(), ());
        let metrics = core_metrics();
        metrics.cache_resident_blocks.add(-1, &*CACHE_CLASS_WARM);
        metrics.cache_resident_blocks.add(1, &*CACHE_CLASS_COLD);
        metrics.cache_block_demotions.add(1, &[]);
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
    }

    #[test]
    fn get_blocks_refreshes_recency_without_changing_class() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert_resident_keys(vec![(key.clone(), make_block())]);

        let _ = cache.get_blocks(std::slice::from_ref(&key));

        let inner = cache.inner.lock();
        assert!(inner.cold.contains_key(&key));
        assert!(!inner.warm.contains_key(&key));
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
        let (prefix_hit, _) = cache.get_prefix_blocks(&keys);
        assert_eq!(prefix_hit, 1);
    }

    #[test]
    fn cold_blocks_are_evicted_before_warm_blocks() {
        let cache = make_cache();
        let warm = BlockKey::new("ns".into(), vec![1]);
        let cold = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(warm.clone(), make_block())]);
        assert_eq!(
            cache.batch_insert_resident_keys(vec![(cold.clone(), make_block())]),
            vec![cold.clone()]
        );

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, cold);
        assert!(cache.contains_keys(&[warm])[0]);
    }

    #[test]
    fn local_hit_promotes_cold_block_to_warm() {
        let cache = make_cache();
        let cold_hit = BlockKey::new("ns".into(), vec![1]);
        let cold_miss = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (cold_hit.clone(), make_block()),
            (cold_miss.clone(), make_block()),
        ]);
        let (hit, _) = cache.get_prefix_blocks(std::slice::from_ref(&cold_hit));
        assert_eq!(hit, 1);

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted[0].0, cold_miss);
        assert!(cache.contains_keys(&[cold_hit])[0]);
    }

    #[test]
    fn rdma_fetch_of_existing_cold_block_promotes_to_warm() {
        let cache = make_cache();
        let existing = BlockKey::new("ns".into(), vec![1]);
        let other = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![(existing.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(other.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(existing.clone(), make_block())]);

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted[0].0, other);
        assert!(cache.contains_keys(&[existing])[0]);
    }

    #[test]
    fn repeated_warm_insert_refreshes_lru_recency() {
        let cache = make_cache();
        let first = BlockKey::new("ns".into(), vec![1]);
        let second = BlockKey::new("ns".into(), vec![2]);
        let first_block = make_block();

        cache.batch_insert(vec![(first.clone(), Arc::clone(&first_block))]);
        cache.batch_insert(vec![(second.clone(), make_block())]);
        cache.batch_insert(vec![(first, first_block)]);

        assert_eq!(cache.remove_lru_batch(1)[0].0, second);
    }

    #[test]
    fn demote_moves_warm_block_to_cold() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        cache.demote(std::slice::from_ref(&key));
        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
    }

    #[test]
    fn remove_all_evicts_resident_blocks() {
        let cache = make_cache();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (key1.clone(), make_block()),
            (key2.clone(), make_block()),
        ]);

        let removed = cache.remove_all();
        assert_eq!(removed.len(), 2);
        assert_eq!(cache.get_blocks(&[key1, key2]).len(), 0);
        assert!(cache.remove_all().is_empty());
    }

    #[test]
    fn batch_insert_resident_keys_excludes_lfu_rejected_blocks() {
        let cache = ReadCache::new(1, true, Some(1));
        let hot_key = BlockKey::new("ns".into(), vec![1]);
        let cold_key = BlockKey::new("ns".into(), vec![2]);

        assert_eq!(
            cache.batch_insert_resident_keys(vec![(hot_key.clone(), make_block())]),
            vec![hot_key.clone()]
        );

        for _ in 0..2 {
            assert_eq!(cache.get_blocks(std::slice::from_ref(&hot_key)).len(), 1);
        }

        assert!(
            cache
                .batch_insert_resident_keys(vec![(cold_key.clone(), make_block())])
                .is_empty()
        );
        assert_eq!(cache.get_blocks(&[hot_key]).len(), 1);
        assert_eq!(cache.get_blocks(&[cold_key]).len(), 0);
    }

    #[test]
    fn batch_insert_resident_keys_includes_already_existing_blocks() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        assert_eq!(
            cache.batch_insert_resident_keys(vec![(key.clone(), make_block())]),
            vec![key]
        );
    }
}
