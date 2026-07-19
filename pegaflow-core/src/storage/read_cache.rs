use std::sync::Arc;

use hashlink::LruCache;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, core_metrics};

const MIN_RECLAIMABLE_OWNER_COUNT: u32 = 3;

pub(super) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    reclaimable: LruCache<BlockKey, ()>,
    retained: LruCache<BlockKey, ()>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum ResidentClass {
    Reclaimable,
    Retained,
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
                reclaimable: LruCache::new_unbounded(),
                retained: LruCache::new_unbounded(),
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
                    refresh_recency(&mut inner, key);
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
            insert_block(&mut inner, key, block, ResidentClass::Retained);
        }
    }

    pub(super) fn batch_insert_resident_keys(
        &self,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut resident_keys = Vec::new();
        for (key, block) in blocks {
            match insert_block(&mut inner, key.clone(), block, ResidentClass::Reclaimable) {
                CacheInsertOutcome::InsertedNew | CacheInsertOutcome::AlreadyExists => {
                    resident_keys.push(key);
                }
                CacheInsertOutcome::Rejected => {}
            }
        }
        resident_keys
    }

    pub(super) fn batch_insert_refs(
        &self,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> Vec<(BlockKey, CacheInsertOutcome)> {
        let mut inner = self.inner.lock();
        let mut resident = Vec::new();
        for (key, block) in blocks {
            let outcome = insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Retained,
            );
            if matches!(
                outcome,
                CacheInsertOutcome::InsertedNew | CacheInsertOutcome::AlreadyExists
            ) {
                resident.push((key.clone(), outcome));
            }
        }
        resident
    }

    /// Look up specific blocks by key without prefix-scan semantics (does not
    /// stop at first miss). Used by the serving side of cross-node transfer.
    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut inner = self.inner.lock();
        let mut found = Vec::new();
        for key in keys {
            if let Some(block) = inner.cache.get(key) {
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
                .reclaimable
                .remove_lru()
                .map(|(key, _)| (key, ResidentClass::Reclaimable))
                .or_else(|| {
                    inner
                        .retained
                        .remove_lru()
                        .map(|(key, _)| (key, ResidentClass::Retained))
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
        let reclaimable_blocks = inner.reclaimable.len() as i64;
        let retained_blocks = inner.retained.len() as i64;
        inner.reclaimable.clear();
        inner.retained.clear();
        let metrics = core_metrics();
        metrics
            .cache_resident_blocks
            .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
        metrics
            .cache_resident_blocks
            .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
        inner.cache.remove_all()
    }

    pub(super) fn mark_reclaimable(&self, keys: &[BlockKey]) {
        let mut inner = self.inner.lock();
        for key in keys {
            mark_reclaimable(&mut inner, key);
        }
    }

    pub(super) fn apply_owner_hints(
        &self,
        resident_saves: &[(BlockKey, CacheInsertOutcome)],
        owner_counts: &[u32],
    ) {
        let mut inner = self.inner.lock();
        for ((key, outcome), owner_count) in resident_saves.iter().zip(owner_counts) {
            if *outcome == CacheInsertOutcome::InsertedNew
                && *owner_count >= MIN_RECLAIMABLE_OWNER_COUNT
            {
                mark_reclaimable(&mut inner, key);
            }
        }
    }
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Reclaimable => &*CACHE_CLASS_RECLAIMABLE,
            Self::Retained => &*CACHE_CLASS_RETAINED,
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
                ResidentClass::Reclaimable => {
                    inner.reclaimable.insert(key, ());
                }
                ResidentClass::Retained => {
                    inner.retained.insert(key, ());
                }
            }
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
        }
        CacheInsertOutcome::AlreadyExists => refresh_recency(inner, &key),
        CacheInsertOutcome::Rejected => {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
        }
    }
    outcome
}

fn refresh_recency(inner: &mut ReadCacheInner, key: &BlockKey) {
    if inner.reclaimable.get(key).is_none() {
        inner.retained.get(key);
    }
}

fn mark_reclaimable(inner: &mut ReadCacheInner, key: &BlockKey) {
    if !inner.cache.contains_key(key) {
        return;
    }

    if inner.retained.remove(key).is_some() {
        inner.reclaimable.insert(key.clone(), ());
        let metrics = core_metrics();
        metrics
            .cache_resident_blocks
            .add(-1, &*CACHE_CLASS_RETAINED);
        metrics
            .cache_resident_blocks
            .add(1, &*CACHE_CLASS_RECLAIMABLE);
        metrics.cache_block_demotions.add(1, &[]);
    } else if inner.reclaimable.contains_key(key) {
        inner.reclaimable.get(key);
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
        assert!(inner.reclaimable.contains_key(&key));
        assert!(!inner.retained.contains_key(&key));
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
    fn reclaimable_blocks_are_evicted_before_retained_blocks() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(retained.clone(), make_block())]);
        assert_eq!(
            cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]),
            vec![reclaimable.clone()]
        );

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].0, reclaimable);
        assert!(cache.contains_keys(&[retained])[0]);
    }

    #[test]
    fn local_hit_refreshes_reclaimable_without_changing_class() {
        let cache = make_cache();
        let reclaimable_hit = BlockKey::new("ns".into(), vec![1]);
        let reclaimable_miss = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (reclaimable_hit.clone(), make_block()),
            (reclaimable_miss.clone(), make_block()),
        ]);
        let (hit, _) = cache.get_prefix_blocks(std::slice::from_ref(&reclaimable_hit));
        assert_eq!(hit, 1);

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted[0].0, reclaimable_miss);
        assert!(cache.contains_keys(std::slice::from_ref(&reclaimable_hit))[0]);
        let inner = cache.inner.lock();
        assert!(inner.reclaimable.contains_key(&reclaimable_hit));
        assert!(!inner.retained.contains_key(&reclaimable_hit));
    }

    #[test]
    fn rdma_fetch_of_existing_block_keeps_existing_class() {
        let cache = make_cache();
        let existing = BlockKey::new("ns".into(), vec![1]);
        let other = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![(existing.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(other.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(existing.clone(), make_block())]);

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted[0].0, other);
        assert!(cache.contains_keys(std::slice::from_ref(&existing))[0]);
        let inner = cache.inner.lock();
        assert!(inner.reclaimable.contains_key(&existing));
        assert!(!inner.retained.contains_key(&existing));
    }

    #[test]
    fn repeated_retained_insert_refreshes_lru_recency() {
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
    fn local_save_reports_existing_resident_for_owner_refresh() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);

        let first = cache.batch_insert_refs(&[(key.clone(), make_block())]);
        assert_eq!(first, vec![(key.clone(), CacheInsertOutcome::InsertedNew)]);

        let second = cache.batch_insert_refs(&[(key.clone(), make_block())]);
        assert_eq!(second, vec![(key, CacheInsertOutcome::AlreadyExists)]);
    }

    #[test]
    fn mark_reclaimable_moves_retained_block() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        cache.mark_reclaimable(std::slice::from_ref(&key));
        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
    }

    #[test]
    fn owner_hint_marks_only_third_local_owner_as_reclaimable() {
        let cache = make_cache();
        let second_owner = BlockKey::new("ns".into(), vec![1]);
        let third_owner = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(second_owner.clone(), make_block())]);
        cache.batch_insert(vec![(third_owner.clone(), make_block())]);
        cache.apply_owner_hints(
            &[
                (second_owner.clone(), CacheInsertOutcome::InsertedNew),
                (third_owner.clone(), CacheInsertOutcome::InsertedNew),
            ],
            &[2, MIN_RECLAIMABLE_OWNER_COUNT],
        );

        let evicted = cache.remove_lru_batch(1);
        assert_eq!(evicted[0].0, third_owner);
        assert!(cache.contains_keys(&[second_owner])[0]);
    }

    #[test]
    fn owner_hint_for_evicted_block_is_noop() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.remove_lru_batch(1);

        cache.apply_owner_hints(
            &[(key, CacheInsertOutcome::InsertedNew)],
            &[MIN_RECLAIMABLE_OWNER_COUNT],
        );

        assert!(cache.remove_lru_batch(1).is_empty());
    }

    #[test]
    fn owner_hint_does_not_demote_existing_local_save() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        cache.apply_owner_hints(
            &[(key.clone(), CacheInsertOutcome::AlreadyExists)],
            &[MIN_RECLAIMABLE_OWNER_COUNT],
        );

        let inner = cache.inner.lock();
        assert!(inner.retained.contains_key(&key));
        assert!(!inner.reclaimable.contains_key(&key));
    }

    #[test]
    fn already_existing_insert_does_not_change_class() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(retained.clone(), make_block())]);

        cache.batch_insert_resident_keys(vec![(retained.clone(), make_block())]);
        let inner = cache.inner.lock();
        assert!(inner.retained.contains_key(&retained));
        assert!(!inner.reclaimable.contains_key(&retained));
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
