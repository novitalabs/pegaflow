use std::{collections::HashMap, sync::Arc, time::Instant};

use hashlink::LruCache;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::{
    CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, CACHE_RESIDENCE_REASON_CLEANUP,
    CACHE_RESIDENCE_REASON_PRESSURE, core_metrics,
};

pub(crate) struct ReadCache {
    inner: Mutex<ReadCacheInner>,
}

struct ReadCacheInner {
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    reclaimable: LruCache<BlockKey, ResidentMetadata>,
    retained: LruCache<BlockKey, ResidentMetadata>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResidentMetadata {
    inserted_at: Instant,
}

struct RemovedResident {
    key: BlockKey,
    block: Arc<SealedBlock>,
    /// Insertion time taken from the block's replacement-class metadata.
    inserted_at: Instant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidentClass {
    Reclaimable,
    Retained,
}

impl ReadCache {
    pub(crate) fn new(
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
    ) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut resident_keys = Vec::new();
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
                resident_keys.push(key.clone());
            }
        }
        resident_keys
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

    /// Position-aligned membership: entry `i` is the block for `keys[i]`, or
    /// `None` on miss. Unlike [`Self::get_prefix_blocks`] this never stops at
    /// the first gap — hybrid-cache checkpoint groups (recurrent state) have
    /// sparse hit patterns by design, where the caller picks the rightmost
    /// hit instead of a prefix.
    pub(super) fn get_blocks_aligned(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let mut inner = self.inner.lock();
        keys.iter()
            .map(|key| {
                inner.cache.get(key).inspect(|_| {
                    refresh_recency(&mut inner, key);
                })
            })
            .collect()
    }

    pub(super) fn remove_lru_batch(&self, batch_size: usize) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let mut removed = Vec::with_capacity(batch_size);
            remove_lru_batch_from_class(
                &mut inner,
                ResidentClass::Reclaimable,
                batch_size,
                &mut removed,
            );
            if removed.len() < batch_size {
                remove_lru_batch_from_class(
                    &mut inner,
                    ResidentClass::Retained,
                    batch_size,
                    &mut removed,
                );
            }
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_PRESSURE)
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let removed = {
            let mut inner = self.inner.lock();
            let reclaimable_blocks = inner.reclaimable.len() as i64;
            let retained_blocks = inner.retained.len() as i64;
            let mut metadata = HashMap::with_capacity(
                inner.reclaimable.len().saturating_add(inner.retained.len()),
            );
            metadata.extend(inner.reclaimable.drain());
            metadata.extend(inner.retained.drain());
            let removed = inner
                .cache
                .remove_all()
                .into_iter()
                .map(|(key, block)| {
                    let inserted_at = metadata.remove(&key).map(|entry| entry.inserted_at);
                    debug_assert!(
                        inserted_at.is_some(),
                        "resident block is missing its replacement metadata"
                    );
                    RemovedResident {
                        inserted_at: inserted_at.unwrap_or_else(Instant::now),
                        key,
                        block,
                    }
                })
                .collect::<Vec<_>>();
            debug_assert_eq!(
                removed.len() as i64,
                reclaimable_blocks + retained_blocks,
                "resident cache and replacement classes diverged"
            );
            debug_assert!(
                metadata.is_empty(),
                "replacement metadata outlives its resident block"
            );
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-reclaimable_blocks, &*CACHE_CLASS_RECLAIMABLE);
            metrics
                .cache_resident_blocks
                .add(-retained_blocks, &*CACHE_CLASS_RETAINED);
            removed
        };
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_CLEANUP)
    }

    pub(crate) fn mark_reclaimable_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) {
        if hashes.is_empty() {
            return;
        }

        let mut inner = self.inner.lock();
        let mut moved = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if mark_reclaimable(&mut inner, &key) {
                moved += 1;
            }
        }
        if moved > 0 {
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-moved, &*CACHE_CLASS_RETAINED);
            metrics
                .cache_resident_blocks
                .add(moved, &*CACHE_CLASS_RECLAIMABLE);
        }
    }

    #[cfg(test)]
    pub(crate) fn insert_retained_for_test(&self, key: BlockKey, block: Arc<SealedBlock>) {
        let mut inner = self.inner.lock();
        insert_block(&mut inner, key, block, ResidentClass::Retained);
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        self.inner.lock().reclaimable.contains_key(key)
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
            class_lru(inner, class).insert(
                key,
                ResidentMetadata {
                    inserted_at: Instant::now(),
                },
            );
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

fn class_lru(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
) -> &mut LruCache<BlockKey, ResidentMetadata> {
    match class {
        ResidentClass::Reclaimable => &mut inner.reclaimable,
        ResidentClass::Retained => &mut inner.retained,
    }
}

fn refresh_recency(inner: &mut ReadCacheInner, key: &BlockKey) {
    let classified = inner.reclaimable.get(key).is_some() || inner.retained.get(key).is_some();
    debug_assert!(
        classified || !inner.cache.contains_key(key),
        "resident block is missing its replacement class"
    );
}

fn mark_reclaimable(inner: &mut ReadCacheInner, key: &BlockKey) -> bool {
    if !inner.cache.contains_key(key) {
        return false;
    }
    if let Some(metadata) = inner.retained.remove(key) {
        inner.reclaimable.insert(key.clone(), metadata);
        true
    } else {
        debug_assert!(
            inner.reclaimable.contains_key(key),
            "resident block is missing its replacement class"
        );
        false
    }
}

fn remove_lru(inner: &mut ReadCacheInner, class: ResidentClass) -> Option<RemovedResident> {
    while let Some((key, metadata)) = class_lru(inner, class).remove_lru() {
        let block = inner.cache.remove(&key);
        debug_assert!(
            block.is_some(),
            "replacement class contains a non-resident block"
        );
        let Some(block) = block else {
            continue;
        };
        let metrics = core_metrics();
        metrics.cache_resident_blocks.add(-1, class.attributes());
        metrics
            .cache_block_evictions_by_class
            .add(1, class.attributes());
        return Some(RemovedResident {
            key,
            block,
            inserted_at: metadata.inserted_at,
        });
    }
    None
}

fn remove_lru_batch_from_class(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    batch_size: usize,
    removed: &mut Vec<RemovedResident>,
) {
    let candidates = class_lru(inner, class).len();
    for _ in 0..candidates {
        if removed.len() == batch_size {
            break;
        }

        let Some(key) = class_lru(inner, class)
            .iter()
            .next()
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        if inner.cache.is_exclusively_owned(&key) {
            let block = remove_lru(inner, class)
                .expect("exclusive LRU candidate must remain resident while locked");
            removed.push(block);
        } else {
            class_lru(inner, class).get(&key);
        }
    }
}

fn record_residence_durations(
    removed: Vec<RemovedResident>,
    attributes: &[opentelemetry::KeyValue],
) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    let removed_at = Instant::now();
    let metrics = core_metrics();
    removed
        .into_iter()
        .map(|entry| {
            metrics.cache_residence_duration.record(
                residence_duration_seconds(entry.inserted_at, removed_at),
                attributes,
            );
            (entry.key, entry.block)
        })
        .collect()
}

fn residence_duration_seconds(inserted_at: Instant, removed_at: Instant) -> f64 {
    removed_at
        .saturating_duration_since(inserted_at)
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn make_cache() -> ReadCache {
        ReadCache::new(1 << 20, false, None)
    }

    fn make_block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    fn assert_class(cache: &ReadCache, key: &BlockKey, expected: ResidentClass) {
        let inner = cache.inner.lock();
        assert!(inner.cache.contains_key(key));
        assert_eq!(
            inner.reclaimable.contains_key(key),
            expected == ResidentClass::Reclaimable
        );
        assert_eq!(
            inner.retained.contains_key(key),
            expected == ResidentClass::Retained
        );
    }

    fn resident_metadata(cache: &ReadCache, key: &BlockKey) -> Option<ResidentMetadata> {
        let inner = cache.inner.lock();
        inner
            .reclaimable
            .peek(key)
            .or_else(|| inner.retained.peek(key))
            .copied()
    }

    fn backdate_resident(cache: &ReadCache, key: &BlockKey, age: Duration) -> Instant {
        let inserted_at = Instant::now() - age;
        let mut inner = cache.inner.lock();
        let metadata = if let Some(metadata) = inner.reclaimable.peek_mut(key) {
            metadata
        } else {
            inner
                .retained
                .peek_mut(key)
                .expect("test resident must have replacement metadata")
        };
        metadata.inserted_at = inserted_at;
        inserted_at
    }

    #[test]
    fn new_blocks_are_classified_by_source() {
        let cache = make_cache();
        let local = BlockKey::new("ns".into(), vec![1]);
        let ssd = BlockKey::new("ns".into(), vec![2]);
        let remote = BlockKey::new("ns".into(), vec![3]);
        let local_block = make_block();

        cache.batch_insert_refs(&[(local.clone(), local_block)]);
        cache.batch_insert(vec![(ssd.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(remote.clone(), make_block())]);

        assert_class(&cache, &local, ResidentClass::Retained);
        assert_class(&cache, &ssd, ResidentClass::Retained);
        assert_class(&cache, &remote, ResidentClass::Reclaimable);
    }

    #[test]
    fn reclaimable_blocks_are_evicted_before_retained_blocks() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(retained.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]);

        let evicted = cache.remove_lru_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![reclaimable, retained]
        );
    }

    #[test]
    fn pressure_reclaim_waits_for_weak_references() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert!(cache.remove_lru_batch(1).is_empty());
        assert_class(&cache, &key, ResidentClass::Retained);

        drop(weak);
        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
    }

    #[test]
    fn local_hit_refreshes_recency_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
        let (count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit));

        assert_eq!(count, 1);
        assert_eq!(
            resident_metadata(&cache, &hit).unwrap().inserted_at,
            inserted_at
        );
        assert_eq!(cache.remove_lru_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Reclaimable);
    }

    #[test]
    fn serving_hit_refreshes_recency_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = backdate_resident(&cache, &hit, Duration::from_secs(60));
        assert_eq!(cache.get_blocks(std::slice::from_ref(&hit)).len(), 1);

        assert_eq!(
            resident_metadata(&cache, &hit).unwrap().inserted_at,
            inserted_at
        );
        assert_eq!(cache.remove_lru_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Retained);
    }

    #[test]
    fn already_existing_insert_keeps_original_class() {
        let cache = make_cache();
        let remote_first = BlockKey::new("ns".into(), vec![1]);
        let remote_other = BlockKey::new("ns".into(), vec![2]);
        let local_first = BlockKey::new("ns".into(), vec![3]);
        let local_other = BlockKey::new("ns".into(), vec![4]);

        cache.batch_insert_resident_keys(vec![(remote_first.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(remote_other.clone(), make_block())]);
        cache.batch_insert(vec![(remote_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_first.clone(), make_block())]);
        cache.batch_insert(vec![(local_other.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(local_first.clone(), make_block())]);

        assert_class(&cache, &remote_first, ResidentClass::Reclaimable);
        assert_class(&cache, &local_first, ResidentClass::Retained);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_other);
        assert_eq!(cache.remove_lru_batch(1)[0].0, remote_first);
        assert_eq!(cache.remove_lru_batch(1)[0].0, local_other);
        assert_class(&cache, &local_first, ResidentClass::Retained);
    }

    #[test]
    fn already_existing_insert_preserves_residence_start() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.batch_insert_resident_keys(vec![(key.clone(), make_block())]);

        assert_eq!(
            resident_metadata(&cache, &key).unwrap().inserted_at,
            inserted_at
        );
        assert_class(&cache, &key, ResidentClass::Retained);
    }

    #[test]
    fn class_migration_preserves_residence_start() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.mark_reclaimable_hashes("ns", std::slice::from_ref(&key.hash));

        assert_eq!(
            resident_metadata(&cache, &key).unwrap().inserted_at,
            inserted_at
        );
        assert_class(&cache, &key, ResidentClass::Reclaimable);
    }

    #[test]
    fn reinsert_after_eviction_starts_new_residence_episode() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let first_inserted_at = backdate_resident(&cache, &key, Duration::from_secs(60));

        cache.remove_lru_batch(1);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        let second_inserted_at = resident_metadata(&cache, &key).unwrap().inserted_at;
        assert!(second_inserted_at > first_inserted_at);
    }

    #[test]
    fn residence_duration_is_non_negative_and_finite() {
        let removed_at = Instant::now();
        let inserted_at = removed_at - Duration::from_secs(60);

        assert_eq!(residence_duration_seconds(inserted_at, removed_at), 60.0);
        assert_eq!(residence_duration_seconds(removed_at, inserted_at), 0.0);
        assert!(residence_duration_seconds(inserted_at, removed_at).is_finite());
    }

    #[test]
    fn local_save_reports_resident_keys() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);

        assert_eq!(
            cache.batch_insert_refs(&[(key.clone(), make_block())]),
            vec![key.clone()]
        );
        assert_eq!(
            cache.batch_insert_refs(&[(key.clone(), make_block())]),
            vec![key]
        );
    }

    #[test]
    fn reclaimable_hashes_move_only_matching_residents() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);
        let other_namespace = BlockKey::new("other".into(), vec![1]);

        cache.batch_insert(vec![
            (retained.clone(), make_block()),
            (other_namespace.clone(), make_block()),
        ]);
        cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]);
        cache.mark_reclaimable_hashes("ns", &[vec![1], vec![2], vec![3]]);

        assert_class(&cache, &retained, ResidentClass::Reclaimable);
        assert_class(&cache, &reclaimable, ResidentClass::Reclaimable);
        assert_class(&cache, &other_namespace, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_hash_for_evicted_block_is_noop() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.remove_lru_batch(1);

        cache.mark_reclaimable_hashes("ns", &[key.hash]);

        assert!(cache.remove_lru_batch(1).is_empty());
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
        let inner = cache.inner.lock();
        assert!(inner.reclaimable.is_empty());
        assert!(inner.retained.is_empty());
        drop(inner);
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
        assert!(!cache.inner.lock().reclaimable.contains_key(&cold_key));
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
