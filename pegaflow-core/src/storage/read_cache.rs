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
    cache: TinyLfuCache,
    window: LruCache<BlockKey, WindowMetadata>,
    window_bytes: u64,
    main_bytes: u64,
    window_budget: Option<u64>,
    main_budget: u64,
    reclaimable: LruCache<BlockKey, ResidentMetadata>,
    retained: LruCache<BlockKey, ResidentMetadata>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ResidentMetadata {
    inserted_at: Instant,
    footprint: u64,
}

#[derive(Copy, Clone)]
struct WindowMetadata {
    resident: ResidentMetadata,
    class: ResidentClass,
}

#[derive(Default)]
pub(super) struct CacheInsertResult {
    pub resident_keys: Vec<BlockKey>,
    pub evicted_keys: Vec<BlockKey>,
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
        window_budget: Option<u64>,
    ) -> Self {
        let cache =
            TinyLfuCache::new_unbounded(capacity_bytes, enable_lfu_admission, value_size_hint);
        Self {
            inner: Mutex::new(ReadCacheInner {
                cache,
                window: LruCache::new_unbounded(),
                window_bytes: 0,
                main_bytes: 0,
                window_budget,
                main_budget: capacity_bytes as u64 - window_budget.unwrap_or(0),
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

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) -> Vec<BlockKey> {
        let mut inner = self.inner.lock();
        let mut evicted_keys = Vec::new();
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key,
                block,
                ResidentClass::Retained,
                &mut evicted_keys,
            );
        }
        evicted_keys
    }

    pub(super) fn batch_insert_resident_keys(
        &self,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> CacheInsertResult {
        let mut inner = self.inner.lock();
        let mut result = CacheInsertResult::default();
        let mut keys = Vec::with_capacity(blocks.len());
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key.clone(),
                block,
                ResidentClass::Reclaimable,
                &mut result.evicted_keys,
            );
            keys.push(key);
        }
        result.resident_keys = keys
            .into_iter()
            .filter(|key| inner.cache.contains_key(key))
            .collect();
        result
    }

    pub(super) fn batch_insert_refs(
        &self,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> CacheInsertResult {
        let mut inner = self.inner.lock();
        let mut result = CacheInsertResult::default();
        for (key, block) in blocks {
            insert_block(
                &mut inner,
                key.clone(),
                Arc::clone(block),
                ResidentClass::Retained,
                &mut result.evicted_keys,
            );
        }
        result.resident_keys = blocks
            .iter()
            .filter(|(key, _)| inner.cache.contains_key(key))
            .map(|(key, _)| key.clone())
            .collect();
        result
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
                remove_lru_batch_from_window(
                    &mut inner,
                    ResidentClass::Reclaimable,
                    batch_size,
                    &mut removed,
                );
            }
            if removed.len() < batch_size {
                remove_lru_batch_from_class(
                    &mut inner,
                    ResidentClass::Retained,
                    batch_size,
                    &mut removed,
                );
            }
            if removed.len() < batch_size {
                remove_lru_batch_from_window(
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
            let window_blocks = inner.window.len() as i64;
            let window_reclaimable = inner
                .window
                .iter()
                .filter(|(_, entry)| entry.class == ResidentClass::Reclaimable)
                .count() as i64;
            let mut metadata = HashMap::with_capacity(
                inner
                    .reclaimable
                    .len()
                    .saturating_add(inner.retained.len())
                    .saturating_add(inner.window.len()),
            );
            metadata.extend(inner.reclaimable.drain());
            metadata.extend(inner.retained.drain());
            metadata.extend(
                inner
                    .window
                    .drain()
                    .map(|(key, entry)| (key, entry.resident)),
            );
            inner.window_bytes = 0;
            inner.main_bytes = 0;
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
                reclaimable_blocks + retained_blocks + window_blocks,
                "resident cache and replacement classes diverged"
            );
            debug_assert!(
                metadata.is_empty(),
                "replacement metadata outlives its resident block"
            );
            let metrics = core_metrics();
            metrics.cache_resident_blocks.add(
                -(reclaimable_blocks + window_reclaimable),
                &*CACHE_CLASS_RECLAIMABLE,
            );
            metrics.cache_resident_blocks.add(
                -(retained_blocks + window_blocks - window_reclaimable),
                &*CACHE_CLASS_RETAINED,
            );
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
        insert_block(
            &mut inner,
            key,
            block,
            ResidentClass::Retained,
            &mut Vec::new(),
        );
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        let inner = self.inner.lock();
        inner.reclaimable.contains_key(key)
            || inner
                .window
                .peek(key)
                .is_some_and(|entry| entry.class == ResidentClass::Reclaimable)
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
    evicted_keys: &mut Vec<BlockKey>,
) -> CacheInsertOutcome {
    let footprint_bytes = block.memory_footprint();
    let outcome = inner.cache.insert(key.clone(), block);
    match outcome {
        CacheInsertOutcome::InsertedNew => {
            let resident = ResidentMetadata {
                inserted_at: Instant::now(),
                footprint: footprint_bytes,
            };
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(footprint_bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
            if inner.window_budget.is_some() {
                inner.window.insert(key, WindowMetadata { resident, class });
                inner.window_bytes = inner.window_bytes.saturating_add(footprint_bytes);
                overflow_window(inner, evicted_keys);
            } else {
                class_lru(inner, class).insert(key, resident);
                inner.main_bytes = inner.main_bytes.saturating_add(footprint_bytes);
            }
        }
        CacheInsertOutcome::AlreadyExists => refresh_recency(inner, &key),
    }
    outcome
}

fn overflow_window(inner: &mut ReadCacheInner, evicted_keys: &mut Vec<BlockKey>) {
    let Some(budget) = inner.window_budget else {
        return;
    };
    if inner.window_bytes <= budget || inner.window.len() <= 1 {
        return;
    }
    let Some((candidate, entry)) = inner.window.remove_lru() else {
        return;
    };
    inner.window_bytes = inner.window_bytes.saturating_sub(entry.resident.footprint);

    if inner.main_bytes.saturating_add(entry.resident.footprint) <= inner.main_budget {
        promote_to_main(inner, candidate, entry);
        return;
    }

    let victim = oldest_eligible(inner, ResidentClass::Reclaimable)
        .or_else(|| oldest_eligible(inner, ResidentClass::Retained));
    if let Some((victim, class)) = victim
        && inner.cache.admits(&candidate, &victim)
        && main_replacement_fits(inner, &victim, entry.resident.footprint)
    {
        if let Some(removed) = remove_main_key(inner, &victim, class) {
            record_policy_removal(removed, class, evicted_keys, false);
            promote_to_main(inner, candidate, entry);
        } else {
            reject_window_candidate(inner, candidate, entry, evicted_keys);
        }
    } else {
        reject_window_candidate(inner, candidate, entry, evicted_keys);
    }
}

fn main_replacement_fits(inner: &ReadCacheInner, victim: &BlockKey, candidate_bytes: u64) -> bool {
    let Some(metadata) = inner
        .reclaimable
        .peek(victim)
        .or_else(|| inner.retained.peek(victim))
    else {
        return false;
    };
    inner
        .main_bytes
        .saturating_sub(metadata.footprint)
        .saturating_add(candidate_bytes)
        <= inner.main_budget
}

fn reject_window_candidate(
    inner: &mut ReadCacheInner,
    candidate: BlockKey,
    entry: WindowMetadata,
    evicted_keys: &mut Vec<BlockKey>,
) {
    let block = inner.cache.remove(&candidate);
    debug_assert!(block.is_some(), "window candidate must be resident");
    if let Some(block) = block {
        record_policy_removal(
            RemovedResident {
                key: candidate,
                block,
                inserted_at: entry.resident.inserted_at,
            },
            entry.class,
            evicted_keys,
            true,
        );
    }
}

fn promote_to_main(inner: &mut ReadCacheInner, key: BlockKey, entry: WindowMetadata) {
    inner.main_bytes = inner.main_bytes.saturating_add(entry.resident.footprint);
    class_lru(inner, entry.class).insert(key, entry.resident);
}

fn oldest_eligible(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
) -> Option<(BlockKey, ResidentClass)> {
    let candidates = class_lru(inner, class).len();
    for _ in 0..candidates {
        let key = class_lru(inner, class)
            .iter()
            .next()
            .map(|(key, _)| key.clone())?;
        if inner.cache.is_cache_owned_only(&key) {
            return Some((key, class));
        }
        class_lru(inner, class).get(&key);
    }
    None
}

fn record_policy_removal(
    removed: RemovedResident,
    class: ResidentClass,
    evicted_keys: &mut Vec<BlockKey>,
    rejected: bool,
) {
    let metrics = core_metrics();
    let bytes = removed.block.memory_footprint();
    metrics.cache_resident_bytes.add(-(bytes as i64), &[]);
    metrics.cache_resident_blocks.add(-1, class.attributes());
    if rejected {
        metrics.cache_block_admission_rejections.add(1, &[]);
        return;
    }
    metrics
        .cache_block_evictions_by_class
        .add(1, class.attributes());
    metrics.cache_block_evictions.add(1, &[]);
    if Arc::strong_count(&removed.block) > 1 {
        metrics.cache_block_evictions_still_referenced.add(1, &[]);
    }
    metrics.cache_residence_duration.record(
        residence_duration_seconds(removed.inserted_at, Instant::now()),
        &*CACHE_RESIDENCE_REASON_PRESSURE,
    );
    evicted_keys.push(removed.key);
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
    let classified = inner.window.get(key).is_some()
        || inner.reclaimable.get(key).is_some()
        || inner.retained.get(key).is_some();
    debug_assert!(
        classified || !inner.cache.contains_key(key),
        "resident block is missing its replacement class"
    );
}

fn mark_reclaimable(inner: &mut ReadCacheInner, key: &BlockKey) -> bool {
    if !inner.cache.contains_key(key) {
        return false;
    }
    if let Some(entry) = inner.window.peek_mut(key) {
        if entry.class == ResidentClass::Retained {
            entry.class = ResidentClass::Reclaimable;
            return true;
        }
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
        inner.main_bytes = inner.main_bytes.saturating_sub(metadata.footprint);
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

fn remove_main_key(
    inner: &mut ReadCacheInner,
    key: &BlockKey,
    class: ResidentClass,
) -> Option<RemovedResident> {
    let metadata = class_lru(inner, class).remove(key)?;
    let block = inner.cache.remove(key)?;
    inner.main_bytes = inner.main_bytes.saturating_sub(metadata.footprint);
    Some(RemovedResident {
        key: key.clone(),
        block,
        inserted_at: metadata.inserted_at,
    })
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
        if inner.cache.is_cache_owned_only(&key) {
            let block = remove_lru(inner, class)
                .expect("cache-owned LRU candidate must remain resident while locked");
            removed.push(block);
        } else {
            class_lru(inner, class).get(&key);
        }
    }
}

fn remove_lru_batch_from_window(
    inner: &mut ReadCacheInner,
    class: ResidentClass,
    batch_size: usize,
    removed: &mut Vec<RemovedResident>,
) {
    let candidates = inner.window.len();
    for _ in 0..candidates {
        if removed.len() == batch_size {
            break;
        }
        let Some(key) = inner.window.iter().next().map(|(key, _)| key.clone()) else {
            break;
        };
        if inner
            .window
            .peek(&key)
            .is_some_and(|entry| entry.class != class)
        {
            inner.window.get(&key);
            continue;
        }
        if inner.cache.is_cache_owned_only(&key) {
            let Some((key, entry)) = inner.window.remove_lru() else {
                break;
            };
            let Some(block) = inner.cache.remove(&key) else {
                continue;
            };
            inner.window_bytes = inner.window_bytes.saturating_sub(entry.resident.footprint);
            let metrics = core_metrics();
            metrics
                .cache_resident_blocks
                .add(-1, entry.class.attributes());
            metrics
                .cache_block_evictions_by_class
                .add(1, entry.class.attributes());
            removed.push(RemovedResident {
                key,
                block,
                inserted_at: entry.resident.inserted_at,
            });
        } else {
            inner.window.get(&key);
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
        ReadCache::new(1 << 20, false, None, None)
    }

    fn make_lfu_cache() -> ReadCache {
        ReadCache::new(1 << 20, true, None, Some(1))
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
    fn pressure_reclaim_ignores_weak_references() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn pressure_reclaim_waits_for_external_strong_reference() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let external = Arc::clone(&block);
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert!(cache.remove_lru_batch(1).is_empty());
        assert_class(&cache, &key, ResidentClass::Retained);

        drop(external);
        assert_eq!(cache.remove_lru_batch(1)[0].0, key);
        assert!(weak.upgrade().is_none());
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
            cache
                .batch_insert_refs(&[(key.clone(), make_block())])
                .resident_keys,
            vec![key.clone()]
        );
        assert_eq!(
            cache
                .batch_insert_refs(&[(key.clone(), make_block())])
                .resident_keys,
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
        let cache = make_lfu_cache();
        let hot_key = BlockKey::new("ns".into(), vec![1]);
        let cold_key = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![(hot_key.clone(), make_block())]);
        {
            let mut inner = cache.inner.lock();
            let mut entry = inner.window.remove(&hot_key).expect("hot key in window");
            entry.resident.footprint = 1;
            inner.window_bytes = 0;
            inner.main_budget = 0;
            inner.main_bytes = 1;
            inner.reclaimable.insert(hot_key.clone(), entry.resident);
            inner.window_budget = Some(0);
            inner.window_bytes = 1;
        }

        for _ in 0..2 {
            assert_eq!(cache.get_blocks(std::slice::from_ref(&hot_key)).len(), 1);
        }

        let result = cache.batch_insert_resident_keys(vec![
            (cold_key.clone(), make_block()),
            (BlockKey::new("ns".into(), vec![3]), make_block()),
        ]);
        assert!(!result.resident_keys.contains(&cold_key));
        assert!(!cache.inner.lock().reclaimable.contains_key(&cold_key));
        assert_eq!(cache.get_blocks(&[hot_key]).len(), 1);
        assert_eq!(cache.get_blocks(&[cold_key]).len(), 0);
    }

    #[test]
    fn window_overflow_promotes_only_the_oldest_entry() {
        let cache = make_lfu_cache();
        let first = BlockKey::new("ns".into(), vec![1]);
        let second = BlockKey::new("ns".into(), vec![2]);
        let third = BlockKey::new("ns".into(), vec![3]);

        cache.batch_insert(vec![(first.clone(), make_block())]);
        {
            let mut inner = cache.inner.lock();
            inner.window_budget = Some(0);
            inner.window_bytes = 1;
        }
        cache.batch_insert(vec![(second.clone(), make_block())]);
        assert_class(&cache, &first, ResidentClass::Retained);
        assert!(!cache.inner.lock().retained.contains_key(&second));

        {
            let mut inner = cache.inner.lock();
            inner.window_bytes = 1;
        }
        cache.batch_insert(vec![(third.clone(), make_block())]);
        assert!(cache.inner.lock().window.contains_key(&third));
    }

    #[test]
    fn owner_hint_migrates_window_entry_before_promotion() {
        let cache = make_lfu_cache();
        let key = BlockKey::new("ns".into(), vec![1]);

        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.mark_reclaimable_hashes("ns", std::slice::from_ref(&key.hash));

        let inner = cache.inner.lock();
        assert_eq!(
            inner.window.peek(&key).expect("window entry").class,
            ResidentClass::Reclaimable
        );
    }

    #[test]
    fn batch_insert_resident_keys_includes_already_existing_blocks() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);

        assert_eq!(
            cache
                .batch_insert_resident_keys(vec![(key.clone(), make_block())])
                .resident_keys,
            vec![key]
        );
    }
}
