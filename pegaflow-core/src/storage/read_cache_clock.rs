use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    time::Instant,
};

use crossbeam_epoch::{self as epoch, Guard};

use crate::block::{BlockKey, SealedBlock};
use crate::cache::{CacheInsertOutcome, TinyLfu};
use crate::metrics::{
    CACHE_CLASS_RECLAIMABLE, CACHE_CLASS_RETAINED, CACHE_RESIDENCE_REASON_CLEANUP,
    CACHE_RESIDENCE_REASON_PRESSURE, core_metrics,
};

use super::read_cache::{DEFAULT_BACKUP_RECLAIMABLE_PERCENT, DemotionReason};

#[path = "clock_resident.rs"]
mod resident;
use resident::Resident;

#[path = "resident_index.rs"]
mod resident_index;
use resident_index::ResidentIndex;

#[path = "clock_fifo.rs"]
mod fifo;
use fifo::{Fifo as ClockFifo, FifoEntry, ScanStep};

/// Resident bytes per replacement class.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct ClassBytes {
    reclaimable: u64,
    retained: u64,
}

const BACKUP_REPORT_USAGE_PERCENT: u64 = 90;
const BACKUP_SCAN_FACTOR: usize = 4;

pub(crate) struct ReadCache {
    index: ResidentIndex,
    reclaimable: ClockFifo,
    retained: ClockFifo,
    next_ticket: AtomicU64,
    // State transitions may finish their accounting out of order. Signed
    // deltas avoid underflow; budget snapshots clamp transient negatives.
    reclaimable_bytes: AtomicI64,
    retained_bytes: AtomicI64,
    pressured: AtomicBool,
    frequency: Option<TinyLfu>,
    capacity_bytes: u64,
    backup_reclaimable_percent: u64,
}

struct IndexedResident<'a> {
    resident: Arc<Resident>,
    index: &'a ResidentIndex,
}

impl IndexedResident<'_> {
    fn value(&self) -> &Arc<Resident> {
        &self.resident
    }

    fn remove(&self) -> bool {
        self.index.remove(&self.resident, &epoch::pin())
    }
}

struct RemovedResident {
    key: BlockKey,
    block: Arc<SealedBlock>,
    inserted_at: Instant,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ResidentClass {
    Reclaimable,
    Retained,
}

impl ResidentClass {
    fn attributes(self) -> &'static [opentelemetry::KeyValue] {
        match self {
            Self::Reclaimable => &*CACHE_CLASS_RECLAIMABLE,
            Self::Retained => &*CACHE_CLASS_RETAINED,
        }
    }
}

impl ReadCache {
    pub(crate) fn new(
        capacity_bytes: usize,
        enable_lfu_admission: bool,
        value_size_hint: Option<usize>,
    ) -> Self {
        let bytes = value_size_hint
            .filter(|size| *size > 0)
            .unwrap_or(128 * 1024 * 1024);
        Self {
            index: ResidentIndex::new(),
            reclaimable: ClockFifo::new((capacity_bytes / bytes).max(1)),
            retained: ClockFifo::new((capacity_bytes / bytes).max(1)),
            next_ticket: AtomicU64::new(0),
            reclaimable_bytes: AtomicI64::new(0),
            retained_bytes: AtomicI64::new(0),
            pressured: AtomicBool::new(false),
            frequency: enable_lfu_admission.then(|| TinyLfu::new((capacity_bytes / bytes).max(1))),
            capacity_bytes: capacity_bytes as u64,
            backup_reclaimable_percent: DEFAULT_BACKUP_RECLAIMABLE_PERCENT,
        }
    }

    pub(crate) fn with_backup_reclaimable_percent(mut self, percent: u64) -> Self {
        self.backup_reclaimable_percent = percent.min(100);
        self
    }

    fn ticket(&self) -> u64 {
        match self
            .next_ticket
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        {
            Ok(previous) => previous + 1,
            Err(_) => panic!("resident generation exhausted"),
        }
    }

    fn clock(&self, class: ResidentClass) -> &ClockFifo {
        match class {
            ResidentClass::Reclaimable => &self.reclaimable,
            ResidentClass::Retained => &self.retained,
        }
    }

    fn bytes(&self, class: ResidentClass) -> &AtomicI64 {
        match class {
            ResidentClass::Reclaimable => &self.reclaimable_bytes,
            ResidentClass::Retained => &self.retained_bytes,
        }
    }

    fn activate(&self, resident: &Resident) {
        if let Some(class) = resident.activate() {
            self.bytes(class)
                .fetch_add(resident.bytes as i64, Ordering::Relaxed);
            let m = core_metrics();
            m.cache_block_insertions.add(1, &[]);
            m.cache_resident_bytes.add(resident.bytes as i64, &[]);
            m.cache_resident_blocks.add(1, class.attributes());
            m.cache_resident_bytes_by_class
                .add(resident.bytes as i64, class.attributes());
        }
    }

    fn lookup(&self, key: &BlockKey) -> Option<IndexedResident<'_>> {
        let guard = &epoch::pin();
        let resident = self.index.get(key, guard)?;
        self.activate(resident);
        resident.is_live().then(|| IndexedResident {
            resident: Arc::clone(resident),
            index: &self.index,
        })
    }

    fn get(&self, key: &BlockKey, touch: bool, guard: &Guard) -> Option<Arc<SealedBlock>> {
        let resident = self.index.get(key, guard)?;
        self.activate(resident);
        let reader = resident.read(touch)?;
        let block = reader.clone_block();
        if touch && let Some(frequency) = &self.frequency {
            frequency.incr(key);
        }
        Some(block)
    }

    pub(super) fn contains_keys(&self, keys: &[BlockKey]) -> Vec<bool> {
        let guard = &epoch::pin();
        keys.iter()
            .map(|key| {
                self.index.get(key, guard).is_some_and(|resident| {
                    self.activate(resident);
                    resident.is_live()
                })
            })
            .collect()
    }

    pub(super) fn peek_blocks(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let guard = &epoch::pin();
        keys.iter().map(|key| self.get(key, false, guard)).collect()
    }

    pub(super) fn get_prefix_blocks(&self, keys: &[BlockKey]) -> (usize, Vec<Arc<SealedBlock>>) {
        let mut blocks = Vec::with_capacity(keys.len());
        let guard = &epoch::pin();
        for key in keys {
            let Some(block) = self.get(key, true, guard) else {
                break;
            };
            blocks.push(block);
        }
        (blocks.len(), blocks)
    }

    pub(super) fn get_blocks(&self, keys: &[BlockKey]) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let guard = &epoch::pin();
        keys.iter()
            .filter_map(|key| self.get(key, true, guard).map(|block| (key.clone(), block)))
            .collect()
    }

    pub(super) fn get_blocks_aligned(&self, keys: &[BlockKey]) -> Vec<Option<Arc<SealedBlock>>> {
        let guard = &epoch::pin();
        keys.iter().map(|key| self.get(key, true, guard)).collect()
    }

    fn admission_victim(&self) -> Option<BlockKey> {
        let guard = &epoch::pin();
        for class in [ResidentClass::Reclaimable, ResidentClass::Retained] {
            let clock = self.clock(class);
            // Observe the CLOCK order without changing bits or FIFO positions.
            for referenced in [false, true] {
                for entry in clock.iter(guard) {
                    let resident = entry.value();
                    if !resident.is_live()
                        || resident.class() != class
                        || resident.ticket(class).load(Ordering::Acquire) != entry.ticket()
                    {
                        continue;
                    }
                    let Some(reader) = resident.read(false) else {
                        continue;
                    };
                    if reader.is_referenced() == referenced && reader.cache_owned_only() {
                        return Some(resident.key.clone());
                    }
                }
            }
        }
        None
    }

    fn insert(
        &self,
        key: BlockKey,
        block: Arc<SealedBlock>,
        class: ResidentClass,
    ) -> CacheInsertOutcome {
        if let Some(frequency) = &self.frequency {
            frequency.incr(&key);
        }
        if let Some(entry) = self.lookup(&key) {
            if let Some(reader) = entry.value().read(true) {
                drop(reader);
                return CacheInsertOutcome::AlreadyExists;
            }
            entry.remove();
        }
        if let Some(frequency) = &self.frequency
            && let Some(victim) = self.admission_victim()
            && frequency.get(&key) < frequency.get(&victim)
        {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
            return CacheInsertOutcome::Rejected;
        }
        let resident = Arc::new(Resident::new(key, self.ticket(), block, class));
        // The FIFO publication precedes index arbitration. Pending FIFO
        // entries cannot be activated until the index confirms their identity.
        self.clock(class)
            .push(resident.generation, Arc::clone(&resident), &epoch::pin());
        self.finish_insert(&resident)
    }

    fn finish_insert(&self, resident: &Arc<Resident>) -> CacheInsertOutcome {
        let guard = &epoch::pin();
        loop {
            let indexed = self.index.get_or_insert(resident, guard);
            if Arc::ptr_eq(indexed, resident) {
                self.activate(resident);
                return CacheInsertOutcome::InsertedNew;
            }
            self.activate(indexed);
            if let Some(reader) = indexed.read(true) {
                drop(reader);
                resident.cancel_pending();
                return CacheInsertOutcome::AlreadyExists;
            }
            // Remove this exact retired node, never a newer same-key entry.
            self.index.remove(indexed, guard);
        }
    }

    pub(super) fn batch_insert(&self, blocks: Vec<(BlockKey, Arc<SealedBlock>)>) {
        for (key, block) in blocks {
            self.insert(key, block, ResidentClass::Retained);
        }
    }

    pub(super) fn batch_insert_refs(
        &self,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> Vec<BlockKey> {
        blocks
            .iter()
            .filter(|(key, block)| {
                self.insert(key.clone(), Arc::clone(block), ResidentClass::Retained)
                    != CacheInsertOutcome::Rejected
            })
            .map(|(key, _)| key.clone())
            .collect()
    }

    pub(super) fn resident_generations(&self, keys: &[BlockKey]) -> Vec<(BlockKey, u64)> {
        keys.iter()
            .filter_map(|key| {
                self.lookup(key)
                    .map(|entry| (key.clone(), entry.value().generation))
            })
            .collect()
    }

    fn prepare_candidate(&self, entry: &FifoEntry<'_>, class: ResidentClass) -> bool {
        let resident = entry.value();
        if !resident.is_live()
            && !resident.is_removed()
            && let Some(indexed) = self.index.get(&resident.key, entry.guard())
            && Arc::ptr_eq(indexed, resident)
        {
            self.activate(resident);
        }
        if resident.is_removed()
            || resident.ticket(class).load(Ordering::Acquire) != entry.ticket()
            || (class == ResidentClass::Retained && resident.class() == ResidentClass::Reclaimable)
        {
            entry.remove();
            return false;
        }
        // A retained resident in the reclaimable FIFO is a pending migration,
        // NOT a stale entry. Leave it until the class CAS or retirement.
        resident.is_live() && resident.class() == class
    }

    fn detach(&self, resident: &Arc<Resident>) {
        self.index.remove(resident, &epoch::pin());
    }

    fn prune_published(&self, entry: &FifoEntry<'_>, class: ResidentClass) {
        let resident = entry.value();
        if resident.is_removed()
            || resident.ticket(class).load(Ordering::Acquire) != entry.ticket()
            || (class == ResidentClass::Retained && resident.class() == ResidentClass::Reclaimable)
        {
            entry.remove();
        }
    }

    fn rotate(&self, entry: &FifoEntry<'_>, class: ResidentClass, reader: &resident::Reader<'_>) {
        self.rotate_with(entry, class, reader, |_| {});
    }

    fn rotate_with(
        &self,
        entry: &FifoEntry<'_>,
        class: ResidentClass,
        _reader: &resident::Reader<'_>,
        after_step: impl Fn(u8),
    ) {
        let resident = entry.value();
        if !resident.is_live() || resident.class() != class || !entry.remove() {
            return;
        }
        // Removing the exact FIFO node claims this rotation, not the resident:
        // readers continue concurrently. The held Reader prevents pressure
        // eviction while the record is temporarily outside this FIFO.
        after_step(0);
        let ticket = entry
            .ticket()
            .checked_add(1)
            .expect("resident position exhausted");
        if resident
            .ticket(class)
            .compare_exchange(entry.ticket(), ticket, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        after_step(1);
        let clock = self.clock(class);
        let published = clock.push(ticket, Arc::clone(resident), entry.guard());
        self.prune_published(&published, class);
    }

    fn removed(
        &self,
        resident: &Resident,
        class: ResidentClass,
        block: Arc<SealedBlock>,
    ) -> RemovedResident {
        self.bytes(class)
            .fetch_sub(resident.bytes as i64, Ordering::Relaxed);
        let m = core_metrics();
        m.cache_resident_blocks.add(-1, class.attributes());
        m.cache_resident_bytes_by_class
            .add(-(resident.bytes as i64), class.attributes());
        RemovedResident {
            key: resident.key.clone(),
            inserted_at: resident.inserted_at,
            block,
        }
    }

    pub(super) fn remove_clock_batch(
        &self,
        batch_size: usize,
    ) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let mut removed = Vec::new();
        let guard = &epoch::pin();
        for class in [ResidentClass::Reclaimable, ResidentClass::Retained] {
            let clock = self.clock(class);
            let mut cursor = clock.iter(guard);
            for _ in 0..clock.scan_budget() {
                if removed.len() == batch_size {
                    break;
                }
                let entry = match cursor.step() {
                    ScanStep::Entry(entry) => entry,
                    ScanStep::Skipped => continue,
                    ScanStep::End => break,
                };
                if !self.prepare_candidate(&entry, class) {
                    continue;
                }
                let resident = entry.value();
                let Some(reader) = resident.read(false) else {
                    continue;
                };
                if !reader.check_second_chance(class) {
                    self.rotate(&entry, class, &reader);
                    continue;
                }
                if let Some(block) = reader.evict_at(class, entry.ticket()) {
                    entry.remove();
                    self.detach(resident);
                    core_metrics()
                        .cache_block_evictions_by_class
                        .add(1, class.attributes());
                    removed.push(self.removed(resident, class, block));
                }
            }
        }
        if !removed.is_empty() {
            self.pressured.store(true, Ordering::Release);
        }
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_PRESSURE)
    }

    pub(super) fn remove_all(&self) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        let boundary = self.next_ticket.load(Ordering::Acquire);
        self.pressured.store(false, Ordering::Release);
        let mut removed = Vec::new();
        let guard = &epoch::pin();
        for resident in self.index.iter(guard) {
            if resident.generation > boundary {
                continue;
            }
            self.activate(resident);
            let Some(reader) = resident.read(false) else {
                continue;
            };
            if let Some((class, block)) = reader.cleanup() {
                self.detach(resident);
                removed.push(self.removed(resident, class, block));
            }
        }
        self.prune_queues(guard);
        record_residence_durations(removed, &*CACHE_RESIDENCE_REASON_CLEANUP)
    }

    fn prune_queues(&self, guard: &Guard) {
        for class in [ResidentClass::Reclaimable, ResidentClass::Retained] {
            for entry in self.clock(class).iter(guard) {
                self.prune_published(&entry, class);
            }
        }
    }

    fn demote(&self, key: &BlockKey, generation: Option<u64>) -> Option<u64> {
        let entry = self.lookup(key)?;
        let resident = entry.value();
        if generation.is_some_and(|expected| expected != resident.generation)
            || resident.class() == ResidentClass::Reclaimable
        {
            return None;
        }
        self.publish_demotion(resident);
        self.finish_demotion(resident)
    }

    fn publish_demotion(&self, resident: &Arc<Resident>) -> u64 {
        self.publish_demotion_with(resident, || {})
    }

    fn publish_demotion_with(
        &self,
        resident: &Arc<Resident>,
        before_publish: impl FnOnce(),
    ) -> u64 {
        let ticket = match resident.reclaim_ticket.load(Ordering::Acquire) {
            0 => match resident.reclaim_ticket.compare_exchange(
                0,
                1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => resident.reclaim_ticket.load(Ordering::Acquire),
                Err(actual) => actual,
            },
            ticket => ticket,
        };
        before_publish();
        // Helpers may publish duplicate positions. The position CAS claims
        // rotation once; retirement claims eviction once. Late old positions
        // are invalid and scans unlink them, never restoring their eligibility.
        let guard = &epoch::pin();
        let published = self.reclaimable.push(ticket, Arc::clone(resident), guard);
        self.prune_published(&published, ResidentClass::Reclaimable);
        ticket
    }

    fn finish_demotion(&self, resident: &Arc<Resident>) -> Option<u64> {
        let moved = resident.demote();
        if moved {
            self.retained_bytes
                .fetch_sub(resident.bytes as i64, Ordering::Relaxed);
            self.reclaimable_bytes
                .fetch_add(resident.bytes as i64, Ordering::Relaxed);
        }
        moved.then_some(resident.bytes)
    }

    pub(crate) fn mark_reclaimable_keys(&self, keys: &[BlockKey], reason: DemotionReason) -> usize {
        let mut moved = Demoted::default();
        for key in keys {
            moved.add(self.demote(key, None));
        }
        moved.record(reason)
    }

    pub(crate) fn mark_reclaimable_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        reason: DemotionReason,
    ) -> usize {
        let keys: Vec<_> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.into(), hash.clone()))
            .collect();
        self.mark_reclaimable_keys(&keys, reason)
    }

    pub(crate) fn mark_reclaimable_hashes_if_generation(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        generations: &HashMap<Vec<u8>, u64>,
    ) {
        let mut moved = Demoted::default();
        for hash in hashes {
            if let Some(&generation) = generations.get(hash) {
                moved.add(self.demote(
                    &BlockKey::new(namespace.into(), hash.clone()),
                    Some(generation),
                ));
            }
        }
        moved.record(DemotionReason::ReclaimHint);
    }

    fn class_bytes(&self) -> ClassBytes {
        ClassBytes {
            reclaimable: self.reclaimable_bytes.load(Ordering::Relaxed).max(0) as u64,
            retained: self.retained_bytes.load(Ordering::Relaxed).max(0) as u64,
        }
    }

    pub(crate) fn backup_report_budget(&self) -> u64 {
        backup_report_budget(
            self.capacity_bytes,
            self.backup_reclaimable_percent,
            self.class_bytes(),
            self.pressured.load(Ordering::Acquire),
        )
    }

    pub(crate) fn backup_candidates(
        &self,
        max_bytes: u64,
        max_blocks: usize,
    ) -> Vec<(BlockKey, u64)> {
        let mut candidates = Vec::new();
        if max_bytes == 0 || max_blocks == 0 {
            return candidates;
        }
        let clock = &self.retained;
        let guard = &epoch::pin();
        let mut cursor = clock.iter(guard);
        let mut total = 0u64;
        let mut seen = HashSet::new();
        for _ in 0..max_blocks
            .saturating_mul(BACKUP_SCAN_FACTOR)
            .min(clock.scan_budget())
        {
            let entry = match cursor.step() {
                ScanStep::Entry(entry) => entry,
                ScanStep::Skipped => continue,
                ScanStep::End => break,
            };
            let resident = entry.value();
            if !self.prepare_candidate(&entry, ResidentClass::Retained) {
                continue;
            }
            let Some(reader) = resident.read(false) else {
                continue;
            };
            if !reader.check_second_chance(ResidentClass::Retained) {
                self.rotate(&entry, ResidentClass::Retained, &reader);
                continue;
            }
            if seen.insert(resident.generation) {
                total = total.saturating_add(resident.bytes);
                candidates.push((resident.key.clone(), resident.bytes));
                if total >= max_bytes || candidates.len() >= max_blocks {
                    break;
                }
            }
        }
        candidates
    }

    #[cfg(test)]
    fn batch_insert_resident_keys(
        &self,
        blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> Vec<BlockKey> {
        blocks
            .into_iter()
            .filter_map(|(key, block)| {
                (self.insert(key.clone(), block, ResidentClass::Reclaimable)
                    != CacheInsertOutcome::Rejected)
                    .then_some(key)
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn insert_retained_for_test(&self, key: BlockKey, block: Arc<SealedBlock>) {
        self.insert(key, block, ResidentClass::Retained);
    }

    #[cfg(test)]
    pub(crate) fn resident_generation_for_test(&self, key: &BlockKey) -> u64 {
        self.resident_generations(std::slice::from_ref(key))[0].1
    }

    #[cfg(test)]
    pub(crate) fn evict_batch_for_test(&self, batch_size: usize) {
        drop(self.remove_clock_batch(batch_size));
    }

    #[cfg(test)]
    pub(crate) fn is_reclaimable_for_test(&self, key: &BlockKey) -> bool {
        self.lookup(key)
            .is_some_and(|entry| entry.value().class() == ResidentClass::Reclaimable)
    }
}
fn backup_report_budget(
    capacity_bytes: u64,
    reclaimable_percent: u64,
    bytes: ClassBytes,
    pressured: bool,
) -> u64 {
    let resident = bytes.reclaimable.saturating_add(bytes.retained);
    if !pressured
        && resident.saturating_mul(100) < capacity_bytes.saturating_mul(BACKUP_REPORT_USAGE_PERCENT)
    {
        return 0;
    }
    (capacity_bytes / 100 * reclaimable_percent).saturating_sub(bytes.reclaimable)
}

/// Blocks and bytes moved from retained to reclaimable in one batch.
#[derive(Default)]
struct Demoted {
    blocks: i64,
    bytes: i64,
}

impl Demoted {
    fn add(&mut self, bytes: Option<u64>) {
        if let Some(bytes) = bytes {
            self.blocks += 1;
            self.bytes += bytes as i64;
        }
    }

    /// Update the class metrics once per batch; returns the block count.
    fn record(self, reason: DemotionReason) -> usize {
        if self.blocks > 0 {
            let m = core_metrics();
            m.cache_class_demotions
                .add(self.blocks as u64, reason.attributes());
            m.cache_resident_blocks
                .add(-self.blocks, &*CACHE_CLASS_RETAINED);
            m.cache_resident_blocks
                .add(self.blocks, &*CACHE_CLASS_RECLAIMABLE);
            m.cache_resident_bytes_by_class
                .add(-self.bytes, &*CACHE_CLASS_RETAINED);
            m.cache_resident_bytes_by_class
                .add(self.bytes, &*CACHE_CLASS_RECLAIMABLE);
        }
        self.blocks as usize
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
#[path = "read_cache_bench.rs"]
mod bench;

#[cfg(test)]
#[path = "read_cache_concurrency.rs"]
mod concurrency_tests;

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

    #[derive(Clone, Copy)]
    struct ResidentMetadata {
        inserted_at: Instant,
        referenced: bool,
    }

    fn assert_class(cache: &ReadCache, key: &BlockKey, expected: ResidentClass) {
        let entry = cache.lookup(key).expect("resident must be indexed");
        assert_eq!(entry.value().class(), expected);
    }

    fn resident_metadata(cache: &ReadCache, key: &BlockKey) -> Option<ResidentMetadata> {
        cache.lookup(key).map(|entry| ResidentMetadata {
            inserted_at: entry.value().inserted_at,
            referenced: entry.value().referenced(),
        })
    }

    fn residence_start(cache: &ReadCache, key: &BlockKey) -> Instant {
        resident_metadata(cache, key)
            .expect("resident must be indexed")
            .inserted_at
    }

    #[tokio::test]
    async fn backup_candidates_follow_retained_tail_and_watermark() {
        use crate::block::{RawBlock, Segment};
        use crate::storage::{StorageConfig, StorageEngine};
        use std::num::NonZeroU64;

        const BLOCK: u64 = 100;
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]).unwrap();
        let sized_block = || {
            let alloc = engine
                .allocate(NonZeroU64::new(BLOCK).unwrap(), None)
                .expect("test pool should have space");
            let ptr = alloc.as_non_null();
            Arc::new(SealedBlock::from_slots(vec![(
                RawBlock::new(vec![Segment::new(ptr, BLOCK as usize, alloc)]),
                pegaflow_common::NumaNode::UNKNOWN,
            )]))
        };
        // Ten 100-byte blocks fill a 1000-byte cache.
        let cache = ReadCache::new(1000, false, None);
        let keys: Vec<BlockKey> = (0..10u8)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();
        let pinned = sized_block();
        for (i, key) in keys.iter().enumerate() {
            let block = if i == 1 {
                Arc::clone(&pinned)
            } else {
                sized_block()
            };
            cache.insert_retained_for_test(key.clone(), block);
        }
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 0,
                retained: 1000
            }
        );

        // Full and nothing reclaimable: ask for the 10% watermark.
        assert_eq!(cache.backup_report_budget(), 100);
        // Oldest first; the block pinned by a load or transfer is skipped.
        let hashes = |candidates: Vec<(BlockKey, u64)>| -> Vec<Vec<u8>> {
            candidates.into_iter().map(|(key, _)| key.hash).collect()
        };
        assert_eq!(
            hashes(cache.backup_candidates(250, usize::MAX)),
            [vec![0], vec![2], vec![3]]
        );
        assert_eq!(
            hashes(cache.backup_candidates(u64::MAX, 2)),
            [vec![0], vec![2]]
        );

        cache.mark_reclaimable_keys(&keys[..1], DemotionReason::TransferRelease);
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 100,
                retained: 900
            }
        );
        // Watermark met: nothing to report.
        assert_eq!(cache.backup_report_budget(), 0);

        drop(pinned);
        cache.evict_batch_for_test(2);
        assert_eq!(
            cache.class_bytes(),
            ClassBytes {
                reclaimable: 0,
                retained: 900 - BLOCK
            }
        );
        // Pressure eviction ran, so the cache counts as full below 90%.
        assert_eq!(cache.backup_report_budget(), 100);
        drop(cache.remove_all());
        assert_eq!(cache.class_bytes(), ClassBytes::default());
        assert_eq!(cache.backup_report_budget(), 0);
    }

    #[test]
    fn backup_candidates_advance_clock_without_removing_blocks() {
        let cache = make_cache();
        let keys: Vec<_> = (0..4)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();
        cache.batch_insert(keys.iter().map(|key| (key.clone(), make_block())).collect());
        drop(cache.get_blocks(&keys[..1]));
        let pinned = cache.get_blocks(&keys[1..2]);

        let candidates: Vec<_> = cache
            .backup_candidates(u64::MAX, 3)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(
            candidates,
            vec![keys[2].clone(), keys[3].clone(), keys[0].clone()]
        );
        assert!(!resident_metadata(&cache, &keys[0]).unwrap().referenced);
        assert!(resident_metadata(&cache, &keys[1]).unwrap().referenced);
        assert_eq!(cache.contains_keys(&keys), vec![true; 4]);
        let evicted: Vec<_> = cache
            .remove_clock_batch(3)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(evicted, candidates);
        drop(pinned);
    }

    #[test]
    fn backup_candidates_make_progress_across_bounded_scans() {
        let cache = make_cache();
        let keys: Vec<_> = (0..5)
            .map(|i| BlockKey::new("ns".into(), vec![i]))
            .collect();
        cache.batch_insert(keys.iter().map(|key| (key.clone(), make_block())).collect());
        drop(cache.get_blocks(&keys));

        assert!(cache.backup_candidates(0, 5).is_empty());
        assert!(cache.backup_candidates(u64::MAX, 0).is_empty());
        // A full scan cannot fit in one report. Keep the hand's progress so
        // the next report can finish the second-chance pass.
        assert!(cache.backup_candidates(u64::MAX, 1).is_empty());
        assert!(resident_metadata(&cache, &keys[4]).unwrap().referenced);
        let candidates: Vec<_> = cache
            .backup_candidates(u64::MAX, 1)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(candidates, vec![keys[0].clone()]);
        assert_eq!(cache.remove_clock_batch(1)[0].0, keys[0]);
    }

    #[test]
    fn second_chance_precedes_later_insert_in_fifo_order() {
        let cache = make_cache();
        let keys: Vec<_> = (0..6)
            .map(|id| BlockKey::new("ns".into(), vec![id]))
            .collect();
        cache.batch_insert(
            keys[..5]
                .iter()
                .map(|key| (key.clone(), make_block()))
                .collect(),
        );
        drop(cache.get_blocks(&keys[..5]));
        // A bounded report rotates the first four hot residents, leaving the
        // fifth at the head. A later insert must follow those four rotations.
        assert!(cache.backup_candidates(u64::MAX, 1).is_empty());
        cache.batch_insert(vec![(keys[5].clone(), make_block())]);
        let removed: Vec<_> = cache
            .remove_clock_batch(6)
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(
            removed,
            vec![
                keys[0].clone(),
                keys[1].clone(),
                keys[2].clone(),
                keys[3].clone(),
                keys[5].clone(),
                keys[4].clone()
            ]
        );
    }

    #[test]
    fn backup_clock_scan_bounds_pinned_entries_to_two_passes() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        cache.batch_insert_refs(&[(key.clone(), Arc::clone(&block))]);
        assert_eq!(cache.retained.scan_budget(), 2);
        assert!(cache.backup_candidates(u64::MAX, 65_536).is_empty());
        assert_eq!(cache.contains_keys(&[key]), vec![true]);
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
        cache.batch_insert_refs(&[(remote.clone(), make_block())]);

        assert_class(&cache, &local, ResidentClass::Retained);
        assert_class(&cache, &ssd, ResidentClass::Retained);
        assert_class(&cache, &remote, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_blocks_are_evicted_before_retained_blocks() {
        let cache = make_cache();
        let retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![(retained.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(reclaimable.clone(), make_block())]);

        let evicted = cache.remove_clock_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![reclaimable, retained]
        );
    }

    #[test]
    fn peeked_blocks_stay_out_of_pressure_eviction_while_held() {
        // A backup target re-registers blocks it already holds; holding the
        // peeked Arc until that insert is queued keeps eviction from queuing a
        // removal ahead of it.
        let cache = make_cache();
        let held = BlockKey::new("ns".into(), vec![1]);
        let other = BlockKey::new("ns".into(), vec![2]);
        let missing = BlockKey::new("ns".into(), vec![3]);
        cache.batch_insert(vec![
            (held.clone(), make_block()),
            (other.clone(), make_block()),
        ]);

        let peeked = cache.peek_blocks(&[held.clone(), missing]);
        assert!(peeked[0].is_some() && peeked[1].is_none());
        // `held` is the next victim, but the reference keeps it resident.
        let evicted = cache.remove_clock_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![other]
        );
        drop(peeked);
        assert_eq!(cache.remove_clock_batch(1)[0].0, held);

        // Peeking does not refresh recency.
        let first = BlockKey::new("ns".into(), vec![4]);
        let second = BlockKey::new("ns".into(), vec![5]);
        cache.batch_insert(vec![
            (first.clone(), make_block()),
            (second.clone(), make_block()),
        ]);
        drop(cache.peek_blocks(std::slice::from_ref(&first)));
        assert_eq!(cache.remove_clock_batch(1)[0].0, first);
    }

    #[test]
    fn stale_reclaimable_hint_does_not_demote_reinserted_generation() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.insert_retained_for_test(key.clone(), make_block());
        let old_generation = cache.resident_generations(std::slice::from_ref(&key))[0].1;

        drop(cache.remove_clock_batch(1));
        cache.insert_retained_for_test(key.clone(), make_block());
        let new_generation = cache.resident_generations(std::slice::from_ref(&key))[0].1;
        assert_ne!(old_generation, new_generation);

        let generations = HashMap::from([(key.hash.clone(), old_generation)]);
        cache.mark_reclaimable_hashes_if_generation(
            "ns",
            std::slice::from_ref(&key.hash),
            &generations,
        );

        assert_class(&cache, &key, ResidentClass::Retained);
    }

    #[test]
    fn pressure_reclaim_ignores_weak_references() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = make_block();
        let weak = Arc::downgrade(&block);
        cache.batch_insert(vec![(key.clone(), block)]);

        assert_eq!(cache.remove_clock_batch(1)[0].0, key);
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

        assert!(cache.remove_clock_batch(1).is_empty());
        assert_class(&cache, &key, ResidentClass::Retained);

        drop(external);
        assert_eq!(cache.remove_clock_batch(1)[0].0, key);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn local_hit_sets_reference_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert_resident_keys(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = residence_start(&cache, &hit);
        let (count, _) = cache.get_prefix_blocks(std::slice::from_ref(&hit));

        assert_eq!(count, 1);
        let metadata = resident_metadata(&cache, &hit).unwrap();
        assert_eq!(metadata.inserted_at, inserted_at);
        assert!(metadata.referenced);
        assert_eq!(cache.remove_clock_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Reclaimable);
    }

    #[test]
    fn serving_hit_sets_reference_without_changing_class() {
        let cache = make_cache();
        let hit = BlockKey::new("ns".into(), vec![1]);
        let oldest = BlockKey::new("ns".into(), vec![2]);

        cache.batch_insert(vec![
            (hit.clone(), make_block()),
            (oldest.clone(), make_block()),
        ]);
        let inserted_at = residence_start(&cache, &hit);
        assert_eq!(cache.get_blocks(std::slice::from_ref(&hit)).len(), 1);

        let metadata = resident_metadata(&cache, &hit).unwrap();
        assert_eq!(metadata.inserted_at, inserted_at);
        assert!(metadata.referenced);
        assert_eq!(cache.remove_clock_batch(1)[0].0, oldest);
        assert_class(&cache, &hit, ResidentClass::Retained);
    }

    #[test]
    fn clock_fifo_hit_order_does_not_replace_fifo_order_after_second_chance() {
        let cache = make_cache();
        let first = BlockKey::new("ns".into(), vec![1]);
        let second = BlockKey::new("ns".into(), vec![2]);
        let third = BlockKey::new("ns".into(), vec![3]);

        cache.batch_insert(vec![
            (first.clone(), make_block()),
            (second.clone(), make_block()),
            (third.clone(), make_block()),
        ]);
        assert_eq!(cache.get_blocks(std::slice::from_ref(&second)).len(), 1);
        assert_eq!(cache.get_blocks(std::slice::from_ref(&first)).len(), 1);

        let evicted = cache.remove_clock_batch(2);
        assert_eq!(
            evicted.into_iter().map(|(key, _)| key).collect::<Vec<_>>(),
            vec![third, first]
        );
        assert_class(&cache, &second, ResidentClass::Retained);
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
        assert_eq!(cache.remove_clock_batch(1)[0].0, remote_other);
        assert_eq!(cache.remove_clock_batch(1)[0].0, remote_first);
        assert_eq!(cache.remove_clock_batch(1)[0].0, local_other);
        assert_class(&cache, &local_first, ResidentClass::Retained);
    }

    #[test]
    fn already_existing_insert_preserves_residence_start() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        let inserted_at = residence_start(&cache, &key);

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
        let inserted_at = residence_start(&cache, &key);

        cache.mark_reclaimable_hashes(
            "ns",
            std::slice::from_ref(&key.hash),
            DemotionReason::ReclaimHint,
        );

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
        let first_inserted_at = residence_start(&cache, &key);

        cache.remove_clock_batch(1);
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
        cache.mark_reclaimable_hashes(
            "ns",
            &[vec![1], vec![2], vec![3]],
            DemotionReason::ReclaimHint,
        );

        assert_class(&cache, &retained, ResidentClass::Reclaimable);
        assert_class(&cache, &reclaimable, ResidentClass::Reclaimable);
        assert_class(&cache, &other_namespace, ResidentClass::Retained);
    }

    #[test]
    fn reclaimable_hash_for_evicted_block_is_noop() {
        let cache = make_cache();
        let key = BlockKey::new("ns".into(), vec![1]);
        cache.batch_insert(vec![(key.clone(), make_block())]);
        cache.remove_clock_batch(1);

        cache.mark_reclaimable_hashes("ns", &[key.hash], DemotionReason::ReclaimHint);

        assert!(cache.remove_clock_batch(1).is_empty());
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
        cache.prune_queues(&epoch::pin());
        assert!(cache.reclaimable.iter(&epoch::pin()).next().is_none());
        assert!(cache.retained.iter(&epoch::pin()).next().is_none());
        assert!(cache.remove_all().is_empty());
    }

    #[test]
    fn batch_insert_resident_keys_excludes_lfu_rejected_blocks() {
        let cache = ReadCache::new(1, true, Some(1));
        let hot_key = BlockKey::new("ns".into(), vec![1]);

        assert_eq!(
            cache.batch_insert_resident_keys(vec![(hot_key.clone(), make_block())]),
            vec![hot_key.clone()]
        );

        for _ in 0..2 {
            assert_eq!(cache.get_blocks(std::slice::from_ref(&hot_key)).len(), 1);
        }
        let frequency = cache.frequency.as_ref().unwrap();
        let cold_key = (2..=255)
            .map(|id| BlockKey::new("ns".into(), vec![id]))
            .find(|key| frequency.get(key) == 0)
            .expect("test needs a key not overestimated by the hot key's sketch counters");

        assert!(
            cache
                .batch_insert_resident_keys(vec![(cold_key.clone(), make_block())])
                .is_empty()
        );
        assert!(!cache.is_reclaimable_for_test(&cold_key));
        assert_eq!(cache.get_blocks(&[hot_key]).len(), 1);
        assert_eq!(cache.get_blocks(&[cold_key]).len(), 0);
    }

    #[test]
    fn lfu_admission_compares_candidate_with_next_clock_victim() {
        let cache = ReadCache::new(3, true, Some(1));
        let hot_retained = BlockKey::new("ns".into(), vec![1]);
        let reclaimable_victim = BlockKey::new("ns".into(), vec![2]);
        let candidate = BlockKey::new("ns".into(), vec![3]);

        cache.batch_insert(vec![(hot_retained.clone(), make_block())]);
        cache.batch_insert_resident_keys(vec![(reclaimable_victim.clone(), make_block())]);
        for _ in 0..2 {
            assert_eq!(
                cache.get_blocks(std::slice::from_ref(&hot_retained)).len(),
                1
            );
        }

        assert_eq!(cache.admission_victim(), Some(reclaimable_victim.clone()));
        cache.batch_insert(vec![(candidate.clone(), make_block())]);
        // Count-min sketches may overestimate due to random hash collisions.
        // Assert the actual admission contract, not collision-free estimates.
        let frequency = cache.frequency.as_ref().unwrap();
        let expected = frequency.get(&candidate) >= frequency.get(&reclaimable_victim);
        assert_eq!(cache.contains_keys(&[candidate]), vec![expected]);
        assert_eq!(cache.remove_clock_batch(1)[0].0, reclaimable_victim);
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
