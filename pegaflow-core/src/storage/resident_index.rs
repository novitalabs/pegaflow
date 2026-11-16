use std::sync::{Arc, atomic::Ordering};

use ahash::RandomState;
use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use smallvec::SmallVec;

use crate::block::BlockKey;

use super::resident::Resident;

#[derive(Clone)]
struct Item {
    hash: u64,
    resident: Arc<Resident>,
}

/// Fixed directory: directory size depends on neither `value_size_hint` nor the
/// configured capacity, so no hint can turn a bucket into a long list. At
/// 32 GiB this is about 0.25 slots per 1 MiB block (32768 residents) and eight
/// slots per 32 KiB block (about one million residents), versus 64 slots per
/// block in the old no-hint layout.
const TOP_BITS: u32 = 17;
const TOP_BUCKETS: usize = 1 << TOP_BITS;

struct Bucket(SmallVec<[Item; 1]>);

/// Immutable buckets make reads independent of writers. Only a successful
/// bucket CAS retires a snapshot; the caller's guard protects borrowed values.
pub(super) struct ResidentIndex {
    buckets: Box<[Atomic<Bucket>]>,
    hasher: RandomState,
}

impl ResidentIndex {
    pub(super) fn new() -> Self {
        Self::from_hasher(RandomState::new())
    }

    #[cfg(test)]
    pub(super) fn with_hasher(hasher: RandomState) -> Self {
        Self::from_hasher(hasher)
    }

    fn from_hasher(hasher: RandomState) -> Self {
        Self {
            buckets: (0..TOP_BUCKETS).map(|_| Atomic::null()).collect(),
            hasher,
        }
    }

    fn bucket(&self, hash: u64) -> &Atomic<Bucket> {
        &self.buckets[hash as usize & (TOP_BUCKETS - 1)]
    }

    pub(super) fn get<'g>(&'g self, key: &BlockKey, guard: &'g Guard) -> Option<&'g Arc<Resident>> {
        let hash = self.hasher.hash_one(key);
        let pointer = self.bucket(hash).load(Ordering::Acquire, guard);
        // SAFETY: published buckets are immutable and pinned by guard.
        unsafe { pointer.as_ref() }?
            .0
            .iter()
            .find(|item| item.hash == hash && item.resident.key == *key)
            .map(|item| &item.resident)
    }

    pub(super) fn get_or_insert<'g>(
        &'g self,
        resident: &Arc<Resident>,
        guard: &'g Guard,
    ) -> &'g Arc<Resident> {
        let hash = self.hasher.hash_one(&resident.key);
        let bucket = self.bucket(hash);
        loop {
            let old = bucket.load(Ordering::Acquire, guard);
            // SAFETY: the guard pins this immutable snapshot.
            let previous = unsafe { old.as_ref() };
            if let Some(item) = previous.and_then(|bucket| {
                bucket
                    .0
                    .iter()
                    .find(|item| item.hash == hash && item.resident.key == resident.key)
            }) {
                return &item.resident;
            }
            let mut items = previous.map(|bucket| bucket.0.clone()).unwrap_or_default();
            items.push(Item {
                hash,
                resident: Arc::clone(resident),
            });
            match bucket.compare_exchange(
                old,
                Owned::new(Bucket(items)),
                Ordering::AcqRel,
                Ordering::Acquire,
                guard,
            ) {
                Ok(new) => {
                    if !old.is_null() {
                        // SAFETY: this CAS uniquely unlinked the old snapshot.
                        unsafe { guard.defer_destroy(old) };
                    }
                    // SAFETY: the published snapshot is immutable and pinned.
                    return &unsafe { new.deref() }
                        .0
                        .last()
                        .expect("inserted item")
                        .resident;
                }
                Err(_) => continue,
            }
        }
    }

    pub(super) fn remove(&self, resident: &Arc<Resident>, guard: &Guard) -> bool {
        let hash = self.hasher.hash_one(&resident.key);
        let bucket = self.bucket(hash);
        loop {
            let old = bucket.load(Ordering::Acquire, guard);
            // SAFETY: the guard pins this immutable snapshot.
            let Some(previous) = (unsafe { old.as_ref() }) else {
                return false;
            };
            let Some(position) = previous
                .0
                .iter()
                .position(|item| Arc::ptr_eq(&item.resident, resident))
            else {
                return false;
            };
            let removed = if previous.0.len() == 1 {
                bucket
                    .compare_exchange(
                        old,
                        Shared::null(),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        guard,
                    )
                    .is_ok()
            } else {
                let mut items = previous.0.clone();
                items.remove(position);
                bucket
                    .compare_exchange(
                        old,
                        Owned::new(Bucket(items)),
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        guard,
                    )
                    .is_ok()
            };
            if removed {
                // SAFETY: only the successful updater retires this snapshot.
                unsafe { guard.defer_destroy(old) };
                return true;
            }
        }
    }

    pub(super) fn iter<'g>(&'g self, guard: &'g Guard) -> impl Iterator<Item = &'g Arc<Resident>> {
        self.buckets.iter().flat_map(move |bucket| {
            let pointer = bucket.load(Ordering::Acquire, guard);
            // SAFETY: each snapshot remains pinned for the iterator lifetime.
            unsafe { pointer.as_ref() }
                .into_iter()
                .flat_map(|bucket| bucket.0.iter().map(|item| &item.resident))
        })
    }

    #[cfg(test)]
    pub(super) fn stats(&self, guard: &Guard) -> (usize, usize, usize, u32) {
        let mut leaves = 0;
        let mut max_leaf = 0;
        for bucket in self.buckets.iter() {
            let pointer = bucket.load(Ordering::Acquire, guard);
            // SAFETY: the test guard pins every snapshot it visits.
            if let Some(bucket) = unsafe { pointer.as_ref() } {
                leaves += 1;
                max_leaf = max_leaf.max(bucket.0.len());
            }
        }
        (leaves, 0, max_leaf, TOP_BITS)
    }
}

impl Drop for ResidentIndex {
    fn drop(&mut self) {
        for bucket in &mut self.buckets {
            // SAFETY: exclusive access excludes any borrowed index values.
            let pointer = bucket.load(Ordering::Relaxed, unsafe { epoch::unprotected() });
            if !pointer.is_null() {
                // SAFETY: this is the last, still-owned snapshot in this slot.
                drop(unsafe { pointer.into_owned() });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::block::SealedBlock;

    fn key(id: u64) -> BlockKey {
        BlockKey::new("index".into(), id.to_le_bytes().to_vec())
    }

    fn resident(id: u64) -> Arc<Resident> {
        Arc::new(Resident::new(
            key(id),
            id,
            Arc::new(SealedBlock::from_slots(Vec::new())),
            super::super::ResidentClass::Retained,
        ))
    }

    fn fixed_hasher() -> RandomState {
        RandomState::with_seeds(0x1234, 0x5678, 0x9abc, 0xdef0)
    }

    fn fixed_index() -> ResidentIndex {
        ResidentIndex::with_hasher(fixed_hasher())
    }

    #[test]
    fn fresh_keys_reuse_a_fixed_directory() {
        let index = fixed_index();
        let guard = &epoch::pin();
        for generation in 0..4u64 {
            let residents: Vec<_> = (0..8_192u64)
                .map(|offset| resident(generation * 8_192 + offset))
                .collect();
            for resident in &residents {
                assert!(Arc::ptr_eq(index.get_or_insert(resident, guard), resident));
            }
            let (leaves, spans, _, _) = index.stats(guard);
            assert!(leaves > 0);
            assert_eq!(spans, 0, "the fixed directory must not publish spans");
            for resident in &residents {
                assert!(index.remove(resident, guard));
            }
            let (leaves, spans, max_leaf, _) = index.stats(guard);
            assert_eq!(leaves, 0, "removing every key must empty every bucket");
            assert_eq!(spans, 0);
            assert_eq!(max_leaf, 0);
        }
    }

    #[test]
    fn concurrent_same_key_inserts_publish_one_resident() {
        const THREADS: usize = 4;
        const KEYS: usize = 32;
        const ROUNDS: usize = 200;
        let index = Arc::new(fixed_index());
        let start = Barrier::new(THREADS);
        let observed: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..THREADS)
                .map(|worker| {
                    let index = &index;
                    let start = &start;
                    scope.spawn(move || {
                        let guard = &epoch::pin();
                        let mut observed = vec![0usize; KEYS];
                        start.wait();
                        for round in 0..ROUNDS {
                            for offset in 0..KEYS {
                                let id = ((offset + worker * 7 + round) % KEYS) as u64;
                                let candidate = resident(id);
                                let published = index.get_or_insert(&candidate, guard);
                                observed[id as usize] = Arc::as_ptr(published) as usize;
                            }
                        }
                        observed
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect()
        });
        for pair in observed.windows(2) {
            assert_eq!(
                pair[0], pair[1],
                "every thread must observe the single published resident per key"
            );
        }
        let guard = &epoch::pin();
        assert_eq!(index.iter(guard).count(), KEYS);
        for id in 0..KEYS as u64 {
            let published = index.get(&key(id), guard).expect("key must be present");
            assert_eq!(Arc::as_ptr(published) as usize, observed[0][id as usize]);
        }

        // Concurrent removal of the same key has exactly one successful owner.
        let winner = index
            .get(&key(0), guard)
            .expect("key must be present")
            .clone();
        let removed = AtomicUsize::new(0);
        let start = Barrier::new(THREADS);
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let index = &index;
                let winner = &winner;
                let removed = &removed;
                let start = &start;
                scope.spawn(move || {
                    let guard = &epoch::pin();
                    start.wait();
                    if index.remove(winner, guard) {
                        removed.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(removed.load(Ordering::Relaxed), 1);
        assert!(index.get(&key(0), guard).is_none());
        assert_eq!(index.iter(guard).count(), KEYS - 1);
    }

    #[test]
    fn concurrent_insert_and_remove_keep_index_consistent() {
        const BASE: u64 = 16_384;
        const THREADS: u64 = 4;
        const ADD_PER_THREAD: u64 = 4_096;
        const BASE_PER_THREAD: u64 = BASE / THREADS;

        let index = Arc::new(fixed_index());
        let base: Vec<_> = (0..BASE).map(resident).collect();
        let guard = &epoch::pin();
        for resident in &base {
            assert!(Arc::ptr_eq(index.get_or_insert(resident, guard), resident));
        }

        // Writers add new keys into the same buckets that removers retire, so
        // insertion and removal CAS the same slots concurrently.
        let start = Barrier::new(THREADS as usize * 2);
        std::thread::scope(|scope| {
            for worker in 0..THREADS {
                let index = &index;
                let start = &start;
                scope.spawn(move || {
                    let guard = &epoch::pin();
                    let first = BASE + worker * ADD_PER_THREAD;
                    start.wait();
                    for offset in 0..ADD_PER_THREAD {
                        let resident = resident(first + offset);
                        assert!(Arc::ptr_eq(
                            index.get_or_insert(&resident, guard),
                            &resident
                        ));
                    }
                });
            }
            for worker in 0..THREADS {
                let index = &index;
                let base = &base;
                let start = &start;
                scope.spawn(move || {
                    let guard = &epoch::pin();
                    let first = worker * BASE_PER_THREAD;
                    start.wait();
                    for offset in (0..BASE_PER_THREAD).step_by(2) {
                        assert!(index.remove(&base[(first + offset) as usize], guard));
                    }
                });
            }
        });
        let guard = &epoch::pin();
        let expected = (BASE - BASE / 2 + THREADS * ADD_PER_THREAD) as usize;
        let mut seen = HashSet::new();
        for resident in index.iter(guard) {
            assert!(
                seen.insert(resident.key.clone()),
                "duplicate key in index: {:?}",
                resident.key
            );
        }
        assert_eq!(seen.len(), expected);
        for worker in 0..THREADS {
            let first = worker * BASE_PER_THREAD;
            for offset in 0..BASE_PER_THREAD {
                let id = first + offset;
                assert_eq!(
                    index.get(&key(id), guard).is_some(),
                    offset % 2 == 1,
                    "pre-existing key {id}"
                );
            }
            let first = BASE + worker * ADD_PER_THREAD;
            for offset in 0..ADD_PER_THREAD {
                let id = first + offset;
                assert!(index.get(&key(id), guard).is_some(), "new key {id}");
            }
        }
        assert_eq!(index.stats(guard).1, 0, "the fixed directory has no spans");
    }
}
