use std::{
    cell::UnsafeCell,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use crate::block::{BlockKey, SealedBlock};

use super::ResidentClass;

const PENDING: u64 = 1;
const REMOVED: u64 = 2;
const REFERENCED: u64 = 4;
const RECLAIMABLE: u64 = 8;
const READER: u64 = 16;
const READERS: u64 = 0xfff0;
const VERSION: u64 = 1 << 16;

/// One state word makes the reference-count sample and retirement CAS immune
/// to readers that enter, clone the block, and leave between those operations.
#[derive(Clone, Copy)]
struct ResidentState(u64);

impl ResidentState {
    fn class(self) -> ResidentClass {
        if self.0 & RECLAIMABLE != 0 {
            ResidentClass::Reclaimable
        } else {
            ResidentClass::Retained
        }
    }

    fn live(self) -> bool {
        self.0 & (PENDING | REMOVED) == 0
    }
    fn readers(self) -> u64 {
        self.0 & READERS
    }

    fn enter(self, touch: bool) -> Self {
        // Exhaustion must never wrap the version or carry into adjacent bits.
        assert!(self.readers() < READERS, "resident reader count exhausted");
        assert!(
            self.0 < u64::MAX - VERSION - READER,
            "resident version exhausted"
        );
        Self((self.0 + VERSION + READER) | if touch { REFERENCED } else { 0 })
    }
}

pub(super) struct Resident {
    pub(super) key: BlockKey,
    pub(super) generation: u64,
    pub(super) inserted_at: Instant,
    pub(super) bytes: u64,
    pub(super) retained_ticket: AtomicU64,
    pub(super) reclaim_ticket: AtomicU64,
    state: AtomicU64,
    block: UnsafeCell<Option<Arc<SealedBlock>>>,
}

// SAFETY: the slot is initialized before publication. Only active Reader
// guards read it. Retirement prevents new readers; only the last reader of a
// retired resident, or the successful sole-reader eviction CAS, may take it.
// AcqRel state RMWs form the release sequence joining all reader exits.
unsafe impl Sync for Resident {}

impl Resident {
    #[cfg(target_arch = "x86_64")]
    pub(super) fn prefetch_for_scan(&self) {
        // SAFETY: both hints point inside this live resident. They neither
        // read the payload slot nor replace the scan's synchronized loads.
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                std::ptr::from_ref(self).cast(),
                std::arch::x86_64::_MM_HINT_T0,
            );
            std::arch::x86_64::_mm_prefetch(
                std::ptr::from_ref(&self.state).cast(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }

    pub(super) fn new(
        key: BlockKey,
        generation: u64,
        block: Arc<SealedBlock>,
        class: ResidentClass,
    ) -> Self {
        Self {
            key,
            generation,
            inserted_at: Instant::now(),
            bytes: block.memory_footprint(),
            retained_ticket: AtomicU64::new(if class == ResidentClass::Retained {
                generation
            } else {
                0
            }),
            reclaim_ticket: AtomicU64::new(if class == ResidentClass::Reclaimable {
                generation
            } else {
                0
            }),
            state: AtomicU64::new(
                PENDING
                    | if class == ResidentClass::Reclaimable {
                        RECLAIMABLE
                    } else {
                        0
                    },
            ),
            block: UnsafeCell::new(Some(block)),
        }
    }

    pub(super) fn ticket(&self, class: ResidentClass) -> &AtomicU64 {
        match class {
            ResidentClass::Retained => &self.retained_ticket,
            ResidentClass::Reclaimable => &self.reclaim_ticket,
        }
    }

    pub(super) fn activate(&self) -> Option<ResidentClass> {
        let state = self.state.load(Ordering::Acquire);
        if state & PENDING == 0 || state & REMOVED != 0 {
            return None;
        }
        self.state
            .compare_exchange(state, state & !PENDING, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|old| ResidentState(old).class())
    }

    pub(super) fn cancel_pending(&self) {
        let state = self.state.load(Ordering::Acquire);
        assert!(state & PENDING != 0 && state & REMOVED == 0);
        assert!(
            self.state
                .compare_exchange(state, state | REMOVED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        );
        // SAFETY: an unpublished loser cannot be activated by FIFO scans.
        // PENDING admits no readers; REMOVED prevents any future activation.
        drop(unsafe { (&mut *self.block.get()).take() });
    }

    pub(super) fn is_live(&self) -> bool {
        ResidentState(self.state.load(Ordering::Acquire)).live()
    }
    pub(super) fn is_removed(&self) -> bool {
        self.state.load(Ordering::Acquire) & REMOVED != 0
    }
    pub(super) fn class(&self) -> ResidentClass {
        ResidentState(self.state.load(Ordering::Acquire)).class()
    }

    pub(super) fn read(&self, touch: bool) -> Option<Reader<'_>> {
        let mut state = ResidentState(self.state.load(Ordering::Acquire));
        loop {
            if !state.live() {
                return None;
            }
            match self.state.compare_exchange_weak(
                state.0,
                state.enter(touch).0,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Some(Reader {
                        resident: self,
                        active: true,
                    });
                }
                Err(actual) => state = ResidentState(actual),
            }
        }
    }

    pub(super) fn demote(&self) -> bool {
        let mut state = self.state.load(Ordering::Acquire);
        loop {
            if !ResidentState(state).live() || state & RECLAIMABLE != 0 {
                return false;
            }
            match self.state.compare_exchange_weak(
                state,
                state | RECLAIMABLE,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(actual) => state = actual,
            }
        }
    }

    #[cfg(test)]
    pub(super) fn referenced(&self) -> bool {
        self.state.load(Ordering::Acquire) & REFERENCED != 0
    }
}

/// This guard is shared, not exclusive. A suspended reader only prevents
/// pressure eviction of its own resident; it never blocks another reader.
pub(super) struct Reader<'a> {
    resident: &'a Resident,
    active: bool,
}

impl Reader<'_> {
    fn block(&self) -> &Arc<SealedBlock> {
        // SAFETY: this active guard contributes one reader until drop or a
        // successful sole-reader retirement. Neither can run through &self.
        unsafe {
            (&*self.resident.block.get())
                .as_ref()
                .expect("active resident has a block")
        }
    }

    pub(super) fn clone_block(&self) -> Arc<SealedBlock> {
        Arc::clone(self.block())
    }
    pub(super) fn cache_owned_only(&self) -> bool {
        Arc::strong_count(self.block()) == 1
    }

    pub(super) fn is_referenced(&self) -> bool {
        self.resident.state.load(Ordering::Acquire) & REFERENCED != 0
    }

    /// Clear a referenced candidate's bit and skip it. Return true only when
    /// this caller may attempt eviction. A CAS failure is a skip, not a wait.
    pub(super) fn check_second_chance(&self, class: ResidentClass) -> bool {
        let state = ResidentState(self.resident.state.load(Ordering::Acquire));
        if !state.live()
            || state.class() != class
            || state.readers() != READER
            || !self.cache_owned_only()
        {
            return false;
        }
        if state.0 & REFERENCED == 0 {
            return true;
        }
        let _ = self.resident.state.compare_exchange(
            state.0,
            state.0 & !REFERENCED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        false
    }

    pub(super) fn evict_at(self, class: ResidentClass, ticket: u64) -> Option<Arc<SealedBlock>> {
        self.evict_after_sample(class, Some(ticket), || {})
    }

    #[cfg(test)]
    pub(super) fn evict(self, class: ResidentClass) -> Option<Arc<SealedBlock>> {
        self.evict_after_sample(class, None, || {})
    }

    fn evict_after_sample(
        mut self,
        class: ResidentClass,
        ticket: Option<u64>,
        after_sample: impl FnOnce(),
    ) -> Option<Arc<SealedBlock>> {
        // Snapshot BEFORE sampling Arc::strong_count, not after it.
        let state = ResidentState(self.resident.state.load(Ordering::Acquire));
        if !state.live()
            || state.class() != class
            || state.readers() != READER
            || state.0 & REFERENCED != 0
            || ticket
                .is_some_and(|ticket| self.resident.ticket(class).load(Ordering::Acquire) != ticket)
            || !self.cache_owned_only()
        {
            return None;
        }
        after_sample();
        if self
            .resident
            .state
            .compare_exchange(
                state.0,
                (state.0 - READER) | REMOVED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return None;
        }
        self.active = false;
        // SAFETY: the CAS consumed the only reader and forbids future readers.
        // The inactive guard cannot subtract or take again in Drop.
        unsafe { (&mut *self.resident.block.get()).take() }
    }

    pub(super) fn cleanup(&self) -> Option<(ResidentClass, Arc<SealedBlock>)> {
        let block = self.clone_block();
        let old = self.resident.state.fetch_or(REMOVED, Ordering::AcqRel);
        (old & REMOVED == 0).then(|| (ResidentState(old).class(), block))
    }
}

impl Drop for Reader<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let old = ResidentState(self.resident.state.fetch_sub(READER, Ordering::AcqRel));
        if old.0 & REMOVED != 0 && old.readers() == READER {
            // SAFETY: retirement disallows new guards; this RMW observed the
            // last prior reader's release. Only this guard sees count == 1.
            drop(unsafe { (&mut *self.resident.block.get()).take() });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident() -> Resident {
        let resident = Resident::new(
            BlockKey::new("ns".into(), vec![1]),
            1,
            Arc::new(SealedBlock::from_slots(Vec::new())),
            ResidentClass::Retained,
        );
        resident.activate();
        resident
    }

    #[test]
    fn completed_reader_invalidates_eviction_sample() {
        let resident = resident();
        let evictor = resident.read(false).unwrap();
        let mut external = None;
        let evicted = evictor.evict_after_sample(ResidentClass::Retained, None, || {
            // Reader count returns to its sampled value, but the returned Arc
            // must pin the block even without setting the CLOCK reference bit.
            external = Some(resident.read(false).unwrap().clone_block());
        });
        assert!(evicted.is_none());
        assert!(resident.is_live());
        assert_eq!(Arc::strong_count(external.as_ref().unwrap()), 2);
        drop(external);
        assert!(
            resident
                .read(false)
                .unwrap()
                .evict(ResidentClass::Retained)
                .is_some()
        );
    }

    #[test]
    fn cleanup_waits_only_for_last_reader_to_release_slot() {
        let resident = resident();
        let paused = resident.read(false).unwrap();
        let weak = Arc::downgrade(paused.block());
        let cleaner = resident.read(false).unwrap();
        let (_, block) = cleaner.cleanup().unwrap();
        drop(cleaner);
        assert!(resident.read(false).is_none());
        assert!(Arc::ptr_eq(&paused.clone_block(), &block));
        drop(block);
        assert!(weak.upgrade().is_some());
        drop(paused);
        assert!(
            weak.upgrade().is_none(),
            "metadata must not retain the retired block"
        );
    }

    #[test]
    fn pressure_retirement_releases_block_before_metadata() {
        let resident = resident();
        let reader = resident.read(false).unwrap();
        let weak = Arc::downgrade(reader.block());
        drop(reader.evict(ResidentClass::Retained).unwrap());
        assert!(resident.is_removed());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn concurrent_readers_and_retirement_release_exactly_once() {
        for force_cleanup in [false, true] {
            for _ in 0..32 {
                let resident = resident();
                let weak = Arc::downgrade(resident.read(false).unwrap().block());
                let start = std::sync::Barrier::new(5);
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        let resident = &resident;
                        let start = &start;
                        scope.spawn(move || {
                            start.wait();
                            for _ in 0..256 {
                                if let Some(reader) = resident.read(false) {
                                    let block = reader.clone_block();
                                    std::hint::black_box(&block);
                                }
                            }
                        });
                    }
                    start.wait();
                    for _ in 0..256 {
                        if let Some(reader) = resident.read(false) {
                            if force_cleanup {
                                drop(reader.cleanup());
                            } else {
                                drop(reader.evict(ResidentClass::Retained));
                            }
                        }
                    }
                });
                if let Some(reader) = resident.read(false) {
                    drop(reader.evict(ResidentClass::Retained));
                }
                assert!(resident.is_removed());
                assert!(
                    weak.upgrade().is_none(),
                    "metadata must not keep a retired block alive"
                );
            }
        }
    }
}
