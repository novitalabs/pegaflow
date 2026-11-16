use std::cell::UnsafeCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Shared};

use super::resident::Resident;

#[path = "clock_node_pool.rs"]
mod node_pool;
use node_pool::NodePool;

struct Record {
    ticket: u64,
    resident: Arc<Resident>,
}

struct NodeData {
    record: Option<Record>,
    sequence: u64,
}

struct Node {
    data: UnsafeCell<NodeData>,
    removed: AtomicBool,
    next: Atomic<Node>,
    free_next: Atomic<Node>,
}

// SAFETY: data is initialized exclusively before link publication, stays
// immutable while linked or pinned, and is reset only after epoch retirement.
// Free-list readers access only the separate atomic free_next field.
unsafe impl Sync for Node {}

impl Node {
    fn empty() -> Self {
        Self {
            data: UnsafeCell::new(NodeData {
                record: None,
                sequence: 0,
            }),
            removed: AtomicBool::new(false),
            next: Atomic::null(),
            free_next: Atomic::null(),
        }
    }

    fn data(&self) -> &NodeData {
        // SAFETY: callers only read published nodes under an epoch guard.
        unsafe { &*self.data.get() }
    }
}

/// Michael-Scott tail publication with marked-link interior reclamation.
/// The permanent head is never removed. A removed last node stays as an
/// append anchor until it has a successor. Sequence numbers only establish
/// that tail has passed a node before unlinking; they never sort entries.
pub(super) struct Fifo {
    head: Atomic<Node>,
    tail: Atomic<Node>,
    unlinked: AtomicUsize,
    pool: Arc<NodePool>,
}

impl Fifo {
    pub(super) fn new(expected: usize) -> Self {
        let guard = &epoch::pin();
        let pool = Arc::new(NodePool::new(expected));
        let head = pool.acquire(guard);
        Self {
            head: Atomic::from(head),
            tail: Atomic::from(head),
            unlinked: AtomicUsize::new(0),
            pool,
        }
    }

    pub(super) fn push<'g>(
        &'g self,
        ticket: u64,
        resident: Arc<Resident>,
        guard: &'g Guard,
    ) -> FifoEntry<'g> {
        self.push_with(ticket, resident, guard, || {})
    }

    fn push_with<'g>(
        &'g self,
        ticket: u64,
        resident: Arc<Resident>,
        guard: &'g Guard,
        after_link: impl FnOnce(),
    ) -> FifoEntry<'g> {
        let node = self.pool.acquire(guard);
        // SAFETY: acquire exclusively removes a reclaimed node from the pool.
        // No queue reader can still access its data until we publish it below.
        unsafe { (*node.deref().data.get()).record = Some(Record { ticket, resident }) };
        loop {
            let tail = self.tail.load(Ordering::Acquire, guard);
            // SAFETY: tail cannot be unlinked until it is advanced; guard
            // protects a tail that changes after this load.
            let last = unsafe { tail.deref() };
            let next = last.next.load(Ordering::Acquire, guard).with_tag(0);
            if !next.is_null() {
                let _ = self.tail.compare_exchange(
                    tail,
                    next,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    guard,
                );
                continue;
            }
            let sequence = last
                .data()
                .sequence
                .checked_add(1)
                .expect("FIFO sequence exhausted");
            // SAFETY: failed link attempts never publish this node.
            unsafe { (*node.deref().data.get()).sequence = sequence };
            match last.next.compare_exchange(
                Shared::null(),
                node,
                Ordering::Release,
                Ordering::Relaxed,
                guard,
            ) {
                Ok(published) => {
                    after_link();
                    let _ = self.tail.compare_exchange(
                        tail,
                        published,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                        guard,
                    );
                    return FifoEntry {
                        _queue: self,
                        node: published,
                        _guard: guard,
                    };
                }
                Err(_) => continue,
            }
        }
    }

    pub(super) fn scan_budget(&self) -> usize {
        let guard = &epoch::pin();
        let tail = self.tail.load(Ordering::Acquire, guard);
        // SAFETY: tail and its successor are pinned. Include a publication
        // whose writer stopped before advancing tail. No per-node size RMW
        // is needed: publication sequences have no gaps, and cursors account
        // physical unlinks in batches. A concurrent snapshot is conservative.
        let last = unsafe { tail.deref() };
        let next = last.next.load(Ordering::Acquire, guard).with_tag(0);
        let sequence = unsafe { next.as_ref() }.unwrap_or(last).data().sequence;
        (sequence as usize)
            .saturating_sub(self.unlinked.load(Ordering::Relaxed))
            .saturating_mul(2)
    }

    pub(super) fn iter<'g>(&'g self, guard: &'g Guard) -> Cursor<'g> {
        let head = self.head.load(Ordering::Acquire, guard);
        #[cfg(target_arch = "x86_64")]
        let lookahead = {
            let mut pointer = head;
            for _ in 0..4 {
                // SAFETY: head and each linked successor are pinned by guard.
                let next = unsafe { pointer.deref() }
                    .next
                    .load(Ordering::Acquire, guard)
                    .with_tag(0);
                if next.is_null() {
                    break;
                }
                pointer = next;
            }
            pointer
        };
        Cursor {
            queue: self,
            predecessor: head,
            #[cfg(target_arch = "x86_64")]
            lookahead,
            sequence: 0,
            guard,
            retired: Shared::null(),
            retired_count: 0,
        }
    }

    fn unlink<'g>(
        &self,
        predecessor: Shared<'g, Node>,
        current: Shared<'g, Node>,
        guard: &'g Guard,
    ) -> bool {
        // SAFETY: both nodes were reached under this guard. Marking next
        // freezes it, so unlinking cannot lose a concurrent append or splice.
        let node = unsafe { current.deref() };
        let next = node.next.load(Ordering::Acquire, guard);
        if next.with_tag(0).is_null() {
            return false;
        }
        loop {
            let tail = self.tail.load(Ordering::Acquire, guard);
            // SAFETY: tail is pinned. It advances monotonically in link order.
            let last = unsafe { tail.deref() };
            if last.data().sequence > node.data().sequence {
                break;
            }
            let successor = last.next.load(Ordering::Acquire, guard).with_tag(0);
            if !successor.is_null() {
                let _ = self.tail.compare_exchange(
                    tail,
                    successor,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    guard,
                );
            }
        }
        let successor = if next.tag() != 0 {
            next.with_tag(0)
        } else {
            match node.next.compare_exchange(
                next,
                next.with_tag(1),
                Ordering::AcqRel,
                Ordering::Acquire,
                guard,
            ) {
                Ok(_) => next,
                Err(_) => return false,
            }
        };
        // SAFETY: a marked predecessor cannot pass this CAS. Every physically
        // unlinked node has a marked next, so a successful CAS is on a reachable
        // predecessor and uniquely retires current.
        unsafe { predecessor.deref() }
            .next
            .compare_exchange(
                current,
                successor,
                Ordering::AcqRel,
                Ordering::Acquire,
                guard,
            )
            .is_ok()
    }

    #[cfg(test)]
    fn tickets(&self, guard: &Guard) -> Vec<u64> {
        self.iter(guard).map(|entry| entry.ticket()).collect()
    }
}

pub(super) struct FifoEntry<'g> {
    _queue: &'g Fifo,
    node: Shared<'g, Node>,
    _guard: &'g Guard,
}

impl FifoEntry<'_> {
    pub(super) fn guard(&self) -> &Guard {
        self._guard
    }
    fn record(&self) -> &Record {
        // SAFETY: entries only expose data nodes, protected by their guard.
        unsafe { self.node.deref() }
            .data()
            .record
            .as_ref()
            .expect("FIFO data node")
    }
    pub(super) fn ticket(&self) -> u64 {
        self.record().ticket
    }
    pub(super) fn value(&self) -> &Arc<Resident> {
        &self.record().resident
    }
    pub(super) fn remove(&self) -> bool {
        // SAFETY: the entry guard protects the node throughout this RMW.
        !unsafe { self.node.deref() }
            .removed
            .swap(true, Ordering::AcqRel)
    }
}

pub(super) struct Cursor<'g> {
    queue: &'g Fifo,
    predecessor: Shared<'g, Node>,
    #[cfg(target_arch = "x86_64")]
    lookahead: Shared<'g, Node>,
    sequence: u64,
    guard: &'g Guard,
    retired: Shared<'g, Node>,
    retired_count: usize,
}

pub(super) enum ScanStep<'g> {
    Entry(FifoEntry<'g>),
    Skipped,
    End,
}

impl<'g> Cursor<'g> {
    fn flush_retired(&mut self) {
        if self.retired_count != 0 {
            self.queue
                .unlinked
                .fetch_add(self.retired_count, Ordering::Relaxed);
            self.queue.pool.retire(self.retired, self.guard);
            self.retired = Shared::null();
            self.retired_count = 0;
        }
    }

    pub(super) fn step(&mut self) -> ScanStep<'g> {
        loop {
            // SAFETY: cursor nodes are protected for this iterator's lifetime.
            let pointer = unsafe { self.predecessor.deref() }
                .next
                .load(Ordering::Acquire, self.guard);
            if pointer.tag() != 0 {
                #[cfg(target_arch = "x86_64")]
                {
                    self.lookahead = Shared::null();
                }
                self.predecessor = self.queue.head.load(Ordering::Acquire, self.guard);
                return ScanStep::Skipped;
            }
            let Some(node) = (unsafe { pointer.as_ref() }) else {
                return ScanStep::End;
            };
            let sequence = node.data().sequence;
            if node.removed.load(Ordering::Acquire) {
                if node
                    .next
                    .load(Ordering::Acquire, self.guard)
                    .with_tag(0)
                    .is_null()
                {
                    return ScanStep::End;
                }
                if self.queue.unlink(self.predecessor, pointer, self.guard) {
                    node.free_next.store(self.retired, Ordering::Relaxed);
                    self.retired = pointer;
                    self.retired_count += 1;
                    if self.retired_count == 64 {
                        self.flush_retired();
                    }
                } else {
                    #[cfg(target_arch = "x86_64")]
                    {
                        self.lookahead = Shared::null();
                    }
                    self.predecessor = self.queue.head.load(Ordering::Acquire, self.guard);
                    return ScanStep::Skipped;
                }
                if sequence > self.sequence {
                    self.sequence = sequence;
                    return ScanStep::Skipped;
                }
                // This is the entry returned on the preceding step; unlinking
                // it does not inspect another candidate or consume its budget.
                continue;
            }
            if sequence <= self.sequence {
                self.predecessor = pointer;
                continue;
            }
            self.sequence = sequence;
            #[cfg(target_arch = "x86_64")]
            {
                // This independent cursor gives prefetches time to complete.
                // After a conflicting unlink, use single-successor prefetch
                // for this cursor. Prefetch never selects candidates or changes
                // the scan's order, state, or budget.
                let source = if self.lookahead.is_null() {
                    pointer
                } else {
                    self.lookahead
                };
                // SAFETY: guard keeps even an unlinked source node alive.
                let next = unsafe { source.deref() }
                    .next
                    .load(Ordering::Acquire, self.guard)
                    .with_tag(0);
                if !self.lookahead.is_null() && !next.is_null() {
                    self.lookahead = next;
                }
                // SAFETY: the same guard pins this successor and its record.
                if let Some(record) =
                    unsafe { next.as_ref() }.and_then(|successor| successor.data().record.as_ref())
                {
                    record.resident.prefetch_for_scan();
                }
            }
            return ScanStep::Entry(FifoEntry {
                _queue: self.queue,
                node: pointer,
                _guard: self.guard,
            });
        }
    }
}

impl<'g> Iterator for Cursor<'g> {
    type Item = FifoEntry<'g>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.step() {
                ScanStep::Entry(entry) => return Some(entry),
                ScanStep::Skipped => continue,
                ScanStep::End => return None,
            }
        }
    }
}

impl Drop for Cursor<'_> {
    fn drop(&mut self) {
        self.flush_retired();
    }
}

#[cfg(test)]
mod tests {
    use super::super::ResidentClass;
    use super::*;
    use crate::block::{BlockKey, SealedBlock};

    fn resident(id: u64) -> Arc<Resident> {
        Arc::new(Resident::new(
            BlockKey::new("fifo".into(), id.to_le_bytes().to_vec()),
            id,
            Arc::new(SealedBlock::from_slots(Vec::new())),
            ResidentClass::Retained,
        ))
    }

    #[test]
    fn publication_order_is_fifo_not_ticket_order() {
        let queue = Fifo::new(32);
        let guard = &crossbeam_epoch::pin();
        for ticket in [3, 1, 2] {
            queue.push(ticket, resident(ticket), guard);
        }
        assert_eq!(queue.tickets(guard), [3, 1, 2]);
    }

    #[test]
    fn paused_tail_publisher_does_not_block_append_or_removal() {
        let queue = Fifo::new(32);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(0);
        let (resume_tx, resume_rx) = std::sync::mpsc::sync_channel(0);
        std::thread::scope(|scope| {
            let publisher = &queue;
            scope.spawn(move || {
                publisher.push_with(1, resident(1), &epoch::pin(), || {
                    ready_tx.send(()).unwrap();
                    resume_rx
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                });
            });
            ready_rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            let guard = &epoch::pin();
            queue.push(2, resident(2), guard);
            assert_eq!(queue.tickets(guard), [1, 2]);
            for entry in queue.iter(guard) {
                assert!(entry.remove());
                assert!(!entry.remove());
            }
            assert!(queue.iter(guard).next().is_none());
            queue.push(3, resident(3), guard);
            assert_eq!(queue.tickets(guard), [3]);
            resume_tx.send(()).unwrap();
        });
        assert_eq!(queue.tickets(&epoch::pin()), [3]);
    }

    #[test]
    fn quiescent_rotation_reuses_nodes_after_warmup() {
        let queue = Fifo::new(32);
        let item = resident(1);
        // Other cache tests deliberately suspend global epoch participants.
        // This contract requires quiescence, so give it an isolated collector.
        let collector = epoch::Collector::new();
        let handle = collector.register();
        let rotate = || {
            let guard = &handle.pin();
            for ticket in 0..32 {
                queue.push(ticket, Arc::clone(&item), guard);
            }
            for entry in queue.iter(guard) {
                entry.remove();
            }
        };
        let collect = || {
            for _ in 0..128 {
                handle.pin().flush();
            }
        };
        for _ in 0..16 {
            rotate();
            collect();
        }
        let allocated = queue.pool.allocated.load(Ordering::Relaxed);
        for _ in 0..16 {
            rotate();
            collect();
        }
        assert_eq!(
            queue.pool.allocated.load(Ordering::Relaxed),
            allocated,
            "quiescent rotation must reuse its prewarmed node storage"
        );
    }

    #[test]
    fn pinned_removed_entry_is_not_reused_and_does_not_stop_growth() {
        let queue = Fifo::new(1);
        let guard = &epoch::pin();
        let held = queue.push(7, resident(7), guard);
        let old = Arc::clone(held.value());
        assert!(held.remove());
        let allocated = queue.pool.allocated.load(Ordering::Relaxed);
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    for id in 0..1024 {
                        let guard = &epoch::pin();
                        queue.push(id + 100, resident(id + 100), guard);
                        for entry in queue.iter(guard) {
                            entry.remove();
                        }
                    }
                })
                .join()
                .unwrap();
        });
        assert_eq!(held.ticket(), 7);
        assert!(Arc::ptr_eq(held.value(), &old));
        assert!(queue.pool.allocated.load(Ordering::Relaxed) > allocated);
        assert!(queue.iter(guard).next().is_none());
    }

    #[test]
    fn concurrent_publication_and_removal_deliver_each_record_once() {
        let queue = Fifo::new(64);
        let start = std::sync::Barrier::new(8);
        let completed = AtomicUsize::new(0);
        let removed = std::thread::scope(|scope| {
            for producer in 0..4 {
                let queue = &queue;
                let start = &start;
                let completed = &completed;
                scope.spawn(move || {
                    start.wait();
                    for offset in 0..2048 {
                        let id = (producer * 2048 + offset) as u64;
                        queue.push(id, resident(id), &epoch::pin());
                    }
                    completed.fetch_add(1, Ordering::Release);
                });
            }
            let tasks: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        let mut ids = Vec::new();
                        loop {
                            let guard = &epoch::pin();
                            for entry in queue.iter(guard).take(64) {
                                if entry.remove() {
                                    ids.push(entry.ticket());
                                }
                            }
                            if completed.load(Ordering::Acquire) == 4
                                && queue.iter(guard).next().is_none()
                            {
                                break;
                            }
                        }
                        ids
                    })
                })
                .collect();
            tasks
                .into_iter()
                .flat_map(|task| task.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(removed.len(), 8192);
        let unique: std::collections::HashSet<_> = removed.into_iter().collect();
        assert_eq!(unique.len(), 8192);
        assert!(queue.tickets(&epoch::pin()).is_empty());
    }

    #[test]
    fn bounded_scan_counts_interior_tombstones_without_moving_live_head() {
        let queue = Fifo::new(66);
        let guard = &epoch::pin();
        let entries: Vec<_> = (0..66)
            .map(|id| queue.push(id, resident(id), guard))
            .collect();
        for entry in &entries[1..65] {
            entry.remove();
        }
        let mut saw_tail = false;
        for round in 0..16 {
            let mut cursor = queue.iter(guard);
            assert!(matches!(cursor.step(), ScanStep::Entry(entry) if entry.ticket() == 0));
            for _ in 0..7 {
                if let ScanStep::Entry(entry) = cursor.step() {
                    assert_eq!(entry.ticket(), 65);
                    assert!(
                        round > 0,
                        "one bounded scan must not process the entire stale chain"
                    );
                    saw_tail = true;
                }
            }
            if saw_tail {
                break;
            }
        }
        assert!(saw_tail);
        assert_eq!(queue.tickets(guard), [0, 65]);
    }

    #[test]
    fn initial_reserve_covers_a_large_pass_without_recycling() {
        const COUNT: usize = 100_000;
        let queue = Fifo::new(COUNT);
        let item = resident(1);
        let guard = &epoch::pin();
        for ticket in 0..COUNT as u64 {
            queue.push(ticket, Arc::clone(&item), guard);
        }
        let allocated = queue.pool.allocated.load(Ordering::Relaxed);
        for entry in queue.iter(guard).take(COUNT) {
            assert!(entry.remove());
            queue.push(entry.ticket() + COUNT as u64, Arc::clone(&item), guard);
        }
        assert_eq!(
            queue.pool.allocated.load(Ordering::Relaxed),
            allocated,
            "reserve must cover an entire pass even when old nodes remain pinned"
        );
        assert_eq!(queue.iter(guard).count(), COUNT);
    }
}
