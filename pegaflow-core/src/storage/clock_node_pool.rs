use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};

use super::Node;

struct Chunk {
    _nodes: Box<[Node]>,
    next: Atomic<Chunk>,
}

/// Chunks stay allocated until the pool and all deferred batches are dropped.
/// A free node may be reused only after its queue retirement grace period.
pub(super) struct NodePool {
    free: Atomic<Node>,
    chunks: Atomic<Chunk>,
    pub(super) allocated: AtomicUsize,
    chunk_size: usize,
}

impl NodePool {
    pub(super) fn new(expected: usize) -> Self {
        let reserve = expected.saturating_mul(2).saturating_add(2).max(128);
        let pool = Self {
            free: Atomic::null(),
            chunks: Atomic::null(),
            allocated: AtomicUsize::new(0),
            chunk_size: reserve.min(131_072),
        };
        while pool.allocated.load(Ordering::Relaxed) < reserve {
            pool.grow(&epoch::pin());
        }
        pool
    }

    fn grow(&self, guard: &Guard) {
        let nodes: Box<[_]> = (0..self.chunk_size).map(|_| Node::empty()).collect();
        let first = Shared::from(&nodes[0] as *const Node);
        let last = Shared::from(&nodes[nodes.len() - 1] as *const Node);
        for pair in nodes.windows(2) {
            pair[0]
                .free_next
                .store(Shared::from(&pair[1] as *const Node), Ordering::Relaxed);
        }
        let mut chunk = Owned::new(Chunk {
            _nodes: nodes,
            next: Atomic::null(),
        });
        loop {
            let previous = self.chunks.load(Ordering::Acquire, guard);
            chunk.next.store(previous, Ordering::Relaxed);
            match self.chunks.compare_exchange(
                previous,
                chunk,
                Ordering::Release,
                Ordering::Relaxed,
                guard,
            ) {
                Ok(_) => break,
                Err(error) => chunk = error.new,
            }
        }
        self.allocated.fetch_add(self.chunk_size, Ordering::Relaxed);
        self.release(first, last, guard);
    }

    fn release(&self, first: Shared<'_, Node>, last: Shared<'_, Node>, guard: &Guard) {
        loop {
            let previous = self.free.load(Ordering::Acquire, guard);
            // SAFETY: this batch is exclusively owned until the release CAS.
            // Stale free-list readers access only this atomic link; their pin
            // prevents a popped node from returning to the free list (ABA).
            unsafe { last.deref() }
                .free_next
                .store(previous, Ordering::Relaxed);
            if self
                .free
                .compare_exchange(previous, first, Ordering::Release, Ordering::Relaxed, guard)
                .is_ok()
            {
                return;
            }
        }
    }

    pub(super) fn acquire<'g>(&self, guard: &'g Guard) -> Shared<'g, Node> {
        loop {
            let first = self.free.load(Ordering::Acquire, guard);
            if first.is_null() {
                self.grow(guard);
                continue;
            }
            // SAFETY: chunks own all nodes; guard prevents free-list ABA.
            let next = unsafe { first.deref() }
                .free_next
                .load(Ordering::Relaxed, guard);
            if self
                .free
                .compare_exchange(first, next, Ordering::AcqRel, Ordering::Acquire, guard)
                .is_ok()
            {
                return first;
            }
        }
    }

    pub(super) fn retire(self: &Arc<Self>, first: Shared<'_, Node>, guard: &Guard) {
        if first.is_null() {
            return;
        }
        let pointer = first.as_raw();
        let pool = Arc::clone(self);
        // SAFETY: every node in this exclusive batch was unlinked exactly once.
        // The captured pool owns its storage. After the grace period, no queue
        // reader can access its data; old free-list readers use atomic links
        // only. The closure is safe to execute on any collector thread.
        unsafe {
            guard.defer_unchecked(move || {
                let guard = &epoch::pin();
                let first = Shared::from(pointer);
                let mut current = first;
                loop {
                    let node = current.deref();
                    (*node.data.get()).record = None;
                    node.next.store(Shared::null(), Ordering::Relaxed);
                    node.removed.store(false, Ordering::Relaxed);
                    let next = node.free_next.load(Ordering::Relaxed, guard);
                    if next.is_null() {
                        pool.release(first, current, guard);
                        break;
                    }
                    current = next;
                }
            });
        };
    }
}

impl Drop for NodePool {
    fn drop(&mut self) {
        // SAFETY: no queue or deferred batch holds this pool anymore.
        let guard = unsafe { epoch::unprotected() };
        let mut current = self.chunks.load(Ordering::Relaxed, guard);
        while !current.is_null() {
            // SAFETY: chunks are never unlinked; this exclusive owner drops all.
            let chunk = unsafe { current.into_owned() };
            current = chunk.next.load(Ordering::Relaxed, guard);
            drop(chunk);
        }
    }
}
