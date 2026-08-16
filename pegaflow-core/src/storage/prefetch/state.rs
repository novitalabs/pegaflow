// One scheduler-owned Mutex guards all prefetch state, so a single guard can
// span the check-then-act sequences that must stay atomic.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::task::JoinHandle;

use crate::metrics::core_metrics;

use super::TaskResult;

struct Entry {
    handle: JoinHandle<TaskResult>,
    started_at: Instant,
}

pub(super) enum ActivePoll {
    Missing,
    Loading,
    Finished(JoinHandle<TaskResult>),
}

pub(super) struct State {
    active: HashMap<String, Entry>,
    /// Reserved SSD prefetch budget for active background tasks.
    ssd_reserved: usize,
    /// req_ids where RDMA delivered fewer blocks than committed (remote
    /// evicted). Prevents re-triggering RDMA on every subsequent poll.
    rdma_failed: HashMap<String, Instant>,
}

impl State {
    pub(super) fn new() -> Self {
        Self {
            active: HashMap::new(),
            ssd_reserved: 0,
            rdma_failed: HashMap::new(),
        }
    }

    /// One operation so "check finished" and "remove" cannot interleave
    /// across lock releases.
    pub(super) fn take_finished(&mut self, req_id: &str) -> ActivePoll {
        let Some(entry) = self.active.get(req_id) else {
            return ActivePoll::Missing;
        };
        if !entry.handle.is_finished() {
            return ActivePoll::Loading;
        }
        let entry = self
            .active
            .remove(req_id)
            .expect("active entry must exist after readiness check");
        ActivePoll::Finished(entry.handle)
    }

    pub(super) fn has_active(&self, req_id: &str) -> bool {
        self.active.contains_key(req_id)
    }

    pub(super) fn insert_active(&mut self, req_id: String, handle: JoinHandle<TaskResult>) {
        self.active.insert(
            req_id,
            Entry {
                handle,
                started_at: Instant::now(),
            },
        );
    }

    pub(super) fn sweep_active(&mut self, max_age: Duration) -> usize {
        let before = self.active.len();
        self.active
            .retain(|_, entry| entry.started_at.elapsed() < max_age);
        before - self.active.len()
    }

    pub(super) fn mark_rdma(&mut self, req_id: &str) {
        self.rdma_failed.insert(req_id.to_string(), Instant::now());
    }

    pub(super) fn rdma_failed(&self, req_id: &str) -> bool {
        self.rdma_failed.contains_key(req_id)
    }

    pub(super) fn sweep_rdma(&mut self, max_age: Duration) -> usize {
        let before = self.rdma_failed.len();
        self.rdma_failed.retain(|_, ts| ts.elapsed() < max_age);
        before - self.rdma_failed.len()
    }

    fn reserve(&mut self, max_blocks: usize, requested: usize, require_full: bool) -> usize {
        let available = max_blocks.saturating_sub(self.ssd_reserved);
        // A partial grant is useless to an all-or-nothing caller, so deny it
        // outright rather than holding budget the task cannot use.
        if require_full && available < requested {
            return 0;
        }
        let granted = requested.min(available);
        self.ssd_reserved += granted;
        granted
    }

    fn release(&mut self, blocks: usize) {
        self.ssd_reserved = self.ssd_reserved.saturating_sub(blocks);
    }

    #[cfg(test)]
    pub(super) fn active_len(&self) -> usize {
        self.active.len()
    }
}

pub(super) struct SsdGuard {
    state: Arc<Mutex<State>>,
    blocks: usize,
}

impl Drop for SsdGuard {
    fn drop(&mut self) {
        self.state.lock().release(self.blocks);
    }
}

/// Reserve SSD prefetch budget, emitting backpressure metrics for whatever
/// could not be granted. `None` means nothing usable was granted: the budget
/// is exhausted, or `require_full` could not be satisfied in full.
pub(super) fn reserve_ssd(
    state: Arc<Mutex<State>>,
    max_prefetch_blocks: usize,
    requested: usize,
    require_full: bool,
) -> Option<(usize, SsdGuard)> {
    let reserved = state
        .lock()
        .reserve(max_prefetch_blocks, requested, require_full);
    let skipped = requested - reserved;
    if skipped > 0 {
        core_metrics()
            .ssd_prefetch_backpressure_blocks
            .add(skipped as u64, &[]);
    }
    if reserved == 0 {
        return None;
    }
    Some((
        reserved,
        SsdGuard {
            state,
            blocks: reserved,
        },
    ))
}
