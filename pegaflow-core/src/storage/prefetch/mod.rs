mod poll;
mod scan;
mod state;
mod task;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{PrefetchStatus, SealedBlock};
use crate::internode::MetaServerClient;

use super::backing_tier::{BackingTier, RdmaFetch, TierSource};
use super::read_cache::ReadCache;
use poll::PollResult;
use scan::tier_lineup;
use state::State;

/// Shared by the task (producer), state (holds the JoinHandle), and poll
/// (consumer) submodules; lives here to keep their dependencies one-way.
pub(super) struct TaskResult {
    pub(super) source: Option<TierSource>,
    pub(super) committed: usize,
    pub(super) inserts: PrefetchResult,
    pub(super) ready_blocks: Vec<Arc<SealedBlock>>,
    pub(super) missing: usize,
}

pub(super) struct PrefixScan<'a> {
    req_id: &'a str,
    namespace: &'a str,
    hashes: &'a [Vec<u8>],
    require_full: bool,
}

impl<'a> PrefixScan<'a> {
    pub(super) fn new(req_id: &'a str, namespace: &'a str, hashes: &'a [Vec<u8>]) -> Self {
        Self {
            req_id,
            namespace,
            hashes,
            require_full: false,
        }
    }

    pub(super) fn require_full(mut self, require_full: bool) -> Self {
        self.require_full = require_full;
        self
    }
}

pub(super) struct PrefetchDeps {
    pub(super) ssd_store: Option<Arc<SsdBackingStore>>,
    pub(super) rdma_fetch: Option<RdmaFetch>,
    pub(super) metaserver_client: Option<Arc<MetaServerClient>>,
    pub(super) max_prefetch_blocks: usize,
}

pub(super) struct Scheduler {
    state: Arc<Mutex<State>>,
    tiers: Vec<BackingTier>,
    metaserver_client: Option<Arc<MetaServerClient>>,
    max_prefetch_blocks: usize,
}

impl Scheduler {
    pub(super) fn new(deps: PrefetchDeps) -> Self {
        let tiers = tier_lineup(
            deps.rdma_fetch.map(BackingTier::Rdma),
            deps.ssd_store.map(BackingTier::Ssd),
        );
        Self {
            state: Arc::new(Mutex::new(State::new())),
            tiers,
            metaserver_client: deps.metaserver_client,
            max_prefetch_blocks: deps.max_prefetch_blocks,
        }
    }

    #[cfg(test)]
    fn with_tiers(tiers: Vec<BackingTier>, max_prefetch_blocks: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State::new())),
            tiers,
            metaserver_client: None,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn query(
        &self,
        read_cache: &ReadCache,
        scan: PrefixScan<'_>,
    ) -> PrefetchStatus {
        match self.poll_task(read_cache, scan.req_id).await {
            PollResult::Idle => {}
            PollResult::StillLoading => {
                return PrefetchStatus::Loading;
            }
            PollResult::Ready(status) => return status,
        }

        self.scan_prefix(read_cache, scan).await
    }

    /// Dropping a `JoinHandle` detaches the task; it keeps running so RDMA
    /// transfer locks can still be released by the normal completion path.
    pub(super) fn gc_stale(&self, active_ttl: Duration, rdma_ttl: Duration) -> (usize, usize) {
        let mut state = self.state.lock();
        (state.sweep_active(active_ttl), state.sweep_rdma(rdma_ttl))
    }
}
