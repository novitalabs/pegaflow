use log::{info, warn};
use tokio::task::JoinHandle;

use crate::block::PrefetchStatus;

use super::super::backing_tier::TierSource;
use super::super::read_cache::ReadCache;
use super::state::ActivePoll;
use super::task::rdma_advert;
use super::{Scheduler, TaskResult};

pub(super) enum PollResult {
    Idle,
    StillLoading,
    Ready(PrefetchStatus),
}

impl Scheduler {
    pub(super) async fn poll_task(&self, read_cache: &ReadCache, req_id: &str) -> PollResult {
        // Guard must drop before any await: the helpers below re-acquire
        // the same mutex (see the `await_holding_lock` lint).
        let active_poll = {
            let mut state = self.state.lock();
            state.take_finished(req_id)
        };
        let handle = match active_poll {
            ActivePoll::Missing => return PollResult::Idle,
            ActivePoll::Loading => return PollResult::StillLoading,
            ActivePoll::Finished(handle) => handle,
        };

        let result = drain_task(req_id, handle).await;
        self.note_shortfall(req_id, &result);
        PollResult::Ready(self.apply_result(read_cache, result))
    }

    /// RDMA can deliver fewer blocks than committed (remote evicted). Don't
    /// re-trigger RDMA on subsequent scans for this request.
    fn note_shortfall(&self, req_id: &str, result: &TaskResult) {
        if result.source == Some(TierSource::Rdma) && result.inserts.len() < result.committed {
            self.state.lock().mark_rdma(req_id);
            info!(
                "RDMA prefetch returned fewer blocks than expected: req_id={} returned={} expected={}",
                req_id,
                result.inserts.len(),
                result.committed
            );
        }
    }

    fn apply_result(&self, read_cache: &ReadCache, result: TaskResult) -> PrefetchStatus {
        let advert = if result.source == Some(TierSource::Rdma) {
            let resident = read_cache.batch_insert_resident_keys(result.inserts);
            rdma_advert(&resident)
        } else {
            read_cache.batch_insert(result.inserts);
            None
        };

        if let Some(client) = &self.metaserver_client
            && let Some((namespace, hashes)) = advert
        {
            client.try_register_namespace(namespace, hashes);
        }

        PrefetchStatus::Ready {
            blocks: result.ready_blocks,
            missing: result.missing,
        }
    }
}

async fn drain_task(req_id: &str, handle: JoinHandle<TaskResult>) -> TaskResult {
    match handle.await {
        Ok(result) => result,
        Err(err) => {
            warn!("Prefetch task failed for req_id={}: {}", req_id, err);
            TaskResult {
                source: None,
                committed: 0,
                inserts: Vec::new(),
                ready_blocks: Vec::new(),
                missing: 0,
            }
        }
    }
}
