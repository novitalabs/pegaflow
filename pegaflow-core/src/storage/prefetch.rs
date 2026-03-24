// Per-request prefetch state machine. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::backing::{PrefetchResult, RdmaFetchStore, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;

struct PrefetchEntry {
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    loading_count: usize,
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
}

impl PrefetchState {
    fn remove_entry(&mut self, req_id: &str) -> Option<PrefetchEntry> {
        if let Some(entry) = self.active.remove(req_id) {
            self.inflight_count = self.inflight_count.saturating_sub(entry.loading_count);
            Some(entry)
        } else {
            None
        }
    }
}

pub(super) struct PrefetchScheduler {
    state: Mutex<PrefetchState>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    rdma_fetch: Option<Arc<RdmaFetchStore>>,
    max_prefetch_blocks: usize,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        rdma_fetch: Option<Arc<RdmaFetchStore>>,
        max_prefetch_blocks: usize,
    ) -> Self {
        Self {
            state: Mutex::new(PrefetchState {
                active: HashMap::new(),
                inflight_count: 0,
            }),
            ssd_store,
            rdma_fetch,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        if let Some(status) = self.poll_existing(read_cache, req_id) {
            match status {
                PollResult::StillLoading => {
                    return PrefetchStatus::Loading { hit: 0, loading: 1 };
                }
                PollResult::Completed => {
                    // Fall through to full scan
                }
            }
        }

        self.full_prefix_scan(
            read_cache,
            instance_id,
            req_id,
            namespace,
            hashes,
            num_workers,
        )
        .await
    }

    fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> Option<PollResult> {
        let mut state = self.state.lock();
        let entry = state.active.get_mut(req_id)?;

        match entry.blocks_rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => Some(PollResult::StillLoading),
            Ok(ssd_blocks) => {
                state.remove_entry(req_id);
                drop(state);
                read_cache.batch_insert(ssd_blocks);
                Some(PollResult::Completed)
            }
            Err(oneshot::error::TryRecvError::Closed) => {
                warn!(
                    "Backing prefetch sender dropped for req_id={}, falling back to re-scan",
                    req_id
                );
                state.remove_entry(req_id);
                Some(PollResult::Completed)
            }
        }
    }

    async fn full_prefix_scan(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();

        let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);

        let remaining = &keys[hit..];
        let mut loading = 0usize;
        let mut blocks_rx: Option<oneshot::Receiver<PrefetchResult>> = None;

        let has_backing = self.ssd_store.is_some();

        let check_keys = if !remaining.is_empty() && has_backing {
            let available = {
                let state = self.state.lock();
                self.max_prefetch_blocks
                    .saturating_sub(state.inflight_count)
            };
            if available > 0 {
                let check_limit = remaining.len().min(available);
                let backpressure_skipped = remaining.len() - check_limit;
                if backpressure_skipped > 0 {
                    core_metrics()
                        .ssd_prefetch_backpressure_blocks
                        .add(backpressure_skipped as u64, &[]);
                }
                Some(remaining[..check_limit].to_vec())
            } else {
                core_metrics()
                    .ssd_prefetch_backpressure_blocks
                    .add(remaining.len() as u64, &[]);
                None
            }
        } else {
            None
        };

        if let Some(check_keys) = check_keys
            && let Some(ssd) = &self.ssd_store
        {
            let (found, rx) = ssd.submit_prefix(check_keys);
            if found > 0 {
                self.state.lock().inflight_count += found;
                loading = found;
                blocks_rx = Some(rx);
            }
        }

        // If nothing is loading from SSD and there are no local hits at all,
        // try RDMA remote fetch from another node.
        if loading == 0
            && hit == 0
            && !remaining.is_empty()
            && let Some(rdma_fetch) = &self.rdma_fetch
        {
            let (found, rx) = rdma_fetch
                .submit_remote_fetch(namespace, remaining.to_vec())
                .await;
            if found > 0 {
                self.state.lock().inflight_count += found;
                loading = found;
                blocks_rx = Some(rx);
            }
        }

        let missing = keys.len() - hit - loading;

        if loading > 0 {
            if let Some(rx) = blocks_rx {
                let mut state = self.state.lock();
                state.active.insert(
                    req_id.to_string(),
                    PrefetchEntry {
                        blocks_rx: rx,
                        loading_count: loading,
                    },
                );
            }
            PrefetchStatus::Loading { hit, loading }
        } else {
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            PrefetchStatus::Done { hit, missing }
        }
    }
}

enum PollResult {
    StillLoading,
    Completed,
}
