// Per-request prefetch state machine. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::backing::{PrefetchResult, RdmaFetchStore, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchSource {
    Ssd,
    Rdma,
}

struct PrefetchEntry {
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    loading_count: usize,
    source: PrefetchSource,
}

/// Result of a single load attempt (SSD or RDMA).
struct LoadResult {
    found: usize,
    rx: oneshot::Receiver<PrefetchResult>,
    source: PrefetchSource,
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
    /// req_ids where RDMA remote fetch returned zero blocks (remote evicted).
    /// Prevents re-triggering RDMA on every subsequent poll for the same request.
    failed_remote: HashMap<String, Instant>,
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
                failed_remote: HashMap::new(),
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
            Err(oneshot::TryRecvError::Empty) => Some(PollResult::StillLoading),
            Ok(prefetched_blocks) => {
                let expected = entry.loading_count;
                let source = entry.source;
                state.remove_entry(req_id);
                // RDMA remote node can return fewer blocks than MetaServer promised
                // (likely evicted). Don't re-trigger RDMA on subsequent scans.
                if source == PrefetchSource::Rdma
                    && prefetched_blocks.len() < expected
                    && expected > 0
                {
                    state
                        .failed_remote
                        .insert(req_id.to_string(), Instant::now());
                    info!(
                        "RDMA prefetch returned fewer blocks than expected: req_id={} returned={} expected={}",
                        req_id,
                        prefetched_blocks.len(),
                        expected
                    );
                }
                drop(state);
                read_cache.batch_insert(prefetched_blocks);
                Some(PollResult::Completed)
            }
            Err(oneshot::TryRecvError::Disconnected) => {
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

        let load = self.try_load(req_id, namespace, remaining).await;
        let loading = load.as_ref().map_or(0, |l| l.found);
        let missing = keys.len() - hit - loading;

        if let Some(load) = load {
            self.register_inflight(req_id, load);
            PrefetchStatus::Loading { hit, loading }
        } else {
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            PrefetchStatus::Done { hit, missing }
        }
    }

    /// Priority fallback: RDMA → SSD. Returns `None` when neither source has blocks.
    async fn try_load(
        &self,
        req_id: &str,
        namespace: &str,
        remaining: &[BlockKey],
    ) -> Option<LoadResult> {
        if remaining.is_empty() {
            return None;
        }

        if let Some(result) = self.try_rdma_load(req_id, namespace, remaining).await {
            return Some(result);
        }

        self.try_ssd_load(remaining)
    }

    fn try_ssd_load(&self, remaining: &[BlockKey]) -> Option<LoadResult> {
        let ssd = self.ssd_store.as_ref()?;
        let check_keys = self.limit_ssd_prefetch(remaining)?;

        let (found, rx) = ssd.submit_prefix(check_keys);
        if found == 0 {
            return None;
        }

        Some(LoadResult {
            found,
            rx,
            source: PrefetchSource::Ssd,
        })
    }

    async fn try_rdma_load(
        &self,
        req_id: &str,
        namespace: &str,
        remaining: &[BlockKey],
    ) -> Option<LoadResult> {
        let rdma = self.rdma_fetch.as_ref()?;

        if self.state.lock().failed_remote.contains_key(req_id) {
            return None;
        }

        let hashes: Vec<Vec<u8>> = remaining.iter().map(|k| k.hash.clone()).collect();
        let (node, found) = rdma.query_prefix(namespace, &hashes).await?;

        let rx = rdma.fetch_blocks(&node, namespace, hashes[..found].to_vec());

        Some(LoadResult {
            found,
            rx,
            source: PrefetchSource::Rdma,
        })
    }

    /// Trim keys to fit inflight capacity, report skipped count to metrics.
    fn limit_ssd_prefetch(&self, remaining: &[BlockKey]) -> Option<Vec<BlockKey>> {
        let available = {
            let state = self.state.lock();
            self.max_prefetch_blocks
                .saturating_sub(state.inflight_count)
        };

        if available == 0 {
            core_metrics()
                .ssd_prefetch_backpressure_blocks
                .add(remaining.len() as u64, &[]);
            return None;
        }

        let check_limit = remaining.len().min(available);
        let skipped = remaining.len() - check_limit;
        if skipped > 0 {
            core_metrics()
                .ssd_prefetch_backpressure_blocks
                .add(skipped as u64, &[]);
        }

        Some(remaining[..check_limit].to_vec())
    }

    fn register_inflight(&self, req_id: &str, load: LoadResult) {
        let mut state = self.state.lock();
        state.inflight_count += load.found;
        state.active.insert(
            req_id.to_string(),
            PrefetchEntry {
                blocks_rx: load.rx,
                loading_count: load.found,
                source: load.source,
            },
        );
    }

    pub(super) fn gc_failed_remote(&self, max_age: std::time::Duration) -> usize {
        let mut state = self.state.lock();
        let before = state.failed_remote.len();
        state.failed_remote.retain(|_, ts| ts.elapsed() < max_age);
        before - state.failed_remote.len()
    }
}

enum PollResult {
    StillLoading,
    Completed,
}
