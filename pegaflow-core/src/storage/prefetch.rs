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
use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchSource {
    Ssd,
    Rdma,
}

impl PrefetchSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ssd => "ssd",
            Self::Rdma => "rdma",
        }
    }

    const fn as_attribution(self) -> AttributionSource {
        match self {
            Self::Ssd => AttributionSource::Ssd,
            Self::Rdma => AttributionSource::Rdma,
        }
    }
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

struct PrefixScan<'a> {
    instance_id: &'a str,
    req_id: &'a str,
    namespace: &'a str,
    hashes: &'a [Vec<u8>],
    num_workers: usize,
    emit_tier_metrics: bool,
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
        // Default: this call may be the first decision and should attribute.
        let mut emit_tier_metrics = true;
        if let Some(status) = self.poll_existing(read_cache, req_id) {
            match status {
                PollResult::StillLoading => {
                    return PrefetchStatus::Loading { hit: 0, loading: 1 };
                }
                PollResult::Completed => {
                    // Backing has just written blocks into read_cache. The
                    // fall-through scan will re-see them as RAM hits; we MUST
                    // NOT attribute again, because we already attributed
                    // them as `rdma`/`ssd` on the first decision.
                    emit_tier_metrics = false;
                }
            }
        }

        self.full_prefix_scan(
            read_cache,
            PrefixScan {
                instance_id,
                req_id,
                namespace,
                hashes,
                num_workers,
                emit_tier_metrics,
            },
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
        scan: PrefixScan<'_>,
    ) -> PrefetchStatus {
        let total_start = Instant::now();

        let key_build_start = Instant::now();
        let keys: Vec<BlockKey> = scan
            .hashes
            .iter()
            .map(|hash| BlockKey::new(scan.namespace.to_string(), hash.clone()))
            .collect();
        let key_build = key_build_start.elapsed();

        let cache_scan_start = Instant::now();
        let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_scan_start.elapsed();
        let remaining = &keys[hit..];

        let load_select_start = Instant::now();
        let load = self.try_load(scan.req_id, scan.namespace, remaining).await;
        let load_select = load_select_start.elapsed();
        let loading = load.as_ref().map_or(0, |l| l.found);
        let missing = keys.len() - hit - loading;

        if let Some(load) = load {
            let source = load.source;
            let register_start = Instant::now();
            self.register_inflight(scan.req_id, load);
            let register = register_start.elapsed();

            self.maybe_record_tier_attribution(
                keys.len(),
                hit,
                loading,
                Some(source.as_attribution()),
                scan.emit_tier_metrics,
            );

            info!(
                "Prefetch scheduling timing: req_id={} source={} total_keys={} hit={} loading={} missing={} key_build={:?} cache_scan={:?} load_select={:?} register_inflight={:?} total={:?}",
                scan.req_id,
                source.as_str(),
                keys.len(),
                hit,
                loading,
                missing,
                key_build,
                cache_scan,
                load_select,
                register,
                total_start.elapsed()
            );
            PrefetchStatus::Loading { hit, loading }
        } else {
            let pin_start = Instant::now();
            read_cache.pin_blocks(scan.instance_id, scan.num_workers, &blocks_to_pin);
            let pin = pin_start.elapsed();

            self.maybe_record_tier_attribution(
                keys.len(),
                hit,
                /* loading = */ 0,
                /* loading_source = */ None,
                scan.emit_tier_metrics,
            );

            info!(
                "Prefetch local-hit timing: req_id={} total_keys={} hit={} missing={} key_build={:?} cache_scan={:?} load_select={:?} pin={:?} total={:?}",
                scan.req_id,
                keys.len(),
                hit,
                missing,
                key_build,
                cache_scan,
                load_select,
                pin,
                total_start.elapsed()
            );
            PrefetchStatus::Done { hit, missing }
        }
    }

    /// Attribute this `query_prefetch` decision. Skips attribution when:
    /// * `emit_tier_metrics == false` (e.g. post-completion fall-through);
    /// * `keys` was empty (no decision to attribute).
    fn maybe_record_tier_attribution(
        &self,
        total: usize,
        hit: usize,
        loading: usize,
        loading_source: Option<AttributionSource>,
        emit_tier_metrics: bool,
    ) {
        if !emit_tier_metrics || total == 0 {
            return;
        }
        let attribution = TierAttribution::classify(total, hit, loading, loading_source);
        record_cache_tier_block_requests(total, attribution);
    }

    /// Priority fallback: RDMA -> SSD. Returns `None` when neither source has blocks.
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

        let rx = rdma.fetch_blocks(&node, req_id, namespace, hashes[..found].to_vec());

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

    /// Sweep `failed_remote` entries older than `max_age`.
    /// Runs under the single `PrefetchState` mutex.
    pub(super) fn gc_failed_remote(&self, max_age: std::time::Duration) -> usize {
        let mut state = self.state.lock();
        let failed_before = state.failed_remote.len();
        state.failed_remote.retain(|_, ts| ts.elapsed() < max_age);
        failed_before - state.failed_remote.len()
    }
}

enum PollResult {
    StillLoading,
    Completed,
}
