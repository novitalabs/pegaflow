// Per-request prefetch coordination. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::backing::{PrefetchResult, RdmaFetchStore, SsdBackingStore};
use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;

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
}

/// Result of a single load attempt (SSD or RDMA).
struct LoadResult {
    found: usize,
    rx: oneshot::Receiver<PrefetchResult>,
    source: PrefetchSource,
}

struct PrefetchState {
    inflight_count: usize,
    /// req_ids where RDMA remote fetch returned zero blocks (remote evicted).
    /// Prevents re-triggering RDMA on every subsequent scan for the same request.
    failed_remote: HashMap<String, Instant>,
}

impl PrefetchState {
    fn register_inflight(&mut self, loading_count: usize) {
        self.inflight_count += loading_count;
    }

    fn complete_inflight(&mut self, loading_count: usize) {
        self.inflight_count = self.inflight_count.saturating_sub(loading_count);
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
                inflight_count: 0,
                failed_remote: HashMap::new(),
            }),
            ssd_store,
            rdma_fetch,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn load_prefix(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> (usize, usize) {
        loop {
            let scan = self
                .full_prefix_scan(read_cache, req_id, namespace, hashes)
                .await;
            let Some(load) = scan.load else {
                read_cache.pin_blocks(instance_id, num_workers, &scan.blocks_to_pin);
                return (scan.hit_blocks, scan.missing_blocks);
            };

            let expected = load.found;
            let source = load.source;
            let prefetched_blocks = self.await_load(req_id, load).await;
            let returned = prefetched_blocks.len();

            if source == PrefetchSource::Rdma && returned < expected && expected > 0 {
                self.state
                    .lock()
                    .failed_remote
                    .insert(req_id.to_string(), Instant::now());
                info!(
                    "RDMA prefetch returned fewer blocks than expected: req_id={} returned={} expected={}",
                    req_id, returned, expected
                );
            }

            read_cache.batch_insert(prefetched_blocks);
            if source == PrefetchSource::Ssd && returned < expected {
                return self.terminal_scan_and_pin(
                    read_cache,
                    instance_id,
                    namespace,
                    hashes,
                    num_workers,
                );
            }
        }
    }

    fn terminal_scan_and_pin(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> (usize, usize) {
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let (hit_blocks, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);
        read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
        (hit_blocks, keys.len() - hit_blocks)
    }

    async fn await_load(&self, req_id: &str, load: LoadResult) -> PrefetchResult {
        let loading_count = load.found;
        let rx = load.rx;
        {
            let mut state = self.state.lock();
            state.register_inflight(loading_count);
        }

        let result = match rx.await {
            Ok(blocks) => blocks,
            Err(_) => {
                warn!(
                    "Backing prefetch sender dropped for req_id={}, falling back to re-scan",
                    req_id
                );
                Vec::new()
            }
        };

        self.state.lock().complete_inflight(loading_count);
        result
    }

    async fn full_prefix_scan(
        &self,
        read_cache: &ReadCache,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefixScan {
        let total_start = Instant::now();

        let key_build_start = Instant::now();
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let key_build = key_build_start.elapsed();

        let cache_scan_start = Instant::now();
        let (hit_blocks, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_scan_start.elapsed();
        let remaining = &keys[hit_blocks..];

        let load_select_start = Instant::now();
        let load = self.try_load(req_id, namespace, remaining).await;
        let load_select = load_select_start.elapsed();
        let loading = load.as_ref().map_or(0, |l| l.found);
        let missing_blocks = keys.len() - hit_blocks - loading;

        match load.as_ref() {
            Some(load) => {
                info!(
                    "Prefetch scheduling timing: req_id={} source={} total_keys={} hit_blocks={} loading_blocks={} missing_blocks={} key_build={:?} cache_scan={:?} load_select={:?} total={:?}",
                    req_id,
                    load.source.as_str(),
                    keys.len(),
                    hit_blocks,
                    loading,
                    missing_blocks,
                    key_build,
                    cache_scan,
                    load_select,
                    total_start.elapsed()
                );
            }
            None => {
                info!(
                    "Prefetch terminal timing: req_id={} total_keys={} hit_blocks={} missing_blocks={} key_build={:?} cache_scan={:?} load_select={:?} total={:?}",
                    req_id,
                    keys.len(),
                    hit_blocks,
                    missing_blocks,
                    key_build,
                    cache_scan,
                    load_select,
                    total_start.elapsed()
                );
            }
        }

        PrefixScan {
            hit_blocks,
            missing_blocks,
            blocks_to_pin,
            load,
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

    pub(super) fn gc_failed_remote(&self, max_age: std::time::Duration) -> usize {
        let mut state = self.state.lock();
        let before = state.failed_remote.len();
        state.failed_remote.retain(|_, ts| ts.elapsed() < max_age);
        before - state.failed_remote.len()
    }
}

struct PrefixScan {
    hit_blocks: usize,
    missing_blocks: usize,
    blocks_to_pin: Vec<(BlockKey, Arc<SealedBlock>)>,
    load: Option<LoadResult>,
}
