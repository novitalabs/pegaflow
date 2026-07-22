// Per-request prefetch state machine. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{info, warn};
use parking_lot::Mutex;
use tokio::task::JoinHandle;

#[cfg(feature = "rdma")]
use crate::backing::RdmaFetchStore;
use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;
use super::tier_attribution::{
    AttributionSource, TierAttribution, record_cache_tier_block_requests,
};

const REMOTE_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REMOTE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[cfg(feature = "rdma")]
#[derive(Clone)]
pub(super) struct RdmaFetch(Arc<RdmaFetchStore>);
#[cfg(not(feature = "rdma"))]
#[derive(Clone)]
pub(super) struct RdmaFetch;

#[cfg(feature = "rdma")]
impl RdmaFetch {
    pub(super) fn new(store: Arc<RdmaFetchStore>) -> Self {
        Self(store)
    }

    async fn try_fetch_prefix(
        &self,
        req_id: &str,
        namespace: &str,
        remaining_hashes: &[Vec<u8>],
        require_full_prefix: bool,
    ) -> Option<(usize, PrefetchResult)> {
        let (node, found) = self.0.query_prefix(namespace, remaining_hashes).await?;
        if require_full_prefix && found != remaining_hashes.len() {
            return None;
        }
        let blocks = self
            .0
            .fetch_blocks(&node, req_id, namespace, &remaining_hashes[..found])
            .await;
        if require_full_prefix && blocks.len() != found {
            // The advertised owner served fewer blocks than the MetaServer
            // promised (stale advertisement or failed fetch). Keep the partial
            // result so poll_existing blacklists RDMA for this request instead
            // of the wait loop retrying the same fetch until timeout.
            warn!(
                "RDMA fetch returned fewer blocks than advertised: req_id={} node={} returned={} advertised={}",
                req_id,
                node,
                blocks.len(),
                found
            );
        }
        Some((found, blocks))
    }
}

#[cfg(not(feature = "rdma"))]
impl RdmaFetch {
    async fn try_fetch_prefix(
        &self,
        _req_id: &str,
        _namespace: &str,
        _remaining_hashes: &[Vec<u8>],
        _require_full_prefix: bool,
    ) -> Option<(usize, PrefetchResult)> {
        None
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum PrefetchSource {
    Ssd,
    Rdma,
}

impl PrefetchSource {
    const fn as_attribution(self) -> AttributionSource {
        match self {
            Self::Ssd => AttributionSource::Ssd,
            Self::Rdma => AttributionSource::Rdma,
        }
    }
}

struct PrefetchEntry {
    handle: JoinHandle<PrefetchTaskResult>,
    started_at: Instant,
}

struct PrefetchTaskResult {
    source: Option<PrefetchSource>,
    found: usize,
    cache_inserts: PrefetchResult,
    ready_blocks: Vec<Arc<SealedBlock>>,
    missing: usize,
}

struct PrefixScan<'a> {
    req_id: &'a str,
    namespace: &'a str,
    hashes: &'a [Vec<u8>],
    emit_tier_metrics: bool,
    wait_for_full_prefix: bool,
}

struct PrefetchStart<'a> {
    req_id: &'a str,
    namespace: &'a str,
    remaining: &'a [BlockKey],
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    hit: usize,
    emit_tier_metrics: bool,
    wait_for_full_prefix: bool,
}

struct PrefetchTaskDeps {
    rdma_fetch: Option<RdmaFetch>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    prefetch_state: Arc<Mutex<PrefetchState>>,
    max_prefetch_blocks: usize,
}

struct PrefetchTaskInput {
    req_id: String,
    namespace: String,
    remaining_keys: Vec<BlockKey>,
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    hit: usize,
    emit_tier_metrics: bool,
    wait_for_full_prefix: bool,
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Reserved SSD prefetch budget for active background tasks.
    reserved_ssd_prefetch_blocks: usize,
    /// req_ids where the advertised RDMA owner served fewer blocks than the
    /// MetaServer promised (stale advertisement or failed fetch). Prevents
    /// re-triggering RDMA on every subsequent poll for the same request.
    failed_remote: HashMap<String, Instant>,
}

impl PrefetchState {
    fn remove_entry(&mut self, req_id: &str) -> Option<PrefetchEntry> {
        self.active.remove(req_id)
    }
}

struct SsdPrefetchReservation {
    state: Arc<Mutex<PrefetchState>>,
    blocks: usize,
}

impl Drop for SsdPrefetchReservation {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        state.reserved_ssd_prefetch_blocks = state
            .reserved_ssd_prefetch_blocks
            .saturating_sub(self.blocks);
    }
}

pub(super) struct PrefetchScheduler {
    state: Arc<Mutex<PrefetchState>>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    rdma_fetch: Option<RdmaFetch>,
    metaserver_client: Option<Arc<MetaServerClient>>,
    max_prefetch_blocks: usize,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        rdma_fetch: Option<RdmaFetch>,
        metaserver_client: Option<Arc<MetaServerClient>>,
        max_prefetch_blocks: usize,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(PrefetchState {
                active: HashMap::new(),
                reserved_ssd_prefetch_blocks: 0,
                failed_remote: HashMap::new(),
            })),
            ssd_store,
            rdma_fetch,
            metaserver_client,
            max_prefetch_blocks,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        wait_for_full_prefix: bool,
    ) -> PrefetchStatus {
        // Default: this call may be the first decision and should attribute.
        match self.poll_existing(read_cache, req_id).await {
            PollResult::NoActivePrefetch => {}
            PollResult::StillLoading => {
                return PrefetchStatus::Loading;
            }
            PollResult::Ready(status) => return status,
        }

        self.full_prefix_scan(
            read_cache,
            PrefixScan {
                req_id,
                namespace,
                hashes,
                emit_tier_metrics: true,
                wait_for_full_prefix,
            },
        )
        .await
    }

    async fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> PollResult {
        let entry = {
            let mut state = self.state.lock();
            let Some(entry) = state.active.get(req_id) else {
                return PollResult::NoActivePrefetch;
            };
            if !entry.handle.is_finished() {
                return PollResult::StillLoading;
            }
            state
                .remove_entry(req_id)
                .expect("active entry must exist after readiness check")
        };

        let result = match entry.handle.await {
            Ok(result) => result,
            Err(err) => {
                warn!("Prefetch task failed for req_id={}: {}", req_id, err);
                PrefetchTaskResult {
                    source: None,
                    found: 0,
                    cache_inserts: Vec::new(),
                    ready_blocks: Vec::new(),
                    missing: 0,
                }
            }
        };

        // RDMA remote node can return fewer blocks than MetaServer promised
        // (likely evicted). Don't re-trigger RDMA on subsequent scans.
        if result.source == Some(PrefetchSource::Rdma)
            && result.cache_inserts.len() < result.found
            && result.found > 0
        {
            self.state
                .lock()
                .failed_remote
                .insert(req_id.to_string(), Instant::now());
            info!(
                "RDMA prefetch returned fewer blocks than expected: req_id={} returned={} expected={}",
                req_id,
                result.cache_inserts.len(),
                result.found
            );
        }

        // RDMA-fetched blocks that survive cache admission are now resident on
        // this node. Re-advertise only those resident blocks to the MetaServer
        // so peers can discover and fetch from here too. SSD prefetch is
        // skipped: those blocks were already registered by this node's own save
        // path, and eviction explicitly unregisters them.
        let rdma_registration = if result.source == Some(PrefetchSource::Rdma) {
            let resident_keys = read_cache.batch_insert_resident_keys(result.cache_inserts);
            rdma_registration_from_resident_keys(result.source, &resident_keys)
        } else {
            read_cache.batch_insert(result.cache_inserts);
            None
        };

        if let Some(client) = &self.metaserver_client
            && let Some((namespace, hashes)) = rdma_registration
        {
            client.try_register_namespace(namespace, hashes);
        }

        PollResult::Ready(PrefetchStatus::Ready {
            blocks: result.ready_blocks,
            missing: result.missing,
        })
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
        let (hit, prefix_blocks) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_scan_start.elapsed();
        let remaining = &keys[hit..];

        let task_start = Instant::now();
        let task_started = !remaining.is_empty()
            && self.start_prefetch_task(PrefetchStart {
                req_id: scan.req_id,
                namespace: scan.namespace,
                remaining,
                prefix_blocks: prefix_blocks.clone(),
                total: keys.len(),
                hit,
                emit_tier_metrics: scan.emit_tier_metrics,
                wait_for_full_prefix: scan.wait_for_full_prefix,
            });
        let task_schedule = task_start.elapsed();

        if task_started {
            info!(
                "Prefetch scheduling timing: req_id={} total_keys={} hit={} remaining={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
                scan.req_id,
                keys.len(),
                hit,
                remaining.len(),
                key_build,
                cache_scan,
                task_schedule,
                total_start.elapsed()
            );
            PrefetchStatus::Loading
        } else {
            let missing = keys.len() - hit;
            record_tier_attribution(
                keys.len(),
                hit,
                /* loading = */ 0,
                /* loading_source = */ None,
                scan.emit_tier_metrics,
            );

            info!(
                "Prefetch local-hit timing: req_id={} total_keys={} hit={} missing={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
                scan.req_id,
                keys.len(),
                hit,
                missing,
                key_build,
                cache_scan,
                task_schedule,
                total_start.elapsed()
            );
            PrefetchStatus::Ready {
                blocks: prefix_blocks,
                missing,
            }
        }
    }

    fn start_prefetch_task(&self, start: PrefetchStart<'_>) -> bool {
        if start.remaining.is_empty() {
            return false;
        }

        let mut state = self.state.lock();
        if state.active.contains_key(start.req_id) {
            return true;
        }

        let rdma_fetch = self
            .rdma_fetch
            .as_ref()
            .filter(|_| !state.failed_remote.contains_key(start.req_id))
            .cloned();

        if rdma_fetch.is_none() && self.ssd_store.is_none() {
            return false;
        }

        let deps = PrefetchTaskDeps {
            rdma_fetch,
            ssd_store: self.ssd_store.clone(),
            prefetch_state: Arc::clone(&self.state),
            max_prefetch_blocks: self.max_prefetch_blocks,
        };
        let input = PrefetchTaskInput {
            req_id: start.req_id.to_string(),
            namespace: start.namespace.to_string(),
            remaining_keys: start.remaining.to_vec(),
            prefix_blocks: start.prefix_blocks,
            total: start.total,
            hit: start.hit,
            emit_tier_metrics: start.emit_tier_metrics,
            wait_for_full_prefix: start.wait_for_full_prefix,
        };

        let handle = tokio::spawn(async move { run_prefetch_task(deps, input).await });

        state.active.insert(
            start.req_id.to_string(),
            PrefetchEntry {
                handle,
                started_at: Instant::now(),
            },
        );
        true
    }

    /// Drop stale active entries and sweep old `failed_remote` entries.
    ///
    /// Dropping a `JoinHandle` detaches the task; it keeps running so RDMA
    /// transfer locks can still be released by the normal completion path.
    pub(super) fn gc_stale_entries(
        &self,
        active_max_age: std::time::Duration,
        failed_remote_max_age: std::time::Duration,
    ) -> (usize, usize) {
        let mut state = self.state.lock();
        let active_before = state.active.len();
        state
            .active
            .retain(|_, entry| entry.started_at.elapsed() < active_max_age);
        let active_removed = active_before - state.active.len();

        let failed_before = state.failed_remote.len();
        state
            .failed_remote
            .retain(|_, ts| ts.elapsed() < failed_remote_max_age);
        (active_removed, failed_before - state.failed_remote.len())
    }
}

/// Attribute this `query_prefetch` decision. Skips attribution when:
/// * `emit_tier_metrics == false` (e.g. post-completion fall-through);
/// * `total` is zero (no decision to attribute).
fn record_tier_attribution(
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

fn reserve_ssd_prefetch_slots(
    state: Arc<Mutex<PrefetchState>>,
    max_prefetch_blocks: usize,
    requested: usize,
    require_full: bool,
) -> Option<(usize, SsdPrefetchReservation)> {
    if requested == 0 {
        return None;
    }

    let mut guard = state.lock();
    let available = max_prefetch_blocks.saturating_sub(guard.reserved_ssd_prefetch_blocks);

    if available == 0 || (require_full && available < requested) {
        core_metrics()
            .ssd_prefetch_backpressure_blocks
            .add(requested as u64, &[]);
        return None;
    }

    let reserved = requested.min(available);
    let skipped = requested - reserved;
    if skipped > 0 {
        core_metrics()
            .ssd_prefetch_backpressure_blocks
            .add(skipped as u64, &[]);
    }

    guard.reserved_ssd_prefetch_blocks += reserved;
    drop(guard);

    Some((
        reserved,
        SsdPrefetchReservation {
            state,
            blocks: reserved,
        },
    ))
}

fn build_ready_result(
    prefix_blocks: Vec<Arc<SealedBlock>>,
    total: usize,
    source: Option<PrefetchSource>,
    found: usize,
    requested_keys: &[BlockKey],
    cache_inserts: PrefetchResult,
) -> PrefetchTaskResult {
    let mut ready_blocks = prefix_blocks;
    let inserts_by_key: HashMap<_, _> = cache_inserts
        .iter()
        .map(|(key, block)| (key, block))
        .collect();
    ready_blocks.extend(
        requested_keys
            .iter()
            .map_while(|key| inserts_by_key.get(key).map(|block| Arc::clone(*block))),
    );
    let missing = total.saturating_sub(ready_blocks.len());
    PrefetchTaskResult {
        source,
        found,
        cache_inserts,
        ready_blocks,
        missing,
    }
}

fn rdma_registration_from_resident_keys(
    source: Option<PrefetchSource>,
    resident_keys: &[BlockKey],
) -> Option<(String, Vec<Vec<u8>>)> {
    if source != Some(PrefetchSource::Rdma) || resident_keys.is_empty() {
        return None;
    }

    let namespace = resident_keys[0].namespace.clone();
    let hashes = resident_keys.iter().map(|key| key.hash.clone()).collect();
    Some((namespace, hashes))
}

async fn run_prefetch_task(deps: PrefetchTaskDeps, input: PrefetchTaskInput) -> PrefetchTaskResult {
    let PrefetchTaskInput {
        req_id,
        namespace,
        remaining_keys,
        prefix_blocks,
        total,
        hit,
        emit_tier_metrics,
        wait_for_full_prefix,
    } = input;
    let remaining_hashes: Vec<Vec<u8>> = remaining_keys.iter().map(|k| k.hash.clone()).collect();

    if let Some(rdma) = deps.rdma_fetch.as_ref()
        && let Some((found, blocks)) = rdma
            .try_fetch_prefix(&req_id, &namespace, &remaining_hashes, wait_for_full_prefix)
            .await
    {
        record_tier_attribution(
            total,
            hit,
            found,
            Some(PrefetchSource::Rdma.as_attribution()),
            emit_tier_metrics,
        );
        return build_ready_result(
            prefix_blocks,
            total,
            Some(PrefetchSource::Rdma),
            found,
            &remaining_keys[..found],
            blocks,
        );
    }

    if let Some(ssd) = deps.ssd_store.as_ref() {
        let found = ssd.prefix_len(&remaining_keys);
        if (!wait_for_full_prefix || found == remaining_keys.len())
            && let Some((reserved, _reservation)) = reserve_ssd_prefetch_slots(
                Arc::clone(&deps.prefetch_state),
                deps.max_prefetch_blocks,
                found,
                wait_for_full_prefix,
            )
        {
            let keys = remaining_keys[..reserved].to_vec();
            let (found, blocks) = ssd.prefetch_prefix(keys).await;
            // wait_for_full_prefix is all-or-nothing: a partial SSD result
            // (backpressured reservation or short read) must not let the
            // caller proceed with a partial prefix.
            if found > 0 && (!wait_for_full_prefix || found == remaining_keys.len()) {
                record_tier_attribution(
                    total,
                    hit,
                    found,
                    Some(PrefetchSource::Ssd.as_attribution()),
                    emit_tier_metrics,
                );
                return build_ready_result(
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Ssd),
                    found,
                    &remaining_keys[..found],
                    blocks,
                );
            }
        }
    }

    if wait_for_full_prefix && let Some(rdma) = deps.rdma_fetch {
        let started_at = Instant::now();
        while started_at.elapsed() < REMOTE_WAIT_TIMEOUT {
            tokio::time::sleep(REMOTE_WAIT_POLL_INTERVAL).await;
            if let Some((found, blocks)) = rdma
                .try_fetch_prefix(&req_id, &namespace, &remaining_hashes, true)
                .await
            {
                record_tier_attribution(
                    total,
                    hit,
                    found,
                    Some(PrefetchSource::Rdma.as_attribution()),
                    emit_tier_metrics,
                );
                return build_ready_result(
                    prefix_blocks,
                    total,
                    Some(PrefetchSource::Rdma),
                    found,
                    &remaining_keys[..found],
                    blocks,
                );
            }
        }
        warn!(
            "Timed out waiting for remote prefix: req_id={} timeout_secs={}",
            req_id,
            REMOTE_WAIT_TIMEOUT.as_secs()
        );
    }

    record_tier_attribution(total, hit, 0, None, emit_tier_metrics);
    build_ready_result(prefix_blocks, total, None, 0, &[], Vec::new())
}

enum PollResult {
    NoActivePrefetch,
    StillLoading,
    Ready(PrefetchStatus),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> BlockKey {
        BlockKey::new("ns".to_string(), vec![n])
    }

    fn block() -> Arc<SealedBlock> {
        Arc::new(SealedBlock::from_slots(Vec::new()))
    }

    #[test]
    fn ready_result_rebuilds_prefix_in_requested_key_order() {
        let local = block();
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b2 = block();
        let b3 = block();

        let result = build_ready_result(
            vec![Arc::clone(&local)],
            4,
            Some(PrefetchSource::Ssd),
            3,
            &[k1.clone(), k2.clone(), k3.clone()],
            vec![
                (k2, Arc::clone(&b2)),
                (k1, Arc::clone(&b1)),
                (k3, Arc::clone(&b3)),
            ],
        );

        assert_eq!(result.ready_blocks.len(), 4);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &local));
        assert!(Arc::ptr_eq(&result.ready_blocks[1], &b1));
        assert!(Arc::ptr_eq(&result.ready_blocks[2], &b2));
        assert!(Arc::ptr_eq(&result.ready_blocks[3], &b3));
        assert_eq!(result.missing, 0);
        assert_eq!(result.cache_inserts.len(), 3);
    }

    #[test]
    fn ready_result_stops_at_first_missing_prefetch_key() {
        let k1 = key(1);
        let k2 = key(2);
        let k3 = key(3);
        let b1 = block();
        let b3 = block();

        let result = build_ready_result(
            Vec::new(),
            3,
            Some(PrefetchSource::Ssd),
            3,
            &[k1.clone(), k2, k3.clone()],
            vec![(k3, b3), (k1, Arc::clone(&b1))],
        );

        assert_eq!(result.ready_blocks.len(), 1);
        assert!(Arc::ptr_eq(&result.ready_blocks[0], &b1));
        assert_eq!(result.missing, 2);
        assert_eq!(result.cache_inserts.len(), 2);
    }

    #[test]
    fn rdma_registration_uses_only_resident_keys() {
        let k1 = key(1);
        let k3 = key(3);

        let (namespace, hashes) =
            rdma_registration_from_resident_keys(Some(PrefetchSource::Rdma), &[k1, k3])
                .expect("RDMA resident keys should register");

        assert_eq!(namespace, "ns");
        assert_eq!(hashes, vec![vec![1], vec![3]]);
    }

    #[test]
    fn rdma_registration_skips_ssd_and_empty_resident_keys() {
        let k1 = key(1);

        assert!(rdma_registration_from_resident_keys(Some(PrefetchSource::Ssd), &[k1]).is_none());
        assert!(rdma_registration_from_resident_keys(Some(PrefetchSource::Rdma), &[]).is_none());
        assert!(rdma_registration_from_resident_keys(None, &[]).is_none());
    }

    /// Feed a finished prefetch task with the given outcome through
    /// `poll_existing` and report whether the request got blacklisted.
    async fn poll_outcome_blacklists_req(
        source: Option<PrefetchSource>,
        found: usize,
        inserts: usize,
    ) -> bool {
        let scheduler = PrefetchScheduler::new(None, None, None, 16);
        let read_cache = ReadCache::new(1 << 20, false, None);
        let result = PrefetchTaskResult {
            source,
            found,
            cache_inserts: (0..inserts).map(|i| (key(i as u8), block())).collect(),
            ready_blocks: Vec::new(),
            missing: 0,
        };
        let handle = tokio::spawn(async move { result });
        while !handle.is_finished() {
            tokio::task::yield_now().await;
        }
        scheduler.state.lock().active.insert(
            "req".to_string(),
            PrefetchEntry {
                handle,
                started_at: Instant::now(),
            },
        );

        let _ = scheduler.poll_existing(&read_cache, "req").await;

        scheduler.state.lock().failed_remote.contains_key("req")
    }

    #[tokio::test]
    async fn short_rdma_result_blacklists_request() {
        // Partial prefix and total failure both mean the advertised owner
        // could not serve what the MetaServer promised.
        assert!(poll_outcome_blacklists_req(Some(PrefetchSource::Rdma), 3, 2).await);
        assert!(poll_outcome_blacklists_req(Some(PrefetchSource::Rdma), 3, 0).await);
    }

    #[tokio::test]
    async fn full_or_non_rdma_result_does_not_blacklist() {
        assert!(!poll_outcome_blacklists_req(Some(PrefetchSource::Rdma), 3, 3).await);
        assert!(!poll_outcome_blacklists_req(Some(PrefetchSource::Ssd), 3, 2).await);
        assert!(!poll_outcome_blacklists_req(None, 0, 0).await);
    }

    #[test]
    fn gc_sweeps_only_expired_failed_remote_entries() {
        let scheduler = PrefetchScheduler::new(None, None, None, 16);
        scheduler
            .state
            .lock()
            .failed_remote
            .insert("req".to_string(), Instant::now());

        let (_, swept) = scheduler.gc_stale_entries(Duration::ZERO, Duration::from_secs(60));
        assert_eq!(swept, 0);

        let (_, swept) = scheduler.gc_stale_entries(Duration::ZERO, Duration::ZERO);
        assert_eq!(swept, 1);
        assert!(scheduler.state.lock().failed_remote.is_empty());
    }

    #[test]
    fn strict_ssd_reservation_is_all_or_nothing() {
        let state = Arc::new(Mutex::new(PrefetchState {
            active: HashMap::new(),
            reserved_ssd_prefetch_blocks: 0,
            failed_remote: HashMap::new(),
        }));
        let (_n, hold) = reserve_ssd_prefetch_slots(Arc::clone(&state), 10, 6, false)
            .expect("reservation within capacity should succeed");
        // 4 of 10 slots remain.

        // Strict request above availability is denied and reserves nothing.
        assert!(reserve_ssd_prefetch_slots(Arc::clone(&state), 10, 5, true).is_none());
        assert_eq!(state.lock().reserved_ssd_prefetch_blocks, 6);

        // Strict request within availability reserves the full amount.
        let (reserved, hold2) = reserve_ssd_prefetch_slots(Arc::clone(&state), 10, 4, true)
            .expect("exact reservation should succeed");
        assert_eq!(reserved, 4);
        assert_eq!(state.lock().reserved_ssd_prefetch_blocks, 10);
        drop(hold2);

        // Non-strict still reserves partially when the full amount is denied.
        let (reserved, _hold3) = reserve_ssd_prefetch_slots(Arc::clone(&state), 10, 5, false)
            .expect("partial reservation should succeed");
        assert_eq!(reserved, 4);
        drop(hold);
    }
}
