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
        let plan = self.0.query_plan(namespace, remaining_hashes).await?;
        let found = plan.block_count();
        if require_full_prefix && found != remaining_hashes.len() {
            return None;
        }
        let blocks = self
            .0
            .fetch_plan(&plan, req_id, namespace, remaining_hashes)
            .await;
        if require_full_prefix && blocks.len() != found {
            // One planned segment served fewer blocks than the MetaServer
            // promised (stale advertisement or failed fetch). Keep the partial
            // result so poll_existing blacklists RDMA for this request instead
            // of the wait loop retrying the same fetch until timeout.
            warn!(
                "RDMA fetch returned fewer blocks than planned: req_id={} returned={} planned={}",
                req_id,
                blocks.len(),
                found
            );
        }
        Some((found, blocks))
    }

    /// How many leading `hashes` peers hold, per the MetaServer.
    pub(super) async fn remote_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> usize {
        self.0
            .query_plan(namespace, hashes)
            .await
            .map_or(0, |plan| plan.block_count())
    }
}

#[cfg(not(feature = "rdma"))]
impl RdmaFetch {
    pub(super) async fn remote_prefix(&self, _namespace: &str, _hashes: &[Vec<u8>]) -> usize {
        0
    }

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

/// Keys of one namespace a parts fetch brings in as a prefix: peers serve
/// what they can from the first key, then SSD from where they stopped.
pub(crate) struct FetchPart {
    pub(crate) keys: Vec<BlockKey>,
    /// Reserve SSD budget for every key SSD should serve, or for none.
    pub(crate) whole: bool,
}

/// Blocks a parts fetch brought in, by the tier that served them.
#[derive(Default)]
struct PartsResult {
    rdma: PrefetchResult,
    ssd: PrefetchResult,
}

struct PartsEntry {
    handle: JoinHandle<PartsResult>,
    started_at: Instant,
}

pub(crate) enum PartsPoll {
    /// No parts fetch of this request.
    Idle,
    Loading,
    /// The fetch ended; its blocks are now in the read cache.
    Done(PrefetchResult),
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Parts fetches, by request.
    parts: HashMap<String, PartsEntry>,
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
                parts: HashMap::new(),
                reserved_ssd_prefetch_blocks: 0,
                failed_remote: HashMap::new(),
            })),
            ssd_store,
            rdma_fetch,
            metaserver_client,
            max_prefetch_blocks,
        }
    }

    /// The RDMA fetcher, if this node fetches from peers.
    pub(super) fn rdma_fetch(&self) -> Option<&RdmaFetch> {
        self.rdma_fetch.as_ref()
    }

    /// Start fetching `parts` for `req_id`, each peers first and then SSD.
    /// False when nothing can serve them; true as well when a fetch of
    /// `req_id` already runs.
    pub(super) fn start_parts(&self, req_id: &str, parts: Vec<FetchPart>) -> bool {
        let parts: Vec<FetchPart> = parts.into_iter().filter(|p| !p.keys.is_empty()).collect();
        if parts.is_empty() || (self.rdma_fetch.is_none() && self.ssd_store.is_none()) {
            return false;
        }
        let mut state = self.state.lock();
        if state.parts.contains_key(req_id) {
            return true;
        }
        let deps = PrefetchTaskDeps {
            rdma_fetch: self.rdma_fetch.clone(),
            ssd_store: self.ssd_store.clone(),
            prefetch_state: Arc::clone(&self.state),
            max_prefetch_blocks: self.max_prefetch_blocks,
        };
        let req = req_id.to_string();
        let handle = tokio::spawn(async move { run_parts_task(deps, req, parts).await });
        state.parts.insert(
            req_id.to_string(),
            PartsEntry {
                handle,
                started_at: Instant::now(),
            },
        );
        true
    }

    /// Poll the parts fetch of `req_id`. A finished one enters the read
    /// cache, and the blocks peers served are announced like local ones.
    pub(super) async fn poll_parts(&self, read_cache: &ReadCache, req_id: &str) -> PartsPoll {
        let entry = {
            let mut state = self.state.lock();
            match state.parts.get(req_id) {
                None => return PartsPoll::Idle,
                Some(e) if !e.handle.is_finished() => return PartsPoll::Loading,
                Some(_) => state.parts.remove(req_id).expect("checked above"),
            }
        };
        let result = entry.handle.await.unwrap_or_else(|e| {
            warn!("Parts prefetch task failed for req_id={req_id}: {e}");
            PartsResult::default()
        });
        // As in `poll_existing`: SSD-served blocks were registered when saved here.
        self.admit_rdma(read_cache, &result.rdma);
        read_cache.batch_insert(result.ssd.clone());
        let mut fetched = result.rdma;
        fetched.extend(result.ssd);
        PartsPoll::Done(fetched)
    }

    /// Admit blocks peers served: retained like local inserts, so cache
    /// pressure does not prefer them over blocks exposed as P2P sources, and
    /// the resident ones re-advertised so peers can fetch them from here too.
    fn admit_rdma(&self, read_cache: &ReadCache, blocks: &PrefetchResult) {
        let resident = read_cache.batch_insert_refs(blocks);
        let Some(client) = &self.metaserver_client else {
            return;
        };
        for (namespace, hashes) in hashes_by_namespace(resident) {
            client.try_register_namespace_without_reclaim_hint(namespace, hashes);
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
        // this node. Keep them retained, like local inserts, so a later cache
        // pressure event does not prefer them over blocks exposed as P2P
        // sources. Re-advertise only those resident blocks to the MetaServer
        // so peers can discover and fetch from here too. SSD prefetch is
        // skipped: those blocks were already registered by this node's own save
        // path, and eviction explicitly unregisters them.
        if result.source == Some(PrefetchSource::Rdma) {
            self.admit_rdma(read_cache, &result.cache_inserts);
        } else {
            read_cache.batch_insert(result.cache_inserts);
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
        let parts_before = state.parts.len();
        state
            .parts
            .retain(|_, entry| entry.started_at.elapsed() < active_max_age);
        let active_removed = active_before - state.active.len() + parts_before - state.parts.len();

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

/// The hashes of `keys` per namespace, in first-seen order: a KV fetch has
/// one namespace, an object fetch may span several.
fn hashes_by_namespace(keys: Vec<BlockKey>) -> Vec<(String, Vec<Vec<u8>>)> {
    let mut out: Vec<(String, Vec<Vec<u8>>)> = Vec::new();
    for key in keys {
        match out.iter_mut().find(|(ns, _)| *ns == key.namespace) {
            Some((_, hashes)) => hashes.push(key.hash),
            None => out.push((key.namespace, vec![key.hash])),
        }
    }
    out
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

async fn run_parts_task(
    deps: PrefetchTaskDeps,
    req_id: String,
    parts: Vec<FetchPart>,
) -> PartsResult {
    let mut out = PartsResult::default();
    let from_peers =
        futures::future::join_all(parts.iter().map(|p| fetch_from_peers(&deps, &req_id, p))).await;
    let Some(ssd) = &deps.ssd_store else {
        out.rdma = from_peers.into_iter().flatten().collect();
        return out;
    };
    // Every part's SSD reads go in one batch, like one KV prefetch: the SSD
    // prefetch queue is shallow and drops batches past its depth.
    let mut reads = Vec::new();
    let mut reservations = Vec::new();
    for (part, peers) in parts.iter().zip(&from_peers) {
        let got: std::collections::HashSet<&BlockKey> = peers.iter().map(|(k, _)| k).collect();
        let rest = &part.keys[part.keys.iter().take_while(|k| got.contains(k)).count()..];
        let found = ssd.prefix_len(rest);
        if (part.whole && found < rest.len()) || found == 0 {
            continue;
        }
        if let Some((reserved, reservation)) = reserve_ssd_prefetch_slots(
            Arc::clone(&deps.prefetch_state),
            deps.max_prefetch_blocks,
            found,
            part.whole,
        ) {
            reads.extend_from_slice(&rest[..reserved]);
            reservations.push(reservation);
        }
    }
    out.rdma = from_peers.into_iter().flatten().collect();
    if !reads.is_empty() {
        out.ssd = ssd.prefetch_prefix(reads).await.1;
    }
    drop(reservations);
    out
}

/// What peers serve of `part`, from its first key.
async fn fetch_from_peers(
    deps: &PrefetchTaskDeps,
    req_id: &str,
    part: &FetchPart,
) -> PrefetchResult {
    let Some(rdma) = &deps.rdma_fetch else {
        return Vec::new();
    };
    let namespace = &part.keys[0].namespace;
    debug_assert!(part.keys.iter().all(|k| &k.namespace == namespace));
    let hashes: Vec<Vec<u8>> = part.keys.iter().map(|k| k.hash.clone()).collect();
    rdma.try_fetch_prefix(req_id, namespace, &hashes, false)
        .await
        .map(|(_, blocks)| blocks)
        .unwrap_or_default()
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
        Arc::new(SealedBlock::from_slots(Vec::new(), 0))
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
    fn rdma_registration_groups_resident_keys_by_namespace() {
        let other = |h: u8| BlockKey::new("other".into(), vec![h]);
        for (case, keys, want) in [
            ("none", vec![], vec![]),
            (
                "one namespace",
                vec![key(1), key(3)],
                vec![("ns", vec![vec![1], vec![3]])],
            ),
            (
                "two namespaces",
                vec![key(1), other(2), key(3)],
                vec![("ns", vec![vec![1], vec![3]]), ("other", vec![vec![2]])],
            ),
        ] {
            let want: Vec<(String, Vec<Vec<u8>>)> = want
                .into_iter()
                .map(|(ns, h)| (ns.to_string(), h))
                .collect();
            assert_eq!(hashes_by_namespace(keys), want, "{case}");
        }
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
            parts: HashMap::new(),
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
