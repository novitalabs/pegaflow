//! Spill tier: hand cold retained blocks to a peer before eviction drops them.
//!
//! A spill source keeps `free + reclaimable` pool bytes at or above a reserve
//! by offering its coldest retained blocks to a spill target. The target pulls
//! them through the regular P2P path (`QueryBlocksForTransfer` + RDMA READ),
//! keeps them retained, and registers them with the MetaServer. Serving that
//! pull demotes the source copies to reclaimable, so pressure eviction drops
//! them first while every peer can still recall the blocks from the target.
//!
//! Custody rule: a node must keep its retained copies; a reclaimable copy has
//! another copy elsewhere. Spilling hands custody to the target the same way a
//! P2P fetch hands it to the fetcher. Eviction itself never waits on a spill:
//! when the source falls behind, eviction drops retained blocks as before.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use log::{debug, warn};
use opentelemetry::KeyValue;
use parking_lot::Mutex;
use pegaflow_common::grpc::{GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL, GRPC_CONNECT_TIMEOUT};
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{SpillOfferRequest, SpillOfferResponse};
use tokio::sync::{Notify, watch};
use tonic::transport::{Channel, Endpoint};

use super::{
    ReadCache, SpillAdoption, SpillSourceConfig, SpillTargetConfig, SpillTargetSelection,
    StorageEngine,
};
use crate::backing::RdmaFetchStore;
use crate::block::BlockKey;
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

/// Fallback re-check of the reserve; evictions wake the source sooner.
const POLL_INTERVAL: Duration = Duration::from_millis(50);
/// Extra wait on an offer beyond this node's transfer lock timeout, which
/// bounds the target's RDMA pull, so a slow pull is never abandoned midway.
const OFFER_TIMEOUT_MARGIN: Duration = Duration::from_secs(10);
const INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const MAX_BACKOFF: Duration = Duration::from_secs(10);

pub(super) struct SpillSource {
    config: SpillSourceConfig,
    advertise_addr: String,
    targets: SpillTargets,
    /// MetaServer assignment; replaces `targets` whenever it changes.
    assigned: Option<watch::Receiver<Vec<String>>>,
}

impl SpillSource {
    pub(super) fn new(
        config: SpillSourceConfig,
        advertise_addr: String,
        assigned: Option<watch::Receiver<Vec<String>>>,
    ) -> Self {
        let targets = match &config.targets {
            SpillTargetSelection::Static(addrs) => SpillTargets::new(addrs),
            SpillTargetSelection::MetaServer => SpillTargets::new(&[]),
        };
        Self {
            config,
            advertise_addr,
            targets,
            assigned,
        }
    }

    fn refresh_targets(&mut self) {
        let Some(assigned) = self.assigned.as_mut() else {
            return;
        };
        if assigned.has_changed().unwrap_or(false) {
            let addrs = assigned.borrow_and_update().clone();
            self.targets.set_addrs(&addrs);
        }
    }
}

/// Background loop of a spill source. Exits once the engine is dropped.
pub(super) async fn run_spill_source(
    engine: Weak<StorageEngine>,
    mut source: SpillSource,
    wake: Arc<Notify>,
) {
    let mut progressed = false;
    loop {
        if !progressed {
            tokio::select! {
                () = wake.notified() => {}
                () = tokio::time::sleep(POLL_INTERVAL) => {}
            }
        }
        let Some(engine) = engine.upgrade() else {
            return;
        };
        progressed = engine.spill_once(&mut source).await;
    }
}

impl StorageEngine {
    /// Offer the coldest retained blocks once when `free + reclaimable` is
    /// below the reserve. Returns true when the target took custody of
    /// anything, so the caller re-checks the reserve without waiting.
    async fn spill_once(&self, source: &mut SpillSource) -> bool {
        source.refresh_targets();
        let deficit = spill_deficit(
            source.config.reserve_bytes,
            &self.allocator.pool_usages(),
            self.read_cache.reclaimable_bytes(),
        );
        if deficit == 0 {
            return false;
        }
        let Some(index) = source.targets.pick(Instant::now()) else {
            return false;
        };
        let mut candidates = self
            .read_cache
            .coldest_retained(deficit.min(source.config.batch_bytes));
        let Some(namespace) = candidates.first().map(|(key, _)| key.namespace.clone()) else {
            return false;
        };
        // One namespace per offer; blocks of other namespaces go in later rounds.
        candidates.retain(|(key, _)| key.namespace == namespace);
        let total_bytes: u64 = candidates.iter().map(|(_, bytes)| bytes).sum();
        let hashes: Vec<Vec<u8>> = candidates.into_iter().map(|(key, _)| key.hash).collect();
        let keys = |hashes: &[Vec<u8>]| -> Vec<BlockKey> {
            hashes
                .iter()
                .map(|hash| BlockKey::new(namespace.clone(), hash.clone()))
                .collect()
        };

        let metrics = core_metrics();
        metrics.spill_offered_bytes.add(total_bytes, &[]);
        let request = SpillOfferRequest {
            source_addr: source.advertise_addr.clone(),
            namespace: namespace.clone(),
            block_hashes: hashes.clone(),
            total_bytes,
        };
        let target = source.targets.addr(index).to_string();
        let timeout = self.transfer_lock.lock_timeout() + OFFER_TIMEOUT_MARGIN;
        match source.targets.offer(index, request, timeout).await {
            Ok(response) if response.accepted => {
                let unclaimed = unclaimed_hashes(
                    &hashes,
                    &response.adopted_hashes,
                    &response.already_held_hashes,
                );
                // Adopted blocks were demoted when the target locked them for
                // its pull; blocks the target already held never were.
                self.read_cache
                    .mark_reclaimable_hashes(&namespace, &response.already_held_hashes);
                self.read_cache.restore_retained_cold(&keys(&unclaimed));

                let adopted = response.adopted_hashes.len();
                let already_held = response.already_held_hashes.len();
                metrics
                    .spill_offers
                    .add(1, &[KeyValue::new("result", "accepted")]);
                record_spill_blocks(adopted, already_held, unclaimed.len());
                debug!(
                    "Spill offer to {target}: namespace={namespace} offered={} bytes={total_bytes} adopted={adopted} already_held={already_held} unclaimed={}",
                    hashes.len(),
                    unclaimed.len(),
                );
                let claimed = adopted + already_held > 0;
                if claimed {
                    source.targets.record_success(index);
                } else if self
                    .read_cache
                    .contains_keys(&keys(&hashes))
                    .contains(&true)
                {
                    // The target could not take blocks this node still holds.
                    source.targets.record_failure(index, Instant::now());
                }
                // Otherwise pressure evicted the whole batch before the target
                // pulled it, which says nothing about the target.
                claimed
            }
            Ok(response) => {
                // Rejected before pulling anything, so no source copy was demoted.
                metrics
                    .spill_offers
                    .add(1, &[KeyValue::new("result", "rejected")]);
                debug!(
                    "Spill offer rejected by {target}: {}",
                    response
                        .status
                        .map(|status| status.message)
                        .unwrap_or_default()
                );
                source.targets.record_failure(index, Instant::now());
                false
            }
            Err(error) => {
                // The target may have demoted part of the offer before failing.
                self.read_cache.restore_retained_cold(&keys(&hashes));
                metrics
                    .spill_offers
                    .add(1, &[KeyValue::new("result", "error")]);
                record_spill_blocks(0, 0, hashes.len());
                warn!("{error}");
                source.targets.record_failure(index, Instant::now());
                false
            }
        }
    }

    /// Spill target: take custody of `hashes` from the source at `source_addr`.
    ///
    /// Blocks already resident here are retained again; the rest are pulled
    /// with the regular RDMA fetch path, kept retained, and advertised to the
    /// MetaServer so peers recall them from this node.
    pub(crate) async fn adopt_spill(
        &self,
        source_addr: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        total_bytes: u64,
    ) -> SpillAdoption {
        let metrics = core_metrics();
        let (Some(admission), Some(store)) = (
            self.spill_admission.as_ref(),
            self.rdma_fetch_store.as_ref(),
        ) else {
            metrics
                .spill_adoptions
                .add(1, &[KeyValue::new("result", "disabled")]);
            return SpillAdoption::Rejected("spill target is not enabled on this node");
        };
        if source_addr == admission.own_addr {
            // Adopting its own blocks would demote this node's only copies.
            metrics
                .spill_adoptions
                .add(1, &[KeyValue::new("result", "disabled")]);
            return SpillAdoption::Rejected("spill offer came from this node itself");
        }
        let Some(permit) = admission.try_admit(total_bytes, Instant::now()) else {
            metrics
                .spill_adoptions
                .add(1, &[KeyValue::new("result", "throttled")]);
            return SpillAdoption::Rejected("spill target is over its pull budget");
        };

        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let held = self.read_cache.retain_keys(&keys);
        let held_keys: HashSet<&BlockKey> = held.iter().collect();
        let missing: Vec<Vec<u8>> = keys
            .iter()
            .filter(|key| !held_keys.contains(key))
            .map(|key| key.hash.clone())
            .collect();
        let already_held: Vec<Vec<u8>> = held.into_iter().map(|key| key.hash).collect();

        // Run the pull detached, as prefetch does: if the source cancels the
        // offer, dropping an RDMA fetch midway can strand its handshake or free
        // staging memory that in-flight READs still write into.
        let pull = tokio::spawn(adopt_blocks(AdoptBlocks {
            store: Arc::clone(store),
            read_cache: Arc::clone(&self.read_cache),
            metaserver_client: self.metaserver_client.clone(),
            source_addr: source_addr.to_string(),
            namespace: namespace.to_string(),
            missing,
            already_held: already_held.clone(),
            _permit: permit,
        }));
        let adopted = pull.await.unwrap_or_else(|error| {
            warn!("Spill adoption from {source_addr} failed: {error}");
            Vec::new()
        });

        metrics
            .spill_adoptions
            .add(1, &[KeyValue::new("result", "accepted")]);
        SpillAdoption::Accepted {
            adopted,
            already_held,
        }
    }
}

/// A spill pull that runs to completion even if its offer is cancelled.
struct AdoptBlocks {
    store: Arc<RdmaFetchStore>,
    read_cache: Arc<ReadCache>,
    metaserver_client: Option<Arc<MetaServerClient>>,
    source_addr: String,
    namespace: String,
    missing: Vec<Vec<u8>>,
    already_held: Vec<Vec<u8>>,
    /// Counts the pulled bytes against the admission budget until done.
    _permit: AdmissionPermit,
}

/// Pull `missing` from the source, keep the blocks retained, and advertise
/// every block this node now keeps for the offer. Returns the adopted hashes.
async fn adopt_blocks(task: AdoptBlocks) -> Vec<Vec<u8>> {
    let mut adopted = Vec::new();
    if !task.missing.is_empty() {
        let fetched = task
            .store
            .fetch_blocks(&task.source_addr, "spill", &task.namespace, &task.missing)
            .await;
        let resident: HashSet<BlockKey> = task
            .read_cache
            .batch_insert_refs(&fetched)
            .into_iter()
            .collect();
        let mut adopted_bytes = 0u64;
        for (key, block) in fetched {
            if resident.contains(&key) {
                adopted_bytes = adopted_bytes.saturating_add(block.memory_footprint());
                adopted.push(key.hash);
            }
        }
        core_metrics().spill_adopted_bytes.add(adopted_bytes, &[]);
    }
    // Already-held blocks are advertised again: their earlier registration may
    // have been dropped or lost with a MetaServer restart.
    if let Some(client) = &task.metaserver_client {
        client.try_register_namespace_without_reclaim_hint(
            task.namespace,
            adopted.iter().chain(&task.already_held).cloned().collect(),
        );
    }
    adopted
}

/// Bytes the source still has to spill for `free + reclaimable` to reach the reserve.
///
/// Each pool (one per NUMA node) owes a capacity-proportional share of the
/// reserve, and only its own free bytes count toward that share: pressure
/// eviction runs per pool, so an idle node's free memory must not hide a full
/// one. Reclaimable bytes are not tracked per node; like eviction, which picks
/// victims by global LRU, the spill tier treats them as one pool.
fn spill_deficit(reserve_bytes: u64, pools: &[(u64, u64)], reclaimable_bytes: u64) -> u64 {
    let capacity: u64 = pools.iter().map(|&(_, total)| total).sum();
    if capacity == 0 {
        return 0;
    }
    let shortfall: u64 = pools
        .iter()
        .map(|&(used, total)| {
            let share =
                (u128::from(reserve_bytes) * u128::from(total) / u128::from(capacity)) as u64;
            share.saturating_sub(total.saturating_sub(used))
        })
        .sum();
    shortfall.saturating_sub(reclaimable_bytes)
}

/// Offered hashes the target neither adopted nor already held, in offer order.
fn unclaimed_hashes(
    offered: &[Vec<u8>],
    adopted: &[Vec<u8>],
    already_held: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    let claimed: HashSet<&[u8]> = adopted
        .iter()
        .chain(already_held)
        .map(Vec::as_slice)
        .collect();
    offered
        .iter()
        .filter(|hash| !claimed.contains(hash.as_slice()))
        .cloned()
        .collect()
}

fn record_spill_blocks(adopted: usize, already_held: usize, unclaimed: usize) {
    let metrics = core_metrics();
    for (outcome, count) in [
        ("adopted", adopted),
        ("already_held", already_held),
        ("unclaimed", unclaimed),
    ] {
        if count > 0 {
            metrics
                .spill_blocks
                .add(count as u64, &[KeyValue::new("outcome", outcome)]);
        }
    }
}

struct SpillTarget {
    addr: String,
    client: Option<EngineClient<Channel>>,
    backoff: Duration,
    retry_at: Option<Instant>,
}

impl SpillTarget {
    fn new(addr: &str) -> Self {
        Self {
            addr: addr.to_string(),
            client: None,
            backoff: INITIAL_BACKOFF,
            retry_at: None,
        }
    }
}

/// Ordered spill targets with per-target backoff. An earlier target wins
/// whenever it is not backing off, so traffic returns to the primary once it
/// recovers.
struct SpillTargets(Vec<SpillTarget>);

impl SpillTargets {
    fn new(addrs: &[String]) -> Self {
        Self(addrs.iter().map(|addr| SpillTarget::new(addr)).collect())
    }

    /// Replace the target list, keeping connection and backoff state of the
    /// targets that stay.
    fn set_addrs(&mut self, addrs: &[String]) {
        let mut previous: HashMap<String, SpillTarget> = self
            .0
            .drain(..)
            .map(|target| (target.addr.clone(), target))
            .collect();
        self.0 = addrs
            .iter()
            .map(|addr| {
                previous
                    .remove(addr)
                    .unwrap_or_else(|| SpillTarget::new(addr))
            })
            .collect();
    }

    fn pick(&self, now: Instant) -> Option<usize> {
        self.0
            .iter()
            .position(|target| target.retry_at.is_none_or(|at| now >= at))
    }

    fn addr(&self, index: usize) -> &str {
        &self.0[index].addr
    }

    fn record_success(&mut self, index: usize) {
        let target = &mut self.0[index];
        target.backoff = INITIAL_BACKOFF;
        target.retry_at = None;
    }

    fn record_failure(&mut self, index: usize, now: Instant) {
        let target = &mut self.0[index];
        target.retry_at = Some(now + target.backoff);
        target.backoff = (target.backoff * 2).min(MAX_BACKOFF);
    }

    async fn offer(
        &mut self,
        index: usize,
        request: SpillOfferRequest,
        timeout: Duration,
    ) -> Result<SpillOfferResponse, String> {
        let target = &mut self.0[index];
        let mut client = match &target.client {
            Some(client) => client.clone(),
            None => {
                let client = connect(&target.addr)?;
                target.client = Some(client.clone());
                client
            }
        };
        match tokio::time::timeout(timeout, client.spill_offer(request)).await {
            Ok(Ok(response)) => Ok(response.into_inner()),
            Ok(Err(status)) => Err(format!("SpillOffer to {} failed: {status}", target.addr)),
            Err(_) => Err(format!(
                "SpillOffer to {} timed out after {timeout:?}",
                target.addr
            )),
        }
    }
}

fn connect(addr: &str) -> Result<EngineClient<Channel>, String> {
    let url = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    let channel = Endpoint::from_shared(url)
        .map_err(|e| format!("invalid spill target address {addr}: {e}"))?
        .connect_timeout(GRPC_CONNECT_TIMEOUT)
        .http2_keep_alive_interval(GRPC_CLIENT_HTTP2_KEEPALIVE_INTERVAL)
        .connect_lazy();
    Ok(EngineClient::new(channel))
}

/// Target-side pull budget: an in-flight cap plus, when a bandwidth is set, a
/// token bucket over bytes per second. An offer larger than the remaining
/// budget is admitted from a non-empty bucket and paid back as debt, so big
/// batches still progress at the configured average rate.
pub(super) struct SpillAdmission {
    /// This node's advertise address; offers from it are refused.
    own_addr: String,
    /// `None`: no rate limit, only the in-flight cap applies.
    rate_bytes_per_sec: Option<f64>,
    max_inflight_bytes: u64,
    state: Mutex<AdmissionState>,
}

struct AdmissionState {
    tokens: f64,
    refilled_at: Instant,
    inflight_bytes: u64,
}

/// Keeps admitted bytes counted as in flight until dropped.
struct AdmissionPermit {
    admission: Arc<SpillAdmission>,
    bytes: u64,
}

impl SpillAdmission {
    pub(super) fn new(config: &SpillTargetConfig, own_addr: String, now: Instant) -> Self {
        let rate = config.max_bandwidth_bytes_per_sec.map(|rate| rate as f64);
        Self {
            own_addr,
            rate_bytes_per_sec: rate,
            max_inflight_bytes: config.max_inflight_bytes,
            state: Mutex::new(AdmissionState {
                tokens: rate.unwrap_or(0.0),
                refilled_at: now,
                inflight_bytes: 0,
            }),
        }
    }

    fn try_admit(self: &Arc<Self>, bytes: u64, now: Instant) -> Option<AdmissionPermit> {
        let mut state = self.state.lock();
        let fits_inflight = state.inflight_bytes == 0
            || state.inflight_bytes.saturating_add(bytes) <= self.max_inflight_bytes;
        if let Some(rate) = self.rate_bytes_per_sec {
            let elapsed = now
                .saturating_duration_since(state.refilled_at)
                .as_secs_f64();
            // Burst capacity is one second of budget.
            state.tokens = (state.tokens + elapsed * rate).min(rate);
            state.refilled_at = state.refilled_at.max(now);
            if state.tokens <= 0.0 || !fits_inflight {
                return None;
            }
            state.tokens -= bytes as f64;
        } else if !fits_inflight {
            return None;
        }
        state.inflight_bytes = state.inflight_bytes.saturating_add(bytes);
        Some(AdmissionPermit {
            admission: Arc::clone(self),
            bytes,
        })
    }
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        let mut state = self.admission.state.lock();
        state.inflight_bytes = state.inflight_bytes.saturating_sub(self.bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deficit_counts_free_and_reclaimable_bytes_per_pool() {
        // Reserve of 100 bytes; pools are (used, total).
        let deficit = |pools: &[(u64, u64)], reclaimable| spill_deficit(100, pools, reclaimable);
        assert_eq!(deficit(&[(0, 1000)], 0), 0, "empty pool");
        assert_eq!(deficit(&[(1000, 1000)], 0), 100, "full pool");
        assert_eq!(deficit(&[(940, 1000)], 0), 40, "partly free");
        assert_eq!(deficit(&[(940, 1000)], 30), 10, "free plus reclaimable");
        assert_eq!(deficit(&[(920, 1000)], 30), 0, "reserve met");
        assert_eq!(
            deficit(&[(1000, 1000), (0, 1000)], 0),
            50,
            "an idle node's free memory does not hide a full node"
        );
        assert_eq!(
            deficit(&[(1000, 1000), (1000, 1000)], 40),
            60,
            "both nodes full"
        );
        assert_eq!(deficit(&[], 0), 0, "no pools");
    }

    #[test]
    fn unclaimed_hashes_keep_offer_order() {
        let offered = vec![vec![1], vec![2], vec![3], vec![4]];
        assert_eq!(
            unclaimed_hashes(&offered, &[vec![3]], &[vec![1]]),
            vec![vec![2], vec![4]]
        );
        assert!(unclaimed_hashes(&offered, &offered, &[]).is_empty());
    }

    #[test]
    fn targets_fail_over_and_return_to_primary() {
        let start = Instant::now();
        let mut targets = SpillTargets::new(&["primary:1".into(), "secondary:1".into()]);
        assert_eq!(targets.pick(start), Some(0));

        targets.record_failure(0, start);
        assert_eq!(
            targets.pick(start),
            Some(1),
            "fail over while primary backs off"
        );
        targets.record_failure(1, start);
        assert_eq!(targets.pick(start), None, "every target backing off");

        let retry = start + INITIAL_BACKOFF;
        assert_eq!(targets.pick(retry), Some(0), "primary is retried first");
        targets.record_failure(0, retry);
        assert_eq!(
            targets.pick(retry + INITIAL_BACKOFF),
            Some(1),
            "primary backoff doubled"
        );
        targets.record_success(0);
        assert_eq!(targets.pick(retry), Some(0));

        // A new assignment keeps the backoff of targets that stay.
        targets.record_failure(1, retry);
        targets.set_addrs(&["secondary:1".into(), "third:1".into()]);
        assert_eq!(targets.addr(0), "secondary:1");
        assert_eq!(targets.pick(retry), Some(1), "secondary still backing off");
    }

    #[test]
    fn admission_charges_debt_and_caps_inflight() {
        let start = Instant::now();
        let admission = |max_bandwidth_bytes_per_sec| {
            Arc::new(SpillAdmission::new(
                &SpillTargetConfig {
                    max_bandwidth_bytes_per_sec,
                    max_inflight_bytes: 1500,
                },
                "target:1".to_string(),
                start,
            ))
        };

        let capped = admission(Some(1000));
        let first = capped
            .try_admit(1200, start)
            .expect("a full bucket admits an oversized offer");
        assert!(capped.try_admit(1, start).is_none(), "bucket in debt");
        // Half a second repays the 200-byte debt, but 1200 + 400 exceeds the cap.
        let later = start + Duration::from_millis(500);
        assert!(capped.try_admit(400, later).is_none(), "in-flight cap");
        drop(first);
        assert!(capped.try_admit(400, later).is_some());

        // Without a bandwidth only the in-flight cap applies.
        let uncapped = admission(None);
        let first = uncapped.try_admit(1200, start).expect("no rate limit");
        assert!(uncapped.try_admit(400, start).is_none(), "in-flight cap");
        assert!(
            uncapped.try_admit(300, start).is_some(),
            "fits beside the first"
        );
        drop(first);
    }
}
