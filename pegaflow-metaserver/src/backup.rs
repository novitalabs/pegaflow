//! Decode-node backup planning.
//!
//! Source nodes (nodes with an inference instance) report the oldest retained
//! blocks of their RAM cache with every heartbeat. Target nodes (nodes without
//! an instance for `target_delay`) poll `PullBackupPlan` and copy the planned
//! blocks with the regular P2P transfer path. The planner only decides who
//! copies what; it never tracks target capacity, because a full target
//! replaces its oldest backups through its own LRU.

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use opentelemetry::KeyValue;
use opentelemetry::metrics::Counter;
use pegaflow_common::BlockKey;

use crate::store::{BlockHashStore, NodeRoles};

pub const DEFAULT_TARGET_DELAY_SECS: u64 = 180;
pub const DEFAULT_INFLIGHT_TTL_SECS: u64 = 180;
pub const DEFAULT_MAX_BYTES_PER_SEC: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub struct BackupConfig {
    /// How long a node must report no instance before it becomes a target.
    pub target_delay: Duration,
    /// Dispatched blocks are not re-dispatched until this expires. Must
    /// outlive the holder's transfer lock timeout so a slow pull finishes and
    /// registers its owner first.
    pub inflight_ttl: Duration,
    /// Per-node dispatch budget, applied to both the source and the target of
    /// every plan. Zero disables the limit.
    pub max_bytes_per_sec: u64,
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            target_delay: Duration::from_secs(DEFAULT_TARGET_DELAY_SECS),
            inflight_ttl: Duration::from_secs(DEFAULT_INFLIGHT_TTL_SECS),
            max_bytes_per_sec: DEFAULT_MAX_BYTES_PER_SEC,
        }
    }
}

/// One namespace's candidates reported by a source, oldest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateGroup {
    pub namespace: String,
    pub block_bytes: u64,
    pub hashes: Vec<Vec<u8>>,
}

/// A batch of blocks one target should copy from one source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupPlan {
    pub source: Arc<str>,
    pub namespace: String,
    pub hashes: Vec<Vec<u8>>,
}

struct QueuedGroup {
    namespace: String,
    block_bytes: u64,
    hashes: VecDeque<Vec<u8>>,
}

struct Inflight {
    dispatched_at: Instant,
    target: Arc<str>,
    bytes: u64,
}

#[derive(Default)]
struct PlannerState {
    /// Latest candidate snapshot per source; replaced on every report.
    candidates: HashMap<Arc<str>, Vec<QueuedGroup>>,
    inflight: HashMap<BlockKey, Inflight>,
    /// Token-bucket style pacing: a node may join a new plan after this time.
    next_allowed: HashMap<Arc<str>, Instant>,
    /// Round-robin cursor over a target's sources.
    cursor: HashMap<Arc<str>, usize>,
}

pub struct BackupPlanner {
    config: BackupConfig,
    state: Mutex<PlannerState>,
}

struct PlannerMetrics {
    dispatched_blocks: Counter<u64>,
    dispatched_bytes: Counter<u64>,
    filtered: Counter<u64>,
    recall_planned_blocks: Counter<u64>,
}

static METRICS: LazyLock<PlannerMetrics> = LazyLock::new(|| {
    let meter = opentelemetry::global::meter("pegaflow_metaserver");
    PlannerMetrics {
        dispatched_blocks: meter
            .u64_counter("pegaflow_metaserver_backup_dispatched_blocks")
            .with_description("Backup blocks dispatched per source/target pair")
            .build(),
        dispatched_bytes: meter
            .u64_counter("pegaflow_metaserver_backup_dispatched_bytes")
            .with_unit("bytes")
            .with_description("Estimated backup bytes dispatched per source/target pair")
            .build(),
        filtered: meter
            .u64_counter("pegaflow_metaserver_backup_candidates_filtered")
            .with_description("Backup candidates skipped (reason=has_backup|source_gone|inflight)")
            .build(),
        recall_planned_blocks: meter
            .u64_counter("pegaflow_metaserver_backup_recall_planned_blocks")
            .with_description(
                "Prefix fetch blocks planned from a backup target, per requester/target pair",
            )
            .build(),
    }
});

fn record_filtered(reason: &'static str, count: u64) {
    if count > 0 {
        METRICS
            .filtered
            .add(count, &[KeyValue::new("reason", reason)]);
    }
}

impl BackupPlanner {
    pub fn new(config: BackupConfig) -> Self {
        Self {
            config,
            state: Mutex::new(PlannerState::default()),
        }
    }

    pub fn config(&self) -> BackupConfig {
        self.config
    }

    pub fn roles(&self, store: &BlockHashStore) -> NodeRoles {
        store.node_roles(self.config.target_delay)
    }

    /// Replace `source`'s candidate snapshot. Returns the reported hashes that
    /// already have a live backup owner, so the source can demote them; they
    /// are dropped from the snapshot.
    pub fn report(
        &self,
        store: &BlockHashStore,
        roles: &NodeRoles,
        source: &str,
        groups: Vec<CandidateGroup>,
    ) -> Vec<CandidateGroup> {
        let source: Arc<str> = Arc::from(source);
        let mut backed = Vec::new();
        let mut queued = Vec::with_capacity(groups.len());
        let mut backed_count = 0u64;
        for group in groups {
            let mut keep = VecDeque::with_capacity(group.hashes.len());
            let mut done = Vec::new();
            for hash in group.hashes {
                let key = BlockKey::new(group.namespace.clone(), hash);
                if has_backup_owner(store, roles, &key) {
                    done.push(key.hash);
                } else {
                    keep.push_back(key.hash);
                }
            }
            backed_count += done.len() as u64;
            if !done.is_empty() {
                backed.push(CandidateGroup {
                    namespace: group.namespace.clone(),
                    block_bytes: group.block_bytes,
                    hashes: done,
                });
            }
            if !keep.is_empty() {
                queued.push(QueuedGroup {
                    namespace: group.namespace,
                    block_bytes: group.block_bytes,
                    hashes: keep,
                });
            }
        }
        record_filtered("has_backup", backed_count);

        let mut state = self.state.lock().expect("backup planner lock poisoned");
        if queued.is_empty() {
            state.candidates.remove(&source);
        } else {
            state.candidates.insert(source, queued);
        }
        backed
    }

    /// Drop a node's snapshot (it stopped being a source).
    pub fn forget_source(&self, source: &str) {
        self.state
            .lock()
            .expect("backup planner lock poisoned")
            .candidates
            .remove(source);
    }

    /// Next batch of at most `max_bytes` (at least one block) for `target`.
    pub fn plan(
        &self,
        store: &BlockHashStore,
        roles: &NodeRoles,
        target: &str,
        max_bytes: u64,
    ) -> Option<BackupPlan> {
        let target = roles.targets.iter().find(|node| node.as_ref() == target)?;
        let sources = pairing(&roles.sources, &roles.targets)
            .into_iter()
            .filter(|(_, paired)| paired == target)
            .map(|(source, _)| source)
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return None;
        }

        let now = Instant::now();
        let mut state = self.state.lock().expect("backup planner lock poisoned");
        let ttl = self.config.inflight_ttl;
        state
            .inflight
            .retain(|_, entry| now.duration_since(entry.dispatched_at) < ttl);
        if state.next_allowed.get(target).is_some_and(|at| *at > now) {
            return None;
        }

        let start = state.cursor.get(target).copied().unwrap_or(0);
        for offset in 0..sources.len() {
            let index = (start + offset) % sources.len();
            let source = &sources[index];
            if state.next_allowed.get(source).is_some_and(|at| *at > now) {
                continue;
            }
            let Some((namespace, block_bytes, hashes)) =
                take_batch(&mut state, store, roles, source, max_bytes)
            else {
                continue;
            };

            let bytes = block_bytes.saturating_mul(hashes.len() as u64);
            for hash in &hashes {
                state.inflight.insert(
                    BlockKey::new(namespace.clone(), hash.clone()),
                    Inflight {
                        dispatched_at: now,
                        target: Arc::clone(target),
                        bytes: block_bytes,
                    },
                );
            }
            if let Some(pause) = self.pace(bytes) {
                for node in [source, target] {
                    let at = state.next_allowed.entry(Arc::clone(node)).or_insert(now);
                    *at = (*at).max(now) + pause;
                }
            }
            state.cursor.insert(Arc::clone(target), index + 1);

            let pair = [
                KeyValue::new("source", source.to_string()),
                KeyValue::new("target", target.to_string()),
            ];
            METRICS.dispatched_blocks.add(hashes.len() as u64, &pair);
            METRICS.dispatched_bytes.add(bytes, &pair);
            return Some(BackupPlan {
                source: Arc::clone(source),
                namespace,
                hashes,
            });
        }
        None
    }

    /// Count prefix-fetch segments planned from a backup target.
    pub fn record_recall(&self, requester: &str, target: &str, blocks: u64) {
        METRICS.recall_planned_blocks.add(
            blocks,
            &[
                KeyValue::new("source", requester.to_string()),
                KeyValue::new("target", target.to_string()),
            ],
        );
    }

    /// Total queued candidate blocks across all sources.
    pub fn queued_candidates(&self) -> u64 {
        self.state
            .lock()
            .expect("backup planner lock poisoned")
            .candidates
            .values()
            .flatten()
            .map(|group| group.hashes.len() as u64)
            .sum()
    }

    /// Estimated in-flight bytes per target.
    pub fn inflight_bytes(&self) -> HashMap<Arc<str>, u64> {
        let now = Instant::now();
        let ttl = self.config.inflight_ttl;
        let state = self.state.lock().expect("backup planner lock poisoned");
        let mut bytes: HashMap<Arc<str>, u64> = HashMap::new();
        for entry in state.inflight.values() {
            if now.duration_since(entry.dispatched_at) < ttl {
                *bytes.entry(Arc::clone(&entry.target)).or_default() += entry.bytes;
            }
        }
        bytes
    }

    fn pace(&self, bytes: u64) -> Option<Duration> {
        if self.config.max_bytes_per_sec == 0 || bytes == 0 {
            return None;
        }
        Some(Duration::from_secs_f64(
            bytes as f64 / self.config.max_bytes_per_sec as f64,
        ))
    }
}

/// Pop the oldest dispatchable blocks of one namespace from `source`'s queue.
fn take_batch(
    state: &mut PlannerState,
    store: &BlockHashStore,
    roles: &NodeRoles,
    source: &Arc<str>,
    max_bytes: u64,
) -> Option<(String, u64, Vec<Vec<u8>>)> {
    let PlannerState {
        candidates,
        inflight,
        ..
    } = state;
    let groups = candidates.get_mut(source)?;
    let mut result = None;
    let (mut skipped_inflight, mut skipped_gone, mut skipped_backed) = (0u64, 0u64, 0u64);
    for group in groups.iter_mut() {
        let mut taken = Vec::new();
        let mut bytes = 0u64;
        while !group.hashes.is_empty() {
            if !taken.is_empty() && bytes.saturating_add(group.block_bytes) > max_bytes {
                break;
            }
            let hash = group.hashes.pop_front().expect("queue is non-empty");
            let key = BlockKey::new(group.namespace.clone(), hash);
            if inflight.contains_key(&key) {
                skipped_inflight += 1;
                continue;
            }
            let owners = store.visible_owners(&key);
            if !owners.iter().any(|owner| owner == source) {
                skipped_gone += 1;
                continue;
            }
            if owners.iter().any(|owner| roles.targets.contains(owner)) {
                skipped_backed += 1;
                continue;
            }
            bytes = bytes.saturating_add(group.block_bytes);
            taken.push(key.hash);
        }
        if !taken.is_empty() {
            result = Some((group.namespace.clone(), group.block_bytes, taken));
            break;
        }
    }
    groups.retain(|group| !group.hashes.is_empty());
    if groups.is_empty() {
        candidates.remove(source);
    }
    record_filtered("inflight", skipped_inflight);
    record_filtered("source_gone", skipped_gone);
    record_filtered("has_backup", skipped_backed);
    result
}

fn has_backup_owner(store: &BlockHashStore, roles: &NodeRoles, key: &BlockKey) -> bool {
    store
        .visible_owners(key)
        .iter()
        .any(|owner| roles.targets.contains(owner))
}

fn rendezvous_score(source: &str, target: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    target.hash(&mut hasher);
    hasher.finish()
}

/// Stable source/target pairing: rendezvous hashing with bounded load.
///
/// The smaller side acts as bins with capacity `ceil(larger / smaller)`; each
/// node on the larger side takes its highest-scoring bin with room. Empty bins
/// then take the best-scoring node from the most loaded bin, so every source
/// has a target and every target has a source. Adding or removing one node
/// moves few pairs because scores depend only on the two node names.
pub fn pairing(sources: &[Arc<str>], targets: &[Arc<str>]) -> Vec<(Arc<str>, Arc<str>)> {
    if sources.is_empty() || targets.is_empty() {
        return Vec::new();
    }
    let sources_are_bins = sources.len() <= targets.len();
    let (bins, items) = if sources_are_bins {
        (sources, targets)
    } else {
        (targets, sources)
    };
    let score = |item: &Arc<str>, bin: &Arc<str>| {
        if sources_are_bins {
            rendezvous_score(bin, item)
        } else {
            rendezvous_score(item, bin)
        }
    };

    let capacity = items.len().div_ceil(bins.len());
    let mut assigned: Vec<Vec<usize>> = vec![Vec::new(); bins.len()];
    let mut sorted_items: Vec<usize> = (0..items.len()).collect();
    sorted_items.sort_by(|a, b| items[*a].cmp(&items[*b]));
    for item in sorted_items {
        let mut ranked: Vec<usize> = (0..bins.len()).collect();
        ranked.sort_by_key(|bin| std::cmp::Reverse(score(&items[item], &bins[*bin])));
        let bin = ranked
            .into_iter()
            .find(|bin| assigned[*bin].len() < capacity)
            .expect("total capacity covers every item");
        assigned[bin].push(item);
    }

    while let Some(empty) = assigned.iter().position(Vec::is_empty) {
        let donor = (0..bins.len())
            .max_by_key(|bin| (assigned[*bin].len(), std::cmp::Reverse(*bin)))
            .expect("bins is non-empty");
        if assigned[donor].len() <= 1 {
            break;
        }
        let (position, _) = assigned[donor]
            .iter()
            .enumerate()
            .max_by_key(|(_, item)| score(&items[**item], &bins[empty]))
            .expect("donor has items");
        let item = assigned[donor].remove(position);
        assigned[empty].push(item);
    }

    let mut pairs = Vec::with_capacity(items.len());
    for (bin, members) in assigned.into_iter().enumerate() {
        for item in members {
            let (source, target) = if sources_are_bins {
                (&bins[bin], &items[item])
            } else {
                (&items[item], &bins[bin])
            };
            pairs.push((Arc::clone(source), Arc::clone(target)));
        }
    }
    pairs.sort();
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::StoreConfig;
    use std::collections::HashSet;
    use uuid::Uuid;

    fn nodes(prefix: &str, count: usize) -> Vec<Arc<str>> {
        (0..count)
            .map(|i| Arc::from(format!("{prefix}{i}:50055")))
            .collect()
    }

    #[test]
    fn pairing_covers_both_sides_with_bounded_load() {
        for (p, d) in [
            (1, 1),
            (1, 3),
            (3, 1),
            (2, 2),
            (4, 3),
            (3, 4),
            (8, 3),
            (3, 8),
        ] {
            let sources = nodes("p", p);
            let targets = nodes("d", d);
            let pairs = pairing(&sources, &targets);
            let covered_sources: HashSet<_> = pairs.iter().map(|(s, _)| s.clone()).collect();
            let covered_targets: HashSet<_> = pairs.iter().map(|(_, t)| t.clone()).collect();
            assert_eq!(covered_sources.len(), p, "{p}P{d}D sources");
            assert_eq!(covered_targets.len(), d, "{p}P{d}D targets");
            assert_eq!(
                pairs.len(),
                p.max(d),
                "{p}P{d}D each larger-side node pairs once"
            );

            let capacity = p.max(d).div_ceil(p.min(d));
            let mut load: HashMap<Arc<str>, usize> = HashMap::new();
            for (s, t) in &pairs {
                *load
                    .entry(if p <= d { s.clone() } else { t.clone() })
                    .or_default() += 1;
            }
            assert!(
                load.values().all(|l| *l <= capacity),
                "{p}P{d}D load {load:?}"
            );
        }
    }

    #[test]
    fn pairing_is_stable_when_a_target_joins() {
        let sources = nodes("p", 4);
        let before = pairing(&sources, &nodes("d", 4));
        let after = pairing(&sources, &nodes("d", 5));
        let kept = before.iter().filter(|pair| after.contains(pair)).count();
        assert!(kept >= 2, "before={before:?} after={after:?}");
    }

    struct Fixture {
        store: BlockHashStore,
        planner: BackupPlanner,
        ids: HashMap<&'static str, Uuid>,
    }

    impl Fixture {
        /// `p*` nodes have an instance, `d*` nodes are idle targets.
        fn new(names: &[&'static str], config: BackupConfig) -> Self {
            let store = BlockHashStore::with_config(StoreConfig::default());
            let mut ids = HashMap::new();
            for name in names {
                let id = Uuid::new_v4();
                store.heartbeat_node(name, id).unwrap();
                store.set_node_has_instance(name, id, name.starts_with('p'));
                ids.insert(*name, id);
            }
            Self {
                store,
                planner: BackupPlanner::new(BackupConfig {
                    target_delay: Duration::ZERO,
                    ..config
                }),
                ids,
            }
        }

        fn own(&self, node: &'static str, hashes: &[u8]) {
            let hashes: Vec<Vec<u8>> = hashes.iter().map(|h| vec![*h]).collect();
            self.store
                .insert_hashes("ns", &hashes, node, self.ids[node])
                .unwrap();
        }

        fn report(&self, node: &'static str, hashes: &[u8]) -> Vec<u8> {
            let roles = self.planner.roles(&self.store);
            let group = CandidateGroup {
                namespace: "ns".into(),
                block_bytes: 10,
                hashes: hashes.iter().map(|h| vec![*h]).collect(),
            };
            self.planner
                .report(&self.store, &roles, node, vec![group])
                .into_iter()
                .flat_map(|group| group.hashes)
                .map(|hash| hash[0])
                .collect()
        }

        fn plan(&self, target: &str, max_bytes: u64) -> Option<(String, Vec<u8>)> {
            let roles = self.planner.roles(&self.store);
            self.planner
                .plan(&self.store, &roles, target, max_bytes)
                .map(|plan| {
                    (
                        plan.source.to_string(),
                        plan.hashes.into_iter().map(|hash| hash[0]).collect(),
                    )
                })
        }
    }

    fn unlimited() -> BackupConfig {
        BackupConfig {
            max_bytes_per_sec: 0,
            ..BackupConfig::default()
        }
    }

    #[test]
    fn plan_filters_backed_gone_and_inflight_blocks() {
        let fx = Fixture::new(&["p0", "d0"], unlimited());
        fx.own("p0", &[1, 2, 3, 4]);
        fx.own("d0", &[2]);

        // Block 2 already has a backup: reported back for demotion.
        assert_eq!(fx.report("p0", &[1, 2, 3, 4, 5]), vec![2]);
        // 5 is not owned by p0 any more; budget 20 bytes = two 10-byte blocks.
        assert_eq!(fx.plan("d0", 20), Some(("p0".into(), vec![1, 3])));
        // Re-reporting in-flight blocks does not dispatch them twice.
        fx.report("p0", &[1, 3, 4, 5]);
        assert_eq!(fx.plan("d0", 100), Some(("p0".into(), vec![4])));
        assert_eq!(fx.plan("d0", 100), None);
    }

    #[test]
    fn plan_redispatches_after_inflight_expiry() {
        let fx = Fixture::new(
            &["p0", "d0"],
            BackupConfig {
                inflight_ttl: Duration::ZERO,
                ..unlimited()
            },
        );
        fx.own("p0", &[1]);
        fx.report("p0", &[1]);
        assert_eq!(fx.plan("d0", 100), Some(("p0".into(), vec![1])));
        fx.report("p0", &[1]);
        assert_eq!(fx.plan("d0", 100), Some(("p0".into(), vec![1])));
    }

    #[test]
    fn plan_requires_target_role_and_paces_per_node() {
        let fx = Fixture::new(
            &["p0", "d0"],
            BackupConfig {
                max_bytes_per_sec: 1,
                ..BackupConfig::default()
            },
        );
        fx.own("p0", &[1, 2]);
        fx.report("p0", &[1, 2]);
        assert_eq!(fx.plan("p0", 100), None, "sources never pull");
        assert_eq!(fx.plan("d0", 10), Some(("p0".into(), vec![1])));
        assert_eq!(fx.plan("d0", 10), None, "10 bytes at 1 B/s pauses the pair");
    }

    #[test]
    fn idle_node_becomes_target_only_after_delay() {
        let store = BlockHashStore::new();
        let id = Uuid::new_v4();
        store.heartbeat_node("n", id).unwrap();
        assert!(
            store
                .node_roles(Duration::from_secs(3600))
                .targets
                .is_empty()
        );
        assert_eq!(store.node_roles(Duration::ZERO).targets.len(), 1);

        store.set_node_has_instance("n", id, true);
        let roles = store.node_roles(Duration::ZERO);
        assert!(roles.targets.is_empty());
        assert_eq!(roles.sources.len(), 1);
    }
}
