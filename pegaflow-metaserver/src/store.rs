use dashmap::{DashMap, mapref::entry::Entry};
use log::{info, warn};
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::hll::{ClusterHllSnapshot, HllNodeReport, HllSchema, HllState, ReceivedHllReport};

const MIN_RECLAIMABLE_OWNER_COUNT: usize = 3;

pub const DEFAULT_NODE_STALE_SECS: u64 = 30;
pub const DEFAULT_TTL_MINUTES: u64 = 120;

/// A prefix query result: one block hash and all live nodes that own it.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub block_hash: Vec<u8>,
    pub nodes: Vec<Arc<str>>,
}

#[derive(Debug, Clone, Copy)]
pub struct StoreConfig {
    pub node_stale_after: Duration,
    pub ttl: Duration,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(DEFAULT_TTL_MINUTES * 60),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SweepStats {
    pub removed_owners: usize,
    pub removed_keys: usize,
    pub removed_nodes: usize,
}

impl SweepStats {
    pub fn is_empty(self) -> bool {
        self.removed_owners == 0 && self.removed_keys == 0 && self.removed_nodes == 0
    }
}

/// Live-owner redundancy distribution over block keys, recomputed each sweep and
/// cached so metric scrapes stay O(1). Keys are bucketed by their number of
/// query-visible owners (1, 2, 3, >=4); keys with zero visible owners are
/// excluded from every bucket. `copies` is the exact total of visible owners, so
/// average redundancy (the cache capacity shrink factor) is
/// `copies / (keys_1 + keys_2 + keys_3 + keys_4plus)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedundancySnapshot {
    pub keys_1: u64,
    pub keys_2: u64,
    pub keys_3: u64,
    pub keys_4plus: u64,
    pub copies: u64,
}

impl RedundancySnapshot {
    fn record(&mut self, visible_owners: u64) {
        match visible_owners {
            0 => {}
            1 => self.keys_1 += 1,
            2 => self.keys_2 += 1,
            3 => self.keys_3 += 1,
            _ => self.keys_4plus += 1,
        }
        self.copies += visible_owners;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    UnknownNode,
    StaleSession,
}

/// Snapshot of the session that wrote this block ownership. Query compares it
/// with the current `NodeRecord.node_id` to filter owners left by old sessions.
#[derive(Debug, Clone)]
struct OwnerRecord {
    node_id: Uuid,
    key_register_time: Instant,
}

/// Authoritative current session for a node URL. `last_seen` gates metadata
/// query visibility and is refreshed by a valid heartbeat or metadata write.
/// `hll_report` is best-effort observability state carried by heartbeats; its
/// own receipt time gates inclusion in the cluster union, so metadata writes
/// cannot keep a stale report alive.
#[derive(Debug, Clone)]
struct NodeRecord {
    node_id: Uuid,
    last_seen: Instant,
    hll_report: Option<ReceivedHllReport>,
}

/// Both lifecycle decisions for one owner record, produced from a single `nodes`
/// lookup: `keep` (survives the TTL purge) and `visible` (query-visible: current
/// session and node still fresh). Lets the sweep tally live redundancy without
/// probing the node map twice per owner.
struct OwnerEval {
    keep: bool,
    visible: bool,
}

/// Async thread-safe block hash storage using DashMap.
///
/// `blocks` maps each block key to node URL ownership records. `nodes` tracks
/// the current MetaServer session and liveness for each node URL.
pub struct BlockHashStore {
    blocks: DashMap<BlockKey, HashMap<Arc<str>, OwnerRecord>>,
    nodes: DashMap<Arc<str>, NodeRecord>,
    config: StoreConfig,
    hll: HllState,
    /// Latest live-owner redundancy distribution, refreshed each sweep and read
    /// by metric callbacks. Decouples the O(N) scan from the scrape path.
    redundancy: Mutex<RedundancySnapshot>,
}

impl BlockHashStore {
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    pub fn with_config(config: StoreConfig) -> Self {
        Self::with_config_and_hll_schema(config, HllSchema::default())
    }

    pub fn with_config_and_hll_schema(config: StoreConfig, hll_schema: HllSchema) -> Self {
        Self {
            blocks: DashMap::new(),
            nodes: DashMap::new(),
            config,
            hll: HllState::new(hll_schema, config.node_stale_after),
            redundancy: Mutex::new(RedundancySnapshot::default()),
        }
    }

    pub fn with_ttl(ttl_minutes: u64) -> Self {
        Self::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(ttl_minutes * 60),
        })
    }

    pub fn config(&self) -> StoreConfig {
        self.config
    }

    pub fn heartbeat_node(
        &self,
        node: &str,
        node_id: Uuid,
        hll_report: HllNodeReport,
    ) -> Result<(), StoreError> {
        let now = Instant::now();
        let validated_report = match self.hll.receive(hll_report, now) {
            Ok(report) => Some(report),
            Err(error) => {
                warn!(
                    "MetaServer heartbeat accepted with invalid HLL report (observability degraded): node={} error={}",
                    node, error
                );
                None
            }
        };

        match self.nodes.entry(Arc::from(node)) {
            Entry::Vacant(entry) => {
                info!("MetaServer node registered: node={node} node_id={node_id}");
                entry.insert(NodeRecord {
                    node_id,
                    last_seen: now,
                    hll_report: validated_report,
                });
            }
            Entry::Occupied(mut entry) => {
                let record = entry.get_mut();
                let same_session = record.node_id == node_id;
                let stale_session =
                    now.duration_since(record.last_seen) > self.config.node_stale_after;
                if same_session || stale_session {
                    if stale_session && !same_session {
                        info!(
                            "MetaServer node session takeover: node={} old_node_id={} new_node_id={}",
                            node, record.node_id, node_id
                        );
                    }
                    record.node_id = node_id;
                    record.last_seen = now;
                    record.hll_report = validated_report;
                } else {
                    warn!(
                        "MetaServer heartbeat rejected stale session: node={} current_node_id={} rejected_node_id={}",
                        node, record.node_id, node_id
                    );
                    return Err(StoreError::StaleSession);
                }
            }
        }
        self.hll.mark_changed();
        Ok(())
    }

    pub fn unregister_node(&self, node: &str, node_id: Uuid) -> Result<usize, StoreError> {
        if self
            .nodes
            .remove_if(node, |_, record| record.node_id == node_id)
            .is_none()
        {
            if self.nodes.contains_key(node) {
                warn!(
                    "MetaServer unregister rejected stale session: node={} rejected_node_id={}",
                    node, node_id
                );
                return Err(StoreError::StaleSession);
            }
            warn!(
                "MetaServer unregister rejected unknown node: node={} node_id={}",
                node, node_id
            );
            return Err(StoreError::UnknownNode);
        }
        self.hll.mark_changed();
        let removed = self.remove_node_owners(node, node_id);
        Ok(removed)
    }

    pub fn insert_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<Vec<Vec<u8>>, StoreError> {
        self.touch_node_session(node, node_id)?;
        let node: Arc<str> = Arc::from(node);
        let now = Instant::now();
        let mut reclaimable_hashes = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let mut owners = self.blocks.entry(key).or_default();
            let previous = owners.insert(
                Arc::clone(&node),
                OwnerRecord {
                    node_id,
                    key_register_time: now,
                },
            );
            let is_new_owner = previous.is_none_or(|owner| owner.node_id != node_id);
            if is_new_owner
                && owners
                    .iter()
                    .filter(|(node, owner)| self.is_owner_visible(node, owner, now))
                    .take(MIN_RECLAIMABLE_OWNER_COUNT)
                    .count()
                    == MIN_RECLAIMABLE_OWNER_COUNT
            {
                reclaimable_hashes.push(hash.clone());
            }
        }
        Ok(reclaimable_hashes)
    }

    pub fn remove_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<usize, StoreError> {
        self.touch_node_session(node, node_id)?;
        let mut removed = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let should_remove_key = if let Some(mut owners) = self.blocks.get_mut(&key) {
                if owners
                    .get(node)
                    .is_some_and(|owner| owner.node_id == node_id)
                {
                    owners.remove(node);
                    removed += 1;
                }
                owners.is_empty()
            } else {
                false
            };
            if should_remove_key {
                self.blocks.remove_if(&key, |_, owners| owners.is_empty());
            }
        }
        Ok(removed)
    }

    /// Query the longest prefix of `hashes` with at least one live owner.
    pub fn query_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<PrefixEntry> {
        let now = Instant::now();
        let mut result = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let Some(owners) = self.blocks.get(&key) else {
                break;
            };

            let live_nodes: Vec<Arc<str>> = owners
                .iter()
                .filter_map(|(node, owner)| {
                    if self.is_owner_visible(node, owner, now) {
                        Some(Arc::clone(node))
                    } else {
                        None
                    }
                })
                .collect();

            if live_nodes.is_empty() {
                break;
            }

            result.push(PrefixEntry {
                block_hash: hash.clone(),
                nodes: live_nodes,
            });
        }
        result
    }

    /// Sweep owners whose node is missing or whose ownership TTL has expired, and
    /// refresh the cached live-owner redundancy snapshot in the same walk.
    pub fn sweep_expired(&self) -> SweepStats {
        let now = Instant::now();
        let mut stats = SweepStats::default();
        let mut snapshot = RedundancySnapshot::default();

        self.blocks.retain(|_, owners| {
            let before = owners.len();
            let mut visible = 0u64;
            owners.retain(|node, owner| {
                let eval = self.eval_owner(node, owner, now);
                if eval.keep && eval.visible {
                    visible += 1;
                }
                eval.keep
            });
            stats.removed_owners += before.saturating_sub(owners.len());
            if owners.is_empty() {
                stats.removed_keys += 1;
                return false;
            }
            snapshot.record(visible);
            true
        });

        let node_before = self.nodes.len();
        let ttl = self.config.ttl;
        self.nodes.retain(|node, record| {
            let keep = now.duration_since(record.last_seen) <= ttl;
            if !keep {
                info!(
                    "MetaServer node swept: node={} node_id={} last_seen_age_secs={}",
                    node,
                    record.node_id,
                    now.duration_since(record.last_seen).as_secs()
                );
            }
            keep
        });
        stats.removed_nodes = node_before.saturating_sub(self.nodes.len());

        *self
            .redundancy
            .lock()
            .expect("redundancy snapshot mutex poisoned") = snapshot;

        stats
    }

    /// Latest cached live-owner redundancy distribution (refreshed each sweep).
    pub fn redundancy_snapshot(&self) -> RedundancySnapshot {
        *self
            .redundancy
            .lock()
            .expect("redundancy snapshot mutex poisoned")
    }

    pub fn entry_count(&self) -> u64 {
        self.blocks.len() as u64
    }

    pub fn owner_count(&self) -> u64 {
        self.blocks
            .iter()
            .map(|entry| entry.value().len() as u64)
            .sum()
    }

    pub fn node_counts(&self) -> (u64, u64) {
        let now = Instant::now();
        let mut active = 0;
        let mut stale = 0;
        for node in &self.nodes {
            let age = now.duration_since(node.last_seen);
            if age <= self.config.node_stale_after {
                active += 1;
            } else if age <= self.config.ttl {
                stale += 1;
            }
        }
        (active, stale)
    }

    pub fn cluster_hll_snapshot(&self) -> ClusterHllSnapshot {
        self.hll.snapshot(|| {
            self.nodes
                .iter()
                .filter_map(|record| record.hll_report.clone())
                .collect()
        })
    }

    #[allow(
        dead_code,
        reason = "maintenance API reserved for explicit store cleanup"
    )]
    pub fn invalidate_all(&self) {
        self.blocks.clear();
        self.nodes.clear();
        self.hll.mark_changed();
        *self
            .redundancy
            .lock()
            .expect("redundancy snapshot mutex poisoned") = RedundancySnapshot::default();
    }

    fn touch_node_session(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let Some(mut record) = self.nodes.get_mut(node) else {
            warn!(
                "MetaServer metadata write rejected unknown node: node={} node_id={}",
                node, node_id
            );
            return Err(StoreError::UnknownNode);
        };
        if record.node_id != node_id {
            warn!(
                "MetaServer metadata write rejected stale session: node={} current_node_id={} rejected_node_id={}",
                node, record.node_id, node_id
            );
            return Err(StoreError::StaleSession);
        }
        // Metadata activity renews owner visibility only; HLL freshness is
        // owned by the heartbeat report timestamp.
        record.last_seen = Instant::now();
        Ok(())
    }

    fn remove_node_owners(&self, node: &str, node_id: Uuid) -> usize {
        let mut removed = 0;
        self.blocks.retain(|_, owners| {
            if owners
                .get(node)
                .is_some_and(|owner| owner.node_id == node_id)
            {
                owners.remove(node);
                removed += 1;
            }
            !owners.is_empty()
        });
        removed
    }

    /// Evaluate one owner against the current node session and clocks with a
    /// single `nodes` lookup. A missing node means the owner is neither kept nor
    /// visible.
    fn eval_owner(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> OwnerEval {
        let Some(record) = self.nodes.get(node.as_ref()) else {
            return OwnerEval {
                keep: false,
                visible: false,
            };
        };
        let node_age = now.duration_since(record.last_seen);
        OwnerEval {
            keep: now.duration_since(owner.key_register_time) <= self.config.ttl
                && node_age <= self.config.ttl,
            visible: record.node_id == owner.node_id && node_age <= self.config.node_stale_after,
        }
    }

    fn is_owner_visible(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> bool {
        self.eval_owner(node, owner, now).visible
    }
}

impl Default for BlockHashStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use pegaflow_common::hll::{HllWindowSnapshot, HyperLogLog};
    use pegaflow_common::hll_config::{
        DEFAULT_HLL_BUCKET_BITS, DEFAULT_HLL_WINDOWS, parse_hll_windows,
    };

    use super::*;

    fn hll_schema(bucket_bits: u8) -> HllSchema {
        HllSchema::new(
            vec![("1m".to_string(), Duration::from_secs(60))],
            bucket_bits,
        )
        .unwrap()
    }

    fn hll_store(config: StoreConfig, bucket_bits: u8) -> BlockHashStore {
        BlockHashStore::with_config_and_hll_schema(config, hll_schema(bucket_bits))
    }

    fn heartbeat_node(store: &BlockHashStore, node: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        store
            .heartbeat_node(node, node_id, test_hll_report())
            .unwrap();
        node_id
    }

    fn test_hll_report() -> HllNodeReport {
        HllNodeReport {
            windows: parse_hll_windows(DEFAULT_HLL_WINDOWS)
                .unwrap()
                .into_iter()
                .map(|(window, duration)| HllWindowSnapshot {
                    window,
                    window_secs: duration.as_secs(),
                    bucket_bits: DEFAULT_HLL_BUCKET_BITS,
                    registers: vec![0; 1 << DEFAULT_HLL_BUCKET_BITS],
                    total_requests: 0,
                })
                .collect(),
        }
    }

    fn hll_report_for(hashes: &[u64], total_requests: u64, bucket_bits: u8) -> HllNodeReport {
        let mut hll = HyperLogLog::new(bucket_bits);
        for hash in hashes {
            let distributed = hash.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            hll.insert(&distributed.to_be_bytes());
        }
        HllNodeReport {
            windows: vec![HllWindowSnapshot {
                window: "1m".into(),
                window_secs: 60,
                bucket_bits,
                registers: hll.registers().to_vec(),
                total_requests,
            }],
        }
    }

    #[test]
    fn cluster_hll_unions_registers_and_sums_observations() {
        let store = hll_store(StoreConfig::default(), 14);
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();
        store
            .heartbeat_node("node-a", node_a, hll_report_for(&[1, 2], 2, 14))
            .unwrap();
        store
            .heartbeat_node("node-b", node_b, hll_report_for(&[2, 3], 2, 14))
            .unwrap();

        let cluster = store.cluster_hll_snapshot();
        assert_eq!(cluster.windows.len(), 1);
        let window = &cluster.windows[0];
        assert_eq!(window.total_requests, 4);
        assert_eq!(window.active_nodes, 2);
        assert!((2.5..3.5).contains(&window.cardinality));
        assert!((0.1..0.4).contains(&window.estimated_hit_rate));
    }

    #[test]
    fn cluster_hll_rejects_incompatible_bucket_bits_without_affecting_liveness() {
        let store = hll_store(StoreConfig::default(), 14);
        store
            .heartbeat_node("node-a", Uuid::new_v4(), hll_report_for(&[1], 1, 14))
            .unwrap();
        store
            .heartbeat_node("node-b", Uuid::new_v4(), hll_report_for(&[2], 1, 13))
            .unwrap();
        let mut incompatible_duration = hll_report_for(&[3], 1, 14);
        incompatible_duration.windows[0].window_secs = 120;
        store
            .heartbeat_node("node-c", Uuid::new_v4(), incompatible_duration)
            .unwrap();

        let cluster = store.cluster_hll_snapshot();
        assert_eq!(store.node_counts(), (3, 0));
        assert_eq!(cluster.windows.len(), 1);
        assert_eq!(cluster.windows[0].active_nodes, 1);
        assert_eq!(cluster.windows[0].total_requests, 1);
    }

    #[test]
    fn unregister_removes_hll_snapshot_from_cluster() {
        let store = hll_store(StoreConfig::default(), 14);
        let node_a = Uuid::new_v4();
        let node_b = Uuid::new_v4();
        store
            .heartbeat_node("node-a", node_a, hll_report_for(&[1], 1, 14))
            .unwrap();
        store
            .heartbeat_node("node-b", node_b, hll_report_for(&[2], 1, 14))
            .unwrap();

        store.unregister_node("node-b", node_b).unwrap();
        let cluster = store.cluster_hll_snapshot();
        assert_eq!(cluster.windows[0].active_nodes, 1);
        assert_eq!(cluster.windows[0].total_requests, 1);
        assert!((0.5..1.5).contains(&cluster.windows[0].cardinality));
    }

    #[test]
    fn cluster_hll_snapshot_excludes_nodes_as_soon_as_they_become_stale() {
        let store = hll_store(
            StoreConfig {
                node_stale_after: Duration::from_millis(10),
                ttl: Duration::from_secs(60),
            },
            14,
        );
        let node_id = Uuid::new_v4();
        store
            .heartbeat_node("node-a", node_id, hll_report_for(&[1], 1, 14))
            .unwrap();

        std::thread::sleep(Duration::from_millis(20));

        assert!(store.cluster_hll_snapshot().windows.is_empty());
    }

    #[test]
    fn cluster_hll_snapshot_age_advances_between_reads() {
        let store = hll_store(StoreConfig::default(), 14);
        store
            .heartbeat_node("node-a", Uuid::new_v4(), hll_report_for(&[1], 1, 14))
            .unwrap();
        let first_age = store.cluster_hll_snapshot().windows[0].snapshot_age_seconds;

        std::thread::sleep(Duration::from_millis(20));

        let second_age = store.cluster_hll_snapshot().windows[0].snapshot_age_seconds;
        assert!(second_age > first_age);
    }

    #[test]
    fn concurrent_initial_reports_use_the_configured_schema() {
        let store = Arc::new(hll_store(StoreConfig::default(), 4));
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let valid_store = Arc::clone(&store);
        let valid_barrier = Arc::clone(&barrier);
        let valid = std::thread::spawn(move || {
            valid_barrier.wait();
            valid_store
                .heartbeat_node("valid", Uuid::new_v4(), hll_report_for(&[1], 1, 4))
                .unwrap();
        });
        let invalid_store = Arc::clone(&store);
        let invalid_barrier = Arc::clone(&barrier);
        let invalid = std::thread::spawn(move || {
            invalid_barrier.wait();
            invalid_store
                .heartbeat_node("invalid", Uuid::new_v4(), hll_report_for(&[2], 1, 5))
                .unwrap();
        });
        barrier.wait();
        valid.join().unwrap();
        invalid.join().unwrap();

        assert_eq!(store.node_counts(), (2, 0));
        let cluster = store.cluster_hll_snapshot();
        assert_eq!(cluster.windows[0].active_nodes, 1);
        assert_eq!(cluster.windows[0].total_requests, 1);
    }

    #[test]
    fn metadata_writes_keep_metadata_live_without_extending_hll_liveness() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_millis(10),
            ttl: Duration::from_secs(60),
        });
        let node_id = heartbeat_node(&store, "node-a");
        std::thread::sleep(Duration::from_millis(20));

        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();

        // Metadata writes keep cache ownership visible, but only a heartbeat
        // refreshes the HLL receipt time.
        assert_eq!(store.node_counts(), (1, 0));
        assert_eq!(store.query_prefix("ns", &[vec![1]]).len(), 1);
        assert!(store.cluster_hll_snapshot().windows.is_empty());
    }

    #[test]
    fn test_register_insert_and_query() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let node = "10.0.0.1:50055";
        let node_id = heartbeat_node(&store, node);

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        let reclaimable = store
            .insert_hashes(namespace, &hashes, node, node_id)
            .unwrap();
        assert!(reclaimable.is_empty());

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 3);
        for entry in &existing {
            assert_eq!(entry.nodes.len(), 1);
            assert_eq!(entry.nodes[0].as_ref(), node);
        }

        let mixed_hashes = vec![vec![1, 2, 3, 4], vec![99, 99, 99, 99], vec![5, 6, 7, 8]];
        let existing = store.query_prefix(namespace, &mixed_hashes);
        assert_eq!(existing.len(), 1);

        let existing = store.query_prefix("other-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_multi_owner() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node_a = "node-a:50055";
        let node_b = "node-b:50055";
        let node_a_id = heartbeat_node(&store, node_a);
        let node_b_id = heartbeat_node(&store, node_b);

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node_a, node_a_id)
            .unwrap();
        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node_b, node_b_id)
            .unwrap();

        let existing = store.query_prefix(namespace, std::slice::from_ref(&hash));
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes.len(), 2);

        let mut node_names: Vec<&str> = existing[0].nodes.iter().map(|n| n.as_ref()).collect();
        node_names.sort();
        assert_eq!(node_names, vec!["node-a:50055", "node-b:50055"]);
    }

    #[test]
    fn insert_returns_only_new_third_owner_hashes() {
        let store = BlockHashStore::new();
        let node_a = heartbeat_node(&store, "node-a:50055");
        let node_b = heartbeat_node(&store, "node-b:50055");
        let node_c = heartbeat_node(&store, "node-c:50055");
        let node_d = heartbeat_node(&store, "node-d:50055");
        let hashes = vec![vec![1], vec![2], vec![1]];

        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-a:50055", node_a)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-b:50055", node_b)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-c:50055", node_c)
                .unwrap(),
            vec![vec![1], vec![2]]
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-c:50055", node_c)
                .unwrap(),
            Vec::<Vec<u8>>::new()
        );
        assert_eq!(
            store
                .insert_hashes("ns", &hashes, "node-d:50055", node_d)
                .unwrap(),
            vec![vec![1], vec![2]]
        );
    }

    #[test]
    fn stale_owner_does_not_count_toward_reclaim_hint() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ttl: Duration::from_secs(60),
        });
        let hash = vec![1];
        let node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let node_c = heartbeat_node(&store, "node-c");
        let node_d = heartbeat_node(&store, "node-d");

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b)
            .unwrap();
        assert!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-c", node_c)
                .unwrap()
                .is_empty(),
            "the stale owner must not make node-c the third live owner"
        );
        assert_eq!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-d", node_d)
                .unwrap(),
            vec![hash]
        );
    }

    #[test]
    fn new_session_at_same_address_counts_as_new_owner() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(30),
            ttl: Duration::from_secs(60),
        });
        let hash = vec![1];
        let old_node_a = heartbeat_node(&store, "node-a");
        let node_b = heartbeat_node(&store, "node-b");
        let node_c = heartbeat_node(&store, "node-c");

        for (node, node_id) in [
            ("node-a", old_node_a),
            ("node-b", node_b),
            ("node-c", node_c),
        ] {
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), node, node_id)
                .unwrap();
        }

        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let new_node_a = heartbeat_node(&store, "node-a");
        assert_ne!(old_node_a, new_node_a);

        assert_eq!(
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", new_node_a,)
                .unwrap(),
            vec![hash]
        );
    }

    #[test]
    fn test_empty_store() {
        let store = BlockHashStore::new();
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_prefix("any-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_remove_own_blocks() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node = "node-a";
        let node_id = heartbeat_node(&store, node);

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), node, node_id)
            .unwrap();

        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), node, node_id)
            .unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 0);
    }

    #[test]
    fn test_remove_other_nodes_block_is_noop() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];
        let node_b_id = heartbeat_node(&store, "node-b");
        let node_a_id = heartbeat_node(&store, "node-a");

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-b", node_b_id)
            .unwrap();

        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 1);
    }

    #[test]
    fn test_remove_one_owner_keeps_others() {
        let store = BlockHashStore::new();
        let hash = vec![1, 2, 3];
        let node_a_id = heartbeat_node(&store, "node-a");
        let node_b_id = heartbeat_node(&store, "node-b");

        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        store
            .insert_hashes("ns", std::slice::from_ref(&hash), "node-b", node_b_id)
            .unwrap();

        let removed = store
            .remove_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
            .unwrap();
        assert_eq!(removed, 1);

        let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-b");
    }

    #[test]
    fn test_remove_nonexistent_is_noop() {
        let store = BlockHashStore::new();
        let node_id = heartbeat_node(&store, "node-a");
        let removed = store
            .remove_hashes("ns", &[vec![9, 9, 9]], "node-a", node_id)
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_heartbeat_rejects_active_different_session() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        let new_id = Uuid::new_v4();
        assert_ne!(old_id, new_id);

        let err = store
            .heartbeat_node("node-a", new_id, test_hll_report())
            .unwrap_err();
        assert_eq!(err, StoreError::StaleSession);
    }

    #[test]
    fn test_query_filters_superseded_node_session() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn test_sweep_keeps_superseded_owner_until_ttl() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let removed = store.sweep_expired();
        assert_eq!(removed, SweepStats::default());
        assert_eq!(store.owner_count(), 1);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
    }

    #[test]
    fn test_sweep_removes_superseded_owner_after_key_purge_age() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::ZERO,
        });
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        let removed = store.sweep_expired();
        assert_eq!(
            removed,
            SweepStats {
                removed_owners: 1,
                removed_keys: 1,
                removed_nodes: 1,
            }
        );
        assert_eq!(store.owner_count(), 0);
    }

    #[test]
    fn test_late_unregister_old_session_does_not_remove_current_node() {
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        store
            .insert_hashes("ns", &[vec![1]], "node-a", new_id)
            .unwrap();

        let err = store.unregister_node("node-a", old_id).unwrap_err();
        assert_eq!(err, StoreError::StaleSession);

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-a");
    }

    #[test]
    fn test_unregistered_insert_is_rejected() {
        let store = BlockHashStore::new();
        let err = store
            .insert_hashes("ns", &[vec![1]], "node-a", Uuid::new_v4())
            .unwrap_err();
        assert_eq!(err, StoreError::UnknownNode);
    }

    #[test]
    fn test_query_filters_stale_node() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_millis(1),
            ttl: Duration::from_secs(60),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();
        std::thread::sleep(Duration::from_millis(2));

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn test_insert_refreshes_node_liveness() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(60),
            ttl: Duration::from_secs(120),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(61);

        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();

        assert_eq!(store.node_counts(), (1, 0));
        let existing = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-a");
    }

    #[test]
    fn test_remove_refreshes_node_liveness() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(60),
            ttl: Duration::from_secs(120),
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(61);

        store
            .remove_hashes("ns", &[vec![2]], "node-a", node_id)
            .unwrap();

        assert_eq!(store.node_counts(), (1, 0));
    }

    #[test]
    fn test_sweep_expired() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::ZERO,
        });
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "node-a", node_id)
            .unwrap();
        assert_eq!(store.entry_count(), 2);
        assert_eq!(store.owner_count(), 2);

        let removed = store.sweep_expired();
        assert_eq!(
            removed,
            SweepStats {
                removed_owners: 2,
                removed_keys: 2,
                removed_nodes: 1,
            }
        );
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));
    }

    #[test]
    fn test_sweep_keeps_fresh_entries() {
        let store = BlockHashStore::new();
        let node_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "node-a", node_id)
            .unwrap();

        let removed = store.sweep_expired();
        assert_eq!(removed, SweepStats::default());
        assert_eq!(store.entry_count(), 2);
    }

    #[test]
    fn test_concurrent_insert_and_remove() {
        use std::sync::Arc;

        let store = Arc::new(BlockHashStore::new());
        let hash = vec![1, 2, 3, 4];

        for _ in 0..100 {
            let node_a_id = heartbeat_node(&store, "node-a");
            let node_b_id = heartbeat_node(&store, "node-b");
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
                .unwrap();

            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);
            let hash_a = hash.clone();
            let hash_b = hash.clone();

            std::thread::scope(|s| {
                s.spawn(|| {
                    store_a
                        .remove_hashes("ns", &[hash_a], "node-a", node_a_id)
                        .unwrap();
                });
                s.spawn(|| {
                    store_b
                        .insert_hashes("ns", &[hash_b], "node-b", node_b_id)
                        .unwrap();
                });
            });

            let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
            assert_eq!(existing.len(), 1, "key must exist after concurrent ops");
            assert!(
                existing[0].nodes.iter().any(|n| n.as_ref() == "node-b"),
                "node-b must be present"
            );

            store.invalidate_all();
        }
    }

    #[test]
    fn test_invalidate_all() {
        let store = BlockHashStore::new();
        let namespace = "model-test";
        let node = "10.0.0.1:50055";
        let node_id = heartbeat_node(&store, node);

        let hashes = vec![vec![1, 2, 3], vec![4, 5, 6]];
        store
            .insert_hashes(namespace, &hashes, node, node_id)
            .unwrap();

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 2);

        store.invalidate_all();

        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 0);
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
        assert_eq!(store.node_counts(), (0, 0));
    }

    #[test]
    fn test_redundancy_snapshot_buckets_live_owners() {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "n-a");
        let b = heartbeat_node(&store, "n-b");
        let c = heartbeat_node(&store, "n-c");
        let d = heartbeat_node(&store, "n-d");

        // 1 owner, 2 owners, and 4 owners across three distinct keys.
        store.insert_hashes("ns", &[vec![1]], "n-a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "n-a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "n-b", b).unwrap();
        for (node, id) in [("n-a", a), ("n-b", b), ("n-c", c), ("n-d", d)] {
            store.insert_hashes("ns", &[vec![3]], node, id).unwrap();
        }

        store.sweep_expired();

        assert_eq!(
            store.redundancy_snapshot(),
            RedundancySnapshot {
                keys_1: 1,
                keys_2: 1,
                keys_3: 0,
                keys_4plus: 1,
                copies: 1 + 2 + 4,
            }
        );
    }

    #[test]
    fn test_redundancy_snapshot_excludes_superseded_owner() {
        // A block whose only owner is a superseded session has zero *visible*
        // owners, so it must not appear in any bucket even though the raw record
        // survives until the TTL purge.
        let store = BlockHashStore::new();
        let old_id = heartbeat_node(&store, "node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen =
            Instant::now() - Duration::from_secs(DEFAULT_NODE_STALE_SECS + 1);
        let new_id = heartbeat_node(&store, "node-a");
        assert_ne!(old_id, new_id);

        store.sweep_expired();

        assert_eq!(store.owner_count(), 1, "raw record still present");
        assert_eq!(store.redundancy_snapshot(), RedundancySnapshot::default());
    }
}
