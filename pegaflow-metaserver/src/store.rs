use dashmap::{DashMap, mapref::entry::Entry};
use log::{info, warn};
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use uuid::Uuid;

const MIN_RECLAIMABLE_OWNER_COUNT: usize = 3;

pub const DEFAULT_NODE_STALE_SECS: u64 = 30;
pub const DEFAULT_TTL_MINUTES: u64 = 120;
pub const MANUAL_CLEANUP_AGE_SECS: u64 = 60 * 60;

/// A prefix query result: one block hash and all live nodes that own it.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub block_hash: Vec<u8>,
    pub nodes: Vec<Arc<str>>,
}

#[derive(Debug, Clone, Copy)]
pub struct StoreConfig {
    pub node_stale_after: Duration,
    /// Node inactivity grace period; never applies to owner registration age.
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

/// Incremental distribution of stored owner records, not query-visible owners.
/// Stale nodes remain counted until TTL cleanup, and superseded sessions until
/// reconciliation. Reads of separate atomic fields may straddle a mutation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RedundancySnapshot {
    pub keys_1: u64,
    pub keys_2: u64,
    pub keys_3: u64,
    pub keys_4plus: u64,
    pub copies: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// Authoritative current session for a node URL. `last_seen` is bumped on
/// heartbeat/insert/remove and gates query visibility.
#[derive(Debug, Clone)]
struct NodeRecord {
    node_id: Uuid,
    last_seen: Instant,
}

#[derive(Default)]
struct RedundancyCounters {
    keys_1: AtomicU64,
    keys_2: AtomicU64,
    keys_3: AtomicU64,
    keys_4plus: AtomicU64,
    copies: AtomicU64,
}

impl RedundancyCounters {
    fn snapshot(&self) -> RedundancySnapshot {
        RedundancySnapshot {
            keys_1: self.keys_1.load(Ordering::Relaxed),
            keys_2: self.keys_2.load(Ordering::Relaxed),
            keys_3: self.keys_3.load(Ordering::Relaxed),
            keys_4plus: self.keys_4plus.load(Ordering::Relaxed),
            copies: self.copies.load(Ordering::Relaxed),
        }
    }

    fn adjust_bucket(&self, count: u64, delta: i64) {
        let counter = match count {
            1 => &self.keys_1,
            2 => &self.keys_2,
            3 => &self.keys_3,
            _ if count >= 4 => &self.keys_4plus,
            _ => return,
        };
        if delta > 0 {
            counter.fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            counter.fetch_sub((-delta) as u64, Ordering::Relaxed);
        }
    }

    fn adjust(&self, before: u64, after: u64) {
        if before == after {
            return;
        }
        self.adjust_bucket(before, -1);
        self.adjust_bucket(after, 1);
        if after > before {
            self.copies.fetch_add(after - before, Ordering::Relaxed);
        } else {
            self.copies.fetch_sub(before - after, Ordering::Relaxed);
        }
    }
}

/// Async thread-safe block hash storage using DashMap.
///
/// `blocks` maps each block key to node URL ownership records. `nodes` tracks
/// the current MetaServer session and liveness for each node URL.
pub struct BlockHashStore {
    blocks: DashMap<BlockKey, HashMap<Arc<str>, OwnerRecord>>,
    nodes: DashMap<Arc<str>, NodeRecord>,
    config: StoreConfig,
    reconcile_needed: AtomicBool,
    /// Incremental stored-owner redundancy counters read by metric callbacks.
    redundancy: RedundancyCounters,
}

impl BlockHashStore {
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            blocks: DashMap::new(),
            nodes: DashMap::new(),
            config,
            reconcile_needed: AtomicBool::new(false),
            redundancy: RedundancyCounters::default(),
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

    pub fn heartbeat_node(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let now = Instant::now();
        match self.nodes.entry(Arc::from(node)) {
            Entry::Vacant(entry) => {
                info!("MetaServer node registered: node={node} node_id={node_id}");
                entry.insert(NodeRecord {
                    node_id,
                    last_seen: now,
                });
                Ok(())
            }
            Entry::Occupied(mut entry) => {
                let record = entry.get_mut();
                let same_session = record.node_id == node_id;
                let stale_session =
                    now.duration_since(record.last_seen) > self.config.node_stale_after;
                if same_session || stale_session {
                    if stale_session && !same_session {
                        self.reconcile_needed.store(true, Ordering::Release);
                        info!(
                            "MetaServer node session takeover: node={} old_node_id={} new_node_id={}",
                            node, record.node_id, node_id
                        );
                    }
                    record.node_id = node_id;
                    record.last_seen = now;
                    return Ok(());
                }
                warn!(
                    "MetaServer heartbeat rejected stale session: node={} current_node_id={} rejected_node_id={}",
                    node, record.node_id, node_id
                );
                Err(StoreError::StaleSession)
            }
        }
    }

    pub fn unregister_node(&self, node: &str, node_id: Uuid) -> Result<usize, StoreError> {
        match self.nodes.entry(Arc::from(node)) {
            Entry::Vacant(_) => return Err(StoreError::UnknownNode),
            Entry::Occupied(entry) => {
                if entry.get().node_id != node_id {
                    return Err(StoreError::StaleSession);
                }
                entry.remove();
            }
        }
        // A concurrent registration may already have written new owners.
        let removed = self.retain_owners(|owner_node, owner| {
            owner_node.as_ref() != node
                || self
                    .nodes
                    .get(node)
                    .is_some_and(|record| record.node_id == owner.node_id)
        });
        Ok(removed.removed_owners)
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
            // Lock blocks before nodes, and keep session validation valid through the write.
            let entry = self.blocks.entry(key);
            let record = self
                .nodes
                .get(node.as_ref())
                .ok_or(StoreError::UnknownNode)?;
            if record.node_id != node_id {
                return Err(StoreError::StaleSession);
            }
            let mut owners = entry.or_default();
            let before = owners.len();
            let previous = owners.insert(
                Arc::clone(&node),
                OwnerRecord {
                    node_id,
                    key_register_time: now,
                },
            );
            self.redundancy.adjust(before as u64, owners.len() as u64);
            // Reclaim hints perform their own node lookups; never nest node guards.
            drop(record);
            let is_new_owner = previous.is_none_or(|owner| owner.node_id != node_id);
            if is_new_owner
                && owners.len() >= MIN_RECLAIMABLE_OWNER_COUNT
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
            if let Entry::Occupied(mut entry) = self.blocks.entry(key) {
                let owners = entry.get_mut();
                let before = owners.len();
                if owners
                    .get(node)
                    .is_some_and(|owner| owner.node_id == node_id)
                {
                    owners.remove(node);
                    removed += 1;
                }
                self.redundancy.adjust(before as u64, owners.len() as u64);
                if owners.is_empty() {
                    entry.remove();
                }
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

    /// Healthy sweeps only inspect nodes. A takeover or node TTL expiry triggers
    /// one block scan; owner registration age never causes background deletion.
    pub fn sweep_expired(&self) -> SweepStats {
        // Consume before scanning so a concurrent takeover remains pending.
        let reconcile = self.reconcile_needed.swap(false, Ordering::AcqRel);
        let now = Instant::now();
        let mut removed_nodes = 0;
        self.nodes.retain(|node, record| {
            let age = now.saturating_duration_since(record.last_seen);
            let keep = age <= self.config.ttl;
            if !keep {
                removed_nodes += 1;
                info!(
                    "MetaServer node swept: node={} node_id={} last_seen_age_secs={}",
                    node,
                    record.node_id,
                    age.as_secs()
                );
            }
            keep
        });
        if removed_nodes == 0 && !reconcile {
            return SweepStats::default();
        }
        let mut stats = self.retain_owners(|node, owner| {
            self.nodes
                .get(node.as_ref())
                .is_some_and(|record| record.node_id == owner.node_id)
        });
        stats.removed_nodes = removed_nodes;
        stats
    }

    /// Remove ownership records older than `max_age`, regardless of node
    /// liveness. This is reserved for explicit operator maintenance.
    pub fn remove_owners_older_than(&self, max_age: Duration) -> SweepStats {
        let now = Instant::now();
        self.retain_owners(|_, owner| {
            now.saturating_duration_since(owner.key_register_time) <= max_age
        })
    }

    /// Latest incrementally maintained stored-owner redundancy distribution.
    pub fn redundancy_snapshot(&self) -> RedundancySnapshot {
        self.redundancy.snapshot()
    }

    pub fn entry_count(&self) -> u64 {
        let snap = self.redundancy.snapshot();
        snap.keys_1 + snap.keys_2 + snap.keys_3 + snap.keys_4plus
    }

    pub fn owner_count(&self) -> u64 {
        self.redundancy.copies.load(Ordering::Relaxed)
    }

    pub fn node_counts(&self) -> (u64, u64) {
        let now = Instant::now();
        let mut active = 0;
        let mut stale = 0;
        for node in &self.nodes {
            let age = now.duration_since(node.last_seen);
            if age <= self.config.node_stale_after {
                active += 1;
            } else {
                stale += 1;
            }
        }
        (active, stale)
    }

    #[allow(
        dead_code,
        reason = "maintenance API reserved for explicit store cleanup"
    )]
    pub fn invalidate_all(&self) {
        self.nodes.clear();
        self.retain_owners(|_, _| false);
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
        record.last_seen = Instant::now();
        Ok(())
    }

    /// Owner changes and accounting share the block shard's write guard.
    fn retain_owners(&self, mut keep: impl FnMut(&Arc<str>, &OwnerRecord) -> bool) -> SweepStats {
        let mut stats = SweepStats::default();
        self.blocks.retain(|_, owners| {
            let before = owners.len();
            owners.retain(|node, owner| keep(node, owner));
            self.redundancy.adjust(before as u64, owners.len() as u64);
            stats.removed_owners += before - owners.len();
            if owners.is_empty() {
                stats.removed_keys += 1;
                return false;
            }
            true
        });
        stats
    }

    fn is_owner_visible(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> bool {
        let Some(record) = self.nodes.get(node.as_ref()) else {
            return false;
        };
        let node_age = now.duration_since(record.last_seen);
        record.node_id == owner.node_id && node_age <= self.config.node_stale_after
    }
}

impl Default for BlockHashStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn manual_cleanup_fixture() -> BlockHashStore {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "a", a)
            .unwrap();
        for mut owners in store.blocks.iter_mut() {
            owners.get_mut("a").unwrap().key_register_time =
                Instant::now() - Duration::from_secs(MANUAL_CLEANUP_AGE_SECS + 1);
        }
        // Refresh one old owner and add a fresh replica to another old key.
        store.insert_hashes("ns", &[vec![3]], "a", a).unwrap();
        store.insert_hashes("ns", &[vec![2]], "b", b).unwrap();
        store
    }

    fn assert_stored_counts(store: &BlockHashStore) {
        let mut expected = RedundancySnapshot::default();
        for owners in &store.blocks {
            match owners.len() {
                0 => panic!("empty key retained"),
                1 => expected.keys_1 += 1,
                2 => expected.keys_2 += 1,
                3 => expected.keys_3 += 1,
                _ => expected.keys_4plus += 1,
            }
            expected.copies += owners.len() as u64;
        }
        assert_eq!(store.owner_count(), expected.copies);
        assert_eq!(store.entry_count(), store.blocks.len() as u64);
        assert_eq!(store.redundancy_snapshot(), expected);
    }

    fn wait_until(mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !ready() {
            assert!(
                Instant::now() < deadline,
                "concurrent operation did not progress"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn insert_rechecks_session_after_waiting_for_block_lock() {
        for occupied in [false, true] {
            let store = BlockHashStore::new();
            let old = heartbeat_node(&store, "a");
            if occupied {
                store.insert_hashes("ns", &[vec![1]], "a", old).unwrap();
            }
            let stale = Instant::now() - Duration::from_secs(31);
            store.nodes.get_mut("a").unwrap().last_seen = stale;
            std::thread::scope(|scope| {
                let mut entry = store.blocks.entry(BlockKey::new("ns".into(), vec![1]));
                let insert = scope.spawn(|| store.insert_hashes("ns", &[vec![1]], "a", old));
                wait_until(|| store.nodes.get("a").unwrap().last_seen > stale);
                store.nodes.get_mut("a").unwrap().last_seen = stale;
                let current = heartbeat_node(&store, "a");
                if let Entry::Occupied(ref mut entry) = entry {
                    // Model a new-session write that wins the block lock first.
                    entry.get_mut().get_mut("a").unwrap().node_id = current;
                }
                drop(entry);
                assert_eq!(insert.join().unwrap(), Err(StoreError::StaleSession));
                assert_eq!(
                    store.query_prefix("ns", &[vec![1]]).len(),
                    usize::from(occupied)
                );
            });
            assert_stored_counts(&store);
        }
    }

    #[test]
    fn takeover_during_sweep_remains_pending() {
        let store = BlockHashStore::new();
        let old = heartbeat_node(&store, "a");
        store.insert_hashes("ns", &[vec![1]], "a", old).unwrap();
        store.reconcile_needed.store(true, Ordering::Release);
        std::thread::scope(|scope| {
            let guard = store
                .blocks
                .get_mut(&BlockKey::new("ns".into(), vec![1]))
                .unwrap();
            let sweep = scope.spawn(|| store.sweep_expired());
            wait_until(|| !store.reconcile_needed.load(Ordering::Acquire));
            store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
            heartbeat_node(&store, "a");
            drop(guard);
            assert_eq!(sweep.join().unwrap().removed_owners, 1);
        });
        assert!(store.reconcile_needed.load(Ordering::Acquire));
        assert!(store.sweep_expired().is_empty());
        assert!(!store.reconcile_needed.load(Ordering::Acquire));
        assert_stored_counts(&store);
    }

    #[test]
    fn unregister_preserves_concurrent_registration() {
        let store = BlockHashStore::new();
        let old = heartbeat_node(&store, "a");
        store.insert_hashes("ns", &[vec![1]], "a", old).unwrap();
        std::thread::scope(|scope| {
            let mut owners = store
                .blocks
                .get_mut(&BlockKey::new("ns".into(), vec![1]))
                .unwrap();
            let unregister = scope.spawn(|| store.unregister_node("a", old));
            wait_until(|| !store.nodes.contains_key("a"));
            let current = heartbeat_node(&store, "a");
            owners.get_mut("a").unwrap().node_id = current;
            drop(owners);
            assert_eq!(unregister.join().unwrap(), Ok(0));
        });
        assert_eq!(store.query_prefix("ns", &[vec![1]]).len(), 1);
        assert_stored_counts(&store);
    }

    #[test]
    fn stale_node_survives_sweep_and_recovers_without_insert() {
        let store = BlockHashStore::new();
        let id = heartbeat_node(&store, "node-a");
        store.insert_hashes("ns", &[vec![1]], "node-a", id).unwrap();
        store.nodes.get_mut("node-a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());

        // Healthy sweeps and entry metrics must not wait for block shard locks.
        let guard = store
            .blocks
            .get_mut(&BlockKey::new("ns".into(), vec![1]))
            .unwrap();
        std::thread::scope(|scope| {
            let (tx, rx) = std::sync::mpsc::channel();
            let store_ref = &store;
            scope.spawn(move || {
                tx.send((store_ref.sweep_expired(), store_ref.entry_count()))
                    .unwrap();
            });
            let result = rx.recv_timeout(Duration::from_secs(2));
            drop(guard);
            assert_eq!(result.unwrap(), (SweepStats::default(), 1));
        });
        assert_eq!(store.node_counts(), (0, 1));
        assert_stored_counts(&store);
        store.heartbeat_node("node-a", id).unwrap();
        assert_eq!(store.query_prefix("ns", &[vec![1]]).len(), 1);
        assert_stored_counts(&store);
    }

    #[test]
    fn node_ttl_cleanup_preserves_old_blocks_on_active_nodes() {
        let store = BlockHashStore::new();
        let a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "a", a)
            .unwrap();
        store.insert_hashes("ns", &[vec![1]], "b", b).unwrap();
        let old = Instant::now() - store.config.ttl - Duration::from_secs(1);
        for mut owners in store.blocks.iter_mut() {
            for owner in owners.values_mut() {
                owner.key_register_time = old;
            }
        }
        store.nodes.get_mut("a").unwrap().last_seen = old;
        assert_eq!(
            store.sweep_expired(),
            SweepStats {
                removed_owners: 2,
                removed_keys: 1,
                removed_nodes: 1,
            }
        );
        assert_eq!(
            store.query_prefix("ns", &[vec![1]])[0].nodes[0].as_ref(),
            "b"
        );
        assert_stored_counts(&store);
    }

    #[test]
    fn mutations_between_takeovers_and_reconciliation_keep_counts_consistent() {
        let store = BlockHashStore::new();
        let mut a = heartbeat_node(&store, "a");
        let b = heartbeat_node(&store, "b");
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "a", a)
            .unwrap();
        store
            .insert_hashes("ns", &[vec![1], vec![2], vec![3]], "b", b)
            .unwrap();
        for _ in 0..2 {
            store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
            a = heartbeat_node(&store, "a");
            store
                .insert_hashes("ns", &[vec![1], vec![1]], "a", a)
                .unwrap();
            store.remove_hashes("ns", &[vec![1]], "b", b).unwrap();
            assert_stored_counts(&store);
        }
        assert_eq!(store.sweep_expired().removed_owners, 2);
        assert_eq!(store.owner_count(), 3);
        assert_stored_counts(&store);
        // A later takeover must set a new flag, even after one sweep consumed it.
        store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let new_a = heartbeat_node(&store, "a");
        assert_ne!(a, new_a);
        assert_eq!(store.sweep_expired().removed_owners, 1);
        assert_stored_counts(&store);
        assert!(store.sweep_expired().is_empty());
    }

    #[test]
    fn unregister_after_takeover_cleans_historical_owners() {
        let store = BlockHashStore::new();
        let old = heartbeat_node(&store, "a");
        store
            .insert_hashes("ns", &[vec![1], vec![2]], "a", old)
            .unwrap();
        store.nodes.get_mut("a").unwrap().last_seen = Instant::now() - Duration::from_secs(31);
        let current = heartbeat_node(&store, "a");
        store.insert_hashes("ns", &[vec![1]], "a", current).unwrap();
        assert_eq!(
            store.unregister_node("a", old),
            Err(StoreError::StaleSession)
        );
        assert_eq!(store.unregister_node("a", current), Ok(2));
        assert!(store.sweep_expired().is_empty());
        assert_stored_counts(&store);
        assert_eq!(store.node_counts(), (0, 0));
    }

    fn heartbeat_node(store: &BlockHashStore, node: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        store.heartbeat_node(node, node_id).unwrap();
        node_id
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

        let err = store.heartbeat_node("node-a", new_id).unwrap_err();
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
    fn test_sweep_reconciles_superseded_owner_before_ttl() {
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
        assert_eq!(removed.removed_owners, 1);
        assert_eq!(removed.removed_keys, 1);
        assert_eq!(removed.removed_nodes, 0);
        assert_eq!(store.owner_count(), 0);
        assert_stored_counts(&store);
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
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
            ttl: Duration::from_secs(60),
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
            ttl: Duration::from_secs(60),
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
    fn test_concurrent_insert_remove_and_cleanup() {
        use std::sync::Arc;

        let store = Arc::new(BlockHashStore::new());
        let hash = vec![1, 2, 3, 4];

        for _ in 0..100 {
            let node_a_id = heartbeat_node(&store, "node-a");
            let node_b_id = heartbeat_node(&store, "node-b");
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a", node_a_id)
                .unwrap();
            store
                .blocks
                .get_mut(&BlockKey::new("ns".into(), hash.clone()))
                .unwrap()
                .get_mut("node-a")
                .unwrap()
                .key_register_time = Instant::now() - Duration::from_secs(3601);

            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);
            let hash_a = hash.clone();
            let hash_b = hash.clone();

            std::thread::scope(|s| {
                s.spawn(|| store.remove_owners_older_than(Duration::from_secs(3600)));
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

            assert_stored_counts(&store);
            store.invalidate_all();
            assert_stored_counts(&store);
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
    fn test_redundancy_counters_follow_owner_mutations() {
        let store = BlockHashStore::new();
        let nodes = ["a", "b", "c", "d", "e"];
        let ids = nodes.map(|node| heartbeat_node(&store, node));
        let hashes = [vec![1], vec![2], vec![3], vec![4], vec![5]];
        assert_stored_counts(&store);
        for (i, node) in nodes.iter().enumerate() {
            for _ in 0..2 {
                store
                    .insert_hashes("ns", &hashes[i..], node, ids[i])
                    .unwrap();
                assert_stored_counts(&store);
            }
        }
        assert_eq!(
            store.redundancy_snapshot(),
            RedundancySnapshot {
                keys_1: 1,
                keys_2: 1,
                keys_3: 1,
                keys_4plus: 2,
                copies: 15,
            }
        );
        for (i, node) in nodes.iter().enumerate() {
            assert_eq!(
                store
                    .remove_hashes("ns", &hashes[i..], node, ids[i])
                    .unwrap(),
                hashes.len() - i
            );
            assert_stored_counts(&store);
        }
        assert_eq!(store.redundancy_snapshot(), RedundancySnapshot::default());
    }
}
