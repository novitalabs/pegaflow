use dashmap::DashMap;
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    UnknownNode,
    StaleSession,
}

#[derive(Debug, Clone)]
struct OwnerRecord {
    node_id: Uuid,
    key_register_time: Instant,
}

#[derive(Debug, Clone)]
struct NodeRecord {
    node_id: Uuid,
    last_seen: Instant,
}

/// Async thread-safe block hash storage using DashMap.
///
/// `blocks` maps each block key to node URL ownership records. `nodes` tracks
/// the current MetaServer session and liveness for each node URL.
pub struct BlockHashStore {
    blocks: DashMap<BlockKey, HashMap<Arc<str>, OwnerRecord>>,
    nodes: DashMap<Arc<str>, NodeRecord>,
    config: StoreConfig,
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
        }
    }

    pub fn with_ttl(ttl_minutes: u64) -> Self {
        Self::with_config(StoreConfig {
            node_stale_after: Duration::from_secs(DEFAULT_NODE_STALE_SECS),
            ttl: Duration::from_secs(ttl_minutes * 60),
        })
    }

    pub fn register_node(&self, node: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        self.nodes.insert(
            Arc::from(node),
            NodeRecord {
                node_id,
                last_seen: Instant::now(),
            },
        );
        node_id
    }

    pub fn heartbeat_node(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let Some(mut record) = self.nodes.get_mut(node) else {
            return Err(StoreError::UnknownNode);
        };
        if record.node_id != node_id {
            return Err(StoreError::StaleSession);
        }
        record.last_seen = Instant::now();
        Ok(())
    }

    pub fn unregister_node(&self, node: &str, node_id: Uuid) -> Result<usize, StoreError> {
        if self
            .nodes
            .remove_if(node, |_, record| record.node_id == node_id)
            .is_none()
        {
            if self.nodes.contains_key(node) {
                return Err(StoreError::StaleSession);
            }
            return Err(StoreError::UnknownNode);
        }
        Ok(self.remove_node_owners(node, node_id))
    }

    pub fn insert_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<usize, StoreError> {
        self.validate_node_session(node, node_id)?;
        let node: Arc<str> = Arc::from(node);
        let now = Instant::now();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            self.blocks.entry(key).or_default().insert(
                Arc::clone(&node),
                OwnerRecord {
                    node_id,
                    key_register_time: now,
                },
            );
        }
        Ok(hashes.len())
    }

    pub fn remove_hashes(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
        node: &str,
        node_id: Uuid,
    ) -> Result<usize, StoreError> {
        self.validate_node_session(node, node_id)?;
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

    /// Sweep owners whose node is missing or whose ownership TTL has expired.
    pub fn sweep_expired(&self) -> SweepStats {
        let now = Instant::now();
        let mut stats = SweepStats::default();

        self.blocks.retain(|_, owners| {
            let before = owners.len();
            owners.retain(|node, owner| self.should_keep_owner(node, owner, now));
            stats.removed_owners += before.saturating_sub(owners.len());
            if owners.is_empty() {
                stats.removed_keys += 1;
                false
            } else {
                true
            }
        });

        let node_before = self.nodes.len();
        let ttl = self.config.ttl;
        self.nodes
            .retain(|_, record| now.duration_since(record.last_seen) <= ttl);
        stats.removed_nodes = node_before.saturating_sub(self.nodes.len());

        stats
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

    #[allow(
        dead_code,
        reason = "maintenance API reserved for explicit store cleanup"
    )]
    pub fn invalidate_all(&self) {
        self.blocks.clear();
        self.nodes.clear();
    }

    fn validate_node_session(&self, node: &str, node_id: Uuid) -> Result<(), StoreError> {
        let Some(record) = self.nodes.get(node) else {
            return Err(StoreError::UnknownNode);
        };
        if record.node_id != node_id {
            return Err(StoreError::StaleSession);
        }
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

    fn is_owner_visible(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> bool {
        let Some(record) = self.nodes.get(node.as_ref()) else {
            return false;
        };
        record.node_id == owner.node_id
            && now.duration_since(record.last_seen) <= self.config.node_stale_after
    }

    fn should_keep_owner(&self, node: &Arc<str>, owner: &OwnerRecord, now: Instant) -> bool {
        let Some(record) = self.nodes.get(node.as_ref()) else {
            return false;
        };
        now.duration_since(owner.key_register_time) <= self.config.ttl
            && now.duration_since(record.last_seen) <= self.config.ttl
    }
}

impl Default for BlockHashStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_insert_and_query() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let node = "10.0.0.1:50055";
        let node_id = store.register_node(node);

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        let inserted = store
            .insert_hashes(namespace, &hashes, node, node_id)
            .unwrap();
        assert_eq!(inserted, 3);

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
        let node_a_id = store.register_node(node_a);
        let node_b_id = store.register_node(node_b);

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
        let node_id = store.register_node(node);

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
        let node_b_id = store.register_node("node-b");
        let node_a_id = store.register_node("node-a");

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
        let node_a_id = store.register_node("node-a");
        let node_b_id = store.register_node("node-b");

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
        let node_id = store.register_node("node-a");
        let removed = store
            .remove_hashes("ns", &[vec![9, 9, 9]], "node-a", node_id)
            .unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_query_filters_superseded_node_session() {
        let store = BlockHashStore::new();
        let old_id = store.register_node("node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        let new_id = store.register_node("node-a");
        assert_ne!(old_id, new_id);

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn test_sweep_keeps_superseded_owner_until_ttl() {
        let store = BlockHashStore::new();
        let old_id = store.register_node("node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        let new_id = store.register_node("node-a");
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
        let old_id = store.register_node("node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", old_id)
            .unwrap();
        let new_id = store.register_node("node-a");
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
        let old_id = store.register_node("node-a");
        let new_id = store.register_node("node-a");
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
            node_stale_after: Duration::ZERO,
            ttl: Duration::from_secs(60),
        });
        let node_id = store.register_node("node-a");
        store
            .insert_hashes("ns", &[vec![1]], "node-a", node_id)
            .unwrap();

        let existing = store.query_prefix("ns", &[vec![1]]);
        assert!(existing.is_empty());
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.owner_count(), 1);
    }

    #[test]
    fn test_sweep_expired() {
        let store = BlockHashStore::with_config(StoreConfig {
            node_stale_after: Duration::ZERO,
            ttl: Duration::ZERO,
        });
        let node_id = store.register_node("node-a");
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
        let node_id = store.register_node("node-a");
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
            let node_a_id = store.register_node("node-a");
            let node_b_id = store.register_node("node-b");
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
        let node_id = store.register_node(node);

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
}
