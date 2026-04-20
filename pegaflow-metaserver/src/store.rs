use dashmap::DashMap;
use log::info;
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default TTL for cache entries (120 minutes)
pub const DEFAULT_TTL_MINUTES: u64 = 120;
pub const DEFAULT_SUSPECT_SECS: u64 = 30;
pub const DEFAULT_HARD_DELETE_SECS: u64 = 90;

/// A prefix query result: one block hash and all nodes that own it.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub block_hash: Vec<u8>,
    pub nodes: Vec<Arc<str>>,
}

pub struct NodeLiveness {
    pub epoch: String,
    pub last_seen: Instant,
}

/// Async thread-safe block hash storage using DashMap.
/// Stores BlockKeys (namespace + hash) mapped to a set of owning node URLs with
/// registration timestamps, enabling multi-owner tracking and periodic TTL sweep.
pub struct BlockHashStore {
    /// Key: BlockKey, Value: { node_url → registration_time }
    map: DashMap<BlockKey, HashMap<Arc<str>, Instant>>,
    ttl: Duration,
    /// Per-node liveness tracking
    nodes: DashMap<Arc<str>, NodeLiveness>,
    suspect_threshold: Duration,
    hard_delete_threshold: Duration,
}

impl BlockHashStore {
    /// Create a new block hash store with default TTL (120 minutes)
    pub fn new() -> Self {
        Self::with_liveness_config(
            DEFAULT_TTL_MINUTES,
            DEFAULT_SUSPECT_SECS,
            DEFAULT_HARD_DELETE_SECS,
        )
    }

    /// Create a new block hash store with specified TTL in minutes
    pub fn with_ttl(ttl_minutes: u64) -> Self {
        Self::with_liveness_config(ttl_minutes, DEFAULT_SUSPECT_SECS, DEFAULT_HARD_DELETE_SECS)
    }

    /// Create a new block hash store with full configuration
    pub fn with_liveness_config(
        ttl_minutes: u64,
        suspect_secs: u64,
        hard_delete_secs: u64,
    ) -> Self {
        Self {
            map: DashMap::new(),
            ttl: Duration::from_secs(ttl_minutes * 60),
            nodes: DashMap::new(),
            suspect_threshold: Duration::from_secs(suspect_secs),
            hard_delete_threshold: Duration::from_secs(hard_delete_secs),
        }
    }

    /// Insert a list of block hashes from a given node.
    /// Re-inserting an existing (block, node) pair refreshes the TTL timestamp.
    /// Returns the number of entries processed.
    pub fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
        let node: Arc<str> = Arc::from(node);
        let now = Instant::now();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            self.map
                .entry(key)
                .or_default()
                .insert(Arc::clone(&node), now);
        }
        hashes.len()
    }

    /// Remove hashes owned by the given node (conditional delete).
    /// Only removes the requesting node's ownership; other nodes' entries are untouched.
    /// Returns the number of removed entries.
    pub fn remove_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
        let mut removed = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let should_remove_key = if let Some(mut nodes) = self.map.get_mut(&key) {
                if nodes.remove(node).is_some() {
                    removed += 1;
                }
                nodes.is_empty()
            } else {
                false
            };
            if should_remove_key {
                // Only remove if still empty (another thread may have inserted)
                self.map.remove_if(&key, |_, nodes| nodes.is_empty());
            }
        }
        removed
    }

    /// Query the longest prefix of `hashes` that exists in the store.
    /// Stops at the first hash not found on any healthy node.
    /// Suspect nodes are filtered out of results.
    pub fn query_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<PrefixEntry> {
        let mut result = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if let Some(owners) = self.map.get(&key) {
                let healthy_nodes: Vec<Arc<str>> = owners
                    .keys()
                    .filter(|n| self.is_node_healthy(n))
                    .cloned()
                    .collect();
                if healthy_nodes.is_empty() {
                    break;
                }
                result.push(PrefixEntry {
                    block_hash: hash.clone(),
                    nodes: healthy_nodes,
                });
            } else {
                break;
            }
        }
        result
    }

    /// Process a heartbeat from a node.
    /// - New node: register with current timestamp
    /// - Same epoch: refresh last_seen
    /// - Different epoch: purge old entries, re-register
    pub fn heartbeat(&self, node: &str, epoch: &str) {
        let node_key: Arc<str> = Arc::from(node);
        let now = Instant::now();

        if let Some(mut entry) = self.nodes.get_mut(&node_key) {
            if entry.epoch == epoch {
                entry.last_seen = now;
            } else {
                let old_epoch = entry.epoch.clone();
                drop(entry);
                let purged = self.purge_node(node);
                info!(
                    "Node {} epoch changed ({} -> {}), purged {} entries",
                    node, old_epoch, epoch, purged
                );
                self.nodes.insert(
                    node_key,
                    NodeLiveness {
                        epoch: epoch.to_string(),
                        last_seen: now,
                    },
                );
            }
        } else {
            self.nodes.insert(
                node_key,
                NodeLiveness {
                    epoch: epoch.to_string(),
                    last_seen: now,
                },
            );
        }
    }

    /// Process a Bye from a node. Purges all entries and removes liveness tracking.
    /// Only acts if the epoch matches (prevents stale Bye from old process).
    pub fn bye(&self, node: &str, epoch: &str) -> usize {
        let node_key: Arc<str> = Arc::from(node);
        if let Some(entry) = self.nodes.get(&node_key) {
            if entry.epoch != epoch {
                return 0;
            }
            drop(entry);
            let purged = self.purge_node(node);
            self.nodes.remove(&node_key);
            purged
        } else {
            0
        }
    }

    /// Remove all block entries owned by a specific node.
    /// Returns the number of block entries removed.
    pub fn purge_node(&self, node: &str) -> usize {
        let mut purged = 0;
        self.map.retain(|_, owners| {
            if owners.remove(node).is_some() {
                purged += 1;
            }
            !owners.is_empty()
        });
        purged
    }

    /// Sweep expired entries and dead nodes.
    ///
    /// 1. Removes per-node block registrations older than TTL.
    /// 2. Purges nodes past `hard_delete_threshold` and removes their block entries.
    ///
    /// Returns (expired_keys_removed, dead_nodes_purged).
    pub fn sweep_expired(&self) -> (usize, usize) {
        let now = Instant::now();
        let ttl = self.ttl;
        let before = self.map.len();
        self.map.retain(|_, nodes| {
            nodes.retain(|_, registered_at| now.duration_since(*registered_at) < ttl);
            !nodes.is_empty()
        });
        let mut expired_keys = before.saturating_sub(self.map.len());

        // Purge dead nodes (past hard_delete_threshold)
        let mut purged_nodes: Vec<Arc<str>> = Vec::new();
        for entry in &self.nodes {
            if entry.last_seen.elapsed() >= self.hard_delete_threshold {
                purged_nodes.push(entry.key().clone());
            }
        }

        // Re-validate with remove_if to avoid TOCTOU with concurrent heartbeats
        purged_nodes.retain(|node_key| {
            self.nodes
                .remove_if(node_key, |_, liveness| {
                    liveness.last_seen.elapsed() >= self.hard_delete_threshold
                })
                .is_some()
        });

        if !purged_nodes.is_empty() {
            let before_purge = self.map.len();
            self.map.retain(|_, owners| {
                for node in &purged_nodes {
                    owners.remove(node.as_ref());
                }
                !owners.is_empty()
            });
            let purge_removed = before_purge.saturating_sub(self.map.len());
            expired_keys += purge_removed;
            for node_key in &purged_nodes {
                info!("Hard-deleted node {}", node_key);
            }
            if purge_removed > 0 {
                info!(
                    "Sweep: purged {} dead nodes, {} block entries",
                    purged_nodes.len(),
                    purge_removed
                );
            }
        }

        (expired_keys, purged_nodes.len())
    }

    /// Get the number of unique block keys
    pub fn entry_count(&self) -> u64 {
        self.map.len() as u64
    }

    /// Get the number of tracked nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn is_node_healthy(&self, node: &Arc<str>) -> bool {
        match self.nodes.get(node) {
            Some(entry) => entry.last_seen.elapsed() < self.suspect_threshold,
            None => false,
        }
    }

    /// Clear all entries (for testing or maintenance)
    #[allow(dead_code)]
    pub fn invalidate_all(&self) {
        self.map.clear();
        self.nodes.clear();
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
    fn test_insert_and_query() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let node = "10.0.0.1:50055";

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        // Insert hashes
        let inserted = store.insert_hashes(namespace, &hashes, node);
        assert_eq!(inserted, 3);

        // Query existing hashes
        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 3);
        for entry in &existing {
            assert_eq!(entry.nodes.len(), 1);
            assert_eq!(entry.nodes[0].as_ref(), node);
        }

        // Query with gap — prefix stops at first miss
        let mixed_hashes = vec![
            vec![1, 2, 3, 4],     // exists
            vec![99, 99, 99, 99], // doesn't exist — prefix stops here
            vec![5, 6, 7, 8],     // exists but unreachable
        ];
        let existing = store.query_prefix(namespace, &mixed_hashes);
        assert_eq!(existing.len(), 1);

        // Query with different namespace
        let existing = store.query_prefix("other-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_multi_owner() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        store.insert_hashes(namespace, std::slice::from_ref(&hash), "node-a:50055");
        store.insert_hashes(namespace, std::slice::from_ref(&hash), "node-b:50055");

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

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_prefix("any-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_remove_own_blocks() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        store.insert_hashes(namespace, std::slice::from_ref(&hash), "node-a");

        // owner matches → should remove
        let removed = store.remove_hashes(namespace, std::slice::from_ref(&hash), "node-a");
        assert_eq!(removed, 1);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 0);
    }

    #[test]
    fn test_remove_other_nodes_block_is_noop() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        store.insert_hashes(namespace, std::slice::from_ref(&hash), "node-b");

        // owner does not match → should not remove
        let removed = store.remove_hashes(namespace, std::slice::from_ref(&hash), "node-a");
        assert_eq!(removed, 0);
        assert_eq!(store.query_prefix(namespace, &[hash]).len(), 1);
    }

    #[test]
    fn test_remove_one_owner_keeps_others() {
        let store = BlockHashStore::new();
        let hash = vec![1, 2, 3];

        store.insert_hashes("ns", std::slice::from_ref(&hash), "node-a");
        store.insert_hashes("ns", std::slice::from_ref(&hash), "node-b");

        let removed = store.remove_hashes("ns", std::slice::from_ref(&hash), "node-a");
        assert_eq!(removed, 1);

        let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
        assert_eq!(existing.len(), 1);
        assert_eq!(existing[0].nodes.len(), 1);
        assert_eq!(existing[0].nodes[0].as_ref(), "node-b");
    }

    #[test]
    fn test_remove_nonexistent_is_noop() {
        let store = BlockHashStore::new();
        let removed = store.remove_hashes("ns", &[vec![9, 9, 9]], "node-a");
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_sweep_expired() {
        // TTL = 0 minutes → everything expires immediately
        let store = BlockHashStore::with_ttl(0);
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");
        assert_eq!(store.entry_count(), 2);

        // Instant::now() is already past the zero-TTL deadline
        let (removed, _) = store.sweep_expired();
        assert_eq!(removed, 2);
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn test_sweep_keeps_fresh_entries() {
        let store = BlockHashStore::with_ttl(120); // 120 min TTL
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");

        let (removed, _) = store.sweep_expired();
        assert_eq!(removed, 0);
        assert_eq!(store.entry_count(), 2);
    }

    #[test]
    fn test_concurrent_insert_and_remove() {
        use std::sync::Arc;

        let store = Arc::new(BlockHashStore::new());
        let hash = vec![1, 2, 3, 4];

        // Run a quick synchronous stress test
        for _ in 0..100 {
            store.insert_hashes("ns", std::slice::from_ref(&hash), "node-a");

            // Simulate concurrent ops from different threads
            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);
            let hash_a = hash.clone();
            let hash_b = hash.clone();

            std::thread::scope(|s| {
                s.spawn(|| {
                    store_a.remove_hashes("ns", &[hash_a], "node-a");
                });
                s.spawn(|| {
                    store_b.insert_hashes("ns", &[hash_b], "node-b");
                });
            });

            // node-b always inserts; node-a's remove only affects node-a's entry.
            // So node-b must always be present.
            let existing = store.query_prefix("ns", std::slice::from_ref(&hash));
            assert_eq!(existing.len(), 1, "key must exist after concurrent ops");
            assert!(
                existing[0].nodes.iter().any(|n| n.as_ref() == "node-b"),
                "node-b must be present"
            );

            // Clean up for next iteration
            store.invalidate_all();
        }
    }

    #[test]
    fn test_invalidate_all() {
        let store = BlockHashStore::new();
        let namespace = "model-test";
        let node = "10.0.0.1:50055";

        let hashes = vec![vec![1, 2, 3], vec![4, 5, 6]];
        store.insert_hashes(namespace, &hashes, node);

        // Verify entries exist
        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 2);

        // Clear all
        store.invalidate_all();

        // Verify entries are gone
        let existing = store.query_prefix(namespace, &hashes);
        assert_eq!(existing.len(), 0);
        assert_eq!(store.entry_count(), 0);
    }

    // --- Heartbeat tests ---

    #[test]
    fn test_heartbeat_new_node() {
        let store = BlockHashStore::new();
        assert_eq!(store.node_count(), 0);

        store.heartbeat("node-a", "epoch-1");
        assert_eq!(store.node_count(), 1);

        let entry = store.nodes.get(&Arc::<str>::from("node-a")).unwrap();
        assert_eq!(entry.epoch, "epoch-1");
    }

    #[test]
    fn test_heartbeat_refresh() {
        let store = BlockHashStore::new();
        store.heartbeat("node-a", "epoch-1");

        let first_seen = store
            .nodes
            .get(&Arc::<str>::from("node-a"))
            .unwrap()
            .last_seen;

        std::thread::sleep(Duration::from_millis(10));
        store.heartbeat("node-a", "epoch-1");

        let second_seen = store
            .nodes
            .get(&Arc::<str>::from("node-a"))
            .unwrap()
            .last_seen;
        assert!(second_seen > first_seen);
    }

    #[test]
    fn test_heartbeat_recovers_stale_node() {
        let store = BlockHashStore {
            suspect_threshold: Duration::from_millis(50),
            ..BlockHashStore::with_liveness_config(120, 1, 3600)
        };
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1]], "node-a");

        // Wait until node-a becomes stale
        std::thread::sleep(Duration::from_millis(60));
        let result = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(result.len(), 0);

        // Heartbeat refreshes last_seen → node visible again
        store.heartbeat("node-a", "epoch-1");
        let result = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].nodes[0].as_ref(), "node-a");
    }

    #[test]
    fn test_heartbeat_new_epoch_purges() {
        let store = BlockHashStore::new();
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");
        assert_eq!(store.entry_count(), 2);

        // Heartbeat with new epoch purges old entries
        store.heartbeat("node-a", "epoch-2");
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.node_count(), 1);

        let entry = store.nodes.get(&Arc::<str>::from("node-a")).unwrap();
        assert_eq!(entry.epoch, "epoch-2");
    }

    // --- Bye tests ---

    #[test]
    fn test_bye_matching_epoch() {
        let store = BlockHashStore::new();
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");

        let purged = store.bye("node-a", "epoch-1");
        assert_eq!(purged, 2);
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn test_bye_mismatched_epoch_is_noop() {
        let store = BlockHashStore::new();
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1]], "node-a");

        let purged = store.bye("node-a", "epoch-wrong");
        assert_eq!(purged, 0);
        assert_eq!(store.entry_count(), 1);
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn test_bye_unknown_node_is_noop() {
        let store = BlockHashStore::new();
        let purged = store.bye("node-unknown", "epoch-1");
        assert_eq!(purged, 0);
    }

    // --- Sweep + dead node purge tests ---

    #[test]
    fn test_sweep_expired_purges_dead_nodes() {
        // hard_delete=0 → immediately dead
        let store = BlockHashStore::with_liveness_config(120, 30, 0);
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");

        let (_, purged_nodes) = store.sweep_expired();
        assert_eq!(purged_nodes, 1);
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn test_sweep_expired_keeps_fresh_nodes() {
        // hard_delete=3600 → node survives
        let store = BlockHashStore::with_liveness_config(120, 30, 3600);
        store.heartbeat("node-a", "epoch-1");
        store.insert_hashes("ns", &[vec![1]], "node-a");

        let (_, purged_nodes) = store.sweep_expired();
        assert_eq!(purged_nodes, 0);
        assert_eq!(store.node_count(), 1);
        assert_eq!(store.entry_count(), 1);
    }

    // --- Query filtering tests ---

    #[test]
    fn test_query_filters_stale_nodes() {
        let store = BlockHashStore {
            suspect_threshold: Duration::from_millis(50),
            ..BlockHashStore::with_liveness_config(120, 1, 3600)
        };
        store.heartbeat("node-a", "epoch-a");
        store.heartbeat("node-b", "epoch-b");
        store.insert_hashes("ns", &[vec![1]], "node-a");
        store.insert_hashes("ns", &[vec![1]], "node-b");

        // Wait until both are stale, then refresh only node-b
        std::thread::sleep(Duration::from_millis(60));
        store.heartbeat("node-b", "epoch-b");

        let result = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].nodes.len(), 1);
        assert_eq!(result[0].nodes[0].as_ref(), "node-b");
    }

    #[test]
    fn test_query_stale_only_owner_breaks_prefix() {
        // suspect_threshold=0 → stale immediately
        let store = BlockHashStore::with_liveness_config(120, 0, 3600);
        store.heartbeat("node-a", "epoch-a");
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");

        // node-a is stale → prefix is empty
        let result = store.query_prefix("ns", &[vec![1], vec![2]]);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_query_mixed_healthy_stale() {
        let store = BlockHashStore {
            suspect_threshold: Duration::from_millis(50),
            ..BlockHashStore::with_liveness_config(120, 1, 3600)
        };

        store.heartbeat("node-a", "epoch-a");
        store.heartbeat("node-b", "epoch-b");

        // h1 owned by both, h2 owned only by node-a
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");
        store.insert_hashes("ns", &[vec![1]], "node-b");

        // Wait until both stale, then refresh only node-b
        std::thread::sleep(Duration::from_millis(60));
        store.heartbeat("node-b", "epoch-b");

        // h1: node-a(stale) + node-b(fresh) → node-b visible
        // h2: node-a(stale) only → prefix stops
        let result = store.query_prefix("ns", &[vec![1], vec![2]]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].nodes.len(), 1);
        assert_eq!(result[0].nodes[0].as_ref(), "node-b");
    }

    // --- purge_node tests ---

    #[test]
    fn test_purge_removes_from_all_blocks() {
        let store = BlockHashStore::new();
        store.insert_hashes("ns", &[vec![1], vec![2], vec![3]], "node-a");

        let purged = store.purge_node("node-a");
        assert_eq!(purged, 3);
    }

    #[test]
    fn test_purge_drops_empty_keys() {
        let store = BlockHashStore::new();
        store.insert_hashes("ns", &[vec![1]], "node-a");

        store.purge_node("node-a");
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn test_purge_keeps_other_owners() {
        let store = BlockHashStore::new();
        store.insert_hashes("ns", &[vec![1]], "node-a");
        store.insert_hashes("ns", &[vec![1]], "node-b");

        store.purge_node("node-a");
        assert_eq!(store.entry_count(), 1);

        let result = store.query_prefix("ns", &[vec![1]]);
        assert_eq!(result[0].nodes.len(), 1);
        assert_eq!(result[0].nodes[0].as_ref(), "node-b");
    }
}
