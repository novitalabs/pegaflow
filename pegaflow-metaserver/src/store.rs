use dashmap::DashMap;
use pegaflow_common::BlockKey;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Default TTL for cache entries (120 minutes)
pub const DEFAULT_TTL_MINUTES: u64 = 120;

/// A prefix query result: one block hash and all nodes that own it.
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub block_hash: Vec<u8>,
    pub nodes: Vec<Arc<str>>,
}

/// Async thread-safe block hash storage using DashMap.
/// Stores BlockKeys (namespace + hash) mapped to a set of owning node URLs with
/// registration timestamps, enabling multi-owner tracking and periodic TTL sweep.
pub struct BlockHashStore {
    /// Key: BlockKey, Value: { node_url → registration_time }
    map: DashMap<BlockKey, HashMap<Arc<str>, Instant>>,
    ttl: Duration,
}

impl BlockHashStore {
    /// Create a new block hash store with default TTL (120 minutes)
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_TTL_MINUTES)
    }

    /// Create a new block hash store with specified TTL in minutes
    pub fn with_ttl(ttl_minutes: u64) -> Self {
        Self {
            map: DashMap::new(),
            ttl: Duration::from_secs(ttl_minutes * 60),
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
    /// Stops at the first hash not found on any node.
    pub fn query_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<PrefixEntry> {
        let mut result = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if let Some(nodes) = self.map.get(&key) {
                if nodes.is_empty() {
                    break;
                }
                result.push(PrefixEntry {
                    block_hash: hash.clone(),
                    nodes: nodes.keys().cloned().collect(),
                });
            } else {
                break;
            }
        }
        result
    }

    /// Sweep expired entries. Removes per-node registrations older than TTL,
    /// then drops block keys with no remaining owners.
    /// Called periodically from a background task.
    /// Returns the number of block keys fully removed.
    pub fn sweep_expired(&self) -> usize {
        let now = Instant::now();
        let ttl = self.ttl;
        let before = self.map.len();
        self.map.retain(|_, nodes| {
            nodes.retain(|_, registered_at| now.duration_since(*registered_at) < ttl);
            !nodes.is_empty()
        });
        before.saturating_sub(self.map.len())
    }

    /// Get the number of unique block keys
    pub fn entry_count(&self) -> u64 {
        self.map.len() as u64
    }

    /// Clear all entries (for testing or maintenance)
    #[allow(dead_code)]
    pub fn invalidate_all(&self) {
        self.map.clear();
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
        let removed = store.sweep_expired();
        assert_eq!(removed, 2);
        assert_eq!(store.entry_count(), 0);
    }

    #[test]
    fn test_sweep_keeps_fresh_entries() {
        let store = BlockHashStore::with_ttl(120); // 120 min TTL
        store.insert_hashes("ns", &[vec![1], vec![2]], "node-a");

        let removed = store.sweep_expired();
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
}
