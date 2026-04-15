use moka::future::Cache;
use moka::policy::EvictionPolicy;
use pegaflow_core::BlockKey;
use std::sync::Arc;
use std::time::Duration;

/// Default max capacity for the cache (512 MB)
pub const DEFAULT_MAX_CAPACITY: u64 = 512 * 1024 * 1024;

/// Default TTL for cache entries (120 minutes)
pub const DEFAULT_TTL_MINUTES: u64 = 120;

/// A block hash with the pegaflow-server node that owns it.
#[derive(Debug, Clone)]
pub struct CrossNodeBlock {
    pub block_hash: Vec<u8>,
    pub node: Arc<str>,
}

/// Async thread-safe block hash storage using Moka cache
/// Stores BlockKeys (namespace + hash) mapped to owning node URL,
/// with LRU eviction, size-aware capacity management, and TTL.
pub struct BlockHashStore {
    /// Moka async cache with LRU eviction, size-aware capacity, and configurable TTL.
    /// Key: BlockKey, Value: node URL (the pegaflow-server that owns this block)
    cache: Arc<Cache<BlockKey, Arc<str>>>,
}

impl BlockHashStore {
    /// Create a new block hash store with default capacity (512 MB) and default TTL (120 minutes)
    pub fn new() -> Self {
        Self::with_capacity_and_ttl(DEFAULT_MAX_CAPACITY, DEFAULT_TTL_MINUTES)
    }

    /// Create a new block hash store with specified max capacity in bytes
    pub fn with_capacity(max_capacity_bytes: u64) -> Self {
        Self::with_capacity_and_ttl(max_capacity_bytes, DEFAULT_TTL_MINUTES)
    }

    /// Create a new block hash store with specified max capacity in bytes and TTL in minutes
    pub fn with_capacity_and_ttl(max_capacity_bytes: u64, ttl_minutes: u64) -> Self {
        let cache = Cache::builder()
            .eviction_policy(EvictionPolicy::lru())
            // Set max capacity based on estimated memory size
            .max_capacity(max_capacity_bytes)
            // Use weigher to estimate the size of each entry (key + node URL)
            .weigher(|key: &BlockKey, node: &Arc<str>| {
                (key.estimated_size() + node.len() as u64 + 16) as u32
            })
            // Set TTL
            .time_to_live(Duration::from_secs(ttl_minutes * 60))
            .build();

        Self {
            cache: Arc::new(cache),
        }
    }

    /// Insert a list of block hashes from a given node asynchronously.
    /// Returns the number of inserted keys.
    pub async fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
        let node: Arc<str> = Arc::from(node);
        let mut inserted = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            self.cache.insert(key, Arc::clone(&node)).await;
            inserted += 1;
        }
        inserted
    }

    /// Remove hashes owned by the given node (conditional delete).
    /// Only removes an entry if its current owner matches `node`, preventing
    /// races where node A evicts but node B has since re-registered the same block.
    /// Returns the number of removed entries.
    pub async fn remove_hashes(&self, namespace: &str, hashes: &[Vec<u8>], node: &str) -> usize {
        use moka::ops::compute::Op;
        let node: Arc<str> = Arc::from(node);
        let mut removed = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            let node_clone = Arc::clone(&node);
            let result = self
                .cache
                .entry(key)
                .and_compute_with(move |maybe_entry| async move {
                    match maybe_entry {
                        Some(entry) if entry.value().as_ref() == node_clone.as_ref() => Op::Remove,
                        _ => Op::Nop,
                    }
                })
                .await;
            if matches!(result, moka::ops::compute::CompResult::Removed(_)) {
                removed += 1;
            }
        }
        removed
    }

    /// Query the longest prefix of `hashes` that exists in the store.
    /// Stops at the first hash not found on any node.
    pub async fn query_prefix(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<CrossNodeBlock> {
        let mut existing = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if let Some(node) = self.cache.get(&key).await {
                existing.push(CrossNodeBlock {
                    block_hash: hash.clone(),
                    node,
                });
            } else {
                break;
            }
        }
        existing
    }

    /// Get the approximate entry count
    /// Note: This may not be exact due to concurrent operations
    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// Get the weighted size (estimated memory usage in bytes)
    pub fn weighted_size(&self) -> u64 {
        self.cache.weighted_size()
    }

    /// Perform cache maintenance operations
    /// This should be called periodically to ensure eviction happens
    pub async fn run_pending_tasks(&self) {
        self.cache.run_pending_tasks().await;
    }

    /// Clear all entries (for testing or maintenance)
    #[allow(dead_code)]
    pub async fn invalidate_all(&self) {
        self.cache.invalidate_all();
        // Wait for invalidation to complete
        self.cache.run_pending_tasks().await;
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

    #[tokio::test]
    async fn test_insert_and_query() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let node = "10.0.0.1:50055";

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        // Insert hashes
        let inserted = store.insert_hashes(namespace, &hashes, node).await;
        assert_eq!(inserted, 3);

        // Run pending tasks to ensure cache is updated
        store.run_pending_tasks().await;

        // Query existing hashes
        let existing = store.query_prefix(namespace, &hashes).await;
        assert_eq!(existing.len(), 3);
        // Verify node is returned
        for entry in &existing {
            assert_eq!(entry.node.as_ref(), node);
        }

        // Query with gap — prefix stops at first miss
        let mixed_hashes = vec![
            vec![1, 2, 3, 4],     // exists
            vec![99, 99, 99, 99], // doesn't exist — prefix stops here
            vec![5, 6, 7, 8],     // exists but unreachable
        ];
        let existing = store.query_prefix(namespace, &mixed_hashes).await;
        assert_eq!(existing.len(), 1);

        // Query with different namespace
        let existing = store.query_prefix("other-namespace", &hashes).await;
        assert_eq!(existing.len(), 0);
    }

    #[tokio::test]
    async fn test_node_overwrite() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        // Insert from node A
        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-a:50055")
            .await;
        store.run_pending_tasks().await;

        let existing = store
            .query_prefix(namespace, std::slice::from_ref(&hash))
            .await;
        assert_eq!(existing[0].node.as_ref(), "node-a:50055");

        // Insert same hash from node B (overwrites)
        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-b:50055")
            .await;
        store.run_pending_tasks().await;

        let existing = store
            .query_prefix(namespace, std::slice::from_ref(&hash))
            .await;
        assert_eq!(existing[0].node.as_ref(), "node-b:50055");
    }

    #[tokio::test]
    async fn test_empty_store() {
        let store = BlockHashStore::new();
        assert_eq!(store.entry_count(), 0);

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_prefix("any-namespace", &hashes).await;
        assert_eq!(existing.len(), 0);
    }

    #[tokio::test]
    async fn test_size_aware_eviction() {
        // Create a store with very small capacity (1 KB)
        let store = BlockHashStore::with_capacity(1024);

        let namespace = "test-namespace";
        let node = "10.0.0.1:50055";

        // Insert many hashes (should trigger eviction)
        let mut hashes = Vec::new();
        for i in 0..100 {
            hashes.push(vec![i as u8; 32]); // 32-byte hash
        }

        let inserted = store.insert_hashes(namespace, &hashes, node).await;
        assert_eq!(inserted, 100);

        // Run pending tasks to trigger eviction
        store.run_pending_tasks().await;

        // Due to size-aware eviction, not all entries should be present
        // The weighted size should be less than or equal to max capacity
        assert!(store.weighted_size() <= 1024);

        // Some entries should have been evicted (LRU)
        let existing = store.query_prefix(namespace, &hashes).await;
        assert!(existing.len() < 100, "Expected some entries to be evicted");
    }

    #[tokio::test]
    async fn test_remove_own_blocks() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-a")
            .await;
        store.run_pending_tasks().await;

        // owner matches → should remove
        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), "node-a")
            .await;
        assert_eq!(removed, 1);
        store.run_pending_tasks().await;
        assert_eq!(store.query_prefix(namespace, &[hash]).await.len(), 0);
    }

    #[tokio::test]
    async fn test_remove_other_nodes_block_is_noop() {
        let store = BlockHashStore::new();
        let namespace = "model-a";
        let hash = vec![1, 2, 3, 4];

        store
            .insert_hashes(namespace, std::slice::from_ref(&hash), "node-b")
            .await;
        store.run_pending_tasks().await;

        // owner does not match → should not remove
        let removed = store
            .remove_hashes(namespace, std::slice::from_ref(&hash), "node-a")
            .await;
        assert_eq!(removed, 0);
        store.run_pending_tasks().await;
        assert_eq!(store.query_prefix(namespace, &[hash]).await.len(), 1);
    }

    #[tokio::test]
    async fn test_remove_nonexistent_is_noop() {
        let store = BlockHashStore::new();

        let removed = store.remove_hashes("ns", &[vec![9, 9, 9]], "node-a").await;
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn test_concurrent_insert_and_remove() {
        let store = Arc::new(BlockHashStore::new());
        let hash = vec![1, 2, 3, 4];

        for _ in 0..100 {
            // Reset: node-a owns the block
            store
                .insert_hashes("ns", std::slice::from_ref(&hash), "node-a")
                .await;
            store.run_pending_tasks().await;

            let store_a = Arc::clone(&store);
            let store_b = Arc::clone(&store);
            let hash_a = hash.clone();
            let hash_b = hash.clone();

            tokio::join!(
                async move {
                    store_a.remove_hashes("ns", &[hash_a], "node-a").await;
                },
                async move {
                    store_b.insert_hashes("ns", &[hash_b], "node-b").await;
                },
            );

            store.run_pending_tasks().await;

            // Both operations completed. insert always runs to completion, so:
            // 1. remove first → node-b inserts after → key exists, owner=node-b
            // 2. insert first → remove sees owner=node-b, skips → key exists, owner=node-b
            let existing = store.query_prefix("ns", std::slice::from_ref(&hash)).await;
            assert_eq!(
                existing.len(),
                1,
                "key must exist after concurrent insert+remove"
            );
            assert_eq!(existing[0].node.as_ref(), "node-b", "owner must be node-b");
        }
    }

    #[tokio::test]
    async fn test_invalidate_all() {
        let store = BlockHashStore::new();
        let namespace = "model-test";
        let node = "10.0.0.1:50055";

        let hashes = vec![vec![1, 2, 3], vec![4, 5, 6]];
        store.insert_hashes(namespace, &hashes, node).await;
        store.run_pending_tasks().await;

        // Verify entries exist
        let existing = store.query_prefix(namespace, &hashes).await;
        assert_eq!(existing.len(), 2);

        // Clear all
        store.invalidate_all().await;

        // Verify entries are gone
        let existing = store.query_prefix(namespace, &hashes).await;
        assert_eq!(existing.len(), 0);
        assert_eq!(store.entry_count(), 0);
    }
}
