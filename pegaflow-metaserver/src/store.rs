use moka::future::Cache;
use pegaflow_core::BlockKey;
use std::sync::Arc;
use std::time::Duration;

/// Default max capacity for the cache (512 MB)
pub const DEFAULT_MAX_CAPACITY: u64 = 512 * 1024 * 1024;

/// Default TTL for cache entries (120 minutes)
pub const DEFAULT_TTL_MINUTES: u64 = 120;

/// Async thread-safe block hash storage using Moka cache
/// Stores BlockKeys (namespace + hash) with LRU eviction, size-aware capacity management, and TTL
pub struct BlockHashStore {
    /// Moka async cache with LRU eviction, size-aware capacity, and 120-minute TTL
    /// Key: BlockKey, Value: () (unit type, we only care about key existence)
    cache: Arc<Cache<BlockKey, ()>>,
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
            // Set max capacity based on estimated memory size
            .max_capacity(max_capacity_bytes)
            // Use weigher to estimate the size of each entry
            .weigher(|key: &BlockKey, _value: &()| key.estimated_size() as u32)
            // Set TTL
            .time_to_live(Duration::from_secs(ttl_minutes * 60))
            .build();

        Self {
            cache: Arc::new(cache),
        }
    }

    /// Insert a list of block hashes asynchronously
    /// Returns the number of inserted keys
    pub async fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> usize {
        let mut inserted = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            // Insert the key (with unit value) into the async cache
            self.cache.insert(key, ()).await;
            inserted += 1;
        }
        inserted
    }

    /// Query which hashes exist in the store asynchronously
    /// Returns a vector of hashes that exist
    pub async fn query_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut existing = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            // Check if the key exists in the cache
            if self.cache.get(&key).await.is_some() {
                existing.push(hash.clone());
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

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        // Insert hashes
        let inserted = store.insert_hashes(namespace, &hashes).await;
        assert_eq!(inserted, 3);

        // Run pending tasks to ensure cache is updated
        store.run_pending_tasks().await;

        // Query existing hashes
        let existing = store.query_hashes(namespace, &hashes).await;
        assert_eq!(existing.len(), 3);

        // Query with mix of existing and non-existing
        let mixed_hashes = vec![
            vec![1, 2, 3, 4],     // exists
            vec![99, 99, 99, 99], // doesn't exist
            vec![5, 6, 7, 8],     // exists
        ];
        let existing = store.query_hashes(namespace, &mixed_hashes).await;
        assert_eq!(existing.len(), 2);

        // Query with different namespace
        let existing = store.query_hashes("other-namespace", &hashes).await;
        assert_eq!(existing.len(), 0);
    }

    #[tokio::test]
    async fn test_empty_store() {
        let store = BlockHashStore::new();
        assert_eq!(store.entry_count(), 0);

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_hashes("any-namespace", &hashes).await;
        assert_eq!(existing.len(), 0);
    }

    #[tokio::test]
    async fn test_size_aware_eviction() {
        // Create a store with very small capacity (1 KB)
        let store = BlockHashStore::with_capacity(1024);

        let namespace = "test-namespace";

        // Insert many hashes (should trigger eviction)
        let mut hashes = Vec::new();
        for i in 0..100 {
            hashes.push(vec![i as u8; 32]); // 32-byte hash
        }

        let inserted = store.insert_hashes(namespace, &hashes).await;
        assert_eq!(inserted, 100);

        // Run pending tasks to trigger eviction
        store.run_pending_tasks().await;

        // Due to size-aware eviction, not all entries should be present
        // The weighted size should be less than or equal to max capacity
        assert!(store.weighted_size() <= 1024);

        // Some entries should have been evicted (LRU)
        let existing = store.query_hashes(namespace, &hashes).await;
        assert!(existing.len() < 100, "Expected some entries to be evicted");
    }

    #[tokio::test]
    async fn test_invalidate_all() {
        let store = BlockHashStore::new();
        let namespace = "model-test";

        let hashes = vec![vec![1, 2, 3], vec![4, 5, 6]];
        store.insert_hashes(namespace, &hashes).await;
        store.run_pending_tasks().await;

        // Verify entries exist
        let existing = store.query_hashes(namespace, &hashes).await;
        assert_eq!(existing.len(), 2);

        // Clear all
        store.invalidate_all().await;

        // Verify entries are gone
        let existing = store.query_hashes(namespace, &hashes).await;
        assert_eq!(existing.len(), 0);
        assert_eq!(store.entry_count(), 0);
    }
}
