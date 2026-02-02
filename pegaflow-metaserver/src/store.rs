use dashmap::DashMap;
use std::sync::Arc;

/// A block key consisting of namespace and hash
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    pub namespace: String,
    pub hash: Vec<u8>,
}

impl BlockKey {
    pub fn new(namespace: String, hash: Vec<u8>) -> Self {
        Self { namespace, hash }
    }
}

/// Thread-safe block hash storage using DashMap
/// Stores BlockKeys (namespace + hash) from all multi-node pegaflow instances
pub struct BlockHashStore {
    /// Set of BlockKeys. DashMap with unit value acts as a concurrent hash set
    /// DashMap provides concurrent read/write access without explicit locking
    store: Arc<DashMap<BlockKey, ()>>,
}

impl BlockHashStore {
    /// Create a new empty block hash store
    pub fn new() -> Self {
        Self {
            store: Arc::new(DashMap::new()),
        }
    }

    /// Insert a list of block hashes
    /// Returns the number of newly inserted keys
    pub fn insert_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> usize {
        let mut inserted = 0;
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            // Insert the key (with unit value)
            self.store.insert(key, ());
            inserted += 1;
        }
        inserted
    }

    /// Query which hashes exist in the store
    /// Returns a vector of hashes that exist
    pub fn query_hashes(&self, namespace: &str, hashes: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let mut existing = Vec::new();
        for hash in hashes {
            let key = BlockKey::new(namespace.to_string(), hash.clone());
            if self.store.contains_key(&key) {
                existing.push(hash.clone());
            }
        }
        existing
    }

    /// Get the total number of stored block keys
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clear all entries (for testing or maintenance)
    #[allow(dead_code)]
    pub fn clear(&self) {
        self.store.clear();
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

        let hashes = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 10, 11, 12]];

        // Insert hashes
        let inserted = store.insert_hashes(namespace, &hashes);
        assert_eq!(inserted, 3);
        assert_eq!(store.len(), 3);

        // Query existing hashes
        let existing = store.query_hashes(namespace, &hashes);
        assert_eq!(existing.len(), 3);

        // Query with mix of existing and non-existing
        let mixed_hashes = vec![
            vec![1, 2, 3, 4],     // exists
            vec![99, 99, 99, 99], // doesn't exist
            vec![5, 6, 7, 8],     // exists
        ];
        let existing = store.query_hashes(namespace, &mixed_hashes);
        assert_eq!(existing.len(), 2);

        // Query with different namespace
        let existing = store.query_hashes("other-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }

    #[test]
    fn test_empty_store() {
        let store = BlockHashStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        let hashes = vec![vec![1, 2, 3]];
        let existing = store.query_hashes("any-namespace", &hashes);
        assert_eq!(existing.len(), 0);
    }
}
