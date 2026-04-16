/// Key for identifying blocks in storage, including namespace for model isolation.
///
/// NOTE: Using String for namespace is simple but adds ~20-50 bytes overhead per key.
/// Future optimization: intern namespaces to u32 IDs (saves memory, faster comparison).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// Namespace for model isolation (e.g., model name, or empty string for shared storage)
    pub namespace: String,
    /// Block content hash
    pub hash: Vec<u8>,
}

impl BlockKey {
    pub fn new(namespace: String, hash: Vec<u8>) -> Self {
        Self { namespace, hash }
    }

    /// Estimate the memory size of this BlockKey in bytes
    /// Used for cache size-aware eviction policies
    pub fn estimated_size(&self) -> u64 {
        // Size = namespace string capacity + hash vec capacity + struct overhead (48 bytes)
        // Using capacity() instead of len() to account for actual heap-allocated memory
        (self.namespace.capacity() + self.hash.capacity() + 48) as u64
    }
}
