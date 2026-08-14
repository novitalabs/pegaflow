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

/// Encode a raw content hash with a hybrid-cache group id.
///
/// Group 0 keeps the raw hash byte-for-byte so existing single-group caches
/// (and every current connector) stay bit-identical. Groups >= 1 append the
/// big-endian group id, which cannot collide with a raw content hash because
/// every real hash family is fixed-length.
pub fn group_hash(hash: &[u8], group_id: u32) -> Vec<u8> {
    if group_id == 0 {
        return hash.to_vec();
    }
    let mut encoded = Vec::with_capacity(hash.len() + 4);
    encoded.extend_from_slice(hash);
    encoded.extend_from_slice(&group_id.to_be_bytes());
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_zero_hash_is_bit_identical_to_raw() {
        // Backward-compat contract: existing single-group deployments must not
        // observe any key change.
        let hash = b"sha256-content-hash";
        assert_eq!(group_hash(hash, 0), hash.to_vec());
        assert_eq!(group_hash(b"", 0), Vec::<u8>::new());
    }

    #[test]
    fn nonzero_group_appends_big_endian_group_id() {
        let hash = [0xAA, 0xBB];
        assert_eq!(group_hash(&hash, 1), vec![0xAA, 0xBB, 0, 0, 0, 1]);
        assert_eq!(group_hash(&hash, 0x01020304), vec![0xAA, 0xBB, 1, 2, 3, 4]);
    }

    #[test]
    fn distinct_groups_do_not_share_keys() {
        // The same content hash in two groups must be two different keys, and
        // an encoded key must never equal a raw hash of any length (length
        // differs, so this holds even across hash families).
        let hash = [1, 2, 3, 4];
        assert_ne!(group_hash(&hash, 0), group_hash(&hash, 1));
        assert_ne!(group_hash(&hash, 1), group_hash(&hash, 2));
        assert_ne!(group_hash(&hash, 1).len(), hash.len());
    }
}
