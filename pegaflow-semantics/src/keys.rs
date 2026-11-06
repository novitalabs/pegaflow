//! Storage keys.

/// Store key of logical block `block_hash` in cache group `group`.
///
/// Group 0 keeps the raw hash byte-for-byte; other groups append the
/// big-endian group id. Real hash families are fixed-length, so an encoded
/// key never equals a raw hash and two groups never share a key.
pub fn key(block_hash: &[u8], group: u32) -> Vec<u8> {
    if group == 0 {
        return block_hash.to_vec();
    }
    let mut encoded = Vec::with_capacity(block_hash.len() + 4);
    encoded.extend_from_slice(block_hash);
    encoded.extend_from_slice(&group.to_be_bytes());
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_zero_is_raw_hash() {
        assert_eq!(key(b"abc", 0), b"abc".to_vec());
    }

    #[test]
    fn other_groups_append_big_endian_id() {
        assert_eq!(key(&[0xAA], 0x0102_0304), vec![0xAA, 1, 2, 3, 4]);
        assert_ne!(key(&[1, 2], 1), key(&[1, 2], 2));
    }
}
