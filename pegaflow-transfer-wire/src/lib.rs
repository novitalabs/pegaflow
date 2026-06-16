//! Wire schema for the serving-side `QueryBlocksForTransfer` response body.
//!
//! The gRPC response carries `transfer_plan` as postcard-encoded bytes of
//! [`TransferPlan`]. Layout is expressed with structs only: contiguous vs
//! split KV is `SlotSchema.segments.len()` (1 or 2).

use std::fmt;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Schema violation found while parsing or validating a transfer plan.
#[derive(Debug)]
pub struct WireError(String);

impl WireError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transfer-wire: {}", self.0)
    }
}

impl std::error::Error for WireError {}

/// One slot (layer): shared segment shape across every block in the plan.
///
/// NUMA placement is expressed per [`RemoteChunk`], not per slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSchema {
    pub segments: SmallVec<[SegmentSchema; 2]>,
}

/// Byte shape of one K or V segment within a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentSchema {
    pub bytes: usize,
    pub block_stride: usize,
}

/// One coalesced remote pinned-memory interval; requester issues one RDMA READ each.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteChunk {
    pub base_ptr: u64,
    pub length: u64,
    pub numa_node: u32,
}

/// Maps a slice inside a [`RemoteChunk`] (after NUMA slab packing) back to blocks.
///
/// `block_count >= 1`. Stride for `block_count >= 2` comes from
/// [`SlotSchema::segments`]; singles use `block_start` only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentPlacement {
    pub slot_idx: u32,
    pub segment_idx: u32,
    pub chunk_idx: u32,
    pub offset_in_chunk: usize,
    pub block_start: u32,
    pub block_count: u32,
}

/// Compact transfer geometry for a batch of provider-side sealed blocks.
///
/// The requester allocates one pinned slab per NUMA node, RDMA-reads each
/// [`RemoteChunk`] into its slab, then rebuilds blocks from [`SegmentPlacement`]
/// entries. Block order is [`Self::block_hashes`]; `block_start` indexes into it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferPlan {
    /// Per-slot (layer) layout shared by every block in the batch.
    pub slot_schemas: Vec<SlotSchema>,
    /// Remote intervals to fetch; one RDMA READ per entry.
    pub remote_chunks: Vec<RemoteChunk>,
    /// How fetched bytes map back to block slots/segments.
    pub placements: Vec<SegmentPlacement>,
    /// Content hashes in transfer order.
    pub block_hashes: Vec<Vec<u8>>,
}

impl TransferPlan {
    pub fn encode_to_vec(&self) -> Result<Vec<u8>, WireError> {
        postcard::to_allocvec(self).map_err(|e| WireError::new(format!("encode failed: {e}")))
    }

    pub fn decode_from_slice(bytes: &[u8]) -> Result<Self, WireError> {
        let plan: Self = postcard::from_bytes(bytes)
            .map_err(|e| WireError::new(format!("decode failed: {e}")))?;
        plan.validate()?;
        Ok(plan)
    }

    pub fn validate(&self) -> Result<(), WireError> {
        if self.block_hashes.is_empty() {
            if !self.slot_schemas.is_empty()
                || !self.remote_chunks.is_empty()
                || !self.placements.is_empty()
            {
                return Err(WireError::new("empty block_hashes with non-empty geometry"));
            }
            return Ok(());
        }

        if self.slot_schemas.is_empty() {
            return Err(WireError::new(
                "block_hashes present but slot_schemas empty",
            ));
        }

        let segment_count = self.slot_schemas[0].segments.len();
        if segment_count == 0 || segment_count > 2 {
            return Err(WireError::new(format!(
                "slot 0 has invalid segment count {segment_count}"
            )));
        }

        for (slot_idx, slot) in self.slot_schemas.iter().enumerate() {
            if slot.segments.len() != segment_count {
                return Err(WireError::new(format!(
                    "slot {slot_idx} segment count {} != {segment_count}",
                    slot.segments.len()
                )));
            }
            for (seg_idx, seg) in slot.segments.iter().enumerate() {
                if seg.bytes == 0 {
                    return Err(WireError::new(format!(
                        "slot {slot_idx} segment {seg_idx} bytes is zero"
                    )));
                }
                if seg.block_stride < seg.bytes {
                    return Err(WireError::new(format!(
                        "slot {slot_idx} segment {seg_idx} stride {} < bytes {}",
                        seg.block_stride, seg.bytes
                    )));
                }
            }
        }

        let block_count = u32::try_from(self.block_hashes.len()).map_err(|_| {
            WireError::new(format!(
                "block_hashes length {} exceeds u32::MAX",
                self.block_hashes.len()
            ))
        })?;

        for chunk in &self.remote_chunks {
            if chunk.length == 0 {
                return Err(WireError::new("remote chunk length is zero"));
            }
        }

        for placement in &self.placements {
            Self::check_slot_segment(placement.slot_idx, placement.segment_idx, segment_count)?;
            Self::check_chunk_idx(placement.chunk_idx, self.remote_chunks.len())?;
            if placement.block_count == 0 {
                return Err(WireError::new(format!(
                    "placement slot={} segment={} block_count is zero",
                    placement.slot_idx, placement.segment_idx
                )));
            }
            if placement.block_start >= block_count {
                return Err(WireError::new("placement block_start out of bounds"));
            }
            let end = placement
                .block_start
                .checked_add(placement.block_count)
                .ok_or_else(|| WireError::new("placement block index overflow"))?;
            if end > block_count {
                return Err(WireError::new("placement extends past block_hashes"));
            }
            let seg = &self.slot_schemas[placement.slot_idx as usize].segments
                [placement.segment_idx as usize];
            let span = if placement.block_count > 1 {
                (u64::from(placement.block_count - 1) * seg.block_stride as u64) + seg.bytes as u64
            } else {
                seg.bytes as u64
            };
            let end_offset = placement.offset_in_chunk as u64 + span;
            if end_offset > self.remote_chunks[placement.chunk_idx as usize].length {
                return Err(WireError::new("placement extends past remote chunk length"));
            }
        }

        Ok(())
    }

    fn check_slot_segment(
        slot_idx: u32,
        segment_idx: u32,
        segment_count: usize,
    ) -> Result<(), WireError> {
        if segment_idx as usize >= segment_count {
            return Err(WireError::new(format!(
                "segment_idx {segment_idx} >= segment_count {segment_count}"
            )));
        }
        let _ = slot_idx;
        Ok(())
    }

    fn check_chunk_idx(chunk_idx: u32, chunk_count: usize) -> Result<(), WireError> {
        if chunk_idx as usize >= chunk_count {
            return Err(WireError::new(format!(
                "chunk_idx {chunk_idx} >= chunk_count {chunk_count}"
            )));
        }
        Ok(())
    }

    /// Total payload bytes referenced by all remote chunks.
    pub fn total_remote_bytes(&self) -> u64 {
        self.remote_chunks.iter().map(|c| c.length).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use smallvec::smallvec;

    fn sample_plan() -> TransferPlan {
        TransferPlan {
            slot_schemas: vec![SlotSchema {
                segments: smallvec![SegmentSchema {
                    bytes: 64,
                    block_stride: 128,
                }],
            }],
            remote_chunks: vec![RemoteChunk {
                base_ptr: 0x5000,
                length: 256,
                numa_node: 0,
            }],
            placements: vec![SegmentPlacement {
                slot_idx: 0,
                segment_idx: 0,
                chunk_idx: 0,
                offset_in_chunk: 0,
                block_start: 0,
                block_count: 2,
            }],
            block_hashes: vec![vec![1], vec![2]],
        }
    }

    #[test]
    fn postcard_round_trip() {
        let plan = sample_plan();
        let bytes = plan.encode_to_vec().expect("encode");
        let decoded = TransferPlan::decode_from_slice(&bytes).expect("decode");
        assert_eq!(plan, decoded);
    }

    #[test]
    fn rejects_placement_with_block_count_zero() {
        let mut plan = sample_plan();
        plan.placements[0].block_count = 0;
        assert!(plan.validate().is_err());
    }
}
