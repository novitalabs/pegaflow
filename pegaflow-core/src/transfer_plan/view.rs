use std::sync::Arc;

use log::warn;
use pegaflow_common::NumaNode;

use crate::BlockKey;
use crate::block::SealedBlock;

use super::schema::TransferPlanError;

/// Per-segment pointer extracted from one block slot, used by chunk coalescing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentPoint {
    pub block_idx: u32,
    pub ptr: u64,
    pub len: usize,
    /// Holder-side NUMA for this block's slot; drives [`RemoteChunk::numa_node`].
    ///
    /// [`RemoteChunk::numa_node`]: pegaflow_transfer_wire::RemoteChunk::numa_node
    pub numa_node: u32,
}

/// Column-major view of a batch of sealed blocks: segment pointers and lengths
/// laid out as `[slot][segment][block]`, NUMA as `[slot][block]`.
///
/// Both encode passes (schema derivation and chunk coalescing) scan one
/// `(slot, segment)` at a time across all blocks, so this layout hands each pass
/// a contiguous block-ordered slice and replaces the per-cell `Vec` allocations
/// of a nested block/slot/segment tree with three flat buffers.
pub(crate) struct BlockMatrix {
    block_count: usize,
    slot_count: usize,
    segment_count: usize,
    hashes: Vec<Vec<u8>>,
    seg_ptr: Vec<u64>,
    seg_len: Vec<usize>,
    numa: Vec<u32>,
}

impl BlockMatrix {
    pub(crate) fn block_count(&self) -> usize {
        self.block_count
    }
    pub(crate) fn slot_count(&self) -> usize {
        self.slot_count
    }
    pub(crate) fn segment_count(&self) -> usize {
        self.segment_count
    }

    fn seg_base(&self, slot: usize, segment: usize) -> usize {
        (slot * self.segment_count + segment) * self.block_count
    }

    /// Block-ordered segment pointers for one `(slot, segment)` column.
    pub(crate) fn seg_ptrs(&self, slot: usize, segment: usize) -> &[u64] {
        let base = self.seg_base(slot, segment);
        &self.seg_ptr[base..base + self.block_count]
    }

    /// Block-ordered segment lengths for one `(slot, segment)` column.
    pub(crate) fn seg_lens(&self, slot: usize, segment: usize) -> &[usize] {
        let base = self.seg_base(slot, segment);
        &self.seg_len[base..base + self.block_count]
    }

    /// Block-ordered NUMA nodes for one slot.
    pub(crate) fn numas(&self, slot: usize) -> &[u32] {
        let base = slot * self.block_count;
        &self.numa[base..base + self.block_count]
    }

    pub(crate) fn into_hashes(self) -> Vec<Vec<u8>> {
        self.hashes
    }
}

/// Build the column-major matrix from found sealed blocks. Validates slot- and
/// segment-count consistency against block 0 up front, so the per-column encode
/// passes can index without re-checking shape.
pub(crate) fn block_matrix_from_found(
    found_blocks: &[(BlockKey, Arc<SealedBlock>)],
) -> Result<BlockMatrix, TransferPlanError> {
    let block_count = found_blocks.len();
    if block_count == 0 {
        return Ok(BlockMatrix {
            block_count: 0,
            slot_count: 0,
            segment_count: 0,
            hashes: Vec::new(),
            seg_ptr: Vec::new(),
            seg_len: Vec::new(),
            numa: Vec::new(),
        });
    }

    let first = found_blocks[0].1.slots();
    let slot_count = first.len();
    if slot_count == 0 {
        return Err(TransferPlanError::new("sealed block has no slots"));
    }
    let segment_count = first[0].num_segments();

    let mut seg_ptr = vec![0u64; block_count * slot_count * segment_count];
    let mut seg_len = vec![0usize; block_count * slot_count * segment_count];
    let mut numa = vec![NumaNode::UNKNOWN.0; block_count * slot_count];
    let mut hashes = Vec::with_capacity(block_count);

    for (block_idx, (key, block)) in found_blocks.iter().enumerate() {
        let slots = block.slots();
        if slots.len() != slot_count {
            return Err(TransferPlanError::new(format!(
                "block {block_idx} slot count {} != {slot_count} \
                 (namespace shared by incompatible KV layouts)",
                slots.len()
            )));
        }
        let slot_numas = block.slot_numas();
        for (slot_idx, raw) in slots.iter().enumerate() {
            if raw.num_segments() != segment_count {
                return Err(TransferPlanError::new(format!(
                    "block {block_idx} slot {slot_idx} segment count {} != {segment_count}",
                    raw.num_segments()
                )));
            }
            let slot_numa = slot_numas
                .get(slot_idx)
                .copied()
                .unwrap_or(NumaNode::UNKNOWN);
            if slot_numa.is_unknown() {
                warn!(
                    "transfer_plan: block {block_idx} slot {slot_idx} NUMA unknown (slot_numas len={})",
                    slot_numas.len()
                );
            }
            numa[slot_idx * block_count + block_idx] = slot_numa.0;

            // One enum-match + tight loop over this slot's inline segment array,
            // rather than re-dispatching `segment_ptr`/`segment_size` per segment.
            for (seg, (ptr, len)) in raw.segment_iovecs().enumerate() {
                if len == 0 {
                    return Err(TransferPlanError::new(format!(
                        "block {block_idx} slot {slot_idx} segment {seg} size is zero"
                    )));
                }
                let idx = (slot_idx * segment_count + seg) * block_count + block_idx;
                seg_ptr[idx] = ptr.as_ptr() as u64;
                seg_len[idx] = len;
            }
        }
        hashes.push(key.hash.clone());
    }

    Ok(BlockMatrix {
        block_count,
        slot_count,
        segment_count,
        hashes,
        seg_ptr,
        seg_len,
        numa,
    })
}

#[cfg(test)]
impl BlockMatrix {
    /// Construct a matrix directly from column-major arrays for unit tests.
    pub(crate) fn from_columns(
        block_count: usize,
        slot_count: usize,
        segment_count: usize,
        seg_ptr: Vec<u64>,
        seg_len: Vec<usize>,
        numa: Vec<u32>,
    ) -> Self {
        assert_eq!(seg_ptr.len(), block_count * slot_count * segment_count);
        assert_eq!(seg_len.len(), seg_ptr.len());
        assert_eq!(numa.len(), block_count * slot_count);
        Self {
            block_count,
            slot_count,
            segment_count,
            hashes: (0..block_count).map(|i| vec![i as u8]).collect(),
            seg_ptr,
            seg_len,
            numa,
        }
    }
}
