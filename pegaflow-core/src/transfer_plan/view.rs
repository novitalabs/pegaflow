use std::sync::Arc;

use log::warn;
use pegaflow_common::NumaNode;

use crate::BlockKey;
use crate::block::{LayerBlock, SealedBlock};

use super::schema::TransferPlanError;

/// Per-segment pointer extracted from a sealed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentPoint {
    pub block_idx: u32,
    pub ptr: u64,
    pub len: usize,
    /// Holder-side NUMA for this block's slot; drives [`RemoteChunk::numa_node`].
    pub numa_node: u32,
}

/// One block row in the transfer matrix.
#[derive(Debug)]
pub(crate) struct BlockView {
    pub hash: Vec<u8>,
    pub slots: Vec<SlotView>,
}

#[derive(Debug)]
pub(crate) struct SlotView {
    pub numa: NumaNode,
    pub segments: Vec<SegmentView>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SegmentView {
    pub ptr: u64,
    pub len: usize,
}

pub(crate) fn block_views_from_found(
    found_blocks: &[(BlockKey, Arc<SealedBlock>)],
) -> Result<Vec<BlockView>, TransferPlanError> {
    if found_blocks.is_empty() {
        return Ok(Vec::new());
    }

    let slot_count = found_blocks[0].1.slots().len();
    if slot_count == 0 {
        return Err(TransferPlanError::new("sealed block has no slots"));
    }

    let mut views = Vec::with_capacity(found_blocks.len());
    for (block_idx, (key, block)) in found_blocks.iter().enumerate() {
        if block.slots().len() != slot_count {
            return Err(TransferPlanError::new(format!(
                "block {block_idx} slot count {} != {slot_count} \
                 (namespace shared by incompatible KV layouts)",
                block.slots().len()
            )));
        }

        let slot_numas = block.slot_numas();
        let mut slots = Vec::with_capacity(slot_count);
        for (slot_idx, raw) in block.slots().iter().enumerate() {
            let numa = slot_numas
                .get(slot_idx)
                .copied()
                .unwrap_or(NumaNode::UNKNOWN);
            if numa.is_unknown() {
                warn!(
                    "transfer_plan: block {block_idx} slot {slot_idx} NUMA unknown (slot_numas len={})",
                    slot_numas.len()
                );
            }
            let layer = LayerBlock::new(raw);
            let mut segments = Vec::new();
            let k_ptr = layer.k_ptr() as u64;
            let k_len = layer.k_size();
            if k_len == 0 {
                return Err(TransferPlanError::new(format!(
                    "block {block_idx} slot {slot_idx} K size is zero"
                )));
            }
            segments.push(SegmentView {
                ptr: k_ptr,
                len: k_len,
            });

            if let Some(v_ptr) = layer.v_ptr() {
                let v_len = layer.v_size().ok_or_else(|| {
                    TransferPlanError::new(format!(
                        "block {block_idx} slot {slot_idx} split layout has V ptr but missing V size"
                    ))
                })?;
                if v_len == 0 {
                    return Err(TransferPlanError::new(format!(
                        "block {block_idx} slot {slot_idx} split layout with zero V size"
                    )));
                }
                segments.push(SegmentView {
                    ptr: v_ptr as u64,
                    len: v_len,
                });
            }

            slots.push(SlotView { numa, segments });
        }

        views.push(BlockView {
            hash: key.hash.clone(),
            slots,
        });
    }

    Ok(views)
}

impl BlockView {
    pub(crate) fn segment_point(
        &self,
        block_idx: u32,
        slot_idx: usize,
        segment_idx: usize,
    ) -> SegmentPoint {
        let seg = &self.slots[slot_idx].segments[segment_idx];
        SegmentPoint {
            block_idx,
            ptr: seg.ptr,
            len: seg.len,
            numa_node: self.slots[slot_idx].numa.0,
        }
    }
}
