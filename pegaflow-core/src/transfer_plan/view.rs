use std::sync::Arc;

use log::warn;
use pegaflow_common::NumaNode;

use crate::BlockKey;
use crate::block::{LayerBlock, SealedBlock};

/// Per-segment pointer extracted from a sealed block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SegmentPoint {
    pub block_idx: u32,
    pub ptr: u64,
    pub len: usize,
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
) -> Vec<BlockView> {
    if found_blocks.is_empty() {
        return Vec::new();
    }

    let slot_count = found_blocks[0].1.slots().len();
    assert!(slot_count > 0, "sealed block must have at least one slot");

    let mut views = Vec::with_capacity(found_blocks.len());
    for (block_idx, (key, block)) in found_blocks.iter().enumerate() {
        assert_eq!(
            block.slots().len(),
            slot_count,
            "block {block_idx} slot count mismatch"
        );

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
            assert!(
                k_len > 0,
                "block {block_idx} slot {slot_idx} K size is zero"
            );
            segments.push(SegmentView {
                ptr: k_ptr,
                len: k_len,
            });

            if let Some(v_ptr) = layer.v_ptr() {
                let v_len = layer
                    .v_size()
                    .expect("split layout has V ptr but missing V size");
                assert!(
                    v_len > 0,
                    "block {block_idx} slot {slot_idx} split layout with zero V size"
                );
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

    views
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
        }
    }
}
