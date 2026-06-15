use std::ptr::NonNull;
use std::sync::Arc;

use pegaflow_transfer_wire::TransferPlan;

use crate::backing::PrefetchResult;
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::pinned_pool::PinnedAllocation;

use super::materialize::{RebuildContext, chunk_ptr_at};

type SegmentSlot = Option<(NonNull<u8>, usize, Arc<PinnedAllocation>)>;

pub(crate) fn rebuild_sealed_blocks(
    plan: &TransferPlan,
    rebuild: &RebuildContext,
    namespace: &str,
) -> Result<PrefetchResult, String> {
    if plan.block_hashes.is_empty() {
        return Ok(Vec::new());
    }

    let block_count = plan.block_hashes.len();
    let slot_count = plan.slot_schemas.len();
    let segment_count = plan.slot_schemas[0].segments.len();

    let mut block_slots: Vec<Vec<Vec<SegmentSlot>>> = (0..block_count)
        .map(|_| {
            (0..slot_count)
                .map(|_| (0..segment_count).map(|_| None).collect())
                .collect()
        })
        .collect();

    for placement in &plan.placements {
        let seg = &plan.slot_schemas[placement.slot_idx as usize].segments
            [placement.segment_idx as usize];
        let seg_len = seg.bytes;
        let stride = seg.block_stride;

        for i in 0..placement.block_count {
            let block_idx = (placement.block_start + i) as usize;
            let offset_in_chunk = placement.offset_in_chunk + (i as usize) * stride;
            let (ptr, allocation) =
                chunk_ptr_at(rebuild, plan, placement.chunk_idx, offset_in_chunk)?;
            block_slots[block_idx][placement.slot_idx as usize][placement.segment_idx as usize] =
                Some((ptr, seg_len, allocation));
        }
    }

    let mut result = Vec::with_capacity(block_count);
    for (block_idx, hash) in plan.block_hashes.iter().enumerate() {
        let slots: Vec<RawBlock> = block_slots[block_idx]
            .iter()
            .map(|slot_segments| {
                let segments: Vec<Segment> = slot_segments
                    .iter()
                    .map(|entry| {
                        let (ptr, len, alloc) = entry
                            .as_ref()
                            .ok_or_else(|| "missing segment during rebuild".to_string())?;
                        Ok(Segment::new(*ptr, *len, Arc::clone(alloc)))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                Ok(RawBlock::new(segments))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        result.push((key, Arc::new(SealedBlock::from_slots(slots))));
    }

    Ok(result)
}
