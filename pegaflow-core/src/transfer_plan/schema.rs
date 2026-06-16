use std::fmt;

use pegaflow_transfer_wire::{SegmentSchema, SlotSchema};
use smallvec::SmallVec;

use super::view::BlockMatrix;

#[derive(Debug)]
pub(crate) struct TransferPlanError(String);

impl TransferPlanError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TransferPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "transfer_plan: {}", self.0)
    }
}

impl std::error::Error for TransferPlanError {}

pub(crate) fn derive_slot_schemas(
    matrix: &BlockMatrix,
) -> Result<Vec<SlotSchema>, TransferPlanError> {
    // No blocks => no schemas. Guard here (not just at the caller) so the
    // per-column `lens[0]` below is self-evidently in bounds: an empty matrix
    // has empty columns.
    let slot_count = matrix.slot_count();
    if slot_count == 0 || matrix.block_count() == 0 {
        return Ok(Vec::new());
    }

    // `block_matrix_from_found` already enforced that every block shares block
    // 0's slot and segment counts; here we only reject a globally invalid
    // segment count (the wire format coalesces at most K/V). A mismatch means a
    // namespace shared by incompatible KV layouts; the load path treats that as
    // a recoverable error, so encode returns Err rather than panicking.
    let segment_count = matrix.segment_count();
    if !(1..=2).contains(&segment_count) {
        return Err(TransferPlanError::new(format!(
            "invalid segment count {segment_count}"
        )));
    }

    let mut schemas = Vec::with_capacity(slot_count);
    for slot_idx in 0..slot_count {
        let mut segment_schemas = SmallVec::<[SegmentSchema; 2]>::new();

        for segment_idx in 0..segment_count {
            let ptrs = matrix.seg_ptrs(slot_idx, segment_idx);
            let lens = matrix.seg_lens(slot_idx, segment_idx);
            let expected_len = lens[0];

            // Single streaming pass over the block-ordered column: validate
            // length uniformity and derive the block stride (largest
            // consecutive-block pointer delta that is at least one segment
            // wide).
            let mut stride = expected_len;
            let mut prev_ptr: Option<u64> = None;
            for (block_idx, (&ptr, &len)) in ptrs.iter().zip(lens).enumerate() {
                if len != expected_len {
                    return Err(TransferPlanError::new(format!(
                        "block {block_idx} slot {slot_idx} segment {segment_idx} len \
                         {len} != {expected_len}"
                    )));
                }
                if let Some(prev) = prev_ptr
                    && ptr > prev
                {
                    let delta = (ptr - prev) as usize;
                    if delta >= expected_len {
                        stride = delta;
                    }
                }
                prev_ptr = Some(ptr);
            }

            segment_schemas.push(SegmentSchema {
                bytes: expected_len,
                block_stride: stride,
            });
        }

        schemas.push(SlotSchema {
            segments: segment_schemas,
        });
    }

    Ok(schemas)
}
