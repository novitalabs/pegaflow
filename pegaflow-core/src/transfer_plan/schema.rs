use std::fmt;

use pegaflow_transfer_wire::{SegmentSchema, SlotSchema};
use smallvec::SmallVec;

use super::view::BlockView;

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
    views: &[BlockView],
) -> Result<Vec<SlotSchema>, TransferPlanError> {
    if views.is_empty() {
        return Ok(Vec::new());
    }

    let slot_count = views[0].slots.len();
    let segment_count = views[0].slots[0].segments.len();
    if !(1..=2).contains(&segment_count) {
        return Err(TransferPlanError::new(format!(
            "invalid segment count {segment_count}"
        )));
    }

    // Validate every block shares block 0's segment geometry up front, so the
    // per-segment indexing below cannot go out of bounds. A mismatch means a
    // namespace shared by incompatible KV layouts; the load path treats that as
    // a recoverable error, so encode returns Err rather than panicking.
    // (Slot-count consistency is already enforced by `block_views_from_found`.)
    for (block_idx, view) in views.iter().enumerate() {
        for (slot_idx, slot) in view.slots.iter().enumerate() {
            if slot.segments.len() != segment_count {
                return Err(TransferPlanError::new(format!(
                    "block {block_idx} slot {slot_idx} segment count {} != {segment_count}",
                    slot.segments.len()
                )));
            }
        }
    }

    let mut schemas = Vec::with_capacity(slot_count);
    for slot_idx in 0..slot_count {
        let mut segment_schemas = SmallVec::<[SegmentSchema; 2]>::new();

        for segment_idx in 0..segment_count {
            let expected_len = views[0].slots[slot_idx].segments[segment_idx].len;
            for (block_idx, view) in views.iter().enumerate() {
                let seg_len = view.slots[slot_idx].segments[segment_idx].len;
                if seg_len != expected_len {
                    return Err(TransferPlanError::new(format!(
                        "block {block_idx} slot {slot_idx} segment {segment_idx} len \
                         {seg_len} != {expected_len}"
                    )));
                }
            }

            let mut stride = expected_len;
            let ptrs: Vec<u64> = views
                .iter()
                .map(|view| view.slots[slot_idx].segments[segment_idx].ptr)
                .collect();
            for window in ptrs.windows(2) {
                if window[1] > window[0] {
                    let delta = (window[1] - window[0]) as usize;
                    if delta >= expected_len {
                        stride = delta;
                    }
                }
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
