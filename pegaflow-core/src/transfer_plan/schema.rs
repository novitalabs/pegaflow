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

pub(crate) fn derive_slot_schemas(views: &[BlockView]) -> Vec<SlotSchema> {
    if views.is_empty() {
        return Vec::new();
    }

    let slot_count = views[0].slots.len();
    let segment_count = views[0].slots[0].segments.len();
    assert!(
        (1..=2).contains(&segment_count),
        "invalid segment count {segment_count}"
    );

    let mut schemas = Vec::with_capacity(slot_count);
    for slot_idx in 0..slot_count {
        let mut segment_schemas = SmallVec::<[SegmentSchema; 2]>::new();

        for segment_idx in 0..segment_count {
            let expected_len = views[0].slots[slot_idx].segments[segment_idx].len;
            let mut stride = expected_len;

            let ptrs: Vec<u64> = views
                .iter()
                .map(|view| view.slots[slot_idx].segments[segment_idx].ptr)
                .collect();

            for (block_idx, view) in views.iter().enumerate() {
                let seg = &view.slots[slot_idx].segments[segment_idx];
                assert_eq!(
                    seg.len, expected_len,
                    "block {block_idx} slot {slot_idx} segment {segment_idx} len mismatch"
                );
            }

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

        for view in views.iter().skip(1) {
            assert_eq!(
                view.slots.len(),
                slot_count,
                "inconsistent slot count across blocks"
            );
            assert_eq!(
                view.slots[slot_idx].segments.len(),
                segment_count,
                "slot {slot_idx} segment count mismatch"
            );
        }

        // NUMA affinity lives on each [`RemoteChunk`]; this field is wire-compat only.
        schemas.push(SlotSchema {
            numa_node: views[0].slots[slot_idx].numa.0,
            segments: segment_schemas,
        });
    }

    schemas
}
