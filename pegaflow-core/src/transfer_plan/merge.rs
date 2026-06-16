use pegaflow_transfer_wire::{RemoteChunk, SegmentPlacement, SlotSchema};

use super::view::SegmentPoint;

type TransferGeometry = (Vec<RemoteChunk>, Vec<SegmentPlacement>);
type PtrInterval = (u64, u64, u32);
type PartitionOutput = (Vec<PendingRun>, Vec<PendingSingle>, Vec<PtrInterval>);

#[derive(Debug, Clone, Copy)]
struct Interval {
    start: u64,
    end: u64,
    numa_node: u32,
}

#[derive(Debug, Clone, Copy)]
struct PendingRun {
    slot_idx: u32,
    segment_idx: u32,
    block_start: u32,
    block_count: u32,
    base_ptr: u64,
}

#[derive(Debug, Clone, Copy)]
struct PendingSingle {
    block_idx: u32,
    slot_idx: u32,
    segment_idx: u32,
    ptr: u64,
}

pub(crate) fn build_transfer_geometry(
    views: &[super::view::BlockView],
    slot_schemas: &[SlotSchema],
) -> TransferGeometry {
    if views.is_empty() {
        return (vec![], vec![]);
    }

    let mut pending_runs = Vec::new();
    let mut pending_singles = Vec::new();
    let mut intervals = Vec::new();

    for (slot_idx, slot_schema) in slot_schemas.iter().enumerate() {
        for (segment_idx, seg_schema) in slot_schema.segments.iter().enumerate() {
            let points: Vec<SegmentPoint> = views
                .iter()
                .enumerate()
                .map(|(block_idx, view)| {
                    view.segment_point(block_idx as u32, slot_idx, segment_idx)
                })
                .collect();

            let (runs, singles, seg_intervals) = partition_points(
                slot_idx as u32,
                segment_idx as u32,
                points,
                seg_schema.block_stride,
                seg_schema.bytes,
            );
            pending_runs.extend(runs);
            pending_singles.extend(singles);
            intervals.extend(seg_intervals);
        }
    }

    assign_chunks(intervals, pending_runs, pending_singles)
}

fn partition_points(
    slot_idx: u32,
    segment_idx: u32,
    points: Vec<SegmentPoint>,
    expected_stride: usize,
    expected_len: usize,
) -> PartitionOutput {
    if points.is_empty() {
        return (vec![], vec![], vec![]);
    }

    let mut sorted = points;
    sorted.sort_by_key(|p| p.ptr);

    let mut runs = Vec::new();
    let mut singles = Vec::new();
    let mut intervals = Vec::new();
    let mut i = 0;

    while i < sorted.len() {
        let start = sorted[i];
        if start.len != expected_len {
            singles.push(PendingSingle {
                block_idx: start.block_idx,
                slot_idx,
                segment_idx,
                ptr: start.ptr,
            });
            intervals.push((start.ptr, start.ptr + start.len as u64, start.numa_node));
            i += 1;
            continue;
        }

        let block_start = start.block_idx;
        let base_ptr = start.ptr;
        let run_numa = start.numa_node;
        let mut count = 1u32;
        let mut j = i + 1;

        while j < sorted.len() {
            let prev = &sorted[j - 1];
            let next = &sorted[j];
            if next.len != expected_len {
                break;
            }
            let expected_ptr = prev.ptr + expected_stride as u64;
            if next.ptr == expected_ptr
                && next.block_idx == prev.block_idx + 1
                && next.numa_node == run_numa
            {
                count += 1;
                j += 1;
            } else {
                break;
            }
        }

        if count >= 2 {
            let end =
                base_ptr + u64::from(count - 1) * expected_stride as u64 + expected_len as u64;
            runs.push(PendingRun {
                slot_idx,
                segment_idx,
                block_start,
                block_count: count,
                base_ptr,
            });
            intervals.push((base_ptr, end, run_numa));
        } else {
            singles.push(PendingSingle {
                block_idx: start.block_idx,
                slot_idx,
                segment_idx,
                ptr: start.ptr,
            });
            intervals.push((start.ptr, start.ptr + start.len as u64, start.numa_node));
        }

        i = j;
    }

    (runs, singles, intervals)
}

fn assign_chunks(
    mut intervals: Vec<PtrInterval>,
    pending_runs: Vec<PendingRun>,
    pending_singles: Vec<PendingSingle>,
) -> TransferGeometry {
    if intervals.is_empty() {
        return (vec![], vec![]);
    }

    intervals.sort_by_key(|(start, _, _)| *start);

    // Sorted by start, so a same-NUMA interval whose start touches or overlaps
    // the previous one (`start <= last.end`, which includes the adjacent
    // `start == last.end` case) extends it. One pass coalesces fully.
    let mut merged: Vec<Interval> = Vec::new();
    for (start, end, numa) in intervals {
        if let Some(last) = merged.last_mut()
            && last.numa_node == numa
            && start <= last.end
        {
            last.end = last.end.max(end);
            continue;
        }
        merged.push(Interval {
            start,
            end,
            numa_node: numa,
        });
    }

    let remote_chunks: Vec<RemoteChunk> = merged
        .iter()
        .map(|iv| RemoteChunk {
            base_ptr: iv.start,
            length: iv.end - iv.start,
            numa_node: iv.numa_node,
        })
        .collect();

    let mut placements = Vec::with_capacity(pending_runs.len() + pending_singles.len());

    for pending in pending_runs {
        let (chunk_idx, offset_in_chunk) = locate_in_chunks(pending.base_ptr, &remote_chunks);
        placements.push(SegmentPlacement {
            slot_idx: pending.slot_idx,
            segment_idx: pending.segment_idx,
            chunk_idx,
            offset_in_chunk,
            block_start: pending.block_start,
            block_count: pending.block_count,
        });
    }

    for pending in pending_singles {
        let (chunk_idx, offset_in_chunk) = locate_in_chunks(pending.ptr, &remote_chunks);
        placements.push(SegmentPlacement {
            slot_idx: pending.slot_idx,
            segment_idx: pending.segment_idx,
            chunk_idx,
            offset_in_chunk,
            block_start: pending.block_idx,
            block_count: 1,
        });
    }

    (remote_chunks, placements)
}

fn locate_in_chunks(ptr: u64, chunks: &[RemoteChunk]) -> (u32, usize) {
    for (idx, chunk) in chunks.iter().enumerate() {
        let end = chunk.base_ptr + chunk.length;
        if ptr >= chunk.base_ptr && ptr < end {
            let offset = (ptr - chunk.base_ptr) as usize;
            return (idx as u32, offset);
        }
    }
    panic!("pointer 0x{ptr:x} not covered by any remote chunk");
}

#[cfg(test)]
mod tests {
    use pegaflow_transfer_wire::SegmentSchema;

    use super::*;
    use crate::transfer_plan::view::{BlockView, SegmentView, SlotView};
    use pegaflow_common::NumaNode;

    use smallvec::smallvec;

    fn contiguous_views(stride: usize, count: usize) -> (Vec<BlockView>, Vec<SlotSchema>) {
        let base = 0x5000u64;
        let bytes = 64usize;
        let views: Vec<BlockView> = (0..count)
            .map(|i| BlockView {
                hash: vec![i as u8],
                slots: vec![SlotView {
                    numa: NumaNode(0),
                    segments: vec![SegmentView {
                        ptr: base + (i as u64) * stride as u64,
                        len: bytes,
                    }],
                }],
            })
            .collect();
        let slot_schemas = vec![SlotSchema {
            segments: smallvec![SegmentSchema {
                bytes,
                block_stride: stride,
            }],
        }];
        (views, slot_schemas)
    }

    #[test]
    fn batch_blocks_form_one_placement() {
        let (views, slot_schemas) = contiguous_views(128, 4);
        let (remote_chunks, placements) = build_transfer_geometry(&views, &slot_schemas);
        assert_eq!(remote_chunks.len(), 1);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].block_count, 4);
    }

    #[test]
    fn broken_stride_produces_singleton_placement() {
        let (mut views, slot_schemas) = contiguous_views(128, 3);
        views[2].slots[0].segments[0].ptr = 0x8000;
        let (_, placements) = build_transfer_geometry(&views, &slot_schemas);
        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].block_count, 2);
        assert_eq!(placements[1].block_count, 1);
        assert_eq!(placements[1].block_start, 2);
    }

    #[test]
    fn mixed_numa_splits_contiguous_run_into_separate_chunks() {
        let base = 0x5000u64;
        let bytes = 64usize;
        let stride = 128usize;
        let views: Vec<BlockView> = (0..4)
            .map(|i| BlockView {
                hash: vec![i as u8],
                slots: vec![SlotView {
                    numa: NumaNode(if i < 2 { 0 } else { 1 }),
                    segments: vec![SegmentView {
                        ptr: base + (i as u64) * stride as u64,
                        len: bytes,
                    }],
                }],
            })
            .collect();
        let slot_schemas = vec![SlotSchema {
            segments: smallvec![SegmentSchema {
                bytes,
                block_stride: stride,
            }],
        }];

        let (remote_chunks, placements) = build_transfer_geometry(&views, &slot_schemas);
        assert_eq!(remote_chunks.len(), 2);
        assert_eq!(remote_chunks[0].numa_node, 0);
        assert_eq!(remote_chunks[1].numa_node, 1);
        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].block_count, 2);
        assert_eq!(placements[1].block_count, 2);
        assert_eq!(placements[1].block_start, 2);
    }
}
