use pegaflow_transfer_wire::{RemoteChunk, SegmentPlacement, SlotSchema};

use super::view::{BlockMatrix, SegmentPoint};

type TransferGeometry = (Vec<RemoteChunk>, Vec<SegmentPlacement>);
type PtrInterval = (u64, u64, u32);

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

/// Accumulates the per-column partition results across every `(slot, segment)`
/// before they are resolved into coalesced chunks + placements.
#[derive(Default)]
struct PartitionSink {
    runs: Vec<PendingRun>,
    singles: Vec<PendingSingle>,
    intervals: Vec<PtrInterval>,
}

pub(crate) fn build_transfer_geometry(
    matrix: &BlockMatrix,
    slot_schemas: &[SlotSchema],
) -> TransferGeometry {
    let block_count = matrix.block_count();
    if block_count == 0 {
        return (vec![], vec![]);
    }

    let mut sink = PartitionSink::default();
    // One scratch buffer reused across every (slot, segment) column instead of
    // allocating a fresh point vector per column.
    let mut points: Vec<SegmentPoint> = Vec::with_capacity(block_count);

    for (slot_idx, slot_schema) in slot_schemas.iter().enumerate() {
        let numas = matrix.numas(slot_idx);
        for (segment_idx, seg_schema) in slot_schema.segments.iter().enumerate() {
            let ptrs = matrix.seg_ptrs(slot_idx, segment_idx);
            let lens = matrix.seg_lens(slot_idx, segment_idx);
            points.clear();
            for block_idx in 0..block_count {
                points.push(SegmentPoint {
                    block_idx: block_idx as u32,
                    ptr: ptrs[block_idx],
                    len: lens[block_idx],
                    numa_node: numas[block_idx],
                });
            }

            partition_points(
                slot_idx as u32,
                segment_idx as u32,
                &mut points,
                seg_schema.block_stride,
                seg_schema.bytes,
                &mut sink,
            );
        }
    }

    assign_chunks(sink)
}

fn partition_points(
    slot_idx: u32,
    segment_idx: u32,
    points: &mut [SegmentPoint],
    expected_stride: usize,
    expected_len: usize,
    sink: &mut PartitionSink,
) {
    if points.is_empty() {
        return;
    }

    let sorted = points;
    sorted.sort_by_key(|p| p.ptr);

    let mut i = 0;

    while i < sorted.len() {
        let start = sorted[i];
        if start.len != expected_len {
            sink.singles.push(PendingSingle {
                block_idx: start.block_idx,
                slot_idx,
                segment_idx,
                ptr: start.ptr,
            });
            sink.intervals
                .push((start.ptr, start.ptr + start.len as u64, start.numa_node));
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
            sink.runs.push(PendingRun {
                slot_idx,
                segment_idx,
                block_start,
                block_count: count,
                base_ptr,
            });
            sink.intervals.push((base_ptr, end, run_numa));
        } else {
            sink.singles.push(PendingSingle {
                block_idx: start.block_idx,
                slot_idx,
                segment_idx,
                ptr: start.ptr,
            });
            sink.intervals
                .push((start.ptr, start.ptr + start.len as u64, start.numa_node));
        }

        i = j;
    }
}

fn assign_chunks(sink: PartitionSink) -> TransferGeometry {
    let PartitionSink {
        runs: pending_runs,
        singles: pending_singles,
        mut intervals,
    } = sink;
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

    use smallvec::smallvec;

    const BASE: u64 = 0x5000;
    const BYTES: usize = 64;

    /// One slot, one segment, `count` blocks at `BASE + i*stride`, all NUMA 0.
    fn contiguous_matrix(stride: usize, count: usize) -> (BlockMatrix, Vec<SlotSchema>) {
        let seg_ptr: Vec<u64> = (0..count)
            .map(|i| BASE + (i as u64) * stride as u64)
            .collect();
        let matrix =
            BlockMatrix::from_columns(count, 1, 1, seg_ptr, vec![BYTES; count], vec![0; count]);
        let slot_schemas = vec![SlotSchema {
            segments: smallvec![SegmentSchema {
                bytes: BYTES,
                block_stride: stride,
            }],
        }];
        (matrix, slot_schemas)
    }

    #[test]
    fn batch_blocks_form_one_placement() {
        let (matrix, slot_schemas) = contiguous_matrix(128, 4);
        let (remote_chunks, placements) = build_transfer_geometry(&matrix, &slot_schemas);
        assert_eq!(remote_chunks.len(), 1);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements[0].block_count, 4);
    }

    #[test]
    fn broken_stride_produces_singleton_placement() {
        let stride = 128usize;
        let mut seg_ptr: Vec<u64> = (0..3).map(|i| BASE + (i as u64) * stride as u64).collect();
        seg_ptr[2] = 0x8000;
        let matrix = BlockMatrix::from_columns(3, 1, 1, seg_ptr, vec![BYTES; 3], vec![0; 3]);
        let slot_schemas = vec![SlotSchema {
            segments: smallvec![SegmentSchema {
                bytes: BYTES,
                block_stride: stride,
            }],
        }];

        let (_, placements) = build_transfer_geometry(&matrix, &slot_schemas);
        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].block_count, 2);
        assert_eq!(placements[1].block_count, 1);
        assert_eq!(placements[1].block_start, 2);
    }

    #[test]
    fn mixed_numa_splits_contiguous_run_into_separate_chunks() {
        let stride = 128usize;
        let seg_ptr: Vec<u64> = (0..4).map(|i| BASE + (i as u64) * stride as u64).collect();
        let numa: Vec<u32> = (0..4).map(|i| if i < 2 { 0 } else { 1 }).collect();
        let matrix = BlockMatrix::from_columns(4, 1, 1, seg_ptr, vec![BYTES; 4], numa);
        let slot_schemas = vec![SlotSchema {
            segments: smallvec![SegmentSchema {
                bytes: BYTES,
                block_stride: stride,
            }],
        }];

        let (remote_chunks, placements) = build_transfer_geometry(&matrix, &slot_schemas);
        assert_eq!(remote_chunks.len(), 2);
        assert_eq!(remote_chunks[0].numa_node, 0);
        assert_eq!(remote_chunks[1].numa_node, 1);
        assert_eq!(placements.len(), 2);
        assert_eq!(placements[0].block_count, 2);
        assert_eq!(placements[1].block_count, 2);
        assert_eq!(placements[1].block_start, 2);
    }
}
