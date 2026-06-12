// Run-encoded segment address tables for cross-node transfer.
//
// A "lane" is one template segment (slot 0 K, slot 0 V if split, slot 1 K,
// ...) across all blocks of a transfer prefix. Holder memory comes from
// sequential bump allocation, so within a lane the addresses normally form
// an arithmetic sequence: addr(block i) = base + i * stride. The wire format
// exploits this — each lane is a handful of affine runs instead of one
// fixed64 per segment.
//
// The requester reverses the encoding and goes further: runs on the same
// NUMA node whose per-block chunks tile densely are grouped, so the common
// fully-bump-allocated holder layout collapses into one RDMA READ per NUMA
// node instead of one per segment.

use std::collections::HashMap;

use pegaflow_common::NumaNode;
use pegaflow_proto::proto::engine::TransferSlotInfo;

/// Lane-major run table mirroring the QueryBlocksForTransfer wire format.
///
/// Lane `l` owns `runs_per_lane[l]` consecutive entries of the three parallel
/// arrays; run `j` covers `count[j]` consecutive blocks at
/// `base[j] + i * stride[j]`. Within a lane, run counts sum to the prefix
/// block count. A run with `count == 1` carries stride 0.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SegmentRunTable {
    pub runs_per_lane: Vec<u32>,
    pub base: Vec<u64>,
    pub stride: Vec<u64>,
    pub count: Vec<u32>,
}

/// Holder-side run accumulator. Feed segment addresses block-major
/// (`push(lane, addr)` for every lane of block 0, then block 1, ...),
/// then `finish()` into the wire table.
pub(crate) struct RunTableBuilder {
    lanes: Vec<LaneAcc>,
}

impl RunTableBuilder {
    pub(crate) fn new(lane_count: usize) -> Self {
        Self {
            lanes: (0..lane_count).map(|_| LaneAcc::default()).collect(),
        }
    }

    pub(crate) fn push(&mut self, lane: usize, addr: u64) {
        self.lanes[lane].push(addr);
    }

    pub(crate) fn finish(mut self) -> SegmentRunTable {
        let mut table = SegmentRunTable::default();
        for lane in &mut self.lanes {
            lane.seal();
            table.runs_per_lane.push(lane.closed.len() as u32);
            for run in &lane.closed {
                table.base.push(run.base);
                table.stride.push(run.stride);
                table.count.push(run.count);
            }
        }
        table
    }
}

#[derive(Default)]
struct LaneAcc {
    closed: Vec<ClosedRun>,
    cur: Option<OpenRun>,
}

struct ClosedRun {
    base: u64,
    stride: u64,
    count: u32,
}

struct OpenRun {
    base: u64,
    /// Fixed by the second address; a single-address run has no stride.
    stride: Option<u64>,
    count: u32,
}

impl LaneAcc {
    fn push(&mut self, addr: u64) {
        let Some(cur) = &mut self.cur else {
            self.cur = Some(OpenRun {
                base: addr,
                stride: None,
                count: 1,
            });
            return;
        };
        let extends = match cur.stride {
            // Second address fixes the stride (ascending only).
            None => match addr.checked_sub(cur.base) {
                Some(stride) => {
                    cur.stride = Some(stride);
                    true
                }
                None => false,
            },
            Some(stride) => {
                stride
                    .checked_mul(cur.count as u64)
                    .and_then(|off| cur.base.checked_add(off))
                    == Some(addr)
            }
        };
        if extends {
            cur.count += 1;
        } else {
            self.seal();
            self.cur = Some(OpenRun {
                base: addr,
                stride: None,
                count: 1,
            });
        }
    }

    fn seal(&mut self) {
        if let Some(cur) = self.cur.take() {
            self.closed.push(ClosedRun {
                base: cur.base,
                stride: cur.stride.unwrap_or(0),
                count: cur.count,
            });
        }
    }
}

/// One RDMA READ: copy `len` bytes from remote `remote_base` into the local
/// NUMA slab at `local_offset`.
pub(crate) struct PlannedDesc {
    pub(crate) numa: NumaNode,
    pub(crate) local_offset: u64,
    pub(crate) remote_base: u64,
    pub(crate) len: u64,
}

/// Local placement of one lane run: blocks `[start_block, start_block+count)`
/// land at `slab[numa] + local_offset + (block - start_block) * local_stride`.
#[derive(Clone)]
pub(crate) struct PlannedRun {
    pub(crate) start_block: u32,
    pub(crate) count: u32,
    pub(crate) local_offset: u64,
    pub(crate) local_stride: u64,
}

pub(crate) struct PlannedLane {
    pub(crate) seg_size: u64,
    pub(crate) numa: NumaNode,
    pub(crate) runs: Vec<PlannedRun>,
}

pub(crate) struct TransferPlan {
    pub(crate) bytes_per_numa: HashMap<NumaNode, u64>,
    pub(crate) descs: Vec<PlannedDesc>,
    /// Lane-major, same order as the wire table.
    pub(crate) lanes: Vec<PlannedLane>,
}

/// A group of runs on one NUMA node whose per-block chunks tile densely:
/// block i contributes `chunk_total` contiguous remote bytes at
/// `remote_base + i * stride`. The local mirror packs those chunks
/// back-to-back, so every member run keeps affine local addressing.
struct Group {
    numa: NumaNode,
    start_block: u32,
    count: u32,
    remote_base: u64,
    /// Remote distance between consecutive block chunks; `None` for
    /// single-block groups where no stride is observable.
    stride: Option<u64>,
    chunk_total: u64,
    /// (run index, byte offset of the run's segment within the chunk)
    members: Vec<(usize, u64)>,
    local_offset: u64,
}

/// Validate a run table against the template and turn it into a transfer
/// plan: per-NUMA slab sizes, coalesced RDMA READ descriptors, and per-lane
/// local placement for rebuilding blocks.
pub(crate) fn plan_transfer(
    template: &[TransferSlotInfo],
    block_count: usize,
    table: &SegmentRunTable,
) -> Result<TransferPlan, String> {
    // Lane metadata in wire order: (segment size, NUMA node). The template
    // is remote input: a slot without a K segment would rebuild into a
    // RawBlock with no or mislabeled segments, so reject it outright.
    let mut lane_meta: Vec<(u64, NumaNode)> = Vec::new();
    for (slot_idx, slot) in template.iter().enumerate() {
        if slot.k_size == 0 {
            return Err(format!("slot {slot_idx} has a zero-size K segment"));
        }
        let numa = NumaNode(slot.numa_node);
        lane_meta.push((slot.k_size, numa));
        if slot.v_size > 0 {
            lane_meta.push((slot.v_size, numa));
        }
    }
    if block_count > 0 && lane_meta.is_empty() {
        return Err(format!("empty slot template for {block_count} blocks"));
    }

    if table.runs_per_lane.len() != lane_meta.len() {
        return Err(format!(
            "run table has {} lanes, template implies {}",
            table.runs_per_lane.len(),
            lane_meta.len()
        ));
    }
    let total_runs: usize = table.runs_per_lane.iter().map(|&n| n as usize).sum();
    if table.base.len() != total_runs
        || table.stride.len() != total_runs
        || table.count.len() != total_runs
    {
        return Err(format!(
            "run arrays out of sync: per-lane counts sum to {total_runs}, base={} stride={} count={}",
            table.base.len(),
            table.stride.len(),
            table.count.len()
        ));
    }

    struct DecodedRun {
        lane: usize,
        start_block: u32,
        count: u32,
        base: u64,
        stride: u64,
    }
    let mut runs: Vec<DecodedRun> = Vec::with_capacity(total_runs);
    let mut lane_run_ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(lane_meta.len());
    let mut next = 0usize;
    for (lane, &nruns) in table.runs_per_lane.iter().enumerate() {
        let begin = runs.len();
        let mut covered = 0u64;
        for j in next..next + nruns as usize {
            let count = table.count[j];
            if count == 0 {
                return Err(format!("lane {lane} contains an empty run"));
            }
            let start_block = u32::try_from(covered)
                .map_err(|_| format!("lane {lane} run start overflows u32"))?;
            // Reject runs whose remote span overflows the address space.
            table.stride[j]
                .checked_mul((count - 1) as u64)
                .and_then(|off| table.base[j].checked_add(off))
                .and_then(|last| last.checked_add(lane_meta[lane].0))
                .ok_or_else(|| format!("lane {lane} run remote span overflows u64"))?;
            runs.push(DecodedRun {
                lane,
                start_block,
                count,
                base: table.base[j],
                stride: table.stride[j],
            });
            covered += count as u64;
        }
        if covered != block_count as u64 {
            return Err(format!(
                "lane {lane} covers {covered} blocks, expected {block_count}"
            ));
        }
        next += nruns as usize;
        lane_run_ranges.push(begin..runs.len());
    }

    // Bucket runs per NUMA node and sort by remote base so dense neighbours
    // meet during grouping and local offsets follow remote order.
    let mut runs_by_numa: HashMap<NumaNode, Vec<usize>> = HashMap::new();
    for (i, run) in runs.iter().enumerate() {
        runs_by_numa
            .entry(lane_meta[run.lane].1)
            .or_default()
            .push(i);
    }
    let mut numas: Vec<NumaNode> = runs_by_numa.keys().copied().collect();
    numas.sort_by_key(|n| n.0);

    let mut bytes_per_numa: HashMap<NumaNode, u64> = HashMap::new();
    let mut descs: Vec<PlannedDesc> = Vec::new();
    let mut planned: Vec<Option<PlannedRun>> = vec![None; runs.len()];

    for numa in numas {
        let mut idxs = runs_by_numa.remove(&numa).expect("numa key from map");
        idxs.sort_by_key(|&i| (runs[i].base, runs[i].start_block));

        let mut groups: Vec<Group> = Vec::new();
        for i in idxs {
            let run = &runs[i];
            let seg_size = lane_meta[run.lane].0;
            if let Some(group) = groups.last_mut()
                && group_accepts(
                    group,
                    run.start_block,
                    run.count,
                    run.base,
                    run.stride,
                    seg_size,
                )
            {
                group.members.push((i, group.chunk_total));
                group.chunk_total += seg_size;
                continue;
            }
            groups.push(Group {
                numa,
                start_block: run.start_block,
                count: run.count,
                remote_base: run.base,
                stride: (run.count > 1).then_some(run.stride),
                chunk_total: seg_size,
                members: vec![(i, 0)],
                local_offset: 0,
            });
        }

        // Assign local offsets in remote order; the slab mirrors the remote
        // layout per group, dense (no per-block padding).
        let mut numa_total = 0u64;
        for group in &mut groups {
            group.local_offset = numa_total;
            let local_len = group
                .chunk_total
                .checked_mul(group.count as u64)
                .ok_or_else(|| "group local size overflows u64".to_string())?;
            numa_total = numa_total
                .checked_add(local_len)
                .ok_or_else(|| "NUMA slab size overflows u64".to_string())?;
        }
        bytes_per_numa.insert(numa, numa_total);

        for group in &groups {
            let dense = group
                .stride
                .is_none_or(|stride| stride == group.chunk_total);
            if dense {
                push_desc(
                    &mut descs,
                    group.numa,
                    group.local_offset,
                    group.remote_base,
                    group.chunk_total * group.count as u64,
                );
            } else {
                let stride = group.stride.expect("non-dense group has a stride");
                for b in 0..group.count as u64 {
                    push_desc(
                        &mut descs,
                        group.numa,
                        group.local_offset + b * group.chunk_total,
                        group.remote_base + b * stride,
                        group.chunk_total,
                    );
                }
            }
            for &(run_idx, chunk_offset) in &group.members {
                let run = &runs[run_idx];
                planned[run_idx] = Some(PlannedRun {
                    start_block: run.start_block,
                    count: run.count,
                    local_offset: group.local_offset + chunk_offset,
                    local_stride: group.chunk_total,
                });
            }
        }
    }

    let lanes = lane_run_ranges
        .iter()
        .enumerate()
        .map(|(lane, range)| {
            let (seg_size, numa) = lane_meta[lane];
            PlannedLane {
                seg_size,
                numa,
                runs: range
                    .clone()
                    .map(|i| planned[i].take().expect("every run belongs to one group"))
                    .collect(),
            }
        })
        .collect();

    Ok(TransferPlan {
        bytes_per_numa,
        descs,
        lanes,
    })
}

/// A run joins a group when its per-block segment starts exactly where the
/// group's chunk currently ends, for the same block range and stride —
/// i.e. the chunks stay contiguous within every block.
fn group_accepts(
    group: &Group,
    start_block: u32,
    count: u32,
    base: u64,
    stride: u64,
    seg_size: u64,
) -> bool {
    if start_block != group.start_block || count != group.count {
        return false;
    }
    if group.remote_base.checked_add(group.chunk_total) != Some(base) {
        return false;
    }
    match group.stride {
        // Single-block group: only base adjacency matters.
        None => true,
        Some(group_stride) => {
            // The grown chunk must still fit between consecutive block bases.
            stride == group_stride
                && group
                    .chunk_total
                    .checked_add(seg_size)
                    .is_some_and(|grown| grown <= group_stride)
        }
    }
}

fn push_desc(
    descs: &mut Vec<PlannedDesc>,
    numa: NumaNode,
    local_offset: u64,
    remote_base: u64,
    len: u64,
) {
    if let Some(last) = descs.last_mut()
        && last.numa == numa
        && last.local_offset + last.len == local_offset
        && last.remote_base + last.len == remote_base
    {
        last.len += len;
        return;
    }
    descs.push(PlannedDesc {
        numa,
        local_offset,
        remote_base,
        len,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(k_size: u64, v_size: u64, numa: u32) -> TransferSlotInfo {
        TransferSlotInfo {
            k_size,
            v_size,
            numa_node: numa,
        }
    }

    fn build(lane_count: usize, addrs_per_block: &[Vec<u64>]) -> SegmentRunTable {
        let mut builder = RunTableBuilder::new(lane_count);
        for block in addrs_per_block {
            assert_eq!(block.len(), lane_count);
            for (lane, &addr) in block.iter().enumerate() {
                builder.push(lane, addr);
            }
        }
        builder.finish()
    }

    /// Replay the plan: for every (lane, block), the local mirror address
    /// must receive exactly the remote segment's bytes through some desc.
    fn assert_plan_replays(addrs_per_block: &[Vec<u64>], plan: &TransferPlan) {
        for (block, addrs) in addrs_per_block.iter().enumerate() {
            for (lane_idx, lane) in plan.lanes.iter().enumerate() {
                let remote = addrs[lane_idx];
                let run = lane
                    .runs
                    .iter()
                    .find(|r| {
                        (r.start_block as usize..r.start_block as usize + r.count as usize)
                            .contains(&block)
                    })
                    .expect("block covered by a run");
                let local =
                    run.local_offset + (block as u64 - run.start_block as u64) * run.local_stride;
                // Find the desc that copies this segment and check the
                // remote->local translation matches the run arithmetic.
                let desc = plan
                    .descs
                    .iter()
                    .find(|d| {
                        d.numa == lane.numa
                            && remote >= d.remote_base
                            && remote + lane.seg_size <= d.remote_base + d.len
                    })
                    .unwrap_or_else(|| {
                        panic!("no desc covers lane {lane_idx} block {block} addr {remote:#x}")
                    });
                assert_eq!(
                    local,
                    desc.local_offset + (remote - desc.remote_base),
                    "lane {lane_idx} block {block}: local placement disagrees with desc copy"
                );
            }
        }
    }

    #[test]
    fn affine_lane_is_one_run() {
        let table = build(1, &[vec![0x1000], vec![0x1100], vec![0x1200]]);
        assert_eq!(table.runs_per_lane, vec![1]);
        assert_eq!(table.base, vec![0x1000]);
        assert_eq!(table.stride, vec![0x100]);
        assert_eq!(table.count, vec![3]);
    }

    #[test]
    fn gap_and_descending_addresses_split_runs() {
        // 0x1000, 0x1100, gap to 0x9000, then descending to 0x800.
        let table = build(1, &[vec![0x1000], vec![0x1100], vec![0x9000], vec![0x800]]);
        assert_eq!(table.runs_per_lane, vec![3]);
        assert_eq!(table.base, vec![0x1000, 0x9000, 0x800]);
        assert_eq!(table.count, vec![2, 1, 1]);
        // Single-block runs carry stride 0.
        assert_eq!(table.stride[1], 0);
        assert_eq!(table.stride[2], 0);
    }

    #[test]
    fn block_major_bump_layout_collapses_to_one_desc() {
        // Holder bump allocation: per block slot0 K,V then slot1 K,V,
        // blocks back-to-back. Lane stride == whole block bytes (16).
        let template = vec![slot(4, 4, 0), slot(4, 4, 0)];
        let addrs: Vec<Vec<u64>> = (0..3)
            .map(|b| {
                let base = 0x1000 + b * 16;
                vec![base, base + 4, base + 8, base + 12]
            })
            .collect();
        let table = build(4, &addrs);
        assert_eq!(table.runs_per_lane, vec![1, 1, 1, 1]);

        let plan = plan_transfer(&template, 3, &table).expect("plan");
        assert_eq!(plan.bytes_per_numa.get(&NumaNode(0)), Some(&48));
        assert_eq!(plan.descs.len(), 1);
        assert_eq!(plan.descs[0].remote_base, 0x1000);
        assert_eq!(plan.descs[0].len, 48);
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn padded_block_stride_emits_per_block_descs() {
        // Block chunks are 16 bytes but 32 apart (something else lives in
        // the gap): per-block descs with a dense local mirror.
        let template = vec![slot(8, 8, 0)];
        let addrs: Vec<Vec<u64>> = (0..3)
            .map(|b| {
                let base = 0x2000 + b * 32;
                vec![base, base + 8]
            })
            .collect();
        let table = build(2, &addrs);

        let plan = plan_transfer(&template, 3, &table).expect("plan");
        assert_eq!(plan.bytes_per_numa.get(&NumaNode(0)), Some(&48));
        assert_eq!(plan.descs.len(), 3);
        assert!(plan.descs.iter().all(|d| d.len == 16));
        assert_eq!(
            plan.descs.iter().map(|d| d.remote_base).collect::<Vec<_>>(),
            vec![0x2000, 0x2020, 0x2040]
        );
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn fragmented_blocks_merge_within_each_block() {
        // Two blocks allocated far apart (eviction churn), in descending
        // order; each block's segments are still adjacent. Expect one desc
        // per block, locals mirroring remote order.
        let template = vec![slot(4, 4, 0), slot(4, 4, 0)];
        let addrs = vec![
            vec![0x9000, 0x9004, 0x9008, 0x900c],
            vec![0x3000, 0x3004, 0x3008, 0x300c],
        ];
        let table = build(4, &addrs);
        assert_eq!(table.runs_per_lane, vec![2, 2, 2, 2]);

        let plan = plan_transfer(&template, 2, &table).expect("plan");
        assert_eq!(plan.descs.len(), 2);
        assert_eq!(
            plan.descs.iter().map(|d| d.remote_base).collect::<Vec<_>>(),
            vec![0x3000, 0x9000]
        );
        assert!(plan.descs.iter().all(|d| d.len == 16));
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn lanes_on_different_numa_nodes_get_separate_slabs() {
        let template = vec![slot(8, 0, 0), slot(8, 0, 1)];
        let addrs: Vec<Vec<u64>> = (0..2)
            .map(|b| vec![0x1000 + b * 8, 0x5000 + b * 8])
            .collect();
        let table = build(2, &addrs);

        let plan = plan_transfer(&template, 2, &table).expect("plan");
        assert_eq!(plan.bytes_per_numa.get(&NumaNode(0)), Some(&16));
        assert_eq!(plan.bytes_per_numa.get(&NumaNode(1)), Some(&16));
        assert_eq!(plan.descs.len(), 2);
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn runs_with_different_strides_do_not_merge() {
        // K and V are base-adjacent at block 0, but V's blocks live far
        // apart, so its stride disagrees with K's: grouping must keep them
        // separate or block 1 would be copied from the wrong place.
        let template = vec![slot(4, 4, 0)];
        let addrs = vec![vec![0x1000, 0x1004], vec![0x1010, 0x2000]];
        let table = build(2, &addrs);
        assert_eq!(table.runs_per_lane, vec![1, 1]);

        let plan = plan_transfer(&template, 2, &table).expect("plan");
        assert_eq!(plan.descs.len(), 4);
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn runs_with_different_block_ranges_do_not_merge() {
        // Block 1's V segment sits right after block 0's K segment, but it
        // covers a different block range than K's two-block run.
        let template = vec![slot(4, 4, 0)];
        let addrs = vec![vec![0x1000, 0x5000], vec![0x1008, 0x1004]];
        let table = build(2, &addrs);
        assert_eq!(table.runs_per_lane, vec![1, 2]);

        let plan = plan_transfer(&template, 2, &table).expect("plan");
        assert_plan_replays(&addrs, &plan);
    }

    #[test]
    fn plan_rejects_inconsistent_tables() {
        let template = vec![slot(4, 0, 0)];

        // Lane count mismatch.
        let table = SegmentRunTable {
            runs_per_lane: vec![1, 1],
            base: vec![0, 0],
            stride: vec![0, 0],
            count: vec![1, 1],
        };
        assert!(plan_transfer(&template, 1, &table).is_err());

        // Block coverage mismatch.
        let table = SegmentRunTable {
            runs_per_lane: vec![1],
            base: vec![0x1000],
            stride: vec![4],
            count: vec![2],
        };
        assert!(plan_transfer(&template, 3, &table).is_err());

        // Parallel arrays out of sync.
        let table = SegmentRunTable {
            runs_per_lane: vec![1],
            base: vec![0x1000],
            stride: vec![],
            count: vec![1],
        };
        assert!(plan_transfer(&template, 1, &table).is_err());

        // Empty run.
        let table = SegmentRunTable {
            runs_per_lane: vec![1],
            base: vec![0x1000],
            stride: vec![0],
            count: vec![0],
        };
        assert!(plan_transfer(&template, 0, &table).is_err());

        // Slot without a K segment.
        let table = SegmentRunTable {
            runs_per_lane: vec![1],
            base: vec![0x1000],
            stride: vec![0],
            count: vec![1],
        };
        assert!(plan_transfer(&[slot(0, 4, 0)], 1, &table).is_err());

        // Empty template for a non-empty prefix.
        assert!(plan_transfer(&[], 1, &SegmentRunTable::default()).is_err());
    }
}
