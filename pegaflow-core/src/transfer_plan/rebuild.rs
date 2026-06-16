use std::ptr::NonNull;
use std::sync::Arc;

use pegaflow_transfer_wire::{SegmentSchema, TransferPlan};

use pegaflow_common::NumaNode;

use crate::backing::PrefetchResult;
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::pinned_pool::PinnedAllocation;

use super::materialize::RebuildContext;

/// Sentinel for "this (block, slot) has no NUMA slab assigned yet".
const UNSET_SLAB: u32 = u32::MAX;

/// Precomputed landing location of one remote chunk inside the local NUMA
/// slabs. Resolving NUMA -> slab once per chunk keeps it out of the per-cell
/// fill loop (millions of iterations for large batches).
struct ChunkLoc {
    slab_idx: u32,
    slab_base: NonNull<u8>,
    slab_len: usize,
    base_offset: usize,
}

impl ChunkLoc {
    /// Validate that a contiguous run of `span` bytes starting at
    /// `offset_in_chunk` fits in the slab, and return the run's host base
    /// pointer. Checking the whole run once keeps the per-cell pointer math in
    /// the fill loop branch-free (the loop runs `block*slot*segment` times).
    fn run_base(&self, offset_in_chunk: usize, span: usize) -> Result<*mut u8, String> {
        let start = self
            .base_offset
            .checked_add(offset_in_chunk)
            .ok_or("chunk offset overflow")?;
        let end = start.checked_add(span).ok_or("segment span overflow")?;
        if end > self.slab_len {
            return Err(format!(
                "segment run {start}..{end} exceeds slab length {}",
                self.slab_len
            ));
        }
        // SAFETY: `start <= end <= slab_len`, so the pointer stays within the slab.
        Ok(unsafe { self.slab_base.as_ptr().add(start) })
    }
}

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

    // Resolve each remote chunk to its local slab up front. Rejects unknown
    // NUMA loudly: the per-slot NUMA source map must not silently land on an
    // arbitrary node.
    let chunk_locs = resolve_chunk_locs(plan, rebuild)?;

    // Two flat scatter buffers replace a block x slot x segment `Vec` tree
    // (~block*slot allocations). `seg_ptrs` holds each segment's host pointer;
    // `slot_slab` records the slab backing each (block, slot). The owning `Arc`
    // is cloned only once per segment, at final construction below.
    let mut seg_ptrs: Vec<Option<NonNull<u8>>> =
        vec![None; block_count * slot_count * segment_count];
    let mut slot_slab: Vec<u32> = vec![UNSET_SLAB; block_count * slot_count];

    for placement in &plan.placements {
        let loc = &chunk_locs[placement.chunk_idx as usize];
        let slot_idx = placement.slot_idx as usize;
        let segment_idx = placement.segment_idx as usize;
        let seg = &plan.slot_schemas[slot_idx].segments[segment_idx];
        let seg_len = seg.bytes;
        let stride = seg.block_stride;
        let count = placement.block_count as usize;

        // Bounds-check the whole run once, then step the base pointer per cell.
        let span = count
            .saturating_sub(1)
            .checked_mul(stride)
            .and_then(|s| s.checked_add(seg_len))
            .ok_or("segment span overflow")?;
        let run_base = loc.run_base(placement.offset_in_chunk, span)?;
        let start_block = placement.block_start as usize;

        for i in 0..count {
            let block_idx = start_block + i;
            // SAFETY: `i * stride < span` and the run is bounded within the slab
            // by `run_base` above, so this pointer stays inside the allocation.
            let ptr = unsafe { NonNull::new_unchecked(run_base.add(i * stride)) };
            seg_ptrs[(block_idx * slot_count + slot_idx) * segment_count + segment_idx] = Some(ptr);

            let slot_pos = block_idx * slot_count + slot_idx;
            match slot_slab[slot_pos] {
                UNSET_SLAB => slot_slab[slot_pos] = loc.slab_idx,
                existing if existing != loc.slab_idx => {
                    return Err(format!(
                        "block {block_idx} slot {slot_idx} NUMA mismatch: {} vs {}",
                        rebuild.slabs[existing as usize].numa,
                        rebuild.slabs[loc.slab_idx as usize].numa,
                    ));
                }
                _ => {}
            }
        }
    }

    let mut result = Vec::with_capacity(block_count);
    for (block_idx, hash) in plan.block_hashes.iter().enumerate() {
        let mut slots = Vec::with_capacity(slot_count);
        let mut slot_numas = Vec::with_capacity(slot_count);
        for slot_idx in 0..slot_count {
            let slab_idx = slot_slab[block_idx * slot_count + slot_idx];
            if slab_idx == UNSET_SLAB {
                return Err(format!(
                    "block {block_idx} slot {slot_idx} missing during rebuild"
                ));
            }
            let slab = &rebuild.slabs[slab_idx as usize];
            let seg_base = (block_idx * slot_count + slot_idx) * segment_count;
            slots.push(build_raw_block(
                &seg_ptrs[seg_base..seg_base + segment_count],
                &plan.slot_schemas[slot_idx].segments,
                &slab.allocation,
                block_idx,
                slot_idx,
            )?);
            slot_numas.push(slab.numa);
        }
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        result.push((
            key,
            Arc::new(SealedBlock::from_slots_with_numas(slots, slot_numas)),
        ));
    }

    Ok(result)
}

fn resolve_chunk_locs(
    plan: &TransferPlan,
    rebuild: &RebuildContext,
) -> Result<Vec<ChunkLoc>, String> {
    let mut locs = Vec::with_capacity(plan.remote_chunks.len());
    for (chunk_idx, chunk) in plan.remote_chunks.iter().enumerate() {
        let numa = NumaNode(chunk.numa_node);
        if numa.is_unknown() {
            return Err(format!(
                "remote chunk {chunk_idx} has unknown NUMA for rebuild"
            ));
        }
        let slab_idx = rebuild
            .slabs
            .iter()
            .position(|slab| slab.numa == numa)
            .ok_or_else(|| format!("missing local slab for NUMA {numa}"))?;
        let base_offset = *rebuild
            .chunk_local_offsets
            .get(chunk_idx)
            .ok_or_else(|| format!("missing local offset for chunk {chunk_idx}"))?;
        let slab = &rebuild.slabs[slab_idx];
        locs.push(ChunkLoc {
            slab_idx: slab_idx as u32,
            slab_base: slab.allocation.as_non_null(),
            slab_len: slab.length,
            base_offset,
        });
    }
    Ok(locs)
}

/// Build one slot's [`RawBlock`] from its segment pointers, cloning the slab's
/// owning `Arc` once per segment. Split K/V (`segment_count == 2`) is the common
/// case and avoids a transient `Vec`.
fn build_raw_block(
    seg_ptrs: &[Option<NonNull<u8>>],
    seg_schemas: &[SegmentSchema],
    allocation: &Arc<PinnedAllocation>,
    block_idx: usize,
    slot_idx: usize,
) -> Result<RawBlock, String> {
    let make = |g: usize| -> Result<Segment, String> {
        let ptr = seg_ptrs[g].ok_or_else(|| {
            format!("block {block_idx} slot {slot_idx} segment {g} missing during rebuild")
        })?;
        Ok(Segment::new(
            ptr,
            seg_schemas[g].bytes,
            Arc::clone(allocation),
        ))
    };
    match seg_ptrs.len() {
        1 => Ok(RawBlock::single_segment(make(0)?)),
        2 => Ok(RawBlock::two_segments(make(0)?, make(1)?)),
        n => {
            let mut segments = Vec::with_capacity(n);
            for g in 0..n {
                segments.push(make(g)?);
            }
            Ok(RawBlock::new(segments))
        }
    }
}
