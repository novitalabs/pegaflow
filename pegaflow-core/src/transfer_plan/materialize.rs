use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::Arc;

use pegaflow_transfer::TransferDesc;
use pegaflow_transfer_wire::TransferPlan;

use pegaflow_common::NumaNode;

use crate::backing::AllocateFn;
use crate::pinned_pool::PinnedAllocation;

/// NUMA slabs and per-chunk offsets used after RDMA completes. Safe to hold across `.await`.
pub(crate) struct RebuildContext {
    pub slabs: Vec<LocalSlab>,
    pub chunk_local_offsets: Vec<usize>,
}

pub(crate) struct MaterializeOutput {
    pub rebuild: RebuildContext,
    pub rdma_descs: Vec<TransferDesc>,
}

pub(crate) struct LocalSlab {
    pub numa: NumaNode,
    pub allocation: Arc<PinnedAllocation>,
    pub length: usize,
}

pub(crate) fn materialize_transfer_plan(
    plan: &TransferPlan,
    allocate_fn: &AllocateFn,
) -> Result<MaterializeOutput, String> {
    if plan.block_hashes.is_empty() {
        return Ok(MaterializeOutput {
            rebuild: RebuildContext {
                slabs: vec![],
                chunk_local_offsets: vec![],
            },
            rdma_descs: vec![],
        });
    }

    let mut bytes_per_numa: BTreeMap<u32, usize> = BTreeMap::new();
    for chunk in &plan.remote_chunks {
        let length = usize::try_from(chunk.length)
            .map_err(|_| format!("remote chunk length exceeds usize: {}", chunk.length))?;
        *bytes_per_numa.entry(chunk.numa_node).or_default() += length;
    }

    let mut slabs = Vec::with_capacity(bytes_per_numa.len());
    for (&numa_id, &total) in &bytes_per_numa {
        let numa = NumaNode(numa_id);
        let allocation = allocate_fn(total as u64, Some(numa))
            .ok_or_else(|| format!("failed to allocate {total} bytes for NUMA slab on {numa}"))?;
        slabs.push(LocalSlab {
            numa,
            allocation,
            length: total,
        });
    }

    let mut numa_cursor: BTreeMap<u32, usize> = BTreeMap::new();
    let mut chunk_local_offsets = Vec::with_capacity(plan.remote_chunks.len());
    let mut rdma_descs = Vec::with_capacity(plan.remote_chunks.len());

    for chunk in &plan.remote_chunks {
        let length = usize::try_from(chunk.length)
            .map_err(|_| format!("remote chunk length exceeds usize: {}", chunk.length))?;
        let local_offset = *numa_cursor.entry(chunk.numa_node).or_default();
        numa_cursor.insert(chunk.numa_node, local_offset + length);
        chunk_local_offsets.push(local_offset);

        let slab = slab_for_numa(&slabs, chunk.numa_node)?;
        let local_ptr = slab_ptr_at(slab, local_offset)?;
        let remote_ptr = NonNull::new(chunk.base_ptr as *mut u8)
            .ok_or_else(|| "remote chunk base ptr is null".to_string())?;

        rdma_descs.push(TransferDesc {
            local_ptr,
            remote_ptr,
            len: length,
        });
    }

    Ok(MaterializeOutput {
        rebuild: RebuildContext {
            slabs,
            chunk_local_offsets,
        },
        rdma_descs,
    })
}

fn slab_for_numa(slabs: &[LocalSlab], numa_node: u32) -> Result<&LocalSlab, String> {
    slabs
        .iter()
        .find(|slab| slab.numa.0 == numa_node)
        .ok_or_else(|| format!("missing local slab for NUMA node {numa_node}"))
}

fn slab_ptr_at(slab: &LocalSlab, offset: usize) -> Result<NonNull<u8>, String> {
    if offset > slab.length {
        return Err(format!(
            "offset {offset} exceeds NUMA slab length {}",
            slab.length
        ));
    }
    let ptr = unsafe { slab.allocation.as_non_null().as_ptr().add(offset) };
    NonNull::new(ptr).ok_or_else(|| "computed slab pointer is null".to_string())
}

pub(crate) fn chunk_ptr_at(
    rebuild: &RebuildContext,
    plan: &TransferPlan,
    chunk_idx: u32,
    offset_in_chunk: usize,
) -> Result<(NonNull<u8>, Arc<PinnedAllocation>), String> {
    let chunk = plan
        .remote_chunks
        .get(chunk_idx as usize)
        .ok_or_else(|| format!("missing remote chunk {chunk_idx}"))?;
    let local_chunk_offset = rebuild
        .chunk_local_offsets
        .get(chunk_idx as usize)
        .copied()
        .ok_or_else(|| format!("missing local offset for chunk {chunk_idx}"))?;
    let slab = slab_for_numa(&rebuild.slabs, chunk.numa_node)?;
    let offset = local_chunk_offset + offset_in_chunk;
    Ok((slab_ptr_at(slab, offset)?, Arc::clone(&slab.allocation)))
}
