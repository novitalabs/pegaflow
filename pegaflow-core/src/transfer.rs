use cudarc::driver::CudaStream;

use crate::KVCacheRegistration;

// ============================================================================
// Transfer Functions Module
//
// This module contains all GPU<->CPU memory transfer operations:
// - Low-level CUDA copy primitives (async)
// - Batched transfer optimization for contiguous memory ranges
// - Helper functions for offset/size calculations
// ============================================================================

#[derive(Clone)]
struct MergedCpuToGpuTransfer {
    gpu_offset: usize,
    cpu_ptr: *const u8,
    size: usize,
}

#[derive(Clone)]
struct MergedGpuToCpuTransfer {
    gpu_offset: usize,
    cpu_ptr: *mut u8,
    size: usize,
}

/// Calculate the byte offset for a given block/segment combination.
pub fn segment_offset(
    registration: &KVCacheRegistration,
    block_idx: usize,
    segment_idx: usize,
) -> Result<usize, String> {
    if segment_idx >= registration.segments {
        return Err("Segment index out of range".to_string());
    }

    let base = block_idx
        .checked_mul(registration.bytes_per_block)
        .ok_or_else(|| "Block offset overflow".to_string())?;

    let segment_offset = segment_idx
        .checked_mul(registration.kv_stride_bytes)
        .ok_or_else(|| "Segment offset overflow".to_string())?;

    let offset = base
        .checked_add(segment_offset)
        .ok_or_else(|| "Combined offset overflow".to_string())?;

    if offset + registration.bytes_per_block > registration.size_bytes {
        return Err(format!(
            "Block {} segment {} exceeds registered memory (offset {}, size {}, limit {})",
            block_idx, segment_idx, offset, registration.bytes_per_block, registration.size_bytes
        ));
    }

    Ok(offset)
}

fn copy_gpu_to_cpu_batch_async(
    transfers: &[MergedGpuToCpuTransfer],
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

    let mut dsts = Vec::with_capacity(transfers.len());
    let mut srcs = Vec::with_capacity(transfers.len());
    let mut sizes = Vec::with_capacity(transfers.len());
    let mut attrs = [sys::CUmemcpyAttributes {
        srcAccessOrder: sys::CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        srcLocHint: sys::CUmemLocation {
            type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        dstLocHint: sys::CUmemLocation {
            type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        flags: 0,
    }];
    let mut attrs_idxs = [0usize];

    for transfer in transfers {
        let src_ptr = registration
            .data_ptr
            .checked_add(transfer.gpu_offset as u64)
            .ok_or_else(|| {
                format!(
                    "GPU source pointer overflow for offset {}",
                    transfer.gpu_offset
                )
            })?;
        dsts.push(transfer.cpu_ptr as usize as sys::CUdeviceptr);
        srcs.push(src_ptr);
        sizes.push(transfer.size);
    }

    // SAFETY: The source GPU addresses are derived from a live registration and checked
    // for overflow. Destination pointers refer to pinned host memory owned by the caller
    // and remain valid until the caller synchronizes the stream. Contiguous ranges have
    // been merged, so each entry is an independent non-overlapping copy.
    unsafe {
        let result = submit_batch_memcpy(
            dsts.as_mut_ptr(),
            srcs.as_mut_ptr(),
            sizes.as_mut_ptr(),
            transfers.len(),
            attrs.as_mut_ptr(),
            attrs_idxs.as_mut_ptr(),
            attrs.len(),
            stream.cu_stream(),
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyBatchAsync DtoH failed: {:?}", result));
        }
    }

    Ok(())
}

#[cfg(not(feature = "cuda-13"))]
#[allow(clippy::too_many_arguments)]
unsafe fn submit_batch_memcpy(
    dsts: *mut cudarc::driver::sys::CUdeviceptr,
    srcs: *mut cudarc::driver::sys::CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    attrs: *mut cudarc::driver::sys::CUmemcpyAttributes,
    attrs_idxs: *mut usize,
    num_attrs: usize,
    stream: cudarc::driver::sys::CUstream,
) -> cudarc::driver::sys::CUresult {
    unsafe {
        cudarc::driver::sys::cuMemcpyBatchAsync(
            dsts,
            srcs,
            sizes,
            count,
            attrs,
            attrs_idxs,
            num_attrs,
            std::ptr::null_mut(),
            stream,
        )
    }
}

#[cfg(feature = "cuda-13")]
#[allow(clippy::too_many_arguments)]
unsafe fn submit_batch_memcpy(
    dsts: *mut cudarc::driver::sys::CUdeviceptr,
    srcs: *mut cudarc::driver::sys::CUdeviceptr,
    sizes: *mut usize,
    count: usize,
    attrs: *mut cudarc::driver::sys::CUmemcpyAttributes,
    attrs_idxs: *mut usize,
    num_attrs: usize,
    stream: cudarc::driver::sys::CUstream,
) -> cudarc::driver::sys::CUresult {
    unsafe {
        cudarc::driver::sys::cuMemcpyBatchAsync_v2(
            dsts, srcs, sizes, count, attrs, attrs_idxs, num_attrs, stream,
        )
    }
}

fn copy_cpu_to_gpu_batch_async(
    transfers: &[MergedCpuToGpuTransfer],
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

    let mut dsts = Vec::with_capacity(transfers.len());
    let mut srcs = Vec::with_capacity(transfers.len());
    let mut sizes = Vec::with_capacity(transfers.len());
    let mut attrs = [sys::CUmemcpyAttributes {
        srcAccessOrder: sys::CUmemcpySrcAccessOrder_enum::CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        srcLocHint: sys::CUmemLocation {
            type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        dstLocHint: sys::CUmemLocation {
            type_: sys::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_INVALID,
            id: 0,
        },
        flags: 0,
    }];
    let mut attrs_idxs = [0usize];

    for transfer in transfers {
        let dst_ptr = registration
            .data_ptr
            .checked_add(transfer.gpu_offset as u64)
            .ok_or_else(|| {
                format!(
                    "GPU destination pointer overflow for offset {}",
                    transfer.gpu_offset
                )
            })?;
        dsts.push(dst_ptr);
        srcs.push(transfer.cpu_ptr as usize as sys::CUdeviceptr);
        sizes.push(transfer.size);
    }

    // SAFETY: Destination GPU addresses are derived from a live registration and checked
    // for overflow. Source pointers refer to pinned host memory owned by the caller and
    // stay valid until the caller synchronizes the stream. Each transfer copies a disjoint
    // segment, so batch execution order does not matter.
    unsafe {
        let result = submit_batch_memcpy(
            dsts.as_mut_ptr(),
            srcs.as_mut_ptr(),
            sizes.as_mut_ptr(),
            transfers.len(),
            attrs.as_mut_ptr(),
            attrs_idxs.as_mut_ptr(),
            attrs.len(),
            stream.cu_stream(),
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyBatchAsync HtoD failed: {:?}", result));
        }
    }

    Ok(())
}

fn merge_cpu_to_gpu_transfers(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
) -> Result<Vec<MergedCpuToGpuTransfer>, String> {
    let mut merged = Vec::with_capacity(transfers.len());
    let mut i = 0;

    while i < transfers.len() {
        let (start_gpu_offset, start_cpu_ptr) = transfers[i];
        let mut count = 1usize;

        while i + count < transfers.len() {
            let (next_gpu_offset, next_cpu_ptr) = transfers[i + count];
            let expected_gpu_offset = start_gpu_offset + count * segment_size;
            // SAFETY: All cpu_ptr values in `transfers` point into the same contiguous
            // allocation. This arithmetic is used only for contiguity comparison.
            let expected_cpu_ptr = unsafe { start_cpu_ptr.add(count * segment_size) };

            if next_gpu_offset == expected_gpu_offset && next_cpu_ptr == expected_cpu_ptr {
                count += 1;
            } else {
                break;
            }
        }

        let size = segment_size
            .checked_mul(count)
            .ok_or_else(|| "merge_cpu_to_gpu_transfers: total_size overflow".to_string())?;

        merged.push(MergedCpuToGpuTransfer {
            gpu_offset: start_gpu_offset,
            cpu_ptr: start_cpu_ptr,
            size,
        });
        i += count;
    }

    Ok(merged)
}

fn merge_gpu_to_cpu_transfers(
    transfers: &[(usize, *mut u8)],
    segment_size: usize,
) -> Result<Vec<MergedGpuToCpuTransfer>, String> {
    let mut merged = Vec::with_capacity(transfers.len());
    let mut i = 0;

    while i < transfers.len() {
        let (start_gpu_offset, start_cpu_ptr) = transfers[i];
        let mut count = 1usize;

        while i + count < transfers.len() {
            let (next_gpu_offset, next_cpu_ptr) = transfers[i + count];
            let expected_gpu_offset = start_gpu_offset + count * segment_size;
            // SAFETY: All cpu_ptr values in `transfers` point into the same contiguous
            // allocation. This arithmetic is used only for contiguity comparison.
            let expected_cpu_ptr = unsafe { start_cpu_ptr.add(count * segment_size) };

            if next_gpu_offset == expected_gpu_offset && next_cpu_ptr == expected_cpu_ptr {
                count += 1;
            } else {
                break;
            }
        }

        let size = segment_size
            .checked_mul(count)
            .ok_or_else(|| "merge_gpu_to_cpu_transfers: total_size overflow".to_string())?;

        merged.push(MergedGpuToCpuTransfer {
            gpu_offset: start_gpu_offset,
            cpu_ptr: start_cpu_ptr,
            size,
        });
        i += count;
    }

    Ok(merged)
}

/// Batch copy segments from CPU to GPU by finding and merging contiguous ranges.
/// Returns the number of merged contiguous transfer ranges submitted.
pub fn batch_copy_segments_to_gpu(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    if transfers.is_empty() {
        return Ok(0);
    }

    let merged_transfers = merge_cpu_to_gpu_transfers(transfers, segment_size)?;
    copy_cpu_to_gpu_batch_async(&merged_transfers, registration, stream)?;
    Ok(merged_transfers.len())
}

/// Batch copy segments from GPU to CPU by finding and merging contiguous ranges.
/// Returns the number of merged contiguous transfer ranges submitted.
pub fn batch_copy_segments_from_gpu(
    transfers: &[(usize, *mut u8)], // (gpu_offset, cpu_dst_ptr)
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    if transfers.is_empty() {
        return Ok(0);
    }

    if transfers.iter().any(|(offset, _)| {
        registration
            .size_bytes
            .checked_sub(segment_size)
            .is_none_or(|max_offset| *offset > max_offset)
    }) {
        return Err("batch_copy_segments_from_gpu: transfer exceeds registered memory".to_string());
    }
    let merged_transfers = merge_gpu_to_cpu_transfers(transfers, segment_size)?;
    copy_gpu_to_cpu_batch_async(&merged_transfers, registration, stream)?;
    Ok(merged_transfers.len())
}
