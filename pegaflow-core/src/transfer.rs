use cudarc::driver::CudaStream;

use crate::KVCacheRegistration;

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "Features `cuda-12` and `cuda-13` are mutually exclusive. \
     Use `--no-default-features --features cuda-13` for CUDA 13 support."
);

// ============================================================================
// Transfer Functions Module
//
// This module contains all GPU<->CPU memory transfer operations:
// - Low-level CUDA copy primitives (async)
// - Batched transfer optimization for contiguous memory ranges
// - Helper functions for offset/size calculations
// ============================================================================

/// Calculate the byte offset for a given block/segment combination.
pub(crate) fn segment_offset(
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

#[cfg(not(feature = "cuda-13"))]
/// Copy data from GPU to CPU asynchronously on the provided stream
fn copy_gpu_to_cpu_async(
    gpu_base_ptr: u64,
    offset: usize,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

    let src_ptr = gpu_base_ptr + offset as u64;

    // SAFETY: src_ptr is gpu_base_ptr + offset, within a valid GPU allocation.
    // dst_ptr points to pinned CPU memory of sufficient size. The stream is valid.
    unsafe {
        let result = sys::cuMemcpyDtoHAsync_v2(
            dst_ptr as *mut std::ffi::c_void,
            src_ptr,
            size,
            stream.cu_stream(),
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoHAsync failed: {:?}", result));
        }
    }

    Ok(())
}

#[cfg(feature = "cuda-13")]
fn copy_gpu_to_cpu_batch_async(
    transfers: &[(usize, *mut u8)],
    segment_size: usize,
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

    for &(offset, dst_ptr) in transfers {
        let src_ptr = registration
            .data_ptr
            .checked_add(offset as u64)
            .ok_or_else(|| format!("GPU source pointer overflow for offset {offset}"))?;
        dsts.push(dst_ptr as usize as sys::CUdeviceptr);
        srcs.push(src_ptr);
        sizes.push(segment_size);
    }

    // SAFETY: The source GPU addresses are derived from a live registration and checked
    // for overflow. Destination pointers refer to pinned host memory owned by the caller
    // and remain valid until the caller synchronizes the stream. All copies are
    // independent, so the batch API's lack of intra-batch ordering is acceptable.
    unsafe {
        let result = sys::cuMemcpyBatchAsync_v2(
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
            return Err(format!("cuMemcpyBatchAsync_v2 DtoH failed: {:?}", result));
        }
    }

    Ok(())
}

#[cfg(not(feature = "cuda-13"))]
/// Copy data from CPU to GPU asynchronously on the provided stream
fn copy_cpu_to_gpu_async(
    gpu_base_ptr: u64,
    offset: usize,
    cpu_buffer: &[u8],
    size: usize,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

    if cpu_buffer.len() < size {
        return Err(format!(
            "CPU buffer too small: {} bytes, need {} bytes",
            cpu_buffer.len(),
            size
        ));
    }

    let dst_ptr = gpu_base_ptr + offset as u64;
    let src_ptr = cpu_buffer.as_ptr();

    // SAFETY: dst_ptr is gpu_base_ptr + offset, within a valid GPU allocation.
    // src_ptr is validated to have at least `size` bytes above. The stream is valid.
    unsafe {
        let result = sys::cuMemcpyHtoDAsync_v2(
            dst_ptr,
            src_ptr as *const std::ffi::c_void,
            size,
            stream.cu_stream(),
        );
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyHtoDAsync failed: {:?}", result));
        }
    }

    Ok(())
}

#[cfg(feature = "cuda-13")]
fn copy_cpu_to_gpu_batch_async(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
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

    for &(offset, src_ptr) in transfers {
        let dst_ptr = registration
            .data_ptr
            .checked_add(offset as u64)
            .ok_or_else(|| format!("GPU destination pointer overflow for offset {offset}"))?;
        dsts.push(dst_ptr);
        srcs.push(src_ptr as usize as sys::CUdeviceptr);
        sizes.push(segment_size);
    }

    // SAFETY: Destination GPU addresses are derived from a live registration and checked
    // for overflow. Source pointers refer to pinned host memory owned by the caller and
    // stay valid until the caller synchronizes the stream. Each transfer copies a disjoint
    // segment, so batch execution order does not matter.
    unsafe {
        let result = sys::cuMemcpyBatchAsync_v2(
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
            return Err(format!("cuMemcpyBatchAsync_v2 HtoD failed: {:?}", result));
        }
    }

    Ok(())
}

/// Batch copy segments from CPU to GPU by finding and merging contiguous ranges.
/// Returns the number of CUDA memcpy calls issued.
pub(crate) fn batch_copy_segments_to_gpu(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    let total_segments = transfers.len();
    if total_segments == 0 {
        return Ok(0);
    }

    #[cfg(feature = "cuda-13")]
    {
        copy_cpu_to_gpu_batch_async(transfers, segment_size, registration, stream)?;
        return Ok(1);
    }

    #[cfg(not(feature = "cuda-13"))]
    {
        let mut batch_count = 0;
        let mut i = 0;

        while i < total_segments {
            let (start_gpu_offset, start_cpu_ptr) = transfers[i];
            let mut count = 1;

            while i + count < total_segments {
                let (next_gpu_offset, next_cpu_ptr) = transfers[i + count];

                let expected_gpu_offset = start_gpu_offset + count * segment_size;
                // SAFETY: All cpu_ptr values in `transfers` point into the same contiguous
                // allocation. This arithmetic is used only for contiguity comparison.
                let expected_cpu_ptr = unsafe { start_cpu_ptr.add(count * segment_size) };

                let gpu_contiguous = next_gpu_offset == expected_gpu_offset;
                let cpu_contiguous = next_cpu_ptr == expected_cpu_ptr;

                if gpu_contiguous && cpu_contiguous {
                    count += 1;
                } else {
                    break;
                }
            }

            let total_size = segment_size
                .checked_mul(count)
                .ok_or_else(|| "batch_copy_segments_to_gpu: total_size overflow".to_string())?;

            // SAFETY: start_cpu_ptr points to valid, initialized pinned memory of at least
            // total_size bytes. The contiguity check above confirms the segments form a
            // single contiguous range. total_size is checked via checked_mul.
            let buffer = unsafe { std::slice::from_raw_parts(start_cpu_ptr, total_size) };
            copy_cpu_to_gpu_async(
                registration.data_ptr,
                start_gpu_offset,
                buffer,
                total_size,
                stream,
            )?;

            batch_count += 1;
            i += count;
        }

        Ok(batch_count)
    }
}

/// Batch copy segments from GPU to CPU by finding and merging contiguous ranges.
/// Returns the number of CUDA memcpy calls issued.
pub(crate) fn batch_copy_segments_from_gpu(
    transfers: &[(usize, *mut u8)], // (gpu_offset, cpu_dst_ptr)
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    let total_segments = transfers.len();
    if total_segments == 0 {
        return Ok(0);
    }

    #[cfg(feature = "cuda-13")]
    {
        if transfers.iter().any(|(offset, _)| {
            registration
                .size_bytes
                .checked_sub(segment_size)
                .is_none_or(|max_offset| *offset > max_offset)
        }) {
            return Err(
                "batch_copy_segments_from_gpu: transfer exceeds registered memory".to_string(),
            );
        }
        copy_gpu_to_cpu_batch_async(transfers, segment_size, registration, stream)?;
        return Ok(1);
    }

    #[cfg(not(feature = "cuda-13"))]
    {
        let mut batch_count = 0;
        let mut i = 0;

        while i < total_segments {
            let (start_gpu_offset, start_cpu_ptr) = transfers[i];
            let mut count = 1;

            while i + count < total_segments {
                let (next_gpu_offset, next_cpu_ptr) = transfers[i + count];

                let expected_gpu_offset = start_gpu_offset + count * segment_size;
                // SAFETY: All cpu_ptr values in `transfers` point into the same contiguous
                // allocation. This arithmetic is used only for contiguity comparison.
                let expected_cpu_ptr = unsafe { start_cpu_ptr.add(count * segment_size) };

                let gpu_contiguous = next_gpu_offset == expected_gpu_offset;
                let cpu_contiguous = next_cpu_ptr == expected_cpu_ptr;

                if gpu_contiguous && cpu_contiguous {
                    count += 1;
                } else {
                    break;
                }
            }

            let total_size = segment_size
                .checked_mul(count)
                .ok_or_else(|| "batch_copy_segments_from_gpu: total_size overflow".to_string())?;

            copy_gpu_to_cpu_async(
                registration.data_ptr,
                start_gpu_offset,
                start_cpu_ptr,
                total_size,
                stream,
            )?;

            batch_count += 1;
            i += count;
        }

        Ok(batch_count)
    }
}
