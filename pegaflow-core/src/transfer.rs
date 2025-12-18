use cudarc::driver::CudaStream;
use tracing::{debug, instrument};

use crate::KVCacheRegistration;

// ============================================================================
// Transfer Functions Module
//
// This module contains all GPU<->CPU memory transfer operations:
// - Low-level CUDA copy primitives (sync/async)
// - Batched transfer optimization for contiguous memory ranges
// - Strided transfer support for non-contiguous layouts
// - Helper functions for offset/size calculations
// ============================================================================

/// Calculate the byte offset for a given block/segment combination.
#[instrument(level = "debug", skip(registration))]
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

/// Check if the layout is contiguous (single segment or no stride)
pub(crate) fn is_contiguous_layout(registration: &KVCacheRegistration) -> bool {
    registration.segments <= 1 || registration.kv_stride_bytes == registration.bytes_per_block
}

/// Copy data from GPU to CPU synchronously
#[instrument(level = "debug", skip(cpu_buffer), fields(offset, size), err)]
pub(crate) fn copy_gpu_to_cpu(
    gpu_base_ptr: u64,
    offset: usize,
    cpu_buffer: &mut [u8],
    size: usize,
) -> Result<(), String> {
    use cudarc::driver::sys;

    let src_ptr = gpu_base_ptr + offset as u64;
    let dst_ptr = cpu_buffer.as_mut_ptr();

    unsafe {
        // Use synchronous copy for simplicity
        let result = sys::cuMemcpyDtoH_v2(dst_ptr as *mut std::ffi::c_void, src_ptr, size);
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoH failed: {:?}", result));
        }
    }

    Ok(())
}

/// Copy data from CPU to GPU asynchronously on the provided stream
#[instrument(level = "debug", skip(cpu_buffer, stream), fields(offset, size), err)]
pub(crate) fn copy_cpu_to_gpu_async(
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


/// Batch copy segments from CPU to GPU by finding and merging contiguous ranges
pub(crate) fn batch_copy_segments_to_gpu(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<(), String> {
    let total_segments = transfers.len();
    if total_segments == 0 {
        debug!("CPU->GPU batch copy: 0 segments -> 0 batches");
        return Ok(());
    }

    let mut batch_count = 0;
    let mut i = 0;
    let mut gpu_discontinuous_count = 0;
    let mut cpu_discontinuous_count = 0;
    let mut both_discontinuous_count = 0;

    while i < total_segments {
        let (start_gpu_offset, start_cpu_ptr) = transfers[i];
        let mut count = 1;

        while i + count < total_segments {
            let (next_gpu_offset, next_cpu_ptr) = transfers[i + count];

            let expected_gpu_offset = start_gpu_offset + count * segment_size;
            let expected_cpu_ptr = unsafe { start_cpu_ptr.add(count * segment_size) };

            let gpu_contiguous = next_gpu_offset == expected_gpu_offset;
            let cpu_contiguous = next_cpu_ptr == expected_cpu_ptr;

            if gpu_contiguous && cpu_contiguous {
                count += 1;
            } else {
                // Track why we couldn't merge
                if !gpu_contiguous && !cpu_contiguous {
                    both_discontinuous_count += 1;
                } else if !gpu_contiguous {
                    gpu_discontinuous_count += 1;
                } else {
                    cpu_discontinuous_count += 1;
                }
                break;
            }
        }

        let total_size = segment_size
            .checked_mul(count)
            .ok_or_else(|| "batch_copy_segments_to_gpu: total_size overflow".to_string())?;

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

    debug!(
        "CPU->GPU batch copy: {} segments -> {} batches ({}x reduction)",
        total_segments,
        batch_count,
        total_segments as f32 / batch_count as f32
    );

    let total_breaks = gpu_discontinuous_count + cpu_discontinuous_count + both_discontinuous_count;
    if total_breaks > 0 {
        debug!(
            "CPU->GPU batch merge breaks: {} total (GPU discontinuous: {}, CPU discontinuous: {}, both: {})",
            total_breaks,
            gpu_discontinuous_count,
            cpu_discontinuous_count,
            both_discontinuous_count
        );
    }

    Ok(())
}

/// Copy a block from GPU to CPU, handling both contiguous and strided layouts
pub(crate) fn copy_block_gpu_to_cpu(
    registration: &KVCacheRegistration,
    block_idx: usize,
    dst_ptr: *mut u8,
) -> Result<(), String> {
    if is_contiguous_layout(registration) {
        let size = registration.block_size_bytes;
        let offset = segment_offset(registration, block_idx, 0)?;
        let buffer = unsafe { std::slice::from_raw_parts_mut(dst_ptr, size) };
        copy_gpu_to_cpu(registration.data_ptr, offset, buffer, size)
    } else {
        copy_gpu_to_cpu_strided(registration, block_idx, dst_ptr)
    }
}

/// Copy a block from CPU to GPU, handling both contiguous and strided layouts
pub(crate) fn copy_block_cpu_to_gpu(
    registration: &KVCacheRegistration,
    block_idx: usize,
    src_ptr: *const u8,
    stream: &CudaStream,
) -> Result<(), String> {
    if is_contiguous_layout(registration) {
        let size = registration.block_size_bytes;
        let offset = segment_offset(registration, block_idx, 0)?;
        let buffer = unsafe { std::slice::from_raw_parts(src_ptr, size) };
        copy_cpu_to_gpu_async(registration.data_ptr, offset, buffer, size, stream)
    } else {
        copy_cpu_to_gpu_strided(registration, block_idx, src_ptr, stream)
    }
}

/// Copy strided block from GPU to CPU (each segment separately)
fn copy_gpu_to_cpu_strided(
    registration: &KVCacheRegistration,
    block_idx: usize,
    dst_ptr: *mut u8,
) -> Result<(), String> {
    // Copy each segment separately using regular cuMemcpy
    for segment_idx in 0..registration.segments {
        let src_offset = segment_offset(registration, block_idx, segment_idx)?;
        let dst_offset = segment_idx * registration.bytes_per_block;
        let dst_segment_ptr = unsafe { dst_ptr.add(dst_offset) };
        let buffer = unsafe {
            std::slice::from_raw_parts_mut(dst_segment_ptr, registration.bytes_per_block)
        };

        copy_gpu_to_cpu(
            registration.data_ptr,
            src_offset,
            buffer,
            registration.bytes_per_block,
        )?;
    }
    Ok(())
}

/// Copy strided block from CPU to GPU (each segment separately)
fn copy_cpu_to_gpu_strided(
    registration: &KVCacheRegistration,
    block_idx: usize,
    src_ptr: *const u8,
    stream: &CudaStream,
) -> Result<(), String> {
    // Copy each segment separately using regular cuMemcpy
    for segment_idx in 0..registration.segments {
        let dst_offset = segment_offset(registration, block_idx, segment_idx)?;
        let src_offset = segment_idx * registration.bytes_per_block;
        let src_segment_ptr = unsafe { src_ptr.add(src_offset) };
        let buffer =
            unsafe { std::slice::from_raw_parts(src_segment_ptr, registration.bytes_per_block) };

        copy_cpu_to_gpu_async(
            registration.data_ptr,
            dst_offset,
            buffer,
            registration.bytes_per_block,
            stream,
        )?;
    }
    Ok(())
}
