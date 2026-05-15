use cudarc::driver::CudaStream;

use crate::KVCacheRegistration;

// ============================================================================
// Transfer Functions Module
//
// GPU<->CPU memory transfer operations:
// - Coalesce contiguous (gpu_offset, cpu_ptr) pairs into larger ranges to
//   reduce the number of CUDA driver calls.
// - Submit each merged range with cuMemcpyHtoDAsync_v2 / cuMemcpyDtoHAsync_v2
//   (explicit direction). cuMemcpyBatchAsync* has shown driver instability on
//   some CUDA stacks, while explicit directional copies preserve the same
//   stream ordering without relying on that batch API path.
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

fn copy_cpu_to_gpu_merged_async(
    transfers: &[MergedCpuToGpuTransfer],
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

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

        // SAFETY: dst_ptr range was checked when the merged transfer was built.
        // cpu_ptr is pinned host memory owned by the caller and stays valid
        // until the caller synchronizes the stream.
        let result = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst_ptr,
                transfer.cpu_ptr as *const std::ffi::c_void,
                transfer.size,
                stream.cu_stream(),
            )
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyHtoDAsync failed: {result:?}"));
        }
    }
    Ok(())
}

fn copy_gpu_to_cpu_merged_async(
    transfers: &[MergedGpuToCpuTransfer],
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<(), String> {
    use cudarc::driver::sys;

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

        // SAFETY: src_ptr range was checked when the merged transfer was built.
        // cpu_ptr is pinned host memory owned by the caller and stays valid
        // until the caller synchronizes the stream.
        let result = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                transfer.cpu_ptr as *mut std::ffi::c_void,
                src_ptr,
                transfer.size,
                stream.cu_stream(),
            )
        };
        if result != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoHAsync failed: {result:?}"));
        }
    }
    Ok(())
}

fn validate_merged_transfer_bounds(
    gpu_offset: usize,
    size: usize,
    registration: &KVCacheRegistration,
    context: &str,
) -> Result<(), String> {
    if gpu_offset
        .checked_add(size)
        .is_none_or(|end| end > registration.size_bytes)
    {
        return Err(format!("{context}: transfer exceeds registered memory"));
    }

    Ok(())
}

fn merge_cpu_to_gpu_transfers(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
    registration: &KVCacheRegistration,
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
        validate_merged_transfer_bounds(
            start_gpu_offset,
            size,
            registration,
            "merge_cpu_to_gpu_transfers",
        )?;

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
    registration: &KVCacheRegistration,
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
        validate_merged_transfer_bounds(
            start_gpu_offset,
            size,
            registration,
            "merge_gpu_to_cpu_transfers",
        )?;

        merged.push(MergedGpuToCpuTransfer {
            gpu_offset: start_gpu_offset,
            cpu_ptr: start_cpu_ptr,
            size,
        });
        i += count;
    }

    Ok(merged)
}

/// Copy segments from CPU to GPU. Contiguous segments are merged into larger
/// ranges and each merged range is submitted as one `cuMemcpyHtoDAsync_v2`.
/// Returns the number of merged ranges submitted.
pub fn batch_copy_segments_to_gpu(
    transfers: &[(usize, *const u8)],
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    if transfers.is_empty() {
        return Ok(0);
    }

    let merged_transfers = merge_cpu_to_gpu_transfers(transfers, segment_size, registration)?;
    copy_cpu_to_gpu_merged_async(&merged_transfers, registration, stream)?;
    Ok(merged_transfers.len())
}

/// Copy segments from GPU to CPU. Contiguous segments are merged into larger
/// ranges and each merged range is submitted as one `cuMemcpyDtoHAsync_v2`.
/// Returns the number of merged ranges submitted.
pub fn batch_copy_segments_from_gpu(
    transfers: &[(usize, *mut u8)], // (gpu_offset, cpu_dst_ptr)
    segment_size: usize,
    registration: &KVCacheRegistration,
    stream: &CudaStream,
) -> Result<usize, String> {
    if transfers.is_empty() {
        return Ok(0);
    }

    let merged_transfers = merge_gpu_to_cpu_transfers(transfers, segment_size, registration)?;
    copy_gpu_to_cpu_merged_async(&merged_transfers, registration, stream)?;
    Ok(merged_transfers.len())
}
