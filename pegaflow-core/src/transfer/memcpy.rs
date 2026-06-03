//! DMA copy-engine backend: coalesce contiguous copies, then one directional
//! `cuMemcpyAsync` per merged range.
//!
//! `cuMemcpyBatchAsync*` has shown driver instability on some CUDA stacks, so
//! explicit directional copies are used instead; they preserve stream ordering
//! without relying on that batch path.

use std::sync::Arc;

use cudarc::driver::{CudaStream, sys};

use super::{CopyDesc, TransferBackend};

/// A run of input copies that are contiguous on both the device and the host
/// side, submitted as a single `cuMemcpyAsync`.
struct Merged {
    device: u64,
    host: *mut u8,
    size: usize,
}

/// Coalesce copies that are adjacent on both sides into larger ranges. Inputs
/// were already bounds-checked when their offsets were built, and merging only
/// joins adjacent validated ranges, so no further bounds check is needed.
fn merge(copies: &[CopyDesc]) -> Vec<Merged> {
    let mut out = Vec::with_capacity(copies.len());
    let mut i = 0;
    while i < copies.len() {
        let start = copies[i];
        let mut size = start.size;
        let mut j = i + 1;
        while j < copies.len() {
            let next = copies[j];
            let device_contiguous = start.device + size as u64 == next.device;
            // SAFETY: pointer arithmetic used only for an address-equality check;
            // the result is never dereferenced.
            let host_contiguous = unsafe { start.host.add(size) } == next.host;
            if device_contiguous && host_contiguous {
                size += next.size;
                j += 1;
            } else {
                break;
            }
        }
        out.push(Merged {
            device: start.device,
            host: start.host,
            size,
        });
        i = j;
    }
    out
}

/// DMA copy-engine transfer backend.
pub struct MemcpyBackend;

impl TransferBackend for MemcpyBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        for m in merge(copies) {
            // SAFETY: `m.device` is a valid device address and `m.host` is pinned
            // host memory kept alive by the caller until the stream is
            // synchronized. The merged range stays within the validated bounds.
            let result = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    m.device,
                    m.host as *const std::ffi::c_void,
                    m.size,
                    stream.cu_stream(),
                )
            };
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoDAsync failed: {result:?}"));
            }
        }
        Ok(())
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        for m in merge(copies) {
            // SAFETY: same invariants as `h2d`, reversed direction.
            let result = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    m.host as *mut std::ffi::c_void,
                    m.device,
                    m.size,
                    stream.cu_stream(),
                )
            };
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoHAsync failed: {result:?}"));
            }
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "direct"
    }
}
