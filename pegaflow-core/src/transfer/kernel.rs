//! Single-launch transfer backend.
//!
//! Instead of one `cuMemcpyAsync` per fragment, a single grid-strided kernel
//! copies the whole batch: one threadblock per descriptor reads from the source
//! address and writes to the destination address. For host<->device copies the
//! host side is mapped pinned memory, which the kernel dereferences directly
//! (zero-copy over PCIe). This collapses N driver submissions into one small
//! descriptor copy plus one launch, which wins when the batch is so fragmented
//! that per-call launch latency on the memcpy path dominates.

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use super::{CopyDesc, TransferBackend};

/// Descriptor layout is a flat `u64` array, 3 entries per copy:
/// `[dst, src, size, dst, src, size, ...]`. A flat array sidesteps any host/
/// device struct-layout mismatch. The 16-byte vectorized path is taken only
/// when both addresses are 16-byte aligned, otherwise a scalar loop is used.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void pega_batch_copy(const unsigned long long* __restrict__ descs, int n) {
    for (int i = blockIdx.x; i < n; i += gridDim.x) {
        char* dst = (char*)descs[3 * i + 0];
        const char* src = (const char*)descs[3 * i + 1];
        unsigned long long size = descs[3 * i + 2];
        if (((((unsigned long long)dst) | ((unsigned long long)src)) & 15ULL) == 0ULL) {
            unsigned long long n16 = size >> 4;
            for (unsigned long long j = threadIdx.x; j < n16; j += blockDim.x) {
                ((int4*)dst)[j] = ((const int4*)src)[j];
            }
            for (unsigned long long j = (n16 << 4) + threadIdx.x; j < size; j += blockDim.x) {
                dst[j] = src[j];
            }
        } else {
            for (unsigned long long j = threadIdx.x; j < size; j += blockDim.x) {
                dst[j] = src[j];
            }
        }
    }
}
"#;

const BLOCK_DIM: u32 = 256;
const MAX_GRID: u32 = 65535;

/// Single-launch transfer backend. Compiled once per worker; the descriptor
/// scratch buffer is reused across calls.
pub struct KernelBackend {
    func: CudaFunction,
    /// Device-side descriptor buffer, grow-only and reused. The owning worker
    /// processes tasks serially and synchronizes after each, so the previous
    /// task's kernel has consumed the buffer before the next call overwrites it.
    scratch: RefCell<Option<CudaSlice<u64>>>,
}

impl KernelBackend {
    /// Compile and load the copy kernel into `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc compile failed: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e:?}"))?;
        let func = module
            .load_function("pega_batch_copy")
            .map_err(|e| format!("load_function failed: {e:?}"))?;
        Ok(Self {
            func,
            scratch: RefCell::new(None),
        })
    }

    /// Enqueue the batch. `host_is_src` selects the direction: H2D reads the
    /// host address and writes the device address, D2H is reversed.
    fn submit(
        &self,
        copies: &[CopyDesc],
        host_is_src: bool,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        if copies.is_empty() {
            return Ok(());
        }

        let n = copies.len();
        let mut descs = Vec::with_capacity(n * 3);
        for c in copies {
            let (dst, src) = if host_is_src {
                (c.device, c.host_device)
            } else {
                (c.host_device, c.device)
            };
            descs.push(dst);
            descs.push(src);
            descs.push(c.size as u64);
        }

        let mut guard = self.scratch.borrow_mut();
        let needs_realloc = guard.as_ref().is_none_or(|s| s.len() < descs.len());
        if needs_realloc {
            *guard = Some(
                stream
                    .alloc_zeros::<u64>(descs.len())
                    .map_err(|e| format!("scratch alloc failed: {e:?}"))?,
            );
        }
        let scratch = guard.as_mut().expect("scratch initialized above");

        // Async H2D of the descriptor array. The source is pageable, so the
        // driver consumes it before returning — the local `descs` is safe to
        // drop. The device buffer outlives the launch because it lives in
        // `self.scratch`, not on this stack frame.
        stream
            .memcpy_htod(&descs, scratch)
            .map_err(|e| format!("descriptor upload failed: {e:?}"))?;

        let grid = (n as u32).clamp(1, MAX_GRID);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_arg = n as i32;
        let mut builder = stream.launch_builder(&self.func);
        builder.arg(&*scratch).arg(&n_arg);
        // SAFETY: the kernel reads exactly `n` descriptors from `scratch` (len
        // >= 3*n) and copies `size` bytes between the device and mapped pinned
        // host addresses, all kept valid by the caller until it synchronizes.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e:?}"))?;
        Ok(())
    }
}

impl TransferBackend for KernelBackend {
    fn h2d(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, true, stream)
    }

    fn d2h(&self, copies: &[CopyDesc], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, false, stream)
    }

    fn name(&self) -> &'static str {
        "kernel"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::MemcpyBackend;
    use cudarc::driver::sys;

    #[derive(Clone, Copy)]
    struct MappedHost {
        host: *mut u8,
        device: u64,
    }

    /// Allocate `len` bytes of mapped pinned host memory.
    fn alloc_mapped_host(len: usize) -> MappedHost {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { sys::cuMemHostAlloc(&mut p, len, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc");

        let mut device: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, p, 0) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer");

        MappedHost {
            host: p as *mut u8,
            device,
        }
    }

    fn alloc_device(len: usize) -> u64 {
        let mut d: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemAlloc_v2(&mut d, len) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
        d
    }

    /// Build `n` descriptors of `seg` bytes each over contiguous device/host
    /// regions. `host_base` is reinterpreted per direction by the backend.
    fn descs(device_base: u64, host_base: MappedHost, n: usize, seg: usize) -> Vec<CopyDesc> {
        (0..n)
            .map(|k| CopyDesc {
                device: device_base + (k * seg) as u64,
                // SAFETY: within the [host_base, host_base+n*seg) allocation.
                host: unsafe { host_base.host.add(k * seg) },
                host_device: host_base.device + (k * seg) as u64,
                size: seg,
            })
            .collect()
    }

    /// The kernel backend must move bytes identically to the direct backend, in
    /// both directions, over mapped pinned host memory.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn kernel_matches_direct_both_directions() {
        const N: usize = 257; // odd, > one grid of blocks would merge nothing
        const SEG: usize = 4096 + 16; // non-power-of-two, 16B-aligned
        let total = N * SEG;

        let ctx = CudaContext::new(0).expect("ctx");
        let stream = ctx.default_stream();
        let kernel = KernelBackend::new(&ctx).expect("kernel backend");
        let memcpy = MemcpyBackend;

        let host = alloc_mapped_host(total);
        let device = alloc_device(total);

        // Distinct pattern per byte so a misrouted copy is caught.
        let pattern: Vec<u8> = (0..total).map(|i| (i * 31 + 7) as u8).collect();
        let host_slice = unsafe { std::slice::from_raw_parts_mut(host.host, total) };

        let read_device = |out: &mut [u8]| {
            let r = unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut _, device, total) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "DtoH");
        };
        let clear_device = || {
            let zeros = vec![0u8; total];
            let r = unsafe { sys::cuMemcpyHtoD_v2(device, zeros.as_ptr() as *const _, total) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD clear");
        };

        let d = descs(device, host, N, SEG);

        // H2D via each backend, then verify device holds the pattern.
        for backend in [&kernel as &dyn TransferBackend, &memcpy] {
            host_slice.copy_from_slice(&pattern);
            clear_device();
            backend.h2d(&d, &stream).expect("h2d");
            stream.synchronize().expect("sync");

            let mut out = vec![0u8; total];
            read_device(&mut out);
            assert_eq!(out, pattern, "h2d mismatch for backend {}", backend.name());
        }

        // D2H via each backend: device holds the pattern, host is cleared first.
        for backend in [&kernel as &dyn TransferBackend, &memcpy] {
            let r = unsafe { sys::cuMemcpyHtoD_v2(device, pattern.as_ptr() as *const _, total) };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "HtoD seed");
            host_slice.fill(0);
            backend.d2h(&d, &stream).expect("d2h");
            stream.synchronize().expect("sync");
            assert_eq!(
                host_slice,
                &pattern[..],
                "d2h mismatch for backend {}",
                backend.name()
            );
        }

        unsafe {
            sys::cuMemFree_v2(device);
            sys::cuMemFreeHost(host.host as *mut _);
        }
    }
}
