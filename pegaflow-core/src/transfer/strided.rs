//! Strided host<->device copies: `rows` rows of `width` bytes, `device_pitch`
//! apart on the device and `host_pitch` apart in mapped pinned host memory.
//!
//! Every batch is one kernel launch: each warp copies a chunk of a row through
//! the mapped host address, so fragments cost no driver call and no shape
//! needs a heuristic, at the price of occupying SMs during the copy.

use std::sync::Arc;

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

/// Bytes one warp copies per work item.
const CHUNK: u64 = 4096;
const BLOCK_DIM: u32 = 256;
const MAX_GRID: u64 = 65_535;
/// u64 words per copy descriptor.
const DESC_WORDS: usize = 7;

/// Descriptor `i` is `descs[7 i ..][..7]`: `dst, src, width, rows, dst_pitch,
/// src_pitch, first`, where `first` is the index of its first chunk; chunks of
/// a row are `CHUNK` bytes, the last one shorter. Warps take chunks in a grid
/// stride and find their copy by binary search on `first`.
fn kernel_src() -> String {
    format!(
        r#"
#define CHUNK {CHUNK}ULL
extern "C" __global__ void pega_strided_copy(const unsigned long long* __restrict__ d,
                                             unsigned long long n, unsigned long long chunks) {{
    const unsigned lane = threadIdx.x & 31;
    const unsigned long long warps = ((unsigned long long)gridDim.x * blockDim.x) >> 5;
    unsigned long long c = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (; c < chunks; c += warps) {{
        unsigned long long lo = 0, hi = n;
        while (hi - lo > 1) {{
            unsigned long long mid = (lo + hi) >> 1;
            if (d[7 * mid + 6] <= c) lo = mid; else hi = mid;
        }}
        const unsigned long long* r = d + 7 * lo;
        const unsigned long long width = r[2];
        const unsigned long long per_row = (width + CHUNK - 1) / CHUNK;
        const unsigned long long k = c - r[6];
        const unsigned long long row = k / per_row;
        const unsigned long long at = (k - row * per_row) * CHUNK;
        const unsigned long long size = width - at < CHUNK ? width - at : CHUNK;
        char* __restrict__ dst = (char*)(r[0] + row * r[4] + at);
        const char* __restrict__ src = (const char*)(r[1] + row * r[5] + at);
        if (((((unsigned long long)dst) | ((unsigned long long)src)) & 15ULL) == 0ULL) {{
            const unsigned long long n16 = size >> 4;
            int4* __restrict__ d4 = (int4*)dst;
            const int4* __restrict__ s4 = (const int4*)src;
            unsigned long long j = lane;
            // Four loads in flight per lane: reads of mapped host memory are
            // latency-bound. Streaming stores keep the copy from evicting L2,
            // which speeds up scattered writes into host memory.
            for (; j + 96 < n16; j += 128) {{
                const int4 a = s4[j], b = s4[j + 32], c = s4[j + 64], e = s4[j + 96];
                __stcs(d4 + j, a); __stcs(d4 + j + 32, b);
                __stcs(d4 + j + 64, c); __stcs(d4 + j + 96, e);
            }}
            for (; j < n16; j += 32) {{
                __stcs(d4 + j, s4[j]);
            }}
            for (unsigned long long j = (n16 << 4) + lane; j < size; j += 32) {{
                dst[j] = src[j];
            }}
        }} else {{
            for (unsigned long long j = lane; j < size; j += 32) {{
                dst[j] = src[j];
            }}
        }}
    }}
}}
"#
    )
}

/// One strided copy between device memory and mapped pinned host memory,
/// which `host_device` addresses from the device. Rows are `device_pitch` and
/// `host_pitch` apart; both are ignored when `rows == 1`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StridedCopy {
    pub(crate) device: u64,
    pub(crate) host_device: u64,
    pub(crate) width: u64,
    pub(crate) rows: u32,
    pub(crate) device_pitch: u64,
    pub(crate) host_pitch: u64,
}

impl StridedCopy {
    /// The same bytes as one row when the rows are back to back on both sides.
    fn flattened(self) -> Self {
        if self.rows > 1 && self.device_pitch == self.width && self.host_pitch == self.width {
            Self {
                width: self.width * u64::from(self.rows),
                rows: 1,
                ..self
            }
        } else {
            self
        }
    }

    /// `next` continues this single-row copy on both sides.
    fn joins(&self, next: &Self) -> bool {
        self.rows == 1
            && next.rows == 1
            && self.device + self.width == next.device
            && self.host_device + self.width == next.host_device
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    HostToDevice,
    DeviceToHost,
}

/// Moves batches of [`StridedCopy`] on one device; holds the compiled kernel
/// and a device buffer for descriptors, so callers serialize use per device.
pub(crate) struct StridedCopier {
    func: CudaFunction,
    scratch: Option<CudaSlice<u64>>,
    /// Most blocks a device -> host launch uses: one per SM.
    d2h_grid: u64,
}

impl StridedCopier {
    /// Compile and load the copy kernel into `ctx`.
    pub(crate) fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = compile_ptx(kernel_src()).map_err(|e| format!("nvrtc compile failed: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module failed: {e:?}"))?;
        let func = module
            .load_function("pega_strided_copy")
            .map_err(|e| format!("load_function failed: {e:?}"))?;
        let sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|e| format!("SM count query failed: {e:?}"))?;
        Ok(Self {
            func,
            scratch: None,
            d2h_grid: u64::try_from(sms).unwrap_or(1).max(1),
        })
    }

    /// Enqueue `batches`, one launch each in order, behind one descriptor
    /// upload; the caller synchronizes `stream`.
    pub(crate) fn copy(
        &mut self,
        batches: &[&[StridedCopy]],
        dir: Direction,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        // (descriptor offset, descriptors, chunks) per launch.
        let mut launches = Vec::with_capacity(batches.len());
        let mut descs: Vec<u64> = Vec::new();
        for batch in batches {
            let at = descs.len();
            let mut chunks = 0u64;
            for c in joined(batch) {
                let (dst, src, dst_pitch, src_pitch) = match dir {
                    Direction::HostToDevice => {
                        (c.device, c.host_device, c.device_pitch, c.host_pitch)
                    }
                    Direction::DeviceToHost => {
                        (c.host_device, c.device, c.host_pitch, c.device_pitch)
                    }
                };
                descs.extend_from_slice(&[
                    dst,
                    src,
                    c.width,
                    u64::from(c.rows),
                    dst_pitch,
                    src_pitch,
                    chunks,
                ]);
                chunks += c.width.div_ceil(CHUNK) * u64::from(c.rows);
            }
            if chunks > 0 {
                launches.push((at, ((descs.len() - at) / DESC_WORDS) as u64, chunks));
            }
        }
        if launches.is_empty() {
            return Ok(());
        }
        if self.scratch.as_ref().is_none_or(|s| s.len() < descs.len()) {
            self.scratch = Some(
                stream
                    .alloc_zeros::<u64>(descs.len().next_power_of_two())
                    .map_err(|e| format!("descriptor alloc failed: {e:?}"))?,
            );
        }
        let scratch = self.scratch.as_mut().expect("allocated above");
        // One upload for every launch of the transfer. The source is pageable,
        // so the driver consumes it before returning, as in `KernelBackend`.
        stream
            .memcpy_htod(&descs, scratch)
            .map_err(|e| format!("descriptor upload failed: {e:?}"))?;
        let warps_per_block = u64::from(BLOCK_DIM / 32);
        // Writes into host memory saturate PCIe with one block per SM; more
        // warps only interleave scattered host rows and lose bandwidth.
        // Reads of host memory are latency-bound and want every warp.
        let max_grid = match dir {
            Direction::HostToDevice => MAX_GRID,
            Direction::DeviceToHost => self.d2h_grid,
        };
        for (at, n, chunks) in launches {
            let grid = chunks.div_ceil(warps_per_block).clamp(1, max_grid) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            let view = scratch.slice(at..);
            let mut builder = stream.launch_builder(&self.func);
            builder.arg(&view).arg(&n).arg(&chunks);
            // SAFETY: the kernel reads `n` descriptors from `view` and copies
            // validated rows between device memory and mapped pinned memory,
            // which the caller keeps alive until it synchronizes the stream.
            unsafe { builder.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e:?}"))?;
        }
        Ok(())
    }
}

/// `copies` with back-to-back rows flattened and adjacent single rows joined.
fn joined(copies: &[StridedCopy]) -> Vec<StridedCopy> {
    let mut out: Vec<StridedCopy> = Vec::with_capacity(copies.len());
    for c in copies.iter().map(|c| c.flattened()) {
        match out.last_mut() {
            Some(prev) if prev.joins(&c) => prev.width += c.width,
            _ => out.push(c),
        }
    }
    out
}
