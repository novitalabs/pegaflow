//! Strided host<->device copies: rows of `width` bytes over up to
//! [`MAX_DIMS`] dimensions, each with its own pitch on the device and in
//! mapped pinned host memory.
//!
//! Every call is one kernel launch: each warp copies a chunk of a row through
//! the mapped host address, or, for rows narrower than a chunk, a chunk's
//! worth of consecutive rows of the first dimension, so fragments cost no
//! driver call and narrow rows still keep every lane's loads in flight, at
//! the price of occupying SMs during the copy.

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
/// Most dimensions of one copy.
pub(crate) const MAX_DIMS: usize = 4;
/// u64 words per copy descriptor.
const DESC_WORDS: usize = 4 + 3 * MAX_DIMS;

/// Descriptor `i` is `descs[16 i ..][..16]`: `dst, src, width, first`, then
/// `count, dst_pitch, src_pitch` per dimension, the first fastest and unused
/// ones a single step. `first` is the index of its first chunk; chunks of a
/// row are `CHUNK` bytes, the last one shorter, and a chunk of a copy whose
/// rows are narrower than `CHUNK` is `CHUNK / width` consecutive rows of the
/// first dimension (see [`chunks_of`]). Warps take chunks in a grid stride
/// and find their copy by binary search on `first`.
fn kernel_src() -> String {
    format!(
        r#"
#define CHUNK {CHUNK}ULL
#define WORDS {DESC_WORDS}
#define DIMS {MAX_DIMS}
extern "C" __global__ void pega_strided_copy(const unsigned long long* __restrict__ d,
                                             unsigned long long n, unsigned long long chunks) {{
    const unsigned lane = threadIdx.x & 31;
    const unsigned long long warps = ((unsigned long long)gridDim.x * blockDim.x) >> 5;
    unsigned long long c = ((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    for (; c < chunks; c += warps) {{
        unsigned long long lo = 0, hi = n;
        while (hi - lo > 1) {{
            unsigned long long mid = (lo + hi) >> 1;
            if (d[WORDS * mid + 3] <= c) lo = mid; else hi = mid;
        }}
        const unsigned long long* r = d + WORDS * lo;
        const unsigned long long width = r[2];
        const unsigned long long k = c - r[3];
        const unsigned long long d0 = r[4];
        if (width < CHUNK && d0 > 1) {{
            // Narrow rows: `per` consecutive rows of dimension 0 per chunk.
            unsigned long long per = CHUNK / width;
            if (per > d0) per = d0;
            const unsigned long long items = (d0 + per - 1) / per;
            unsigned long long row = k / items;
            const unsigned long long first = (k - row * items) * per;
            const unsigned long long rows = d0 - first < per ? d0 - first : per;
            const unsigned long long dp = r[5], sp = r[6];
            unsigned long long dst_at = r[0] + first * dp, src_at = r[1] + first * sp;
            #pragma unroll
            for (int i = 1; i < DIMS; ++i) {{
                const unsigned long long* m = r + 4 + 3 * i;
                if (m[0] == 1) break;
                const unsigned long long q = row / m[0];
                const unsigned long long x = row - q * m[0];
                row = q;
                dst_at += x * m[1];
                src_at += x * m[2];
            }}
            if (((dst_at | src_at | width | dp | sp) & 15ULL) == 0ULL) {{
                const unsigned upr = (unsigned)(width >> 4);
                const unsigned units = (unsigned)rows * upr;
                unsigned j = lane;
                for (; j + 96 < units; j += 128) {{
                    int4 v[4];
                    int4* to[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {{
                        const unsigned u = j + 32 * t, q = u / upr;
                        const unsigned long long off = (unsigned long long)(u - q * upr) << 4;
                        v[t] = *(const int4*)(src_at + q * sp + off);
                        to[t] = (int4*)(dst_at + q * dp + off);
                    }}
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) __stcs(to[t], v[t]);
                }}
                for (; j < units; j += 32) {{
                    const unsigned q = j / upr;
                    const unsigned long long off = (unsigned long long)(j - q * upr) << 4;
                    __stcs((int4*)(dst_at + q * dp + off), *(const int4*)(src_at + q * sp + off));
                }}
            }} else {{
                for (unsigned long long q = 0; q < rows; ++q) {{
                    char* __restrict__ dst = (char*)(dst_at + q * dp);
                    const char* __restrict__ src = (const char*)(src_at + q * sp);
                    for (unsigned long long j = lane; j < width; j += 32) dst[j] = src[j];
                }}
            }}
            continue;
        }}
        const unsigned long long per_row = (width + CHUNK - 1) / CHUNK;
        unsigned long long row = k / per_row;
        const unsigned long long at = (k - row * per_row) * CHUNK;
        const unsigned long long size = width - at < CHUNK ? width - at : CHUNK;
        unsigned long long dst_at = r[0] + at, src_at = r[1] + at;
        #pragma unroll
        for (int i = 0; i < DIMS; ++i) {{
            const unsigned long long* m = r + 4 + 3 * i;
            // Used dimensions come first.
            if (m[0] == 1) break;
            const unsigned long long q = row / m[0];
            const unsigned long long x = row - q * m[0];
            row = q;
            dst_at += x * m[1];
            src_at += x * m[2];
        }}
        char* __restrict__ dst = (char*)dst_at;
        const char* __restrict__ src = (const char*)src_at;
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

/// One dimension of a [`StridedCopy`]: `count` steps of `device_pitch` and
/// `host_pitch` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dim {
    pub(crate) count: u64,
    pub(crate) device_pitch: u64,
    pub(crate) host_pitch: u64,
}

/// One strided copy between device memory and mapped pinned host memory,
/// which `host_device` addresses from the device: a row of `width` bytes at
/// every index of `dims`, the first fastest. No dimensions is one row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StridedCopy {
    device: u64,
    host_device: u64,
    width: u64,
    dims: [Dim; MAX_DIMS],
    /// Dimensions used, from the first.
    ndims: usize,
}

impl StridedCopy {
    /// A copy of `width` bytes per row over `dims`, at most [`MAX_DIMS`].
    pub(crate) fn new(device: u64, host_device: u64, width: u64, dims: &[Dim]) -> Self {
        let mut c = Self {
            device,
            host_device,
            width,
            dims: [Dim {
                count: 1,
                device_pitch: 0,
                host_pitch: 0,
            }; MAX_DIMS],
            ndims: 0,
        };
        for d in dims.iter().filter(|d| d.count != 1) {
            c.dims[c.ndims] = *d;
            c.ndims += 1;
        }
        c
    }

    fn rows(&self) -> u64 {
        self.dims[..self.ndims].iter().map(|d| d.count).product()
    }

    /// The same bytes with leading dimensions whose rows are back to back on
    /// both sides folded into the row.
    fn flattened(mut self) -> Self {
        while self.ndims > 0
            && self.dims[0].device_pitch == self.width
            && self.dims[0].host_pitch == self.width
        {
            self.width *= self.dims[0].count;
            self.dims.copy_within(1..self.ndims, 0);
            self.ndims -= 1;
            self.dims[self.ndims] = Dim {
                count: 1,
                device_pitch: 0,
                host_pitch: 0,
            };
        }
        self
    }

    /// `next` continues this single-row copy on both sides.
    fn joins(&self, next: &Self) -> bool {
        self.ndims == 0
            && next.ndims == 0
            && self.device + self.width == next.device
            && self.host_device + self.width == next.host_device
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Enqueue `copies` as one launch; the caller synchronizes `stream`.
    pub(crate) fn copy(
        &mut self,
        copies: &[StridedCopy],
        dir: Direction,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let h2d = dir == Direction::HostToDevice;
        let mut descs: Vec<u64> = Vec::new();
        let mut chunks = 0u64;
        for c in joined(copies) {
            let (dst, src) = if h2d {
                (c.device, c.host_device)
            } else {
                (c.host_device, c.device)
            };
            descs.extend_from_slice(&[dst, src, c.width, chunks]);
            for d in &c.dims {
                let (dp, sp) = if h2d {
                    (d.device_pitch, d.host_pitch)
                } else {
                    (d.host_pitch, d.device_pitch)
                };
                descs.extend_from_slice(&[d.count, dp, sp]);
            }
            chunks += chunks_of(&c);
        }
        if chunks == 0 {
            return Ok(());
        }
        let n = (descs.len() / DESC_WORDS) as u64;
        if self.scratch.as_ref().is_none_or(|s| s.len() < descs.len()) {
            self.scratch = Some(
                stream
                    .alloc_zeros::<u64>(descs.len().next_power_of_two())
                    .map_err(|e| format!("descriptor alloc failed: {e:?}"))?,
            );
        }
        let scratch = self.scratch.as_mut().expect("allocated above");
        // The source is pageable, so the driver consumes it before returning,
        // as in `KernelBackend`.
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
        let grid = chunks.div_ceil(warps_per_block).clamp(1, max_grid) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&self.func);
        builder.arg(&*scratch).arg(&n).arg(&chunks);
        // SAFETY: the kernel reads `n` descriptors from `scratch` and copies
        // validated rows between device memory and mapped pinned memory,
        // which the caller keeps alive until it synchronizes the stream.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e:?}"))?;
        Ok(())
    }
}

/// Work items of `c` as the kernel cuts it: rows narrower than [`CHUNK`]
/// go `CHUNK / width` consecutive rows of the first dimension to an item,
/// wider ones a [`CHUNK`] of one row.
fn chunks_of(c: &StridedCopy) -> u64 {
    let d0 = if c.ndims > 0 { c.dims[0].count } else { 1 };
    if c.width < CHUNK && d0 > 1 {
        let per = (CHUNK / c.width).min(d0);
        d0.div_ceil(per) * (c.rows() / d0)
    } else {
        c.width.div_ceil(CHUNK) * c.rows()
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
