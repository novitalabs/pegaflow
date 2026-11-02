//! Strided host<->device copies: `rows` rows of `width` bytes, `device_pitch`
//! apart on the device and `host_pitch` apart in mapped pinned host memory.
//!
//! [`StridedMode::Dma`] issues one `cuMemcpy2DAsync` per copy (one 1D copy when
//! the rows are back to back); the copy engines run it beside compute, but each
//! call costs a few microseconds, so small copies are call-bound.
//! [`StridedMode::Kernel`] moves every copy in one launch: each warp copies a
//! chunk of a row through the mapped host address, so fragments cost no driver
//! call but the copy occupies SMs. [`StridedMode::Auto`] loads with the kernel,
//! which reads mapped host memory at the PCIe rate for every shape, and saves
//! copies of at least `DMA_MIN_BYTES` in rows of at least `DMA_MIN_WIDTH` on
//! the copy engines, which write host memory faster than the kernel, and the
//! rest with the kernel.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use cudarc::nvrtc::compile_ptx;

/// Bytes one warp copies per work item.
const CHUNK: u64 = 4096;
/// Copies at least this large go to the copy engines in `Auto`.
const DMA_MIN_BYTES: u64 = 256 << 10;
/// ... if their rows are at least this wide (2D copies of narrow rows are slow).
const DMA_MIN_WIDTH: u64 = 2 << 10;
/// cuMemcpy2D rejects pitches at or above this; such copies go row by row.
const MAX_PITCH: u64 = 1 << 31;
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
            // Four loads in flight per lane: reads of mapped host memory are latency-bound.
            for (; j + 96 < n16; j += 128) {{
                const int4 a = s4[j], b = s4[j + 32], c = s4[j + 64], e = s4[j + 96];
                d4[j] = a; d4[j + 32] = b; d4[j + 64] = c; d4[j + 96] = e;
            }}
            for (; j < n16; j += 32) {{
                d4[j] = s4[j];
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

/// How a [`StridedCopier`] moves copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StridedMode {
    /// One `cuMemcpy2DAsync` (or 1D copy) per copy.
    Dma,
    /// One kernel launch for every copy.
    Kernel,
    /// Loads in one launch; saves of large copies of wide rows on the copy
    /// engines, the rest in one launch.
    #[default]
    Auto,
}

impl std::str::FromStr for StridedMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "dma" => Ok(Self::Dma),
            "kernel" => Ok(Self::Kernel),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("unknown copy mode {s:?} (dma, kernel, auto)")),
        }
    }
}

/// One strided copy between device memory and mapped pinned host memory.
///
/// `host` and `host_device` address the same byte, from the host and from the
/// device. Rows are `device_pitch` and `host_pitch` apart; both are ignored
/// when `rows == 1`.
#[derive(Debug, Clone, Copy)]
pub struct StridedCopy {
    pub device: u64,
    pub host: *mut u8,
    pub host_device: u64,
    pub width: u64,
    pub rows: u32,
    pub device_pitch: u64,
    pub host_pitch: u64,
}

// SAFETY: `host` points into pinned memory the caller keeps alive until the
// stream synchronizes; only CUDA dereferences it.
unsafe impl Send for StridedCopy {}

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

    fn bytes(&self) -> u64 {
        self.width * u64::from(self.rows)
    }

    /// `next` continues this single-row copy on both sides.
    fn joins(&self, next: &Self) -> bool {
        self.rows == 1
            && next.rows == 1
            && self.device + self.width == next.device
            && self.host_device + self.width == next.host_device
    }
}

/// A copy `Auto` sends to the copy engines when saving.
fn large(c: &StridedCopy) -> bool {
    c.bytes() >= DMA_MIN_BYTES && c.width >= DMA_MIN_WIDTH
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Dir {
    HostToDevice,
    DeviceToHost,
}

/// Moves batches of [`StridedCopy`] on one device; holds the compiled kernel
/// and a reusable descriptor buffer, so callers serialize use per device.
pub struct StridedCopier {
    mode: StridedMode,
    func: Option<CudaFunction>,
    scratch: Option<CudaSlice<u64>>,
}

impl StridedCopier {
    /// A copier for `ctx`; the kernel is compiled unless `mode` is `Dma`.
    pub fn new(ctx: &Arc<CudaContext>, mode: StridedMode) -> Result<Self, String> {
        let func = if mode == StridedMode::Dma {
            None
        } else {
            let ptx =
                compile_ptx(kernel_src()).map_err(|e| format!("nvrtc compile failed: {e:?}"))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| format!("load_module failed: {e:?}"))?;
            Some(
                module
                    .load_function("pega_strided_copy")
                    .map_err(|e| format!("load_function failed: {e:?}"))?,
            )
        };
        Ok(Self {
            mode,
            func,
            scratch: None,
        })
    }

    pub fn mode(&self) -> StridedMode {
        self.mode
    }

    /// Enqueue host -> device copies; the caller synchronizes `stream`.
    pub fn h2d(&mut self, copies: &[StridedCopy], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, Dir::HostToDevice, stream)
    }

    /// Enqueue device -> host copies; the caller synchronizes `stream`.
    pub fn d2h(&mut self, copies: &[StridedCopy], stream: &Arc<CudaStream>) -> Result<(), String> {
        self.submit(copies, Dir::DeviceToHost, stream)
    }

    fn submit(
        &mut self,
        copies: &[StridedCopy],
        dir: Dir,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        let mut dma: Vec<StridedCopy> = Vec::new();
        let mut kernel: Vec<StridedCopy> = Vec::new();
        for c in copies.iter().map(|c| c.flattened()) {
            let to_dma = match self.mode {
                StridedMode::Dma => true,
                StridedMode::Kernel => false,
                StridedMode::Auto => dir == Dir::DeviceToHost && large(&c),
            };
            let out = if to_dma { &mut dma } else { &mut kernel };
            match out.last_mut() {
                Some(prev) if prev.joins(&c) => prev.width += c.width,
                _ => out.push(c),
            }
        }
        // Small copies `Auto` set aside may join into a large one.
        if self.mode == StridedMode::Auto && dir == Dir::DeviceToHost {
            kernel.retain(|c| {
                if large(c) {
                    dma.push(*c);
                }
                !large(c)
            });
        }
        for c in &dma {
            dma_copy(c, dir, stream)?;
        }
        self.launch(&kernel, dir, stream)
    }

    fn launch(
        &mut self,
        copies: &[StridedCopy],
        dir: Dir,
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        if copies.is_empty() {
            return Ok(());
        }
        let func = self
            .func
            .as_ref()
            .ok_or("kernel copy requested in dma mode")?;
        let mut descs = Vec::with_capacity(copies.len() * DESC_WORDS);
        let mut chunks = 0u64;
        for c in copies {
            let (dst, src, dst_pitch, src_pitch) = match dir {
                Dir::HostToDevice => (c.device, c.host_device, c.device_pitch, c.host_pitch),
                Dir::DeviceToHost => (c.host_device, c.device, c.host_pitch, c.device_pitch),
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
        if self.scratch.as_ref().is_none_or(|s| s.len() < descs.len()) {
            self.scratch = Some(
                stream
                    .alloc_zeros::<u64>(descs.len().next_power_of_two())
                    .map_err(|e| format!("descriptor alloc failed: {e:?}"))?,
            );
        }
        let scratch = self.scratch.as_mut().expect("allocated above");
        // The source is pageable: the driver stages it before returning.
        stream
            .memcpy_htod(&descs, scratch)
            .map_err(|e| format!("descriptor upload failed: {e:?}"))?;
        let warps_per_block = u64::from(BLOCK_DIM / 32);
        let grid = chunks.div_ceil(warps_per_block).clamp(1, MAX_GRID) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let n = copies.len() as u64;
        let mut builder = stream.launch_builder(func);
        builder.arg(&*scratch).arg(&n).arg(&chunks);
        // SAFETY: the kernel reads `n` descriptors of `scratch` and copies
        // validated rows between device memory and mapped pinned memory, which
        // the caller keeps alive until it synchronizes the stream.
        unsafe { builder.launch(cfg) }.map_err(|e| format!("kernel launch failed: {e:?}"))?;
        Ok(())
    }
}

fn dma_copy(c: &StridedCopy, dir: Dir, stream: &CudaStream) -> Result<(), String> {
    if c.rows == 1 {
        // SAFETY: both sides were validated by the caller; the host side is
        // pinned memory kept alive until the stream synchronizes.
        let r = unsafe {
            match dir {
                Dir::HostToDevice => sys::cuMemcpyHtoDAsync_v2(
                    c.device,
                    c.host.cast_const().cast(),
                    c.width as usize,
                    stream.cu_stream(),
                ),
                Dir::DeviceToHost => sys::cuMemcpyDtoHAsync_v2(
                    c.host.cast(),
                    c.device,
                    c.width as usize,
                    stream.cu_stream(),
                ),
            }
        };
        return check(r, "cuMemcpyAsync");
    }
    if c.device_pitch >= MAX_PITCH || c.host_pitch >= MAX_PITCH {
        for r in 0..u64::from(c.rows) {
            let row = StridedCopy {
                device: c.device + r * c.device_pitch,
                // SAFETY: row `r` lies inside the copy, validated by the caller.
                host: unsafe { c.host.add((r * c.host_pitch) as usize) },
                host_device: c.host_device + r * c.host_pitch,
                rows: 1,
                ..*c
            };
            dma_copy(&row, dir, stream)?;
        }
        return Ok(());
    }
    use sys::CUmemorytype::{CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST};
    let (src_type, src_pitch, dst_type, dst_pitch) = match dir {
        Dir::DeviceToHost => (
            CU_MEMORYTYPE_DEVICE,
            c.device_pitch,
            CU_MEMORYTYPE_HOST,
            c.host_pitch,
        ),
        Dir::HostToDevice => (
            CU_MEMORYTYPE_HOST,
            c.host_pitch,
            CU_MEMORYTYPE_DEVICE,
            c.device_pitch,
        ),
    };
    // CUDA reads only the pointer fields matching each side's memory type.
    // The struct holds enums, so it is built field by field, not zeroed.
    let p = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: src_type,
        srcHost: c.host.cast_const().cast(),
        srcDevice: c.device,
        srcArray: std::ptr::null_mut(),
        srcPitch: src_pitch as usize,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: dst_type,
        dstHost: c.host.cast(),
        dstDevice: c.device,
        dstArray: std::ptr::null_mut(),
        dstPitch: dst_pitch as usize,
        WidthInBytes: c.width as usize,
        Height: c.rows as usize,
    };
    // SAFETY: as above.
    let r = unsafe { sys::cuMemcpy2DAsync_v2(&p, stream.cu_stream()) };
    check(r, "cuMemcpy2DAsync")
}

fn check(r: sys::CUresult, what: &str) -> Result<(), String> {
    if r == sys::cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {r:?}"))
    }
}
