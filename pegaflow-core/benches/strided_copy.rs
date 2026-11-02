//! Strided host<->device copies in the shape of v2 runs: `rows` rows of `width`
//! bytes, contiguous on the device and `host_pitch` apart in the object.
//!
//! Compares the [`StridedCopier`] modes object copies can run (one `cuMemcpy2DAsync`
//! per run, one strided kernel launch, auto), one `cuMemcpyAsync` per row, the
//! v1 copy kernel with one descriptor per row, and one contiguous copy of the
//! same bytes as the PCIe ceiling.
//!
//! `cargo bench --bench strided_copy` (needs a GPU; `STRIDED_COPY_GPU` picks one).

use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use pegaflow_core::transfer::{
    CopyDesc, KernelBackend, MemcpyBackend, StridedCopier, StridedCopy, StridedMode,
    TransferBackend,
};

const TOTAL: usize = 64 << 20;
const ITERS: usize = 5;

fn ok(r: sys::CUresult, what: &str) {
    assert_eq!(r, sys::cudaError_enum::CUDA_SUCCESS, "{what}");
}

struct Buffers {
    host: *mut u8,
    host_device: u64,
    device: u64,
}

impl Buffers {
    fn new(host_len: usize, device_len: usize) -> Self {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: out parameter of a fresh mapped pinned allocation.
        ok(
            unsafe { sys::cuMemHostAlloc(&mut p, host_len, sys::CU_MEMHOSTALLOC_DEVICEMAP) },
            "cuMemHostAlloc",
        );
        let mut host_device = 0;
        // SAFETY: `p` is mapped pinned memory of the current context.
        ok(
            unsafe { sys::cuMemHostGetDevicePointer_v2(&mut host_device, p, 0) },
            "cuMemHostGetDevicePointer",
        );
        // SAFETY: `p` holds `host_len` bytes.
        unsafe { std::ptr::write_bytes(p.cast::<u8>(), 7, host_len) };
        let mut device = 0;
        // SAFETY: out parameter of a fresh device allocation.
        ok(
            unsafe { sys::cuMemAlloc_v2(&mut device, device_len) },
            "cuMemAlloc",
        );
        Self {
            host: p.cast(),
            host_device,
            device,
        }
    }
}

impl Drop for Buffers {
    fn drop(&mut self) {
        // SAFETY: both were allocated above and are freed once.
        unsafe {
            sys::cuMemFree_v2(self.device);
            sys::cuMemFreeHost(self.host.cast());
        }
    }
}

/// `TOTAL` bytes as runs of `rows` rows of `width` bytes; rows are
/// `device_pitch` and `host_pitch` apart, runs follow each other.
#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    width: usize,
    rows: usize,
    device_pitch: usize,
    host_pitch: usize,
}

impl Shape {
    fn runs(&self) -> usize {
        TOTAL / (self.width * self.rows)
    }

    fn host_len(&self) -> usize {
        self.runs() * self.rows * self.host_pitch
    }

    fn device_len(&self) -> usize {
        self.runs() * self.rows * self.device_pitch
    }

    /// Device and host offset of row `r` of run `k`.
    fn row(&self, k: usize, r: usize) -> (usize, usize) {
        let i = k * self.rows + r;
        (i * self.device_pitch, i * self.host_pitch)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    H2D,
    D2H,
}

fn runs(b: &Buffers, s: &Shape) -> Vec<StridedCopy> {
    (0..s.runs())
        .map(|k| {
            let (d, h) = s.row(k, 0);
            StridedCopy {
                device: b.device + d as u64,
                // SAFETY: the run lies inside the host buffer.
                host: unsafe { b.host.add(h) },
                host_device: b.host_device + h as u64,
                width: s.width as u64,
                rows: s.rows as u32,
                device_pitch: s.device_pitch as u64,
                host_pitch: s.host_pitch as u64,
            }
        })
        .collect()
}

fn descs(b: &Buffers, s: &Shape) -> Vec<CopyDesc> {
    let mut out = Vec::with_capacity(s.runs() * s.rows);
    for k in 0..s.runs() {
        for r in 0..s.rows {
            let (d, h) = s.row(k, r);
            out.push(CopyDesc {
                device: b.device + d as u64,
                // SAFETY: the row lies inside the host buffer.
                host: unsafe { b.host.add(h) },
                host_device: b.host_device + h as u64,
                size: s.width,
                device_allocation: 0,
                host_allocation: 0,
            });
        }
    }
    out
}

fn contiguous(b: &Buffers, dir: Dir, stream: &CudaStream) {
    // SAFETY: TOTAL bytes fit both buffers.
    let r = unsafe {
        match dir {
            Dir::H2D => {
                sys::cuMemcpyHtoDAsync_v2(b.device, b.host.cast(), TOTAL, stream.cu_stream())
            }
            Dir::D2H => {
                sys::cuMemcpyDtoHAsync_v2(b.host.cast(), b.device, TOTAL, stream.cu_stream())
            }
        }
    };
    ok(r, "contiguous copy");
}

/// Median GB/s over `ITERS` timed runs after one warmup.
fn time(stream: &Arc<CudaStream>, mut f: impl FnMut()) -> f64 {
    f();
    stream.synchronize().expect("sync");
    let mut t: Vec<f64> = (0..ITERS)
        .map(|_| {
            let start = Instant::now();
            f();
            stream.synchronize().expect("sync");
            start.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    TOTAL as f64 / t[ITERS / 2] / 1e9
}

fn main() {
    let gpu = std::env::var("STRIDED_COPY_GPU").map_or(0, |g| g.parse().expect("GPU ordinal"));
    let ctx = CudaContext::new(gpu).expect("CUDA context");
    ctx.bind_to_thread().expect("bind");
    let stream = ctx.new_stream().expect("stream");
    let kernel = KernelBackend::new(&ctx).expect("kernel backend");
    let memcpy = MemcpyBackend;
    let mut modes: Vec<StridedCopier> = [StridedMode::Dma, StridedMode::Kernel, StridedMode::Auto]
        .into_iter()
        .map(|m| StridedCopier::new(&ctx, m).expect("strided copier"))
        .collect();

    let shape = |name, width, rows, device_pitch, host_pitch| Shape {
        name,
        width,
        rows,
        device_pitch,
        host_pitch,
    };
    // Row-major objects scatter on the host; head-major ones gather on the device.
    let shapes = [
        shape("256B x256 host pitch 2KiB", 256, 256, 256, 2048),
        shape("256B x256 dev pitch 2KiB", 256, 256, 2048, 256),
        shape("1KiB x256 host pitch 2KiB", 1024, 256, 1024, 2048),
        shape("1KiB x256 dev pitch 2KiB", 1024, 256, 2048, 1024),
        shape("1152B x64 host pitch 9KiB", 1152, 64, 1152, 9216),
        shape("128B x256 dev pitch 1KiB", 128, 256, 1024, 128),
        shape("512KiB contiguous", 2048, 256, 2048, 2048),
        shape("64KiB x1", 65536, 1, 65536, 65536),
        shape("8KiB x96 host pitch 64KiB", 8192, 96, 8192, 65536),
    ];
    println!("{TOTAL} bytes per measurement, GB/s (median of {ITERS})");
    println!(
        "{:<38} {:>4} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "shape", "dir", "runs", "dma2d", "strided", "auto", "1d/row", "krn/row", "contig"
    );
    for s in shapes {
        let b = Buffers::new(s.host_len().max(TOTAL), s.device_len().max(TOTAL));
        let d = descs(&b, &s);
        let r = runs(&b, &s);
        for dir in [Dir::H2D, Dir::D2H] {
            let mut by_mode = [0.0; 3];
            for (i, m) in modes.iter_mut().enumerate() {
                by_mode[i] = time(&stream, || match dir {
                    Dir::H2D => m.h2d(&r, &stream).expect("h2d"),
                    Dir::D2H => m.d2h(&r, &stream).expect("d2h"),
                });
            }
            let rows = time(&stream, || match dir {
                Dir::H2D => memcpy.h2d(&d, &stream).expect("h2d"),
                Dir::D2H => memcpy.d2h(&d, &stream).expect("d2h"),
            });
            let kern = time(&stream, || match dir {
                Dir::H2D => kernel.h2d(&d, &stream).expect("h2d"),
                Dir::D2H => kernel.d2h(&d, &stream).expect("d2h"),
            });
            let flat = time(&stream, || contiguous(&b, dir, &stream));
            println!(
                "{:<38} {:>4} {:>7} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                s.name,
                if dir == Dir::H2D { "h2d" } else { "d2h" },
                s.runs(),
                by_mode[0],
                by_mode[1],
                by_mode[2],
                rows,
                kern,
                flat
            );
        }
    }
}
