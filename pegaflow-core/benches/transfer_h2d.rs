//! Fully fragmented H2D throughput: direct (per-fragment `cuMemcpyAsync`) vs
//! kernel (single grid-strided launch over mapped pinned memory).
//!
//! The batch is `n` copies of `seg` bytes (4 KiB by default) whose descriptor
//! order is randomly shuffled, so the direct path's coalescing finds nothing to
//! merge and must issue one driver submission per fragment — the regime where
//! per-call launch latency, not bandwidth, decides throughput.
//!
//! Run (this box is CUDA 13):
//!   cargo bench -p pegaflow-core --no-default-features \
//!       --features cuda-13,rdma --bench transfer_h2d

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use pegaflow_core::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend};

const SEG: usize = 4096; // 4 KiB fragments
const WARMUP: usize = 3;
const ITERS: usize = 20;
const BLOCK_COUNTS: &[usize] = &[1024, 4096, 16384, 65536];

#[derive(Clone, Copy, Debug)]
enum HostMemory {
    CudaHostAlloc,
    HugePages,
}

struct BenchConfig {
    host_memory: HostMemory,
    alloc_bytes: usize,
}

struct MappedHost {
    host: *mut u8,
    device: u64,
    len: usize,
    host_memory: HostMemory,
}

impl Drop for MappedHost {
    fn drop(&mut self) {
        unsafe {
            match self.host_memory {
                HostMemory::CudaHostAlloc => {
                    let r = sys::cuMemFreeHost(self.host as *mut _);
                    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemFreeHost");
                }
                HostMemory::HugePages => {
                    let r = sys::cuMemHostUnregister(self.host as *mut _);
                    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostUnregister");
                    let r = libc::munmap(self.host as *mut _, self.len);
                    assert_eq!(r, 0, "munmap hugepage host allocation");
                }
            }
        }
    }
}

fn parse_config() -> BenchConfig {
    let mut host_memory = HostMemory::CudaHostAlloc;
    let mut alloc_bytes = max_total_bytes();
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--hugepages" => {
                host_memory = HostMemory::HugePages;
                alloc_bytes = 4 * 1024 * 1024 * 1024;
            }
            "--alloc-gib" => {
                let value = args.next().expect("--alloc-gib requires an integer value");
                let gib: usize = value.parse().expect("--alloc-gib must be an integer");
                alloc_bytes = gib
                    .checked_mul(1024 * 1024 * 1024)
                    .expect("alloc-gib overflow");
            }
            "--bench" => {}
            "--help" | "-h" => {
                println!(
                    "usage: cargo bench -p pegaflow-core --bench transfer_h2d -- [--hugepages] [--alloc-gib N]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument: {other}"),
        }
    }

    BenchConfig {
        host_memory,
        alloc_bytes: alloc_bytes.max(max_total_bytes()),
    }
}

fn max_total_bytes() -> usize {
    BLOCK_COUNTS
        .iter()
        .copied()
        .max()
        .expect("BLOCK_COUNTS must not be empty")
        * SEG
}

fn alloc_mapped_host(len: usize, host_memory: HostMemory) -> MappedHost {
    match host_memory {
        HostMemory::CudaHostAlloc => alloc_cuda_host_alloc(len),
        HostMemory::HugePages => alloc_hugepage_host(len),
    }
}

fn alloc_cuda_host_alloc(len: usize) -> MappedHost {
    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemHostAlloc(&mut p, len, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostAlloc");

    let mut device: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, p, 0) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer");

    MappedHost {
        host: p as *mut u8,
        device,
        len,
        host_memory: HostMemory::CudaHostAlloc,
    }
}

fn alloc_hugepage_host(len: usize) -> MappedHost {
    let len = align_to_hugepage(len);
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
            -1,
            0,
        )
    };
    assert_ne!(p, libc::MAP_FAILED, "mmap(MAP_HUGETLB) failed");

    let r = unsafe { sys::cuMemHostRegister_v2(p, len, sys::CU_MEMHOSTREGISTER_DEVICEMAP) };
    if r != sys::CUresult::CUDA_SUCCESS {
        unsafe {
            libc::munmap(p, len);
        }
    }
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostRegister");

    let mut device: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, p, 0) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemHostGetDevicePointer");

    MappedHost {
        host: p as *mut u8,
        device,
        len,
        host_memory: HostMemory::HugePages,
    }
}

fn align_to_hugepage(len: usize) -> usize {
    let hugepage = hugepage_size().unwrap_or(2 * 1024 * 1024);
    len.div_ceil(hugepage) * hugepage
}

fn hugepage_size() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            let kb: usize = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

fn alloc_device(len: usize) -> u64 {
    let mut d: sys::CUdeviceptr = 0;
    let r = unsafe { sys::cuMemAlloc_v2(&mut d, len) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc");
    d
}

/// A deterministic random permutation of `0..n` (Fisher-Yates over a seeded
/// xorshift64), so the fragmentation pattern is stable across runs.
fn shuffled_slots(n: usize) -> Vec<usize> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    perm
}

/// `iters` timed H2D submissions (each synchronized) after `warmup` untimed
/// ones, returned as GiB/s.
fn measure(
    backend: &dyn TransferBackend,
    copies: &[CopyDesc],
    stream: &Arc<CudaStream>,
    total_bytes: usize,
) -> (f64, f64) {
    for _ in 0..WARMUP {
        backend.h2d(copies, stream).expect("h2d warmup");
        stream.synchronize().expect("sync warmup");
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        backend.h2d(copies, stream).expect("h2d");
        stream.synchronize().expect("sync");
    }
    let secs = start.elapsed().as_secs_f64();
    let avg_ms = secs * 1e3 / ITERS as f64;
    let gibps = (total_bytes * ITERS) as f64 / secs / (1024.0 * 1024.0 * 1024.0);
    (avg_ms, gibps)
}

fn main() {
    let config = parse_config();
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let kernel = KernelBackend::new(&ctx).expect("kernel backend");
    let memcpy = MemcpyBackend;

    println!(
        "fragment={} B, warmup={}, iters={}, host_memory={:?}, host_alloc={} MiB\n",
        SEG,
        WARMUP,
        ITERS,
        config.host_memory,
        config.alloc_bytes / (1024 * 1024)
    );
    println!(
        "{:>8}  {:>9}  {:>11}  {:>11}  {:>11}  {:>11}  {:>8}",
        "blocks", "total", "direct_ms", "direct_GiBs", "kernel_ms", "kernel_GiBs", "speedup"
    );

    for &n in BLOCK_COUNTS {
        let total = n * SEG;
        let host = alloc_mapped_host(config.alloc_bytes, config.host_memory);
        let device = alloc_device(total);

        let copies: Vec<CopyDesc> = shuffled_slots(n)
            .into_iter()
            .map(|slot| CopyDesc {
                device: device + (slot * SEG) as u64,
                host: unsafe { host.host.add(slot * SEG) },
                host_device: host.device + (slot * SEG) as u64,
                size: SEG,
            })
            .collect();

        let (d_ms, d_gibs) = measure(&memcpy, &copies, &stream, total);
        let (k_ms, k_gibs) = measure(&kernel, &copies, &stream, total);

        println!(
            "{:>8}  {:>8.0}M  {:>11.3}  {:>11.2}  {:>11.3}  {:>11.2}  {:>7.2}x",
            n,
            total as f64 / (1024.0 * 1024.0),
            d_ms,
            d_gibs,
            k_ms,
            k_gibs,
            k_gibs / d_gibs,
        );

        unsafe {
            sys::cuMemFree_v2(device);
        }
    }
}
