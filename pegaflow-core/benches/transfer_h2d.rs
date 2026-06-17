//! Native GPU<->CPU transfer benchmark for the transfer_h2d bench target.
//!
//! This exercises the same `CopyDesc` batches used by PegaFlow GPU workers:
//! direct DMA copies (`MemcpyBackend`) and the mapped-host single-kernel path
//! (`KernelBackend`). It is intentionally scoped to local transfer mechanics;
//! it does not model P/D RDMA handshakes, request scheduling, or decode-side
//! completion behavior.
//!
//! Run (this box is CUDA 13):
//!   cargo bench -p pegaflow-core --no-default-features \
//!       --features cuda-13,rdma --bench transfer_h2d
//!
//! Optional host allocation flags:
//!   cargo bench -p pegaflow-core --bench transfer_h2d -- --hugepages
//!   cargo bench -p pegaflow-core --bench transfer_h2d -- --alloc-gib 8

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream, sys};
use pegaflow_core::transfer::{CopyDesc, KernelBackend, MemcpyBackend, TransferBackend};

const SEG: usize = 4096;
const STRIDE_FACTOR: usize = 2;
const WARMUP: usize = 3;
const ITERS: usize = 20;
const BLOCK_COUNTS: &[usize] = &[1024, 4096, 16384, 65536];
const LAYOUTS: &[FragmentLayout] = &[
    FragmentLayout::Contiguous,
    FragmentLayout::StridedDevice,
    FragmentLayout::Shuffled,
];
const DIRECTIONS: &[Direction] = &[Direction::D2h, Direction::H2d];

#[derive(Clone, Copy, Debug)]
enum Direction {
    D2h,
    H2d,
}

impl Direction {
    fn name(self) -> &'static str {
        match self {
            Self::D2h => "D2H",
            Self::H2d => "H2D",
        }
    }

    fn submit(
        self,
        backend: &dyn TransferBackend,
        copies: &[CopyDesc],
        stream: &Arc<CudaStream>,
    ) -> Result<(), String> {
        match self {
            Self::D2h => backend.d2h(copies, stream),
            Self::H2d => backend.h2d(copies, stream),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FragmentLayout {
    /// Sequential block ids with adjacent GPU and host ranges. The direct
    /// backend should coalesce this to one large CUDA memcpy.
    Contiguous,
    /// Sequential host blocks copied to every other GPU block id. This mirrors
    /// sparse block lists where device-side KV slots are not adjacent.
    StridedDevice,
    /// A deterministic random block order. Host and device offsets match each
    /// slot, but descriptor order prevents direct-path coalescing.
    Shuffled,
}

impl FragmentLayout {
    fn name(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::StridedDevice => "strided_device",
            Self::Shuffled => "shuffled",
        }
    }

    fn device_bytes(self, blocks: usize) -> usize {
        match self {
            Self::Contiguous | Self::Shuffled => blocks * SEG,
            Self::StridedDevice => ((blocks - 1) * STRIDE_FACTOR + 1) * SEG,
        }
    }

    fn host_bytes(self, blocks: usize) -> usize {
        blocks * SEG
    }

    fn copies(self, device: u64, host: &MappedHost, blocks: usize) -> Vec<CopyDesc> {
        match self {
            Self::Contiguous => (0..blocks)
                .map(|slot| copy_desc(device, host, slot * SEG, slot * SEG))
                .collect(),
            Self::StridedDevice => (0..blocks)
                .map(|slot| copy_desc(device, host, slot * STRIDE_FACTOR * SEG, slot * SEG))
                .collect(),
            Self::Shuffled => shuffled_slots(blocks)
                .into_iter()
                .map(|slot| copy_desc(device, host, slot * SEG, slot * SEG))
                .collect(),
        }
    }
}

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
    let mut alloc_bytes = max_host_bytes();
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
        alloc_bytes: alloc_bytes.max(max_host_bytes()),
    }
}

fn max_host_bytes() -> usize {
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

fn fill_host(host: &MappedHost, len: usize) {
    assert!(len <= host.len);
    let slice = unsafe { std::slice::from_raw_parts_mut(host.host, len) };
    for (idx, byte) in slice.iter_mut().enumerate() {
        *byte = (idx.wrapping_mul(31).wrapping_add(7) & 0xFF) as u8;
    }
}

fn fill_device(device: u64, len: usize) {
    check_cuda(
        unsafe { sys::cuMemsetD8_v2(device, 0xA5, len) },
        "cuMemsetD8_v2",
    );
}

fn copy_desc(device: u64, host: &MappedHost, device_offset: usize, host_offset: usize) -> CopyDesc {
    CopyDesc {
        device: device + device_offset as u64,
        host: unsafe { host.host.add(host_offset) },
        host_device: host.device + host_offset as u64,
        size: SEG,
    }
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

fn measure(
    backend: &dyn TransferBackend,
    direction: Direction,
    copies: &[CopyDesc],
    stream: &Arc<CudaStream>,
    total_bytes: usize,
) -> (f64, f64) {
    for _ in 0..WARMUP {
        direction
            .submit(backend, copies, stream)
            .expect("transfer warmup");
        stream.synchronize().expect("sync warmup");
    }
    let start = Instant::now();
    for _ in 0..ITERS {
        direction.submit(backend, copies, stream).expect("transfer");
        stream.synchronize().expect("sync");
    }
    let secs = start.elapsed().as_secs_f64();
    let avg_ms = secs * 1e3 / ITERS as f64;
    let gibps = (total_bytes * ITERS) as f64 / secs / (1024.0 * 1024.0 * 1024.0);
    (avg_ms, gibps)
}

fn print_result(
    layout: FragmentLayout,
    blocks: usize,
    direction: Direction,
    backend: &dyn TransferBackend,
    avg_ms: f64,
    gibps: f64,
) {
    let total_mib = blocks * SEG / (1024 * 1024);
    println!(
        "{:>15}  {:>8}  {:>8}M  {:>9}  {:>7}  {:>11.3}  {:>11.2}",
        layout.name(),
        blocks,
        total_mib,
        direction.name(),
        backend.name(),
        avg_ms,
        gibps,
    );
}

fn main() {
    let config = parse_config();
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();
    let kernel = KernelBackend::new(&ctx).expect("kernel backend");
    let memcpy = MemcpyBackend;

    println!(
        "fragment={} B, stride_factor={}, warmup={}, iters={}, host_memory={:?}, host_alloc={} MiB\n",
        SEG,
        STRIDE_FACTOR,
        WARMUP,
        ITERS,
        config.host_memory,
        config.alloc_bytes / (1024 * 1024)
    );
    println!(
        "{:>15}  {:>8}  {:>9}  {:>9}  {:>7}  {:>11}  {:>11}",
        "layout", "blocks", "total", "direction", "backend", "avg_ms", "GiB/s"
    );

    let backends: [&dyn TransferBackend; 2] = [&memcpy, &kernel];

    for &layout in LAYOUTS {
        for &blocks in BLOCK_COUNTS {
            let host_bytes = layout.host_bytes(blocks);
            let device_bytes = layout.device_bytes(blocks);
            let total_bytes = blocks * SEG;
            let host = alloc_mapped_host(config.alloc_bytes.max(host_bytes), config.host_memory);
            let device = alloc_device(device_bytes);
            fill_host(&host, host_bytes);
            fill_device(device, device_bytes);

            let copies = layout.copies(device, &host, blocks);
            for &direction in DIRECTIONS {
                for backend in backends {
                    let (avg_ms, gibps) =
                        measure(backend, direction, &copies, &stream, total_bytes);
                    print_result(layout, blocks, direction, backend, avg_ms, gibps);
                }
            }

            check_cuda(unsafe { sys::cuMemFree_v2(device) }, "cuMemFree_v2");
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}
