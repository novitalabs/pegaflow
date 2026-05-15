//! Cold-start cost of pinned memory allocation under different strategies.
//! Motivates the `mmap + parallel_pre_touch + cudaHostRegister` path now used
//! by `PinnedMemory::allocate_regular`/`allocate_hugepages`.
//!
//! Non-hugepage (size = BENCH_SIZE_GB):
//!   A. cudaHostAlloc(flags=0)                       — old Regular path
//!   B. mmap(MAP_POPULATE) + cudaHostRegister        — single-thread populate
//!   C. mmap + parallel_pre_touch(N) + register      — current Regular path
//!
//! Hugepage (size = BENCH_HUGE_MB, requires reserved hugepages):
//!   D. mmap(MAP_HUGETLB) + cudaHostRegister         — old HugePages path
//!   E. mmap(MAP_HUGETLB) + parallel_touch(N) + register — current HugePages path
//!
//! ## Measured results
//!
//! ### 48-core single-socket (RTX 5070ti, EPYC 7402), 20 GB, median of 3
//!
//! | strategy     | total    | notes                         |
//! |--------------|---------:|-------------------------------|
//! | A            | 17.4 s   | old baseline                  |
//! | B            | 13.5 s   |                               |
//! | C[1]         | 14.0 s   | == B                          |
//! | C[2]         |  8.9 s   |                               |
//! | C[4]         |  5.8 s   |                               |
//! | C[8]         |  4.4 s   |                               |
//! | C[16]        |  3.9 s   | sweet spot                    |
//! | C[32]        |  3.85 s  | plateaus                      |
//!
//! ### 240-core dual-socket (h20, 2× 120 hw-threads, 2 NUMA nodes), 500 GB, single rep
//!
//! Non-hugepage:
//!
//! | strategy     | total    | touch    | register | notes        |
//! |--------------|---------:|---------:|---------:|--------------|
//! | A            | 397.7 s  |          |          | old baseline |
//! | B            | 121.2 s  |          |  20.1 s  |              |
//! | C[1]         | 125.3 s  | 105.2 s  |  20.1 s  |              |
//! | C[16]        |  27.3 s  |   7.9 s  |  19.4 s  |              |
//! | C[64]        |  26.1 s  |   8.3 s  |  17.8 s  |              |
//! | C[128]       |  21.3 s  |   4.3 s  |  17.0 s  | best         |
//! | C[240]       |  24.4 s  |   6.0 s  |  18.4 s  | oversubscribed|
//!
//! Hugepage (the more dramatic win — cudaHostRegister hands its serial
//! page-fault work to parallel pre-touch instead of doing it inline):
//!
//! | strategy     | total    | touch    | register | notes        |
//! |--------------|---------:|---------:|---------:|--------------|
//! | D            |  88.5 s  |          |  88.5 s  | old baseline |
//! | E[1]         |  89.8 s  |  78.3 s  |  11.6 s  |              |
//! | E[16]        |  17.7 s  |   6.4 s  |  11.4 s  |              |
//! | E[64]        |  14.8 s  |   3.4 s  |  11.4 s  | best         |
//! | E[128]       |  16.2 s  |   4.7 s  |  11.5 s  |              |
//! | E[240]       |  18.0 s  |   6.6 s  |  11.4 s  | oversubscribed|
//!
//! ## NUMA-aware touch (verified separately on h20, 200 GB, 64 threads)
//!
//! Pre-touch threads must run on the target NUMA node so first-touch places
//! every page locally. With explicit `pin_thread_to_numa_node`: 1000/1000
//! sampled pages on the target node, no timing penalty vs unpinned. The
//! production path now plumbs `NumaNode` through `allocate_regular` /
//! `allocate_hugepages` so each pre-touch worker pins itself before faulting.
//!
//! ## Takeaways
//!
//! - cudaHostAlloc serializes page allocation + zero + pin → slowest.
//! - `cudaHostRegister` on HUGETLB has ~12 s of "real" IOMMU pinning;
//!   the rest of its cost on a cold mmap is serial fault-in we can offload.
//! - Sweet spot scales with NUMA-local CPU count, not total HW threads.
//!   Production uses `available_parallelism()` (affinity-aware) so a pinned
//!   thread sees only its node's CPUs.
//!
//! ## Running
//!
//! ```bash
//! cargo run --release --example pinned_alloc_parallel
//! BENCH_SIZE_GB=500 BENCH_HUGE_MB=512000 BENCH_REPS=1 \
//!   BENCH_THREADS=1,16,64,128,240 cargo run --release --example pinned_alloc_parallel
//! ```
//!
//! Env:
//!   BENCH_SIZE_GB  non-hugepage size, default 20
//!   BENCH_HUGE_MB  hugepage size, default 256 (0 to skip)
//!   BENCH_REPS     default 3
//!   BENCH_THREADS  default "1,2,4,8,16,32"

use std::ptr;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::runtime::sys as rt;

const GB: usize = 1024 * 1024 * 1024;

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    ctx.bind_to_thread().expect("bind CUDA context");

    let size_gb: usize = env_parse("BENCH_SIZE_GB", 20);
    let reps: usize = env_parse("BENCH_REPS", 3);
    let thread_sweep: Vec<usize> = std::env::var("BENCH_THREADS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect::<Vec<usize>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 4, 8, 16, 32]);

    let size = size_gb * GB;
    println!(
        "size = {} GB    reps = {}    page = {} B",
        size_gb,
        reps,
        page_size()
    );
    println!();

    println!("=== A: cudaHostAlloc(flags=0) ===");
    let mut a_totals = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = bench_cuda_host_alloc(size);
        a_totals.push(t.total);
        println!("  total: {:>9.2} ms", ms(t.total));
    }
    println!("  median total: {:>9.2} ms", ms(median(&mut a_totals)));
    println!();

    println!("=== B: mmap(MAP_POPULATE) + cudaHostRegister ===");
    let mut b_totals = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = bench_mmap_populate(size);
        b_totals.push(t.total);
        println!(
            "  mmap: {:>9.2} ms   register: {:>9.2} ms   total: {:>9.2} ms",
            ms(t.mmap),
            ms(t.register),
            ms(t.total)
        );
    }
    println!("  median total: {:>9.2} ms", ms(median(&mut b_totals)));
    println!();

    for &n in &thread_sweep {
        println!(
            "=== C[{}]: mmap + parallel_touch(threads={}) + register ===",
            n, n
        );
        let mut totals = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = bench_parallel_touch(size, n);
            totals.push(t.total);
            println!(
                "  mmap: {:>7.2} ms   touch: {:>9.2} ms   register: {:>9.2} ms   total: {:>9.2} ms",
                ms(t.mmap),
                ms(t.touch),
                ms(t.register),
                ms(t.total),
            );
        }
        println!("  median total: {:>9.2} ms", ms(median(&mut totals)));
        println!();
    }

    let huge_mb: usize = env_parse("BENCH_HUGE_MB", 256);
    if huge_mb == 0 {
        return;
    }
    let huge_page = hugepage_size().unwrap_or(2 * 1024 * 1024);
    let huge_size = ((huge_mb * 1024 * 1024) + huge_page - 1) & !(huge_page - 1);
    println!(
        "--- Hugepage bench: size = {} MB    hugepage = {} KB ---",
        huge_size / (1024 * 1024),
        huge_page / 1024,
    );
    println!();

    println!("=== D: mmap(MAP_HUGETLB) + cudaHostRegister ===");
    let mut d_totals = Vec::with_capacity(reps);
    for _ in 0..reps {
        match bench_huge(huge_size, 0) {
            Some(t) => {
                d_totals.push(t.total);
                println!(
                    "  mmap: {:>6.2} ms   register: {:>9.2} ms   total: {:>9.2} ms",
                    ms(t.mmap),
                    ms(t.register),
                    ms(t.total)
                );
            }
            None => {
                println!("  mmap(MAP_HUGETLB) FAILED — not enough reserved hugepages");
                return;
            }
        }
    }
    println!("  median total: {:>9.2} ms", ms(median(&mut d_totals)));
    println!();

    for &n in &thread_sweep {
        println!(
            "=== E[{}]: mmap(MAP_HUGETLB) + parallel_touch(threads={}) + register ===",
            n, n
        );
        let mut totals = Vec::with_capacity(reps);
        for _ in 0..reps {
            let Some(t) = bench_huge(huge_size, n) else {
                println!("  mmap(MAP_HUGETLB) FAILED");
                return;
            };
            totals.push(t.total);
            println!(
                "  mmap: {:>6.2} ms   touch: {:>7.2} ms   register: {:>9.2} ms   total: {:>9.2} ms",
                ms(t.mmap),
                ms(t.touch),
                ms(t.register),
                ms(t.total),
            );
        }
        println!("  median total: {:>9.2} ms", ms(median(&mut totals)));
        println!();
    }
}

#[derive(Default)]
struct Timing {
    mmap: Duration,
    touch: Duration,
    register: Duration,
    total: Duration,
}

fn bench_cuda_host_alloc(size: usize) -> Timing {
    let mut ptr: *mut libc::c_void = ptr::null_mut();
    let start = Instant::now();
    // SAFETY: &mut ptr is a valid out-param; size validated > 0 by caller config.
    let r = unsafe { rt::cudaHostAlloc(&mut ptr, size, 0) };
    let total = start.elapsed();
    assert_eq!(
        r,
        rt::cudaError::cudaSuccess,
        "cudaHostAlloc failed: {:?}",
        r
    );
    // SAFETY: ptr returned successfully by cudaHostAlloc above.
    unsafe { rt::cudaFreeHost(ptr) };
    Timing {
        total,
        ..Default::default()
    }
}

fn bench_mmap_populate(size: usize) -> Timing {
    let t0 = Instant::now();
    // SAFETY: null hint + anon mapping with valid prot/flags; result checked vs MAP_FAILED.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
            -1,
            0,
        )
    };
    let mmap = t0.elapsed();
    assert!(
        p != libc::MAP_FAILED,
        "mmap failed: {}",
        std::io::Error::last_os_error()
    );

    let t1 = Instant::now();
    // SAFETY: p points to a valid mapping of `size` bytes from mmap above.
    let r = unsafe { rt::cudaHostRegister(p, size, rt::cudaHostRegisterDefault) };
    let register = t1.elapsed();
    assert_eq!(
        r,
        rt::cudaError::cudaSuccess,
        "cudaHostRegister failed: {:?}",
        r
    );

    // SAFETY: matched unregister/munmap for the pointer registered/mapped above.
    unsafe { rt::cudaHostUnregister(p) };
    // SAFETY: same.
    unsafe { libc::munmap(p, size) };

    Timing {
        mmap,
        register,
        total: mmap + register,
        touch: Duration::ZERO,
    }
}

fn bench_parallel_touch(size: usize, threads: usize) -> Timing {
    let t0 = Instant::now();
    // SAFETY: null hint + anon mapping with valid prot/flags; result checked vs MAP_FAILED.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    let mmap = t0.elapsed();
    assert!(
        p != libc::MAP_FAILED,
        "mmap failed: {}",
        std::io::Error::last_os_error()
    );

    let t1 = Instant::now();
    parallel_pre_touch(p.cast::<u8>(), size, threads);
    let touch = t1.elapsed();

    let t2 = Instant::now();
    // SAFETY: p points to a valid mapping of `size` bytes from mmap above.
    let r = unsafe { rt::cudaHostRegister(p, size, rt::cudaHostRegisterDefault) };
    let register = t2.elapsed();
    assert_eq!(
        r,
        rt::cudaError::cudaSuccess,
        "cudaHostRegister failed: {:?}",
        r
    );

    // SAFETY: matched unregister/munmap for the pointer registered/mapped above.
    unsafe { rt::cudaHostUnregister(p) };
    // SAFETY: same.
    unsafe { libc::munmap(p, size) };

    Timing {
        mmap,
        touch,
        register,
        total: mmap + touch + register,
    }
}

/// Fault in every page of [ptr, ptr+size) across N threads.
/// Each thread owns a disjoint chunk; one byte per page + a tail byte per chunk
/// forces the kernel to materialize all pages, including the last partial one.
fn parallel_pre_touch(ptr: *mut u8, size: usize, threads: usize) {
    let page = page_size();
    let threads = threads.max(1);
    let chunk = size.div_ceil(threads);
    // *mut u8 is not Send; smuggle as usize and reconstruct inside the scope.
    // Lifetimes are bounded by thread::scope.
    let base = ptr as usize;

    std::thread::scope(|s| {
        for i in 0..threads {
            let off = i * chunk;
            if off >= size {
                break;
            }
            let len = chunk.min(size - off);
            s.spawn(move || touch_chunk(base + off, len, page));
        }
    });
}

fn touch_chunk(base: usize, len: usize, page: usize) {
    let p = base as *mut u8;
    let mut off = 0usize;
    while off < len {
        // SAFETY: caller guarantees [base, base+len) lies inside a valid mmap.
        // Volatile keeps the compiler from elimating the write.
        unsafe { p.add(off).write_volatile(0u8) };
        off += page;
    }
    // Ensure the tail page (which `off += page` may overshoot past) is faulted.
    // SAFETY: len > 0 is enforced by parallel_pre_touch's chunk bookkeeping.
    unsafe { p.add(len - 1).write_volatile(0u8) };
}

/// Hugepage variant. If `touch_threads` > 0, pre-touches before register;
/// otherwise relies on `cudaHostRegister` to materialize pages. Returns None
/// if MAP_HUGETLB allocation fails (insufficient reserved hugepages).
fn bench_huge(size: usize, touch_threads: usize) -> Option<Timing> {
    let t0 = Instant::now();
    // SAFETY: null hint + anon HUGETLB mapping; result checked vs MAP_FAILED.
    let p = unsafe {
        libc::mmap(
            ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
            -1,
            0,
        )
    };
    let mmap = t0.elapsed();
    if p == libc::MAP_FAILED {
        return None;
    }

    let touch = if touch_threads > 0 {
        let t1 = Instant::now();
        parallel_pre_touch(p.cast::<u8>(), size, touch_threads);
        t1.elapsed()
    } else {
        Duration::ZERO
    };

    let t2 = Instant::now();
    // SAFETY: p points to a valid hugepage mapping of `size` bytes.
    let r = unsafe { rt::cudaHostRegister(p, size, rt::cudaHostRegisterDefault) };
    let register = t2.elapsed();
    assert_eq!(
        r,
        rt::cudaError::cudaSuccess,
        "cudaHostRegister failed: {:?}",
        r
    );

    // SAFETY: matched unregister/munmap for the pointer registered/mapped above.
    unsafe { rt::cudaHostUnregister(p) };
    // SAFETY: same.
    unsafe { libc::munmap(p, size) };

    Some(Timing {
        mmap,
        touch,
        register,
        total: mmap + touch + register,
    })
}

fn hugepage_size() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Hugepagesize:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if parts.len() == 2 && parts[1] == "kB" {
                return parts[0].parse::<usize>().ok().map(|kb| kb * 1024);
            }
        }
    }
    None
}

fn page_size() -> usize {
    // SAFETY: sysconf with a valid name; non-positive result is fallback-handled.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
}

fn env_parse(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn median(v: &mut [Duration]) -> Duration {
    v.sort();
    v[v.len() / 2]
}
