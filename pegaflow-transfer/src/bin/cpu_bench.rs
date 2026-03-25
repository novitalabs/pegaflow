// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-memory RDMA benchmark: measures per-task RDMA READ vs WRITE latency across NUMA nodes and NICs.
//!
//! Models a realistic workload: each "task" transfers N random-sized batches of fixed-size blocks
//! via RDMA, measuring per-task latency. Runs single-NIC baselines then multi-NIC aggregate.

use std::ptr::NonNull;
use std::time::Instant;
use std::{mem, ptr, thread};

use clap::Parser;
use pegaflow_common::read_cpu_topology_from_sysfs;
use pegaflow_transfer::rdma_topo::SystemTopology;
use pegaflow_transfer::{MemoryRegion, TransferDesc, TransferEngine, TransferOp, init_logging};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "pegaflow-cpu-bench",
    version,
    about = "RDMA CPU memory block-task latency benchmark"
)]
struct Cli {
    /// Block size (e.g. "4mb", "2mb").
    #[arg(long, default_value = "4mb")]
    block_size: String,

    /// Blocks per task: single number (e.g. "150") or range (e.g. "100-200").
    /// When a range, each task randomly picks a count (deterministic seed).
    #[arg(long, default_value = "150")]
    blocks_per_task: String,

    /// Number of measured tasks.
    #[arg(long, default_value_t = 50)]
    tasks: usize,

    /// Number of warmup tasks (not measured).
    #[arg(long, default_value_t = 5)]
    warmup_tasks: usize,

    /// Benchmark mode.
    #[arg(long, default_value = "both", value_parser = ["read", "write", "both"])]
    mode: String,

    /// Restrict to a single NIC (e.g. "mlx5_0").
    #[arg(long)]
    nic: Option<String>,

    /// Exclude a NIC (e.g. "mlx5_0").
    #[arg(long)]
    exclude_nic: Option<String>,

    /// Restrict to a single NUMA node (e.g. 0).
    #[arg(long)]
    numa: Option<u32>,
}

// ---------------------------------------------------------------------------
// Deterministic RNG (SplitMix64)
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi] inclusive.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if lo == hi {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        (lo as u64 + self.next_u64() % span) as usize
    }
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------

fn parse_size(s: &str) -> usize {
    let s = s.trim().to_lowercase();
    let (num_str, multiplier) = if s.ends_with("tb") {
        (&s[..s.len() - 2], 1usize << 40)
    } else if s.ends_with("gb") {
        (&s[..s.len() - 2], 1usize << 30)
    } else if s.ends_with("mb") {
        (&s[..s.len() - 2], 1usize << 20)
    } else if s.ends_with("kb") {
        (&s[..s.len() - 2], 1usize << 10)
    } else {
        (s.as_str(), 1usize)
    };
    let num: f64 = num_str.parse().expect("invalid size number");
    (num * multiplier as f64) as usize
}

/// Parse "150" or "100-200" into (lo, hi).
fn parse_block_range(s: &str) -> (usize, usize) {
    if let Some((lo, hi)) = s.split_once('-') {
        let lo: usize = lo.trim().parse().expect("invalid blocks-per-task low");
        let hi: usize = hi.trim().parse().expect("invalid blocks-per-task high");
        assert!(lo <= hi, "blocks-per-task: low must <= high");
        assert!(lo > 0, "blocks-per-task: must be > 0");
        (lo, hi)
    } else {
        let n: usize = s.trim().parse().expect("invalid blocks-per-task");
        assert!(n > 0, "blocks-per-task: must be > 0");
        (n, n)
    }
}

// ---------------------------------------------------------------------------
// NUMA-aware CPU buffer
// ---------------------------------------------------------------------------

struct NumaBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for NumaBuffer {}
unsafe impl Sync for NumaBuffer {}

impl NumaBuffer {
    fn alloc(numa_node: u32, len: usize) -> Self {
        assert!(len > 0);

        let cpu_topo =
            read_cpu_topology_from_sysfs().expect("failed to read NUMA CPU topology from sysfs");
        let cpus = cpu_topo
            .get(&numa_node)
            .unwrap_or_else(|| panic!("no CPUs found for NUMA{}", numa_node));

        let cpus = cpus.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("numa{}-alloc", numa_node))
            .spawn(move || {
                unsafe {
                    let mut cpu_set: libc::cpu_set_t = mem::zeroed();
                    for &cpu in &cpus {
                        libc::CPU_SET(cpu, &mut cpu_set);
                    }
                    let ret =
                        libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &cpu_set);
                    assert_eq!(
                        ret,
                        0,
                        "sched_setaffinity failed: {}",
                        std::io::Error::last_os_error()
                    );
                }

                let p = unsafe {
                    libc::mmap(
                        ptr::null_mut(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                assert_ne!(p, libc::MAP_FAILED, "mmap failed");
                // Touch all pages to fault them on the correct NUMA node.
                unsafe {
                    ptr::write_bytes(p as *mut u8, 0xAB, len);
                }
                tx.send(p as u64).unwrap();
            })
            .expect("failed to spawn NUMA alloc thread");
        handle.join().expect("NUMA alloc thread panicked");
        let ptr = NonNull::new(rx.recv().unwrap() as *mut u8).expect("mmap returned null");

        Self { ptr, len }
    }

    fn fill(&self, pattern: u8) {
        unsafe {
            ptr::write_bytes(self.ptr.as_ptr(), pattern, self.len);
        }
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.len);
        }
    }
}

// ---------------------------------------------------------------------------
// Engine context: a server/client engine pair with handshake metadata
// ---------------------------------------------------------------------------

struct EngineContext {
    label: String,
    #[allow(dead_code)]
    server: TransferEngine,
    client: TransferEngine,
    server_buf: NumaBuffer,
    client_buf: NumaBuffer,
}

// ---------------------------------------------------------------------------
// Stats helpers
// ---------------------------------------------------------------------------

fn gib_per_sec(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    bytes as f64 / secs / (1024.0 * 1024.0 * 1024.0)
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / secs / 1e9
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// Task schedule
// ---------------------------------------------------------------------------

fn generate_task_schedule(
    total_tasks: usize,
    block_range: (usize, usize),
    seed: u64,
) -> Vec<usize> {
    let mut rng = SimpleRng::new(seed);
    (0..total_tasks)
        .map(|_| rng.range(block_range.0, block_range.1))
        .collect()
}

// ---------------------------------------------------------------------------
// Build scatter lists for a task
// ---------------------------------------------------------------------------

fn build_block_scatter(
    local_base: NonNull<u8>,
    remote_base: NonNull<u8>,
    nblocks: usize,
    block_size: usize,
) -> Vec<TransferDesc> {
    (0..nblocks)
        .map(|i| {
            let off = i * block_size;
            TransferDesc {
                local_ptr: unsafe { NonNull::new_unchecked(local_base.as_ptr().add(off)) },
                remote_ptr: unsafe { NonNull::new_unchecked(remote_base.as_ptr().add(off)) },
                len: block_size,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

struct TaskResult {
    latency_ms: f64,
    bytes: usize,
}

struct BenchResult {
    label: String,
    tasks: Vec<TaskResult>,
}

// ---------------------------------------------------------------------------
// Bench runner (works for both single-NIC and multi-NIC engines)
// ---------------------------------------------------------------------------

fn run_bench(
    ctx: &EngineContext,
    op: TransferOp,
    schedule: &[usize],
    warmup: usize,
    block_size: usize,
) -> BenchResult {
    let mut tasks = Vec::with_capacity(schedule.len().saturating_sub(warmup));

    for (i, &nblocks) in schedule.iter().enumerate() {
        let descs =
            build_block_scatter(ctx.client_buf.ptr, ctx.server_buf.ptr, nblocks, block_size);

        let start = Instant::now();
        let rx = ctx
            .client
            .batch_transfer_async(op, "bench-server", &descs)
            .expect("RDMA submit failed");
        rx.recv()
            .expect("completion channel dropped")
            .expect("RDMA transfer failed");
        let elapsed = start.elapsed();

        if i >= warmup {
            tasks.push(TaskResult {
                latency_ms: elapsed.as_secs_f64() * 1000.0,
                bytes: nblocks * block_size,
            });
        }
    }

    BenchResult {
        label: ctx.label.clone(),
        tasks,
    }
}

// ---------------------------------------------------------------------------
// Print results
// ---------------------------------------------------------------------------

fn print_bench_result(
    result: &BenchResult,
    op: TransferOp,
    block_size: usize,
    numa: u32,
    nic_count: usize,
) {
    let mut latencies: Vec<f64> = result.tasks.iter().map(|t| t.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);

    let avg_bytes = result.tasks.iter().map(|t| t.bytes).sum::<usize>() / result.tasks.len().max(1);
    let avg_blocks = avg_bytes / block_size;

    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);

    println!();
    if nic_count == 1 {
        println!(
            "=== RDMA {:?} — {} (NUMA{}, single NIC) ===",
            op, result.label, numa,
        );
    } else {
        println!(
            "=== RDMA {:?} — NUMA{}, {} NICs aggregate [{}] ===",
            op, numa, nic_count, result.label,
        );
    }
    if nic_count > 1 {
        println!(
            "  avg {} blocks x {}  ({:.1} MB/task, split across {} NICs)",
            avg_blocks,
            format_size(block_size),
            avg_bytes as f64 / (1024.0 * 1024.0),
            nic_count,
        );
    } else {
        println!(
            "  avg {} blocks x {}  ({:.1} MB/task)",
            avg_blocks,
            format_size(block_size),
            avg_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    println!("  p50={:.2}ms  p95={:.2}ms  p99={:.2}ms", p50, p95, p99);
    println!(
        "  p50 equiv: {:.1} Gbps ({:.2} GiB/s)",
        gbps(avg_bytes, p50 / 1000.0),
        gib_per_sec(avg_bytes, p50 / 1000.0),
    );
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.0}KB", bytes as f64 / (1u64 << 10) as f64)
    } else {
        format!("{}B", bytes)
    }
}

// ---------------------------------------------------------------------------
// Helper: create an EngineContext for a set of NIC names
// ---------------------------------------------------------------------------

fn create_engine_context(
    label: String,
    nic_names: &[String],
    numa_node: u32,
    buf_size: usize,
) -> EngineContext {
    let server_buf = NumaBuffer::alloc(numa_node, buf_size);
    let client_buf = NumaBuffer::alloc(numa_node, buf_size);
    server_buf.fill(0xAA);
    client_buf.fill(0xBB);

    let server = TransferEngine::new(nic_names).expect("server init failed");
    server
        .register_memory(&[MemoryRegion {
            ptr: server_buf.ptr,
            len: buf_size,
        }])
        .expect("server register_memory failed");

    let client = TransferEngine::new(nic_names).expect("client init failed");
    client
        .register_memory(&[MemoryRegion {
            ptr: client_buf.ptr,
            len: buf_size,
        }])
        .expect("client register_memory failed");

    // Handshake: both sides prepare, then complete handshake to each other.
    let server_meta = match server
        .get_or_prepare("bench-client")
        .expect("server get_or_prepare failed")
    {
        pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
        pegaflow_transfer::ConnectionStatus::Existing
        | pegaflow_transfer::ConnectionStatus::Connecting => {
            panic!("unexpected state on fresh engine")
        }
    };
    let client_meta = match client
        .get_or_prepare("bench-server")
        .expect("client get_or_prepare failed")
    {
        pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
        pegaflow_transfer::ConnectionStatus::Existing
        | pegaflow_transfer::ConnectionStatus::Connecting => {
            panic!("unexpected state on fresh engine")
        }
    };
    server
        .complete_handshake("bench-client", &server_meta, &client_meta)
        .expect("server complete_handshake failed");
    client
        .complete_handshake("bench-server", &client_meta, &server_meta)
        .expect("client complete_handshake failed");

    let nic_count = nic_names.len();
    assert_eq!(
        server.num_qps(),
        nic_count,
        "server should have {nic_count} QP(s) after handshake, got {}",
        server.num_qps()
    );
    assert_eq!(
        client.num_qps(),
        nic_count,
        "client should have {nic_count} QP(s) after handshake, got {}",
        client.num_qps()
    );

    EngineContext {
        label,
        server,
        client,
        server_buf,
        client_buf,
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    init_logging();
    let cli = Cli::parse();

    let block_size = parse_size(&cli.block_size);
    let block_range = parse_block_range(&cli.blocks_per_task);
    let total_tasks = cli.warmup_tasks + cli.tasks;

    let schedule = generate_task_schedule(total_tasks, block_range, 0x42);
    let max_blocks = *schedule.iter().max().unwrap();

    println!(
        "pegaflow-cpu-bench: block_size={} blocks_per_task={} tasks={} warmup={}",
        cli.block_size, cli.blocks_per_task, cli.tasks, cli.warmup_tasks,
    );
    if block_range.0 != block_range.1 {
        println!(
            "  schedule: min={} max={} blocks (deterministic seed)",
            schedule.iter().min().unwrap(),
            max_blocks,
        );
    }
    if let Some(ref nic) = cli.nic {
        println!("  filter: nic={}", nic);
    }
    if let Some(ref nic) = cli.exclude_nic {
        println!("  exclude: nic={}", nic);
    }
    if let Some(numa) = cli.numa {
        println!("  filter: numa={}", numa);
    }

    let buf_size = max_blocks * block_size;

    // Detect topology.
    let topo = SystemTopology::detect();
    topo.log_summary();

    let groups: Vec<_> = topo
        .groups()
        .iter()
        .filter(|g| cli.numa.is_none_or(|n| g.node.0 == n))
        .filter(|g| !g.nics.is_empty())
        .collect();

    if groups.is_empty() {
        eprintln!("error: no NUMA groups with RDMA NICs found");
        std::process::exit(1);
    }

    for group in &groups {
        let nics: Vec<_> = group
            .nics
            .iter()
            .filter(|nic| cli.nic.as_ref().is_none_or(|f| &nic.name == f))
            .filter(|nic| cli.exclude_nic.as_ref() != Some(&nic.name))
            .collect();

        if nics.is_empty() {
            continue;
        }

        let nic_count = nics.len();

        println!();
        println!(
            "--- NUMA{}: {} NIC(s) [{}], buf={} ---",
            group.node.0,
            nic_count,
            nics.iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            format_size(buf_size),
        );

        // Build single-NIC contexts for baselines.
        let single_contexts: Vec<EngineContext> = nics
            .iter()
            .map(|nic| {
                let ctx = create_engine_context(
                    nic.name.clone(),
                    std::slice::from_ref(&nic.name),
                    group.node.0,
                    buf_size,
                );
                println!("  {} ready (handshake complete)", nic.name);
                ctx
            })
            .collect();

        // Build multi-NIC context (if >1 NIC).
        let multi_context = if nic_count > 1 {
            let nic_names: Vec<String> = nics.iter().map(|n| n.name.clone()).collect();
            let label = nic_names.join(", ");
            let ctx = create_engine_context(label, &nic_names, group.node.0, buf_size);
            println!(
                "  multi-NIC engine ready ({} NICs, handshake complete)",
                nic_count
            );
            Some(ctx)
        } else {
            None
        };

        let run_mode = |op: TransferOp| {
            // Phase 1: single-NIC baselines.
            for ctx in &single_contexts {
                if op == TransferOp::Read {
                    ctx.client_buf.fill(0x00);
                }
                let result = run_bench(ctx, op, &schedule, cli.warmup_tasks, block_size);
                print_bench_result(&result, op, block_size, group.node.0, 1);
            }

            // Phase 2: multi-NIC aggregate (only if >1 NIC).
            if let Some(ref ctx) = multi_context {
                if op == TransferOp::Read {
                    ctx.client_buf.fill(0x00);
                }
                let result = run_bench(ctx, op, &schedule, cli.warmup_tasks, block_size);
                print_bench_result(&result, op, block_size, group.node.0, nic_count);
            }
        };

        match cli.mode.as_str() {
            "write" => run_mode(TransferOp::Write),
            "read" => run_mode(TransferOp::Read),
            _ => {
                run_mode(TransferOp::Write);
                run_mode(TransferOp::Read);
            }
        }

        // Cleanup.
        for ctx in &single_contexts {
            ctx.client.unregister_memory(&[ctx.client_buf.ptr]).ok();
            ctx.server.unregister_memory(&[ctx.server_buf.ptr]).ok();
        }
        if let Some(ref ctx) = multi_context {
            ctx.client.unregister_memory(&[ctx.client_buf.ptr]).ok();
            ctx.server.unregister_memory(&[ctx.server_buf.ptr]).ok();
        }
    }

    println!();
    println!("bench complete.");
}
