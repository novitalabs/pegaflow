// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-memory RDMA benchmark: measures per-task RDMA READ vs WRITE latency across NUMA nodes and NICs.
//!
//! Models a realistic workload: each "task" transfers N random-sized batches of fixed-size blocks
//! via RDMA, measuring per-task latency. Runs single-NIC baselines then multi-NIC aggregate.

use std::sync::{Arc, Barrier};
use std::time::Instant;
use std::{mem, ptr, thread};

use clap::Parser;
use pegaflow_common::read_cpu_topology_from_sysfs;
use pegaflow_transfer::rdma_topo::SystemTopology;
use pegaflow_transfer::{HandshakeMetadata, TransferEngine, TransferOp, init_logging};

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
    ptr: *mut u8,
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
        let raw = rx.recv().unwrap() as *mut u8;

        Self { ptr: raw, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr as u64
    }

    fn fill(&self, pattern: u8) {
        unsafe {
            ptr::write_bytes(self.ptr, pattern, self.len);
        }
    }
}

impl Drop for NumaBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-NIC context: a server/client engine pair with handshake metadata
// ---------------------------------------------------------------------------

struct NicContext {
    nic_name: String,
    server: TransferEngine,
    client: TransferEngine,
    server_hs: HandshakeMetadata,
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
    local_base: u64,
    remote_base: u64,
    nblocks: usize,
    block_size: usize,
) -> (Vec<u64>, Vec<u64>, Vec<usize>) {
    let mut local_ptrs = Vec::with_capacity(nblocks);
    let mut remote_ptrs = Vec::with_capacity(nblocks);
    let lens = vec![block_size; nblocks];
    for i in 0..nblocks {
        let off = (i * block_size) as u64;
        local_ptrs.push(local_base + off);
        remote_ptrs.push(remote_base + off);
    }
    (local_ptrs, remote_ptrs, lens)
}

/// Distribute `nblocks` round-robin across `nic_count` NICs.
fn distribute_blocks(nblocks: usize, nic_count: usize) -> Vec<usize> {
    let base = nblocks / nic_count;
    let remainder = nblocks % nic_count;
    (0..nic_count)
        .map(|i| base + if i < remainder { 1 } else { 0 })
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
    nic_name: String,
    tasks: Vec<TaskResult>,
}

// ---------------------------------------------------------------------------
// Single-NIC bench
// ---------------------------------------------------------------------------

fn run_single_nic_bench(
    ctx: &NicContext,
    op: TransferOp,
    schedule: &[usize],
    warmup: usize,
    block_size: usize,
) -> BenchResult {
    let mut tasks = Vec::with_capacity(schedule.len().saturating_sub(warmup));

    for (i, &nblocks) in schedule.iter().enumerate() {
        let (local_ptrs, remote_ptrs, lens) = build_block_scatter(
            ctx.client_buf.as_u64(),
            ctx.server_buf.as_u64(),
            nblocks,
            block_size,
        );

        let start = Instant::now();
        ctx.client
            .batch_transfer(op, &ctx.server_hs, &local_ptrs, &remote_ptrs, &lens)
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
        nic_name: ctx.nic_name.clone(),
        tasks,
    }
}

// ---------------------------------------------------------------------------
// Multi-NIC aggregate bench
// ---------------------------------------------------------------------------

fn run_multi_nic_bench(
    contexts: &[NicContext],
    op: TransferOp,
    schedule: &[usize],
    warmup: usize,
    block_size: usize,
) -> Vec<BenchResult> {
    let n = contexts.len();
    if n == 1 {
        return vec![run_single_nic_bench(
            &contexts[0],
            op,
            schedule,
            warmup,
            block_size,
        )];
    }

    let barrier = Arc::new(Barrier::new(n));
    let schedule = Arc::new(schedule.to_vec());

    thread::scope(|s| {
        let handles: Vec<_> = contexts
            .iter()
            .enumerate()
            .map(|(nic_idx, ctx)| {
                let barrier = Arc::clone(&barrier);
                let schedule = Arc::clone(&schedule);
                s.spawn(move || {
                    let mut tasks = Vec::with_capacity(schedule.len().saturating_sub(warmup));

                    for (i, &nblocks) in schedule.iter().enumerate() {
                        let per_nic = distribute_blocks(nblocks, n);
                        let my_blocks = per_nic[nic_idx];

                        let (local_ptrs, remote_ptrs, lens) = build_block_scatter(
                            ctx.client_buf.as_u64(),
                            ctx.server_buf.as_u64(),
                            my_blocks,
                            block_size,
                        );

                        barrier.wait();

                        let start = Instant::now();
                        if my_blocks > 0 {
                            ctx.client
                                .batch_transfer(
                                    op,
                                    &ctx.server_hs,
                                    &local_ptrs,
                                    &remote_ptrs,
                                    &lens,
                                )
                                .expect("RDMA transfer failed");
                        }
                        let elapsed = start.elapsed();

                        if i >= warmup {
                            tasks.push(TaskResult {
                                latency_ms: elapsed.as_secs_f64() * 1000.0,
                                bytes: my_blocks * block_size,
                            });
                        }
                    }

                    BenchResult {
                        nic_name: ctx.nic_name.clone(),
                        tasks,
                    }
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

// ---------------------------------------------------------------------------
// Print results
// ---------------------------------------------------------------------------

fn print_single_nic_result(result: &BenchResult, op: TransferOp, block_size: usize, numa: u32) {
    let mut latencies: Vec<f64> = result.tasks.iter().map(|t| t.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);

    let avg_bytes = result.tasks.iter().map(|t| t.bytes).sum::<usize>() / result.tasks.len().max(1);
    let avg_blocks = avg_bytes / block_size;

    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);

    println!();
    println!(
        "=== RDMA {:?} — {} (NUMA{}, single NIC) ===",
        op, result.nic_name, numa,
    );
    println!(
        "  avg {} blocks x {}  ({:.1} MB/task)",
        avg_blocks,
        format_size(block_size),
        avg_bytes as f64 / (1024.0 * 1024.0),
    );
    println!("  p50={:.2}ms  p95={:.2}ms  p99={:.2}ms", p50, p95, p99);
    println!(
        "  p50 equiv: {:.1} Gbps ({:.2} GiB/s)",
        gbps(avg_bytes, p50 / 1000.0),
        gib_per_sec(avg_bytes, p50 / 1000.0),
    );
}

fn print_multi_nic_results(
    results: &[BenchResult],
    op: TransferOp,
    block_size: usize,
    schedule: &[usize],
    warmup: usize,
    numa: u32,
) {
    let nic_count = results.len();
    let measured = &schedule[warmup..];
    let ntasks = measured.len();
    if ntasks == 0 {
        return;
    }

    // Aggregate latency = max across NICs per task.
    let mut agg_latencies: Vec<f64> = (0..ntasks)
        .map(|i| {
            results
                .iter()
                .map(|r| r.tasks[i].latency_ms)
                .fold(0.0_f64, f64::max)
        })
        .collect();
    agg_latencies.sort_by(f64::total_cmp);

    let avg_total_blocks = measured.iter().sum::<usize>() / ntasks;
    let avg_bytes = avg_total_blocks * block_size;

    let p50 = percentile(&agg_latencies, 0.50);
    let p95 = percentile(&agg_latencies, 0.95);
    let p99 = percentile(&agg_latencies, 0.99);

    let nic_names: Vec<&str> = results.iter().map(|r| r.nic_name.as_str()).collect();

    println!();
    println!(
        "=== RDMA {:?} — NUMA{}, {} NICs aggregate [{}] ===",
        op,
        numa,
        nic_count,
        nic_names.join(", "),
    );
    println!(
        "  avg {} blocks x {}  ({:.1} MB/task, split across {} NICs)",
        avg_total_blocks,
        format_size(block_size),
        avg_bytes as f64 / (1024.0 * 1024.0),
        nic_count,
    );
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

    let buf_per_nic = max_blocks * block_size;

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
            "--- NUMA{}: {} NIC(s) [{}], buf={} per NIC ---",
            group.node.0,
            nic_count,
            nics.iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            format_size(buf_per_nic),
        );

        // Build NIC contexts with handshake-based connection.
        let mut contexts: Vec<NicContext> = Vec::with_capacity(nic_count);

        for nic in &nics {
            let server_buf = NumaBuffer::alloc(group.node.0, buf_per_nic);
            let client_buf = NumaBuffer::alloc(group.node.0, buf_per_nic);
            server_buf.fill(0xAA);
            client_buf.fill(0xBB);

            let server = TransferEngine::new(nic.name.clone()).expect("server init failed");
            server
                .register_memory(&[server_buf.as_u64()], &[buf_per_nic])
                .expect("server register_memory failed");

            let client = TransferEngine::new(nic.name.clone()).expect("client init failed");
            client
                .register_memory(&[client_buf.as_u64()], &[buf_per_nic])
                .expect("client register_memory failed");

            // Handshake: both sides prepare, then accept each other.
            let server_hs = server
                .prepare_handshake()
                .expect("server prepare_handshake failed");
            let client_hs = client
                .prepare_handshake()
                .expect("client prepare_handshake failed");
            server
                .accept_handshake(&client_hs)
                .expect("server accept_handshake failed");
            client
                .accept_handshake(&server_hs)
                .expect("client accept_handshake failed");

            println!("  {} ready (handshake complete)", nic.name);

            contexts.push(NicContext {
                nic_name: nic.name.clone(),
                server,
                client,
                server_hs,
                server_buf,
                client_buf,
            });
        }

        let run_mode = |op: TransferOp| {
            // Phase 1: single-NIC baselines.
            for ctx in &contexts {
                if op == TransferOp::Read {
                    ctx.client_buf.fill(0x00);
                }
                let result = run_single_nic_bench(ctx, op, &schedule, cli.warmup_tasks, block_size);
                print_single_nic_result(&result, op, block_size, group.node.0);
            }

            // Phase 2: multi-NIC aggregate (only if >1 NIC).
            if contexts.len() > 1 {
                for ctx in &contexts {
                    if op == TransferOp::Read {
                        ctx.client_buf.fill(0x00);
                    }
                }
                let results =
                    run_multi_nic_bench(&contexts, op, &schedule, cli.warmup_tasks, block_size);
                print_multi_nic_results(
                    &results,
                    op,
                    block_size,
                    &schedule,
                    cli.warmup_tasks,
                    group.node.0,
                );
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
        for ctx in &contexts {
            ctx.client
                .unregister_memory(&[ctx.client_buf.as_u64()])
                .ok();
            ctx.server
                .unregister_memory(&[ctx.server_buf.as_u64()])
                .ok();
        }
    }

    println!();
    println!("bench complete.");
}
