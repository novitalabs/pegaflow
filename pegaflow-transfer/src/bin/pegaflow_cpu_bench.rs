// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-memory RDMA benchmark: measures RDMA READ vs WRITE throughput across NUMA nodes and NICs.
//!
//! Spawns server + client engine pairs per NIC, allocates NUMA-local CPU memory,
//! and measures single-NIC and aggregate multi-NIC bandwidth.

use std::sync::{Arc, Barrier};
use std::time::Instant;
use std::{mem, ptr, thread};

use clap::Parser;
use pegaflow_core::numa::read_cpu_topology_from_sysfs;
use pegaflow_transfer::rdma_topo::SystemTopology;
use pegaflow_transfer::{MooncakeTransferEngine, init_logging};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "pegaflow_cpu_bench", about = "RDMA CPU memory bandwidth benchmark")]
struct Cli {
    /// Total transfer size (e.g. "1gb", "512mb"). Split evenly across NICs.
    #[arg(long, default_value = "1gb")]
    size: String,

    /// Number of measured iterations.
    #[arg(long, default_value_t = 10)]
    iters: usize,

    /// Number of warmup iterations.
    #[arg(long, default_value_t = 3)]
    warmup: usize,

    /// Base RPC port. Each engine pair gets consecutive ports from here.
    #[arg(long, default_value_t = 56100)]
    base_port: u16,

    /// Restrict to a single NIC (e.g. "mlx5_0"). Default: all NICs per NUMA.
    #[arg(long)]
    nic: Option<String>,

    /// Restrict to a single NUMA node (e.g. 0). Default: all NUMA nodes.
    #[arg(long)]
    numa: Option<u32>,
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
    /// Allocate `len` bytes on the given NUMA node using thread-affinity + first-touch.
    fn alloc(numa_node: u32, len: usize) -> Self {
        assert!(len > 0);

        let cpu_topo =
            read_cpu_topology_from_sysfs().expect("failed to read NUMA CPU topology from sysfs");
        let cpus = cpu_topo
            .get(&numa_node)
            .unwrap_or_else(|| panic!("no CPUs found for NUMA{}", numa_node));

        // Spawn a thread pinned to the target NUMA node, mmap + touch there.
        let cpus = cpus.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("numa{}-alloc", numa_node))
            .spawn(move || {
                // Pin to NUMA CPUs.
                unsafe {
                    let mut cpu_set: libc::cpu_set_t = mem::zeroed();
                    for &cpu in &cpus {
                        libc::CPU_SET(cpu, &mut cpu_set);
                    }
                    let ret = libc::sched_setaffinity(
                        0,
                        mem::size_of::<libc::cpu_set_t>(),
                        &cpu_set,
                    );
                    assert_eq!(ret, 0, "sched_setaffinity failed: {}", std::io::Error::last_os_error());
                }

                // mmap + first-touch so pages land on this NUMA node.
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
// Per-NIC context: holds server + client engines and their buffers
// ---------------------------------------------------------------------------

struct NicContext {
    nic_name: String,
    server: MooncakeTransferEngine,
    client: MooncakeTransferEngine,
    server_buf: NumaBuffer,
    client_buf: NumaBuffer,
    per_nic_bytes: usize,
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

// ---------------------------------------------------------------------------
// Bench runner
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct NicResult {
    nic_name: String,
    latencies_ms: Vec<f64>,
}

/// Max bytes per single RDMA WR (IB max_msg_size is typically 1 GiB).
const MAX_RDMA_MSG_BYTES: usize = 1 << 30;

/// Build chunked (local_ptr, remote_ptr, len) vectors for a single buffer pair.
fn chunk_transfer(
    local_base: u64,
    remote_base: u64,
    total: usize,
) -> (Vec<u64>, Vec<u64>, Vec<usize>) {
    let mut local_ptrs = Vec::new();
    let mut remote_ptrs = Vec::new();
    let mut lens = Vec::new();
    let mut offset = 0usize;
    while offset < total {
        let chunk = MAX_RDMA_MSG_BYTES.min(total - offset);
        local_ptrs.push(local_base + offset as u64);
        remote_ptrs.push(remote_base + offset as u64);
        lens.push(chunk);
        offset += chunk;
    }
    (local_ptrs, remote_ptrs, lens)
}

/// Run one direction (READ or WRITE) across all NIC contexts in parallel.
fn run_bench(
    contexts: &[NicContext],
    mode: &str,
    warmup: usize,
    iters: usize,
) -> Vec<NicResult> {
    let n = contexts.len();
    let barrier = Arc::new(Barrier::new(n));
    let total_iters = warmup + iters;

    thread::scope(|s| {
        let handles: Vec<_> = contexts
            .iter()
            .map(|ctx| {
                let barrier = Arc::clone(&barrier);
                s.spawn(move || {
                    let server_session = ctx.server.get_session_id();
                    let mut latencies_ms = Vec::with_capacity(iters);

                    let (local_ptrs, remote_ptrs, lens) = chunk_transfer(
                        ctx.client_buf.as_u64(),
                        ctx.server_buf.as_u64(),
                        ctx.per_nic_bytes,
                    );

                    for iter in 0..total_iters {
                        // Sync all NICs for each iteration.
                        barrier.wait();

                        let start = Instant::now();
                        let result = match mode {
                            "write" => ctx.client.batch_transfer_sync_write(
                                &server_session,
                                &local_ptrs,
                                &remote_ptrs,
                                &lens,
                            ),
                            "read" => ctx.client.batch_transfer_sync_read(
                                &server_session,
                                &local_ptrs,
                                &remote_ptrs,
                                &lens,
                            ),
                            _ => unreachable!(),
                        };
                        let elapsed = start.elapsed();
                        result.expect("RDMA transfer failed");

                        if iter >= warmup {
                            latencies_ms.push(elapsed.as_secs_f64() * 1000.0);
                        }
                    }

                    NicResult {
                        nic_name: ctx.nic_name.clone(),
                        latencies_ms,
                    }
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn print_results(results: &[NicResult], mode: &str, per_nic_bytes: usize, numa_node: u32) {
    let nic_count = results.len();
    let total_bytes = per_nic_bytes * nic_count;

    println!();
    println!(
        "=== RDMA {} (NUMA{}, {} NIC{}) ===",
        mode.to_uppercase(),
        numa_node,
        nic_count,
        if nic_count > 1 { "s" } else { "" },
    );

    // Per-NIC stats.
    for r in results {
        let mut sorted = r.latencies_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let p50_ms = percentile(&sorted, 0.50);
        let p50_bw = gbps(per_nic_bytes, p50_ms / 1000.0);
        println!(
            "  {}: p50={:.2}ms ({:.1} Gbps)",
            r.nic_name, p50_ms, p50_bw,
        );
    }

    // Aggregate: for each iteration, sum the max latency across NICs (barrier-synced,
    // so aggregate time is max across NICs per iter).
    let iters = results[0].latencies_ms.len();
    let mut agg_latencies_ms: Vec<f64> = (0..iters)
        .map(|i| {
            results
                .iter()
                .map(|r| r.latencies_ms[i])
                .fold(0.0_f64, f64::max)
        })
        .collect();
    agg_latencies_ms.sort_by(f64::total_cmp);

    let p50_ms = percentile(&agg_latencies_ms, 0.50);
    let p95_ms = percentile(&agg_latencies_ms, 0.95);
    let p50_gib = gib_per_sec(total_bytes, p50_ms / 1000.0);
    let p50_gbps = gbps(total_bytes, p50_ms / 1000.0);

    println!(
        "  Aggregate: {:.1} Gbps ({:.2} GiB/s)  p50={:.2}ms p95={:.2}ms",
        p50_gbps, p50_gib, p50_ms, p95_ms,
    );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    init_logging();
    let cli = Cli::parse();
    let total_size = parse_size(&cli.size);

    println!(
        "pegaflow_cpu_bench: size={} iters={} warmup={} base_port={}",
        cli.size, cli.iters, cli.warmup, cli.base_port
    );
    if let Some(ref nic) = cli.nic {
        println!("  filter: nic={}", nic);
    }
    if let Some(numa) = cli.numa {
        println!("  filter: numa={}", numa);
    }

    // Detect topology.
    let topo = SystemTopology::detect();
    topo.log_summary();

    // Select NUMA groups.
    let groups: Vec<_> = topo
        .groups()
        .iter()
        .filter(|g| {
            if let Some(numa) = cli.numa {
                g.node.0 == numa
            } else {
                true
            }
        })
        .filter(|g| !g.nics.is_empty())
        .collect();

    if groups.is_empty() {
        eprintln!("error: no NUMA groups with RDMA NICs found");
        std::process::exit(1);
    }

    let mut port_cursor = cli.base_port;

    for group in &groups {
        // Filter NICs.
        let nics: Vec<_> = group
            .nics
            .iter()
            .filter(|nic| {
                if let Some(ref filter) = cli.nic {
                    &nic.name == filter
                } else {
                    true
                }
            })
            .collect();

        if nics.is_empty() {
            continue;
        }

        let nic_count = nics.len();
        let per_nic_bytes = total_size / nic_count;
        if per_nic_bytes == 0 {
            eprintln!(
                "warning: NUMA{} skipped, size too small to split across {} NICs",
                group.node.0, nic_count
            );
            continue;
        }

        println!();
        println!(
            "--- NUMA{}: {} NIC(s) [{}], {:.2} GiB per NIC ---",
            group.node.0,
            nic_count,
            nics.iter()
                .map(|n| n.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            per_nic_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        // Build NIC contexts.
        let mut contexts: Vec<NicContext> = Vec::with_capacity(nic_count);

        for nic in &nics {
            let server_port = port_cursor;
            port_cursor += 1;
            let client_port = port_cursor;
            port_cursor += 1;

            let server_buf = NumaBuffer::alloc(group.node.0, per_nic_bytes);
            let client_buf = NumaBuffer::alloc(group.node.0, per_nic_bytes);

            // Fill server buffer with data (for READ bench), client with different pattern (for WRITE bench).
            server_buf.fill(0xAA);
            client_buf.fill(0xBB);

            let mut server = MooncakeTransferEngine::new();
            server
                .initialize(nic.name.clone(), server_port)
                .expect("server init failed");
            server
                .register_memory(server_buf.as_u64(), per_nic_bytes)
                .expect("server register_memory failed");

            let mut client = MooncakeTransferEngine::new();
            client
                .initialize(nic.name.clone(), client_port)
                .expect("client init failed");
            client
                .register_memory(client_buf.as_u64(), per_nic_bytes)
                .expect("client register_memory failed");

            println!(
                "  {} ready: server_port={} client_port={}",
                nic.name, server_port, client_port
            );

            contexts.push(NicContext {
                nic_name: nic.name.clone(),
                server,
                client,
                server_buf,
                client_buf,
                per_nic_bytes,
            });
        }

        // Pre-establish RC connections sequentially (avoid concurrent UD handshake storms).
        // Use a small transfer (4 KiB) just to trigger connection setup.
        for ctx in &contexts {
            let server_session = ctx.server.get_session_id();
            ctx.client
                .transfer_sync_write(
                    &server_session,
                    ctx.client_buf.as_u64(),
                    ctx.server_buf.as_u64(),
                    4096,
                )
                .expect("pre-connect WRITE failed");
            println!("  {} connected", ctx.nic_name);
        }

        // Run WRITE bench.
        let write_results = run_bench(&contexts, "write", cli.warmup, cli.iters);
        print_results(&write_results, "write", per_nic_bytes, group.node.0);

        // Reset client buffers before READ (clear any stale data).
        for ctx in &contexts {
            ctx.client_buf.fill(0x00);
        }

        // Run READ bench.
        let read_results = run_bench(&contexts, "read", cli.warmup, cli.iters);
        print_results(&read_results, "read", per_nic_bytes, group.node.0);

        // Cleanup: unregister memory.
        for ctx in &contexts {
            ctx.client
                .unregister_memory(ctx.client_buf.as_u64())
                .ok();
            ctx.server
                .unregister_memory(ctx.server_buf.as_u64())
                .ok();
        }
    }

    println!();
    println!("bench complete.");
}
