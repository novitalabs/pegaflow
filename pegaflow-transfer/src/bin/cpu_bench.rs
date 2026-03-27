// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-memory RDMA benchmark: measures per-task RDMA READ vs WRITE latency across NUMA nodes and NICs.
//!
//! Models a realistic workload: each "task" transfers N random-sized batches of fixed-size blocks
//! via RDMA, measuring per-task latency. Runs single-NIC baselines then multi-NIC aggregate.

use std::time::Instant;

use clap::Parser;
use pegaflow_transfer::bench_support::{
    NumaBuffer, block_recv, build_block_scatter, build_numa_scatter, format_size, gbps,
    generate_task_schedule, gib_per_sec, parse_block_range, parse_size, percentile,
};
use pegaflow_transfer::rdma_topo::SystemTopology;
use pegaflow_transfer::{MemoryRegion, TransferEngine, TransferOp, init_logging};

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
        let receivers = ctx
            .client
            .batch_transfer_async(op, "bench-server", &descs)
            .expect("RDMA submit failed");
        for rx in receivers {
            block_recv(rx)
                .expect("completion channel dropped")
                .expect("RDMA transfer failed");
        }
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

    // -----------------------------------------------------------------
    // Phase: All-NIC aggregate (cross-NUMA)
    // -----------------------------------------------------------------
    if cli.nic.is_none() && cli.numa.is_none() {
        let mut all_nic_names: Vec<String> = Vec::new();
        let mut numa_nodes_used: Vec<u32> = Vec::new();
        for group in &groups {
            let nics: Vec<_> = group
                .nics
                .iter()
                .filter(|nic| cli.exclude_nic.as_ref() != Some(&nic.name))
                .collect();
            if !nics.is_empty() {
                for nic in &nics {
                    all_nic_names.push(nic.name.clone());
                }
                numa_nodes_used.push(group.node.0);
            }
        }

        if all_nic_names.len() > 1 && numa_nodes_used.len() > 1 {
            let total_nics = all_nic_names.len();
            let num_nodes = numa_nodes_used.len();
            let blocks_per_node = max_blocks.div_ceil(num_nodes);
            let per_node_buf_size = blocks_per_node * block_size;

            println!();
            println!(
                "--- ALL NICs: {} NIC(s) across {} NUMA nodes [{}] ---",
                total_nics,
                num_nodes,
                all_nic_names.join(", "),
            );

            let server_bufs: Vec<NumaBuffer> = numa_nodes_used
                .iter()
                .map(|&node| {
                    let buf = NumaBuffer::alloc(node, per_node_buf_size);
                    buf.fill(0xAA);
                    buf
                })
                .collect();
            let client_bufs: Vec<NumaBuffer> = numa_nodes_used
                .iter()
                .map(|&node| {
                    let buf = NumaBuffer::alloc(node, per_node_buf_size);
                    buf.fill(0xBB);
                    buf
                })
                .collect();

            let server = TransferEngine::new(&all_nic_names).expect("all-NIC server init failed");
            let client = TransferEngine::new(&all_nic_names).expect("all-NIC client init failed");

            for buf in &server_bufs {
                server
                    .register_memory(&[MemoryRegion {
                        ptr: buf.ptr,
                        len: per_node_buf_size,
                    }])
                    .expect("all-NIC server register_memory failed");
            }
            for buf in &client_bufs {
                client
                    .register_memory(&[MemoryRegion {
                        ptr: buf.ptr,
                        len: per_node_buf_size,
                    }])
                    .expect("all-NIC client register_memory failed");
            }

            let server_meta = match server
                .get_or_prepare("bench-client")
                .expect("all-NIC server get_or_prepare failed")
            {
                pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
                _ => panic!("unexpected state on fresh engine"),
            };
            let client_meta = match client
                .get_or_prepare("bench-server")
                .expect("all-NIC client get_or_prepare failed")
            {
                pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
                _ => panic!("unexpected state on fresh engine"),
            };
            server
                .complete_handshake("bench-client", &server_meta, &client_meta)
                .expect("all-NIC server complete_handshake failed");
            client
                .complete_handshake("bench-server", &client_meta, &server_meta)
                .expect("all-NIC client complete_handshake failed");

            assert_eq!(server.num_qps(), total_nics);
            assert_eq!(client.num_qps(), total_nics);
            println!("  all-NIC engine ready (handshake complete)");

            let numas_label = numa_nodes_used
                .iter()
                .map(|n| format!("NUMA{n}"))
                .collect::<Vec<_>>()
                .join("+");
            let nic_label = all_nic_names.join(", ");

            let run_all_nic = |op: TransferOp| {
                if op == TransferOp::Read {
                    for buf in &client_bufs {
                        buf.fill(0x00);
                    }
                }

                let mut tasks = Vec::with_capacity(schedule.len().saturating_sub(cli.warmup_tasks));
                for (i, &nblocks) in schedule.iter().enumerate() {
                    let descs = build_numa_scatter(&client_bufs, &server_bufs, nblocks, block_size);

                    let start = Instant::now();
                    let receivers = client
                        .batch_transfer_async(op, "bench-server", &descs)
                        .expect("RDMA submit failed");
                    for rx in receivers {
                        block_recv(rx)
                            .expect("completion channel dropped")
                            .expect("RDMA transfer failed");
                    }
                    let elapsed = start.elapsed();

                    if i >= cli.warmup_tasks {
                        tasks.push(TaskResult {
                            latency_ms: elapsed.as_secs_f64() * 1000.0,
                            bytes: nblocks * block_size,
                        });
                    }
                }

                let mut latencies: Vec<f64> = tasks.iter().map(|t| t.latency_ms).collect();
                latencies.sort_by(f64::total_cmp);
                let avg_bytes = tasks.iter().map(|t| t.bytes).sum::<usize>() / tasks.len().max(1);
                let avg_blocks = avg_bytes / block_size;
                let p50 = percentile(&latencies, 0.50);
                let p95 = percentile(&latencies, 0.95);
                let p99 = percentile(&latencies, 0.99);

                println!();
                println!(
                    "=== RDMA {:?} — ALL NICs ({}, {} NICs) [{}] ===",
                    op, numas_label, total_nics, nic_label,
                );
                println!(
                    "  avg {} blocks x {}  ({:.1} MB/task, NUMA-interleaved across {} nodes)",
                    avg_blocks,
                    format_size(block_size),
                    avg_bytes as f64 / (1024.0 * 1024.0),
                    num_nodes,
                );
                println!("  p50={:.2}ms  p95={:.2}ms  p99={:.2}ms", p50, p95, p99);
                println!(
                    "  p50 equiv: {:.1} Gbps ({:.2} GiB/s)",
                    gbps(avg_bytes, p50 / 1000.0),
                    gib_per_sec(avg_bytes, p50 / 1000.0),
                );
            };

            match cli.mode.as_str() {
                "write" => run_all_nic(TransferOp::Write),
                "read" => run_all_nic(TransferOp::Read),
                _ => {
                    run_all_nic(TransferOp::Write);
                    run_all_nic(TransferOp::Read);
                }
            }

            for buf in &client_bufs {
                client.unregister_memory(&[buf.ptr]).ok();
            }
            for buf in &server_bufs {
                server.unregister_memory(&[buf.ptr]).ok();
            }
        }
    }

    println!();
    println!("bench complete.");
}
