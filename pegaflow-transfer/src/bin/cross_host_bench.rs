use std::error::Error;
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

use clap::Parser;
use serde::{Deserialize, Serialize};

use pegaflow_transfer::bench_support::{
    BufferSpec, NumaBuffer, block_recv, build_segmented_numa_scatter, format_size, gbps,
    generate_task_schedule, gib_per_sec, parse_block_range, parse_size, percentile, recv_framed,
    send_framed,
};
use pegaflow_transfer::rdma_topo::SystemTopology;
use pegaflow_transfer::{
    ConnectionStatus, HandshakeMetadata, MemoryRegion, TransferEngine, TransferOp, init_logging,
};

#[derive(Parser)]
#[command(
    name = "pegaflow-cross-host-bench",
    version,
    about = "Cross-host RDMA benchmark with the same transfer engine/workload model"
)]
struct Cli {
    /// Control plane listen address, e.g. 0.0.0.0:18515.
    #[arg(long, conflicts_with = "connect")]
    listen: Option<String>,

    /// Control plane server address, e.g. 10.96.191.100:18515.
    #[arg(long, conflicts_with = "listen")]
    connect: Option<String>,

    /// Per-block logical payload size before splitting into descriptors.
    #[arg(long, default_value = "23mb")]
    block_size: String,

    /// Blocks per task: single number (e.g. "160") or range (e.g. "128-192").
    #[arg(long, default_value = "160")]
    blocks_per_task: String,

    /// Descriptors per block. 160 blocks x 8 descs = 1280 descs/task.
    #[arg(long, default_value_t = 8)]
    descs_per_block: usize,

    /// Number of measured tasks.
    #[arg(long, default_value_t = 30)]
    tasks: usize,

    /// Number of warmup tasks.
    #[arg(long, default_value_t = 5)]
    warmup_tasks: usize,

    /// Benchmark mode.
    #[arg(long, default_value = "read", value_parser = ["read", "write", "both"])]
    mode: String,

    /// Restrict to specific NICs. Repeat to include multiple NICs.
    #[arg(long)]
    nic: Vec<String>,

    /// Exclude a NIC. Repeat to exclude multiple NICs.
    #[arg(long)]
    exclude_nic: Vec<String>,

    /// Restrict to specific NUMA nodes. Repeat to include multiple nodes.
    #[arg(long)]
    numa: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchHello {
    nic_names: Vec<String>,
    numa_nodes: Vec<u32>,
    handshake: Vec<u8>,
    buffers: Vec<BufferSpec>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum ControlMessage {
    Ready,
    Done,
}

struct BenchState {
    nic_names: Vec<String>,
    numa_nodes: Vec<u32>,
    local_buffers: Vec<NumaBuffer>,
    local_specs: Vec<BufferSpec>,
    engine: TransferEngine,
}

struct TaskResult {
    latency_ms: f64,
    bytes: usize,
    descs: usize,
}

fn parse_mode(mode: &str) -> Vec<TransferOp> {
    match mode {
        "read" => vec![TransferOp::Read],
        "write" => vec![TransferOp::Write],
        _ => vec![TransferOp::Write, TransferOp::Read],
    }
}

fn select_nics(cli: &Cli) -> Result<(Vec<String>, Vec<u32>), Box<dyn Error>> {
    let topo = SystemTopology::detect();
    topo.log_summary();

    let mut nic_names = Vec::new();
    let mut numa_nodes = Vec::new();
    for group in topo.groups() {
        if !cli.numa.is_empty() && !cli.numa.contains(&group.node.0) {
            continue;
        }

        let selected: Vec<String> = group
            .nics
            .iter()
            .filter(|nic| cli.nic.is_empty() || cli.nic.iter().any(|name| name == &nic.name))
            .filter(|nic| !cli.exclude_nic.iter().any(|name| name == &nic.name))
            .map(|nic| nic.name.clone())
            .collect();
        if selected.is_empty() {
            continue;
        }

        numa_nodes.push(group.node.0);
        nic_names.extend(selected);
    }

    if nic_names.is_empty() {
        return Err("no RDMA NICs selected".into());
    }
    Ok((nic_names, numa_nodes))
}

fn create_state(
    nic_names: Vec<String>,
    numa_nodes: Vec<u32>,
    per_node_buf_size: usize,
) -> Result<BenchState, Box<dyn Error>> {
    let local_buffers: Vec<NumaBuffer> = numa_nodes
        .iter()
        .map(|&node| {
            let buf = NumaBuffer::alloc(node, per_node_buf_size);
            buf.fill(0xAB);
            buf
        })
        .collect();
    let local_specs: Vec<BufferSpec> = local_buffers
        .iter()
        .zip(&numa_nodes)
        .map(|(buf, &numa_node)| BufferSpec {
            numa_node,
            base_ptr: buf.ptr.as_ptr() as u64,
            len: buf.len,
        })
        .collect();

    let engine = TransferEngine::new(&nic_names)?;
    let regions: Vec<MemoryRegion> = local_buffers
        .iter()
        .map(|buf| MemoryRegion {
            ptr: buf.ptr,
            len: buf.len,
        })
        .collect();
    engine.register_memory(&regions)?;

    Ok(BenchState {
        nic_names,
        numa_nodes,
        local_buffers,
        local_specs,
        engine,
    })
}

fn prepare_local_meta(
    engine: &TransferEngine,
    peer_name: &str,
) -> Result<HandshakeMetadata, Box<dyn Error>> {
    match engine.get_or_prepare(peer_name)? {
        ConnectionStatus::Prepared(meta) => Ok(meta),
        ConnectionStatus::Existing | ConnectionStatus::Connecting => {
            Err("unexpected connection state on fresh bench engine".into())
        }
    }
}

fn make_hello(state: &BenchState, local_meta: &HandshakeMetadata) -> BenchHello {
    BenchHello {
        nic_names: state.nic_names.clone(),
        numa_nodes: state.numa_nodes.clone(),
        handshake: local_meta.to_bytes(),
        buffers: state.local_specs.clone(),
    }
}

fn validate_peer(local: &BenchState, peer: &BenchHello) -> Result<(), Box<dyn Error>> {
    if local.nic_names != peer.nic_names {
        return Err(format!(
            "NIC mismatch: local={:?}, peer={:?}",
            local.nic_names, peer.nic_names
        )
        .into());
    }
    if local.numa_nodes != peer.numa_nodes {
        return Err(format!(
            "NUMA mismatch: local={:?}, peer={:?}",
            local.numa_nodes, peer.numa_nodes
        )
        .into());
    }
    if local.local_buffers.len() != peer.buffers.len() {
        return Err(format!(
            "buffer group mismatch: local={}, peer={}",
            local.local_buffers.len(),
            peer.buffers.len()
        )
        .into());
    }
    Ok(())
}

fn print_results(
    op: TransferOp,
    task_results: &[TaskResult],
    block_size: usize,
    descs_per_block: usize,
) {
    let mut latencies: Vec<f64> = task_results.iter().map(|t| t.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);
    let avg_bytes = task_results.iter().map(|t| t.bytes).sum::<usize>() / task_results.len().max(1);
    let avg_descs = task_results.iter().map(|t| t.descs).sum::<usize>() / task_results.len().max(1);
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);

    println!();
    println!("=== Cross-host RDMA {:?} ===", op);
    println!(
        "  avg payload: {:.1} MiB/task  block={}  descs/block={}  avg desc={:.1} KiB",
        avg_bytes as f64 / (1024.0 * 1024.0),
        format_size(block_size),
        descs_per_block,
        avg_bytes as f64 / avg_descs.max(1) as f64 / 1024.0,
    );
    println!("  p50={:.2}ms  p95={:.2}ms  p99={:.2}ms", p50, p95, p99);
    println!(
        "  p50 equiv: {:.1} Gbps ({:.2} GiB/s)",
        gbps(avg_bytes, p50 / 1000.0),
        gib_per_sec(avg_bytes, p50 / 1000.0),
    );
}

fn run_client(
    cli: &Cli,
    mut stream: TcpStream,
    state: &BenchState,
    per_node_buf_size: usize,
    block_size: usize,
    block_range: (usize, usize),
) -> Result<(), Box<dyn Error>> {
    let local_meta = prepare_local_meta(&state.engine, "bench-server")?;
    let peer_hello: BenchHello = recv_framed(&mut stream)?;
    validate_peer(state, &peer_hello)?;
    let remote_meta = HandshakeMetadata::from_bytes(&peer_hello.handshake)?;
    let local_hello = make_hello(state, &local_meta);
    send_framed(&mut stream, &local_hello)?;
    state
        .engine
        .complete_handshake("bench-server", &local_meta, &remote_meta)?;
    let ready: ControlMessage = recv_framed(&mut stream)?;
    if !matches!(ready, ControlMessage::Ready) {
        return Err("peer did not acknowledge handshake".into());
    }

    println!(
        "client ready: peer={} nics=[{}] numa_nodes={:?} per_node_buf={}",
        stream.peer_addr()?,
        state.nic_names.join(", "),
        state.numa_nodes,
        format_size(per_node_buf_size),
    );

    let schedule = generate_task_schedule(cli.warmup_tasks + cli.tasks, block_range, 0x4242);
    let modes = parse_mode(&cli.mode);
    for op in modes {
        if op == TransferOp::Read {
            for buf in &state.local_buffers {
                buf.fill(0);
            }
        }

        let mut task_results = Vec::with_capacity(cli.tasks);
        for (idx, &nblocks) in schedule.iter().enumerate() {
            let descs = build_segmented_numa_scatter(
                &state.local_buffers,
                &peer_hello.buffers,
                nblocks,
                block_size,
                cli.descs_per_block,
            );
            let start = Instant::now();
            let receivers = state
                .engine
                .batch_transfer_async(op, "bench-server", &descs)?;
            for rx in receivers {
                block_recv(rx)??;
            }
            let elapsed = start.elapsed();
            if idx >= cli.warmup_tasks {
                task_results.push(TaskResult {
                    latency_ms: elapsed.as_secs_f64() * 1000.0,
                    bytes: nblocks * block_size,
                    descs: descs.len(),
                });
            }
        }

        print_results(op, &task_results, block_size, cli.descs_per_block);
    }

    send_framed(&mut stream, &ControlMessage::Done)?;
    Ok(())
}

fn run_server(
    listen_addr: &str,
    state: &BenchState,
    per_node_buf_size: usize,
) -> Result<(), Box<dyn Error>> {
    let listener = TcpListener::bind(listen_addr)?;
    println!(
        "server listening on {}  nics=[{}] numa_nodes={:?} per_node_buf={}",
        listener.local_addr()?,
        state.nic_names.join(", "),
        state.numa_nodes,
        format_size(per_node_buf_size),
    );

    let (mut stream, peer_addr) = listener.accept()?;
    stream.set_nodelay(true)?;
    let local_meta = prepare_local_meta(&state.engine, "bench-client")?;
    let local_hello = make_hello(state, &local_meta);
    send_framed(&mut stream, &local_hello)?;
    let peer_hello: BenchHello = recv_framed(&mut stream)?;
    validate_peer(state, &peer_hello)?;
    let remote_meta = HandshakeMetadata::from_bytes(&peer_hello.handshake)?;
    state
        .engine
        .complete_handshake("bench-client", &local_meta, &remote_meta)?;
    send_framed(&mut stream, &ControlMessage::Ready)?;
    println!("server handshake complete with {}", peer_addr);

    let done: ControlMessage = recv_framed(&mut stream)?;
    if !matches!(done, ControlMessage::Done) {
        return Err("client terminated without done message".into());
    }
    println!("server done");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    init_logging();
    let cli = Cli::parse();

    if cli.listen.is_none() == cli.connect.is_none() {
        return Err("exactly one of --listen or --connect must be set".into());
    }

    let block_size = parse_size(&cli.block_size);
    let block_range = parse_block_range(&cli.blocks_per_task);
    let max_blocks = block_range.1;
    if cli.descs_per_block == 0 {
        return Err("--descs-per-block must be > 0".into());
    }
    if block_size % cli.descs_per_block != 0 {
        return Err("block_size must be divisible by descs_per_block".into());
    }

    let (nic_names, numa_nodes) = select_nics(&cli)?;
    let per_node_blocks = max_blocks.div_ceil(numa_nodes.len());
    let per_node_buf_size = per_node_blocks * block_size;
    let state = create_state(nic_names, numa_nodes, per_node_buf_size)?;

    println!(
        "bench config: block_size={} blocks_per_task={} descs_per_block={} tasks={} warmup={}",
        cli.block_size, cli.blocks_per_task, cli.descs_per_block, cli.tasks, cli.warmup_tasks
    );
    println!(
        "  task max payload={} total_descs={} desc_size={}",
        format_size(max_blocks * block_size),
        max_blocks * cli.descs_per_block,
        format_size(block_size / cli.descs_per_block),
    );

    if let Some(listen_addr) = cli.listen.as_deref() {
        run_server(listen_addr, &state, per_node_buf_size)?;
    } else if let Some(connect_addr) = cli.connect.as_deref() {
        let stream = TcpStream::connect(connect_addr)?;
        stream.set_nodelay(true)?;
        run_client(
            &cli,
            stream,
            &state,
            per_node_buf_size,
            block_size,
            block_range,
        )?;
    }

    let ptrs: Vec<_> = state.local_buffers.iter().map(|buf| buf.ptr).collect();
    state.engine.unregister_memory(&ptrs).ok();
    Ok(())
}
