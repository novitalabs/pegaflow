//! End-to-end p2p RDMA fetch benchmark and integrity test.
//!
//! Drives the production cross-node path without vLLM. The holder saves
//! blocks through the real GPU save path with a vLLM-shaped layout
//! (`layers x tp_ranks` slots per block, split K/V segments), registers them
//! in an in-process MetaServer, and serves the production Engine gRPC
//! service. The requester registers the same layout, then walks the real
//! prefetch path: MetaServer discovery -> QueryBlocksForTransfer ->
//! RDMA READ (requester reads holder slabs) -> SealedBlock rebuild, and
//! optionally loads the blocks to GPU and verifies every byte.
//!
//! Holder (node A):
//!   p2p_bench --role holder --advertise-ip <A> --nics mlx5_0,...
//! Requester (node B):
//!   p2p_bench --role requester --holder-ip <A> --advertise-ip <B> \
//!       --nics mlx5_0,... --verify

use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use cudarc::driver::{CudaContext, sys};
use log::info;
use pegaflow_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use pegaflow_core::*;
use pegaflow_metaserver::{BlockHashStore, GrpcMetaService};
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService, RegistryHandle};
use tokio::sync::Notify;
use tonic::transport::Server;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Role {
    Holder,
    Requester,
}

#[derive(Parser)]
#[command(
    name = "p2p_bench",
    about = "End-to-end p2p RDMA fetch benchmark (production path, no vLLM)"
)]
struct Cli {
    #[arg(long, value_enum)]
    role: Role,

    /// This node's routable IP, used for gRPC advertise / MetaServer identity.
    #[arg(long)]
    advertise_ip: String,

    /// Holder's routable IP (requester only).
    #[arg(long)]
    holder_ip: Option<String>,

    /// Engine gRPC port (holder listens; both sides use it as identity).
    #[arg(long, default_value_t = 50855)]
    port: u16,

    /// MetaServer port (holder hosts it in-process).
    #[arg(long, default_value_t = 50856)]
    meta_port: u16,

    /// RDMA NIC names, comma separated. Empty = all RDMA NICs.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    nics: Vec<String>,

    /// Model shape: transformer layers (block slots = layers * tp).
    #[arg(long, default_value_t = 36)]
    layers: usize,

    /// Model shape: tensor-parallel ranks; rank r uses CUDA device r.
    #[arg(long, default_value_t = 8)]
    tp: usize,

    /// K segment bytes per (block, layer, rank); V segment is the same size.
    #[arg(long, default_value_t = 4096)]
    kv_bytes: usize,

    /// Blocks per set (one set ~= one request's KV prefix).
    #[arg(long, default_value_t = 1000)]
    blocks: usize,

    /// Distinct block sets. Set 0 is warmup; sets 1.. are measured.
    #[arg(long, default_value_t = 4)]
    sets: usize,

    /// Pinned pool size in GiB. 0 = auto (sets * set bytes + 25% slack).
    #[arg(long, default_value_t = 0)]
    pool_gib: usize,

    /// Allocate the pinned pool with huge pages.
    #[arg(long)]
    use_hugepages: bool,

    /// Requester: load every fetched block to GPU and verify every byte.
    #[arg(long)]
    verify: bool,
}

const NAMESPACE: &str = "p2p-bench";
const INSTANCE: &str = "p2p-bench-inst";

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

struct GpuBuffer {
    ctx: Arc<CudaContext>,
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(ctx: Arc<CudaContext>, len: usize) -> Self {
        ctx.bind_to_thread().expect("bind CUDA context");
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ctx, ptr, len }
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        self.ctx.bind_to_thread().expect("bind CUDA context");
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        self.ctx.bind_to_thread().expect("bind CUDA context");
        let mut out = vec![0u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        out
    }

    fn zero(&self) {
        self.ctx.bind_to_thread().expect("bind CUDA context");
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.ptr, 0, self.len) },
            "memset",
        );
    }
}

/// One tp rank's GPU state: a single buffer backing all layers,
/// layer l at offset `l * layer_bytes`, K run then V run inside a layer.
struct RankBuffers {
    device: usize,
    buf: GpuBuffer,
}

struct Shape {
    layers: usize,
    tp: usize,
    kv_bytes: usize,
    blocks: usize,
}

impl Shape {
    fn layer_bytes(&self) -> usize {
        2 * self.blocks * self.kv_bytes
    }

    fn rank_bytes(&self) -> usize {
        self.layers * self.layer_bytes()
    }

    fn set_bytes(&self) -> usize {
        self.tp * self.rank_bytes()
    }

    fn layer_names(&self) -> Vec<String> {
        (0..self.layers).map(|l| format!("layer_{l}")).collect()
    }
}

/// Deterministic content for one (set, rank, layer, block, segment) cell —
/// both sides derive it independently, so the requester can verify bytes
/// without shipping the expected data out of band.
fn cell_fill(set: usize, rank: usize, layer: usize, block: usize, seg: usize) -> u8 {
    ((set * 131 + rank * 17 + layer * 7 + block * 3 + seg * 11) % 251 + 1) as u8
}

/// Fill one rank's host image for `set` in the registered GPU layout.
fn fill_rank_image(shape: &Shape, set: usize, rank: usize, out: &mut [u8]) {
    assert_eq!(out.len(), shape.rank_bytes());
    let kv = shape.kv_bytes;
    for layer in 0..shape.layers {
        let layer_base = layer * shape.layer_bytes();
        for seg in 0..2 {
            let seg_base = layer_base + seg * shape.blocks * kv;
            for block in 0..shape.blocks {
                let fill = cell_fill(set, rank, layer, block, seg);
                out[seg_base + block * kv..seg_base + (block + 1) * kv].fill(fill);
            }
        }
    }
}

fn make_block_hashes(blocks: usize, set: usize) -> Vec<Vec<u8>> {
    (0..blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(8);
            hash.extend_from_slice(&(set as u32).to_le_bytes());
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

fn register_ranks(engine: &PegaEngine, shape: &Shape) -> Vec<RankBuffers> {
    let layer_names = shape.layer_names();
    (0..shape.tp)
        .map(|rank| {
            let ctx = CudaContext::new(rank).expect("CUDA context");
            let buf = GpuBuffer::alloc(ctx, shape.rank_bytes());
            let ptrs: Vec<u64> = (0..shape.layers)
                .map(|l| buf.ptr + (l * shape.layer_bytes()) as u64)
                .collect();
            engine
                .register_context_layer_batch(
                    INSTANCE,
                    NAMESPACE,
                    rank as i32, // device_id
                    rank,        // tp_rank
                    0,           // pp_rank
                    shape.tp,
                    shape.tp, // world_size
                    &layer_names,
                    &ptrs,
                    &vec![shape.layer_bytes(); shape.layers],
                    &vec![shape.blocks; shape.layers],
                    &vec![shape.kv_bytes; shape.layers],
                    &vec![shape.blocks * shape.kv_bytes; shape.layers], // kv_stride
                    &vec![2; shape.layers],                             // split K/V
                    TransferMode::Direct,
                    false,
                )
                .expect("register rank");
            RankBuffers { device: rank, buf }
        })
        .collect()
}

async fn wait_until<F: FnMut() -> Option<usize>>(
    what: &str,
    expected: usize,
    timeout: Duration,
    mut probe: F,
) {
    let deadline = Instant::now() + timeout;
    let mut last = 0;
    loop {
        if let Some(n) = probe() {
            last = n;
            if n >= expected {
                return;
            }
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what} ({last}/{expected})"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn cached_blocks(engine: &PegaEngine, req_id: &str, hashes: &[Vec<u8>]) -> usize {
    match engine
        .count_prefix_hit_blocks_with_prefetch(INSTANCE, req_id, hashes)
        .await
        .expect("count_prefix_hit_blocks_with_prefetch")
    {
        PrefetchStatus::Ready { blocks, .. } => blocks.len(),
        PrefetchStatus::Loading => 0,
    }
}

async fn run_holder(cli: &Cli, shape: &Shape, pool_bytes: usize) {
    let meta_addr: SocketAddr = ([0, 0, 0, 0], cli.meta_port).into();
    let meta_store = Arc::new(BlockHashStore::new());
    let meta_service = GrpcMetaService::new(Arc::clone(&meta_store));
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServerServer::new(meta_service))
            .serve(meta_addr)
            .await
            .expect("MetaServer gRPC serve");
    });

    let config = StorageConfig {
        metaserver_addr: Some(format!("http://127.0.0.1:{}", cli.meta_port)),
        advertise_addr: Some(format!("{}:{}", cli.advertise_ip, cli.port)),
        rdma_nic_names: nic_config(cli),
        max_prefetch_blocks: shape.blocks + 100,
        ..StorageConfig::default()
    };
    let engine = Arc::new(
        PegaEngine::new_with_config(pool_bytes, cli.use_hugepages, config).expect("holder engine"),
    );

    // The bench saves in-process (no gRPC IPC saves), so the registry stays
    // empty; `empty()` avoids dragging an embedded-Python torch import into
    // a bench that never registers IPC tensors.
    let registry = RegistryHandle::spawn(CudaTensorRegistry::empty());
    let shutdown = Arc::new(Notify::new());
    let hll = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            vec![("24h".into(), Duration::from_secs(86400))],
            14,
        ),
    ));
    let service = GrpcEngineService::new(Arc::clone(&engine), registry, shutdown, hll);
    let listen: SocketAddr = ([0, 0, 0, 0], cli.port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(listen)
            .await
            .expect("Engine gRPC serve");
    });

    let ranks = register_ranks(&engine, shape);
    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    let layer_names = shape.layer_names();

    let mut image = vec![0u8; shape.rank_bytes()];
    for set in 0..cli.sets {
        let hashes = make_block_hashes(shape.blocks, set);
        for (rank, rb) in ranks.iter().enumerate() {
            fill_rank_image(shape, set, rank, &mut image);
            rb.buf.copy_from_host(&image);
            let saves = layer_names
                .iter()
                .map(|name| LayerSave {
                    layer_name: name.clone(),
                    block_ids: block_ids.clone(),
                    block_hashes: hashes.clone(),
                })
                .collect();
            engine
                .batch_save_kv_blocks_from_ipc(INSTANCE, rank, 0, rb.device as i32, saves)
                .await
                .expect("save");
        }
        let seal_deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let hit = cached_blocks(&engine, &format!("seal-{set}"), &hashes).await;
            if hit >= shape.blocks {
                break;
            }
            assert!(
                Instant::now() < seal_deadline,
                "timed out sealing set {set} ({hit}/{} blocks)",
                shape.blocks
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        wait_until(
            "metaserver registration",
            shape.blocks,
            Duration::from_secs(60),
            || Some(meta_store.query_prefix(NAMESPACE, &hashes).len()),
        )
        .await;
        info!(
            "holder: set {set}/{} sealed and registered ({:.1} MiB)",
            cli.sets,
            shape.set_bytes() as f64 / (1024.0 * 1024.0)
        );
    }

    println!(
        "HOLDER_READY sets={} blocks={} set_mib={:.1}",
        cli.sets,
        shape.blocks,
        shape.set_bytes() as f64 / (1024.0 * 1024.0)
    );
    tokio::signal::ctrl_c().await.expect("ctrl_c");
}

async fn run_requester(cli: &Cli, shape: &Shape, pool_bytes: usize) {
    let holder_ip = cli
        .holder_ip
        .as_deref()
        .expect("--holder-ip is required for the requester");
    let config = StorageConfig {
        metaserver_addr: Some(format!("http://{holder_ip}:{}", cli.meta_port)),
        advertise_addr: Some(format!("{}:{}", cli.advertise_ip, cli.port)),
        rdma_nic_names: nic_config(cli),
        max_prefetch_blocks: shape.blocks + 100,
        ..StorageConfig::default()
    };
    let engine = Arc::new(
        PegaEngine::new_with_config(pool_bytes, cli.use_hugepages, config)
            .expect("requester engine"),
    );
    let ranks = register_ranks(&engine, shape);
    let set_mib = shape.set_bytes() as f64 / (1024.0 * 1024.0);

    let mut results: Vec<(usize, f64)> = Vec::new();
    for set in 0..cli.sets {
        let hashes = make_block_hashes(shape.blocks, set);
        let t0 = Instant::now();
        let deadline = t0 + Duration::from_secs(120);
        loop {
            let hit = cached_blocks(&engine, &format!("fetch-{set}"), &hashes).await;
            if hit >= shape.blocks {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timed out fetching set {set} ({hit}/{} blocks)",
                shape.blocks
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let elapsed = t0.elapsed();
        let gib_s = shape.set_bytes() as f64 / (1024.0 * 1024.0 * 1024.0) / elapsed.as_secs_f64();
        let label = if set == 0 { "warmup" } else { "measure" };
        println!(
            "FETCH {label} set={set} mib={set_mib:.1} ms={:.2} gib_s={gib_s:.2}",
            elapsed.as_secs_f64() * 1000.0
        );
        if set > 0 {
            results.push((set, elapsed.as_secs_f64()));
        }

        if cli.verify {
            verify_set(&engine, shape, set, &hashes, &ranks).await;
        }
    }

    if !results.is_empty() {
        let mut times: Vec<f64> = results.iter().map(|(_, t)| *t).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[times.len() / 2];
        println!(
            "SUMMARY sets_measured={} set_mib={set_mib:.1} median_ms={:.2} median_gib_s={:.2}",
            times.len(),
            median * 1000.0,
            shape.set_bytes() as f64 / (1024.0 * 1024.0 * 1024.0) / median
        );
    }
}

/// Load `set` to GPU through the production load path and compare every byte
/// against the generator pattern.
async fn verify_set(
    engine: &PegaEngine,
    shape: &Shape,
    set: usize,
    hashes: &[Vec<u8>],
    ranks: &[RankBuffers],
) {
    let blocks = match engine
        .count_prefix_hit_blocks_with_prefetch(INSTANCE, &format!("verify-{set}"), hashes)
        .await
        .expect("verify query")
    {
        PrefetchStatus::Ready { blocks, .. } => blocks,
        PrefetchStatus::Loading => panic!("verify: set {set} still loading"),
    };
    assert_eq!(blocks.len(), shape.blocks, "verify: incomplete set {set}");
    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    let layer_names = shape.layer_names();
    let layer_name_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

    let mut expected = vec![0u8; shape.rank_bytes()];
    for (rank, rb) in ranks.iter().enumerate() {
        let lease = engine
            .create_query_lease(INSTANCE, blocks.clone())
            .expect("create lease");
        rb.buf.zero();
        let load_state = LoadState::new().expect("LoadState");
        engine
            .batch_load_kv_blocks_multi_layer(
                INSTANCE,
                rank,
                rb.device as i32,
                load_state.shm_name(),
                &layer_name_refs,
                &[(lease, block_ids.clone())],
            )
            .expect("batch_load");
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let state = load_state.get();
            if state == LOAD_STATE_SUCCESS {
                break;
            }
            assert!(state != LOAD_STATE_ERROR, "load reported ERROR");
            assert!(Instant::now() < deadline, "timed out waiting for load");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let loaded = rb.buf.copy_to_host();
        fill_rank_image(shape, set, rank, &mut expected);
        assert!(
            loaded == expected,
            "verify FAILED: set {set} rank {rank} differs from generator pattern"
        );
    }
    println!(
        "VERIFY set={set} ok (all {} ranks, every byte)",
        ranks.len()
    );
}

fn nic_config(cli: &Cli) -> Option<Vec<String>> {
    if cli.nics.is_empty() {
        None
    } else {
        Some(cli.nics.clone())
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() {
    let cli = Cli::parse();
    pegaflow_common::logging::init_stdout_colored("info");

    let shape = Shape {
        layers: cli.layers,
        tp: cli.tp,
        kv_bytes: cli.kv_bytes,
        blocks: cli.blocks,
    };
    // Generous slack: if the pool runs tight the engine evicts earlier sets
    // (and deregisters them from MetaServer), which breaks the run.
    let pool_bytes = if cli.pool_gib > 0 {
        cli.pool_gib << 30
    } else {
        (cli.sets * shape.set_bytes() + shape.set_bytes() / 2).max(1 << 30)
    };
    info!(
        "p2p_bench role={:?} shape: layers={} tp={} kv_bytes={} blocks={} sets={} set_bytes={:.1}MiB pool={:.1}GiB",
        cli.role,
        shape.layers,
        shape.tp,
        shape.kv_bytes,
        shape.blocks,
        cli.sets,
        shape.set_bytes() as f64 / (1024.0 * 1024.0),
        pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    match cli.role {
        Role::Holder => run_holder(&cli, &shape, pool_bytes).await,
        Role::Requester => run_requester(&cli, &shape, pool_bytes).await,
    }
}
