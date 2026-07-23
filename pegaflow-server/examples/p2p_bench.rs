//! End-to-end p2p RDMA fetch benchmark and integrity test.
//!
//! Drives the production cross-node path without vLLM. The holder saves
//! blocks through the real GPU save path, registers them in an in-process
//! MetaServer, and serves the production Engine gRPC service. The requester
//! registers the same layout, then walks the real prefetch path: MetaServer
//! discovery -> QueryBlocksForTransfer -> RDMA READ (requester reads holder
//! slabs) -> SealedBlock rebuild, and optionally loads to GPU and verifies
//! every byte.
//!
//! Two layouts:
//!   * `--model uniform` (default): vLLM dense-attention shape, `layers`
//!     equal-size split-K/V layers (`--segments 2`). One host slot per
//!     (layer, tp_rank).
//!   * `--model glm51`: GLM-5.1 (glm_moe_dsa) MLA shape. Each transformer
//!     layer contributes two single-segment KV layers -- an MLA latent cache
//!     (kv_lora_rank 512 + qk_rope_head_dim 64 = 576 dims) and a DSA sparse
//!     indexer k_cache (index_head_dim 128), both bf16. MLA replicates KV
//!     across TP, so the realistic transfer shape is `--tp 1` (one effective
//!     copy).
//!
//! `--page-first` collapses host slots from (layers x tp) to (tp): one
//! contiguous page per (block, tp_rank) instead of one slot per
//! (block, layer, tp_rank). This cuts RDMA descriptors and the
//! QueryBlocksForTransfer response payload by ~num_layers. Page-first requires
//! single-segment layers (it rejects split K/V).
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

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq)]
enum Model {
    /// vLLM dense-attention: `layers` equal-size split-K/V layers.
    Uniform,
    /// GLM-5.1 (glm_moe_dsa): per transformer layer, an MLA latent cache +
    /// a DSA indexer k_cache, both single-segment bf16.
    Glm51,
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

    /// Layout model.
    #[arg(long, value_enum, default_value_t = Model::Uniform)]
    model: Model,

    /// `uniform`: transformer layers (block slots = layers * tp).
    #[arg(long, default_value_t = 36)]
    layers: usize,

    /// `glm51`: transformer layers incl. MTP; KV layers = 2 * this.
    #[arg(long, default_value_t = 79)]
    glm_layers: usize,

    /// Tokens per block (KV page size). `glm51` derives per-layer bytes from it.
    #[arg(long, default_value_t = 64)]
    block_size: usize,

    /// `uniform`: K segment bytes per (block, layer, rank). V segment matches.
    #[arg(long, default_value_t = 4096)]
    kv_bytes: usize,

    /// `uniform`: segments per slot (2 = split K/V, 1 = single). Page-first
    /// requires 1.
    #[arg(long, default_value_t = 2)]
    segments: usize,

    /// Tensor-parallel ranks; rank r uses CUDA device r. MLA uses tp=1.
    #[arg(long, default_value_t = 8)]
    tp: usize,

    /// Store one contiguous page per (block, tp_rank) instead of one slot per
    /// (block, layer, tp_rank). Collapses metadata ~num_layers. Single-segment only.
    #[arg(long)]
    page_first: bool,

    /// Model the KV as a single MLA replica (effective_tp_size=1): register all
    /// `--tp` GPUs into one slot space (tp_rank=0, unique device_id) and
    /// block-stripe the save (GPU r stores blocks where block_id % tp == r, all
    /// layers). Stores ONE logical copy striped across the GPUs / both NUMA
    /// nodes -- the real GLM-5.1 MLA-TP transfer shape. Without it, `--tp` is
    /// dense (N independent full copies).
    #[arg(long)]
    mla_replica: bool,

    /// Blocks per set (one set ~= one request's KV prefix = seq_len / block_size).
    #[arg(long, default_value_t = 1000)]
    blocks: usize,

    /// Distinct block sets. Set 0 is warmup; sets 1.. are measured.
    #[arg(long, default_value_t = 4)]
    sets: usize,

    /// Pinned pool size in GiB. 0 = auto (sets * set bytes + 50% slack).
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

/// One tp rank's GPU state: a single buffer backing all layers, layer `i` at
/// `layer_offset(i)`, segments laid out back to back inside a layer.
struct RankBuffers {
    device: usize,
    buf: GpuBuffer,
}

/// One KV layer's per-block geometry.
#[derive(Clone, Copy)]
struct LayerSpec {
    /// Bytes per block for ONE segment of this layer.
    kv_bytes: usize,
    /// Segments per slot: 1 (single latent) or 2 (split K/V).
    segments: usize,
}

impl LayerSpec {
    /// Bytes one rank stores for this layer across all blocks and segments.
    fn total_bytes(&self, blocks: usize) -> usize {
        self.segments * blocks * self.kv_bytes
    }
}

struct Shape {
    layers: Vec<LayerSpec>,
    /// Physical GPU / device count.
    tp: usize,
    blocks: usize,
    /// One MLA replica striped across the `tp` GPUs (effective_tp_size=1).
    mla_replica: bool,
}

impl Shape {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Effective tp_size the engine sees: 1 for an MLA replica, else the
    /// physical GPU count (dense TP).
    fn eng_tp_size(&self) -> usize {
        if self.mla_replica { 1 } else { self.tp }
    }

    /// Byte offset of layer `i` inside a rank's buffer (prefix sum).
    fn layer_offset(&self, i: usize) -> usize {
        self.layers[..i]
            .iter()
            .map(|l| l.total_bytes(self.blocks))
            .sum()
    }

    fn rank_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.total_bytes(self.blocks)).sum()
    }

    /// Bytes stored and transferred for one set: dense = `tp` independent
    /// copies; mla-replica = one copy striped across the `tp` GPUs.
    fn set_bytes(&self) -> usize {
        if self.mla_replica {
            self.rank_bytes()
        } else {
            self.tp * self.rank_bytes()
        }
    }

    fn layer_names(&self) -> Vec<String> {
        (0..self.num_layers())
            .map(|l| format!("layer_{l}"))
            .collect()
    }

    /// Host slots a single block occupies on the fetch path.
    fn slots_per_block(&self, page_first: bool) -> usize {
        let eng_tp = self.eng_tp_size();
        if page_first {
            eng_tp
        } else {
            self.num_layers() * eng_tp
        }
    }

    /// RDMA read descriptors the requester issues for one set.
    fn descriptors(&self, page_first: bool) -> usize {
        let eng_tp = self.eng_tp_size();
        if page_first {
            // one contiguous page per (block, slot), single segment
            self.blocks * eng_tp
        } else {
            let segs_per_rank: usize = self.layers.iter().map(|l| l.segments).sum();
            self.blocks * segs_per_rank * eng_tp
        }
    }
}

/// Build the per-layer geometry from CLI flags.
fn build_layers(cli: &Cli) -> Vec<LayerSpec> {
    match cli.model {
        Model::Uniform => (0..cli.layers)
            .map(|_| LayerSpec {
                kv_bytes: cli.kv_bytes,
                segments: cli.segments,
            })
            .collect(),
        Model::Glm51 => {
            // GLM-5.1 (glm_moe_dsa) per-transformer-layer caches, both single
            // segment, bf16. MTP is counted in glm_layers.
            const MLA_DIM: usize = 512 + 64; // kv_lora_rank + qk_rope_head_dim
            const IDX_DIM: usize = 128; // index_head_dim
            const ELEM: usize = 2; // bf16
            let attn = LayerSpec {
                kv_bytes: cli.block_size * MLA_DIM * ELEM,
                segments: 1,
            };
            let idx = LayerSpec {
                kv_bytes: cli.block_size * IDX_DIM * ELEM,
                segments: 1,
            };
            (0..cli.glm_layers).flat_map(|_| [attn, idx]).collect()
        }
    }
}

/// Deterministic content for one (set, rank, layer, block, segment) cell --
/// both sides derive it independently, so the requester can verify bytes
/// without shipping the expected data out of band.
fn cell_fill(set: usize, rank: usize, layer: usize, block: usize, seg: usize) -> u8 {
    ((set * 131 + rank * 17 + layer * 7 + block * 3 + seg * 11) % 251 + 1) as u8
}

/// Fill one rank's host image for `set` in the registered GPU layout.
fn fill_rank_image(shape: &Shape, set: usize, rank: usize, out: &mut [u8]) {
    assert_eq!(out.len(), shape.rank_bytes());
    for (layer, spec) in shape.layers.iter().enumerate() {
        let layer_base = shape.layer_offset(layer);
        let kv = spec.kv_bytes;
        for seg in 0..spec.segments {
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

fn register_ranks(engine: &PegaEngine, shape: &Shape, page_first: bool) -> Vec<RankBuffers> {
    let layer_names = shape.layer_names();
    let size_bytes: Vec<usize> = shape
        .layers
        .iter()
        .map(|l| l.total_bytes(shape.blocks))
        .collect();
    let bytes_per_block: Vec<usize> = shape.layers.iter().map(|l| l.kv_bytes).collect();
    let kv_stride: Vec<usize> = shape
        .layers
        .iter()
        .map(|l| shape.blocks * l.kv_bytes)
        .collect();
    let segments: Vec<usize> = shape.layers.iter().map(|l| l.segments).collect();
    let num_blocks = vec![shape.blocks; shape.num_layers()];
    // MLA replica: every GPU registers tp_rank=0, tp_size=1 (one slot space),
    // distinguished only by device_id. Dense: tp_rank=device, tp_size=tp.
    let eng_tp = shape.eng_tp_size();

    (0..shape.tp)
        .map(|rank| {
            let ctx = CudaContext::new(rank).expect("CUDA context");
            let buf = GpuBuffer::alloc(ctx, shape.rank_bytes());
            let ptrs: Vec<u64> = (0..shape.num_layers())
                .map(|l| buf.ptr + shape.layer_offset(l) as u64)
                .collect();
            let tp_rank = if shape.mla_replica { 0 } else { rank };
            engine
                .register_context_layer_batch(
                    INSTANCE,
                    NAMESPACE,
                    rank as i32, // device_id (the worker key)
                    tp_rank,
                    0, // pp_rank
                    eng_tp,
                    shape.tp, // world_size = physical GPU count
                    &layer_names,
                    &ptrs,
                    &size_bytes,
                    &num_blocks,
                    &bytes_per_block,
                    &kv_stride,
                    &segments,
                    TransferMode::Direct,
                    page_first,
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
        .count_prefix_hit_blocks_with_prefetch(INSTANCE, req_id, hashes, false)
        .await
        .expect("count_prefix_hit_blocks_with_prefetch")
    {
        PrefetchStatus::Ready { blocks, .. } => blocks.len(),
        PrefetchStatus::Loading => 0,
    }
}

/// Print the layout's metadata footprint -- the quantities page-first shrinks.
fn report_metadata(shape: &Shape, page_first: bool) {
    let slots = shape.blocks * shape.slots_per_block(page_first);
    let descs = shape.descriptors(page_first);
    // QueryBlocksForTransferResponse carries one TransferSlotInfo per host
    // slot (5 fields; v_ptr/v_size are 0 for single-segment) plus a per-block
    // hash. ~25 B/slot + ~12 B/block on the wire is a close estimate.
    let query_bytes = slots * 25 + shape.blocks * 12;
    println!(
        "META page_first={page_first} model_layers={} tp={} blocks={} \
         set_mib={:.1} slots_per_block={} total_slots={} rdma_descriptors={} \
         query_resp_kib={:.1}",
        shape.num_layers(),
        shape.tp,
        shape.blocks,
        shape.set_bytes() as f64 / (1024.0 * 1024.0),
        shape.slots_per_block(page_first),
        slots,
        descs,
        query_bytes as f64 / 1024.0,
    );
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
    // Bench saves in-process; no native VMM clients, so no fd side-channel.
    let service = GrpcEngineService::new(Arc::clone(&engine), registry, shutdown, hll, None);
    let listen: SocketAddr = ([0, 0, 0, 0], cli.port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(listen)
            .await
            .expect("Engine gRPC serve");
    });

    let ranks = register_ranks(&engine, shape, cli.page_first);
    let layer_names = shape.layer_names();

    let mut image = vec![0u8; shape.rank_bytes()];
    for set in 0..cli.sets {
        let hashes = make_block_hashes(shape.blocks, set);
        for (rank, rb) in ranks.iter().enumerate() {
            // MLA replica: all GPUs hold identical content and each saves only
            // its block stripe (block_id % tp == rank), so the union is exactly
            // one copy. Dense: each GPU has distinct content and saves every block.
            let content_rank = if cli.mla_replica { 0 } else { rank };
            fill_rank_image(shape, set, content_rank, &mut image);
            rb.buf.copy_from_host(&image);
            let (block_ids, block_hashes): (Vec<usize>, Vec<Vec<u8>>) = if cli.mla_replica {
                (0..shape.blocks)
                    .filter(|b| b % shape.tp == rank)
                    .map(|b| (b, hashes[b].clone()))
                    .unzip()
            } else {
                ((0..shape.blocks).collect(), hashes.clone())
            };
            let tp_rank = if cli.mla_replica { 0 } else { rank };
            let saves = layer_names
                .iter()
                .map(|name| LayerSave {
                    layer_name: name.clone(),
                    block_ids: block_ids.clone(),
                    block_hashes: block_hashes.clone(),
                })
                .collect();
            engine
                .batch_save_kv_blocks_from_ipc(INSTANCE, tp_rank, 0, rb.device as i32, saves)
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

    report_metadata(shape, cli.page_first);
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
    let ranks = register_ranks(&engine, shape, cli.page_first);
    let set_mib = shape.set_bytes() as f64 / (1024.0 * 1024.0);
    report_metadata(shape, cli.page_first);

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
            "FETCH {label} page_first={} set={set} mib={set_mib:.1} ms={:.2} \
             gib_s={gib_s:.2} descriptors={}",
            cli.page_first,
            elapsed.as_secs_f64() * 1000.0,
            shape.descriptors(cli.page_first),
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
            "SUMMARY page_first={} model_layers={} tp={} blocks={} set_mib={set_mib:.1} \
             descriptors={} median_ms={:.2} median_gib_s={:.2}",
            cli.page_first,
            shape.num_layers(),
            shape.tp,
            shape.blocks,
            shape.descriptors(cli.page_first),
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
        .count_prefix_hit_blocks_with_prefetch(INSTANCE, &format!("verify-{set}"), hashes, false)
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

    // MLA replica: one stored copy loadable from any worker (tp_rank=0); verify
    // it once on device 0. Dense: verify each rank's distinct shard.
    let targets: Vec<(usize, &RankBuffers)> = if shape.mla_replica {
        vec![(0usize, &ranks[0])]
    } else {
        ranks.iter().enumerate().collect()
    };

    let mut expected = vec![0u8; shape.rank_bytes()];
    for (rank, rb) in targets {
        let tp_rank = if shape.mla_replica { 0 } else { rank };
        let content_rank = if shape.mla_replica { 0 } else { rank };
        let lease = engine
            .create_query_lease(INSTANCE, blocks.clone())
            .expect("create lease");
        rb.buf.zero();
        let load_state = LoadState::new().expect("LoadState");
        engine
            .batch_load_kv_blocks_multi_layer(
                INSTANCE,
                tp_rank,
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
        fill_rank_image(shape, set, content_rank, &mut expected);
        assert!(
            loaded == expected,
            "verify FAILED: set {set} rank {rank} differs from generator pattern"
        );
    }
    let n = if shape.mla_replica { 1 } else { ranks.len() };
    println!("VERIFY set={set} ok ({n} target(s), every byte)");
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
        layers: build_layers(&cli),
        tp: cli.tp,
        blocks: cli.blocks,
        mla_replica: cli.mla_replica,
    };
    // Generous slack: if the pool runs tight the engine evicts earlier sets
    // (and deregisters them from MetaServer), which breaks the run.
    let pool_bytes = if cli.pool_gib > 0 {
        cli.pool_gib << 30
    } else {
        (cli.sets * shape.set_bytes() + shape.set_bytes() / 2).max(1 << 30)
    };
    info!(
        "p2p_bench role={:?} model={:?} layers={} tp={} blocks={} sets={} \
         set_bytes={:.1}MiB pool={:.1}GiB page_first={} mla_replica={}",
        cli.role,
        cli.model,
        shape.num_layers(),
        shape.tp,
        shape.blocks,
        cli.sets,
        shape.set_bytes() as f64 / (1024.0 * 1024.0),
        pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        cli.page_first,
        cli.mla_replica,
    );

    match cli.role {
        Role::Holder => run_holder(&cli, &shape, pool_bytes).await,
        Role::Requester => run_requester(&cli, &shape, pool_bytes).await,
    }
}
