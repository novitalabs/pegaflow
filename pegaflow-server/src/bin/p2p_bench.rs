//! P2P RDMA block transfer benchmark.
//!
//! Spins up an in-process MetaServer and two PegaEngines on the same host,
//! writes blocks from GPU into the owner's pinned pool, then reads them all
//! back on a requester engine via the full P2P path:
//!
//!   MetaServer query → AcquireLease RPC → batch RDMA READ → ReleaseLease
//!
//! Reports write throughput, RDMA read throughput, and optionally verifies
//! a sample of blocks for data integrity.
//!
//! Requirements:
//!   - RDMA NIC (default: mlx5_0)
//!   - CUDA GPU (device 0)
//!
//! Example:
//!   cargo run -p pegaflow-server --bin p2p_bench --release -- \
//!       --nic mlx5_0 --block-size 262144 --total-gb 10

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use cudarc::driver::{CudaContext, sys};
use log::{info, warn};
use pegaflow_core::metaserver::{BlockHashStore, GrpcMetaService};
use pegaflow_core::sync_state::LOAD_STATE_SUCCESS;
use pegaflow_core::{
    BakingStoreConfig, LayerSave, LoadState, PegaEngine, PrefetchStatus, StorageConfig,
};
use pegaflow_server::GrpcRdmaTransferService;
use pegaflow_server::proto::engine::meta_server_server::MetaServerServer;
use pegaflow_server::proto::engine::rdma_transfer_server::RdmaTransferServer;
use pegaflow_transfer::TransferEngine;
use tokio::sync::Notify;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

const NAMESPACE: &str = "bench-ns";
const INSTANCE_ID: &str = "bench-inst";
const LAYER_NAME: &str = "layer_0";
const META_PORT: u16 = 50200;
const OWNER_GRPC_PORT: u16 = 50199;
const DEVICE_ID: i32 = 0;

// ============================================================================
// CLI
// ============================================================================

#[derive(Parser, Debug, Clone)]
#[command(
    name = "p2p_bench",
    about = "P2P RDMA block transfer benchmark (requires RDMA NIC + CUDA GPU)"
)]
struct Args {
    /// RDMA NIC name (e.g. mlx5_0)
    #[arg(long, default_value = "mlx5_0")]
    nic: String,

    /// RDMA UDP port for the owner engine
    #[arg(long, default_value_t = 56070)]
    owner_rdma_port: u16,

    /// RDMA UDP port for the requester engine
    #[arg(long, default_value_t = 56080)]
    requester_rdma_port: u16,

    /// KV cache block size in bytes
    #[arg(long, default_value_t = 256 * 1024)]
    block_size: usize,

    /// Total data volume to transfer, in GiB
    #[arg(long, default_value_t = 1.0)]
    total_gb: f64,

    /// Skip data integrity verification (faster)
    #[arg(long)]
    skip_verify: bool,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, default_value = "info")]
    log_level: String,
}

// ============================================================================
// GPU buffer — raw CUDA, same pattern as rdma_lease_e2e
// ============================================================================

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        let mut ptr: sys::CUdeviceptr = 0;
        let r = unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc_v2: {r:?}");
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        let r = unsafe {
            sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const std::ffi::c_void, self.len)
        };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemcpyHtoD_v2: {r:?}");
    }

    fn copy_to_host(&self, out: &mut [u8]) {
        assert_eq!(out.len(), self.len);
        let r = unsafe {
            sys::cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                self.len,
            )
        };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "cuMemcpyDtoH_v2: {r:?}");
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { sys::cuMemFree_v2(self.ptr) };
        }
    }
}

// ============================================================================
// Data helpers
// ============================================================================

/// Block i is filled with `((i % 251) + 1) as u8`, matching the test harness.
fn fill_test_pattern(buf: &mut [u8], block_size: usize) {
    for (i, block) in buf.chunks_exact_mut(block_size).enumerate() {
        block.fill(((i % 251) + 1) as u8);
    }
}

fn make_hashes(n: usize) -> Vec<Vec<u8>> {
    (0..n)
        .map(|i| {
            let mut h = vec![0xABu8];
            h.extend_from_slice(&(i as u32).to_le_bytes());
            h
        })
        .collect()
}

fn fmt_bytes(n: usize) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    if n as f64 >= GIB {
        format!("{:.2} GiB", n as f64 / GIB)
    } else if n as f64 >= MIB {
        format!("{:.2} MiB", n as f64 / MIB)
    } else {
        format!("{} B", n)
    }
}

// ============================================================================
// Bench infrastructure
// ============================================================================

async fn start_metaserver() -> (String, Arc<BlockHashStore>) {
    let store = Arc::new(BlockHashStore::with_capacity_and_ttl(512 << 20, 300));
    let shutdown = Arc::new(Notify::new());
    let svc = GrpcMetaService::new(Arc::clone(&store), shutdown);
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{META_PORT}"))
        .await
        .expect("bind metaserver");
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServerServer::new(svc))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("metaserver serve");
    });
    let url = format!("http://{addr}");
    info!("MetaServer: {url}");
    (url, store)
}

fn init_rdma(nic: &str, port: u16, label: &str) -> Arc<TransferEngine> {
    info!("[{label}] initializing RDMA engine on {nic}:{port}");
    let mut eng = TransferEngine::new();
    eng.initialize(nic, port)
        .unwrap_or_else(|e| panic!("[{label}] RDMA initialize: {e}"));
    Arc::new(eng)
}

fn create_engine(
    pool_bytes: usize,
    meta_addr: &str,
    p2p_node_addr: &str,
    block_size: usize,
    max_prefetch_blocks: usize,
    rdma: Arc<TransferEngine>,
) -> Arc<PegaEngine> {
    Arc::new(PegaEngine::new_with_config(
        pool_bytes,
        false,
        StorageConfig {
            enable_lfu_admission: false,
            hint_value_size_bytes: Some(block_size),
            max_prefetch_blocks,
            baking_store_config: Some(BakingStoreConfig {
                p2p_coordinator_addr: meta_addr.to_string(),
                p2p_node_addr: p2p_node_addr.to_string(),
                node_id: String::new(),
            }),
            ssd_cache_config: None,
            enable_numa_affinity: false,
            transfer_engine: Some(rdma),
        },
    ))
}

fn register_pool_regions(rdma: &TransferEngine, engine: &PegaEngine, label: &str) {
    for (ptr, size) in engine.pinned_pool_regions() {
        rdma.register_memory(ptr, size)
            .unwrap_or_else(|e| panic!("[{label}] register pool region {ptr:#x}: {e}"));
        info!(
            "[{label}] RDMA registered region ptr={ptr:#x} size={}",
            fmt_bytes(size)
        );
    }
}

fn unregister_pool_regions(rdma: &TransferEngine, engine: &PegaEngine) {
    for (ptr, _) in engine.pinned_pool_regions() {
        rdma.unregister_memory(ptr)
            .unwrap_or_else(|e| warn!("unregister_memory {ptr:#x}: {e}"));
    }
}

// ============================================================================
// Bench phases
// ============================================================================

/// Phase 1: save all blocks from GPU to owner's pinned pool via GPU workers.
/// Timing starts when `batch_save_kv_blocks_from_ipc` is submitted.
async fn write_phase(
    engine: &PegaEngine,
    gpu_ptr: u64,
    total_bytes: usize,
    num_blocks: usize,
    block_size: usize,
    hashes: &[Vec<u8>],
) -> Duration {
    let block_ids: Vec<i32> = (0..num_blocks as i32).collect();

    engine
        .register_context_layer(
            INSTANCE_ID,
            NAMESPACE,
            DEVICE_ID,
            LAYER_NAME.to_string(),
            gpu_ptr,
            total_bytes,
            num_blocks,
            block_size,
            0, // kv_stride_bytes: contiguous
            1, // segments
            0, // tp_rank
            1, // tp_size
            1, // world_size
            1, // num_layers
        )
        .expect("register_context_layer owner");

    let t0 = Instant::now();

    engine
        .batch_save_kv_blocks_from_ipc(
            INSTANCE_ID,
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER_NAME.to_string(),
                block_ids,
                block_hashes: hashes.to_vec(),
            }],
        )
        .await
        .expect("batch_save_kv_blocks_from_ipc");

    // Wait until all blocks are visible in owner's cache.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let (hit, _) = engine
            .count_prefix_hit_blocks(INSTANCE_ID, hashes)
            .expect("count_prefix_hit_blocks");
        if hit == num_blocks {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "write timeout: {hit}/{num_blocks} blocks cached"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    t0.elapsed()
}

/// Phase 2: fetch all blocks on the requester via P2P RDMA, optionally loading
/// them into GPU memory for verification.
///
/// Returns `(rdma_elapsed, load_elapsed)` where `load_elapsed` is `None` when
/// `skip_verify` is true.
async fn read_phase(
    engine: &PegaEngine,
    gpu_ptr: u64,
    total_bytes: usize,
    num_blocks: usize,
    block_size: usize,
    hashes: &[Vec<u8>],
    skip_verify: bool,
) -> (Duration, Option<Duration>) {
    let block_ids: Vec<i32> = (0..num_blocks as i32).collect();

    engine
        .register_context_layer(
            INSTANCE_ID,
            NAMESPACE,
            DEVICE_ID,
            LAYER_NAME.to_string(),
            gpu_ptr,
            total_bytes,
            num_blocks,
            block_size,
            0,
            1,
            0,
            1,
            1,
            1,
        )
        .expect("register_context_layer requester");

    // First call triggers the RDMA fetch; subsequent calls poll the same entry.
    const REQ_ID: &str = "p2p-bench-req";
    let t0 = Instant::now();

    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        match engine
            .count_prefix_hit_blocks_with_prefetch(INSTANCE_ID, REQ_ID, hashes)
            .expect("count_prefix_hit_blocks_with_prefetch")
        {
            PrefetchStatus::Done { hit, missing } => {
                if hit == num_blocks {
                    break;
                }
                if missing > 0 {
                    // MetaServer may have a partial prefix or some RDMA reads
                    // failed; re-arm with a fresh req_id to retry.
                    warn!("P2P read: {hit} cached, {missing} missing; re-arming prefetch");
                }
            }
            PrefetchStatus::Loading { hit, loading } => {
                if hit + loading < num_blocks {
                    // Backpressure: backing store didn't pick up all blocks on
                    // this call — will be retried on the next poll.
                }
            }
        }
        assert!(
            Instant::now() < deadline,
            "read timeout: P2P transfer did not complete within 300 s"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let rdma_elapsed = t0.elapsed();

    if skip_verify {
        // Unpin the blocks that were reserved by the Done call.
        engine
            .unpin_blocks(INSTANCE_ID, hashes)
            .expect("unpin_blocks");
        return (rdma_elapsed, None);
    }

    // Load pinned blocks → requester GPU for host-side data comparison.
    let load_state = LoadState::new().expect("LoadState::new");
    let t_load = Instant::now();

    engine
        .batch_load_kv_blocks_multi_layer(
            INSTANCE_ID,
            0,
            DEVICE_ID,
            load_state.shm_name(),
            &[LAYER_NAME],
            &block_ids,
            hashes,
        )
        .expect("batch_load_kv_blocks_multi_layer");

    let load_deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if load_state.get() == LOAD_STATE_SUCCESS {
            break;
        }
        assert!(
            Instant::now() < load_deadline,
            "load timeout after P2P read"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    (rdma_elapsed, Some(t_load.elapsed()))
}

// ============================================================================
// Orchestration
// ============================================================================

async fn run_bench(
    args: &Args,
    num_blocks: usize,
    total_bytes: usize,
    owner_gpu_ptr: u64,
    req_gpu_ptr: u64,
    hashes: &[Vec<u8>],
) -> (Duration, Duration, Option<Duration>) {
    let pool_bytes = (total_bytes as f64 * 1.15) as usize;

    // ---- MetaServer ----
    let (meta_addr, meta_store) = start_metaserver().await;

    // ---- Owner ----
    let owner_rdma = init_rdma(&args.nic, args.owner_rdma_port, "owner");
    let owner_engine = create_engine(
        pool_bytes,
        &meta_addr,
        &format!("127.0.0.1:{OWNER_GRPC_PORT}"),
        args.block_size,
        num_blocks + 64,
        Arc::clone(&owner_rdma),
    );
    register_pool_regions(&owner_rdma, &owner_engine, "owner");

    // Owner's RdmaTransfer gRPC service.
    let grpc_addr: SocketAddr = format!("127.0.0.1:{OWNER_GRPC_PORT}").parse().unwrap();
    let rdma_svc =
        GrpcRdmaTransferService::new_with_rdma(Arc::clone(&owner_engine), Arc::clone(&owner_rdma));
    tokio::spawn(async move {
        Server::builder()
            .add_service(RdmaTransferServer::new(rdma_svc))
            .serve(grpc_addr)
            .await
            .expect("RdmaTransfer gRPC serve");
    });
    tokio::time::sleep(Duration::from_millis(50)).await; // let server bind
    info!("Owner RdmaTransfer gRPC: {grpc_addr}");

    // ---- Write phase ----
    eprintln!(
        "\n[write] saving {} blocks ({})...",
        num_blocks,
        fmt_bytes(total_bytes)
    );
    let write_elapsed = write_phase(
        &owner_engine,
        owner_gpu_ptr,
        total_bytes,
        num_blocks,
        args.block_size,
        hashes,
    )
    .await;
    eprintln!(
        "[write] {:.3} s  |  {:.2} GB/s  (GPU → pinned pool)",
        write_elapsed.as_secs_f64(),
        total_bytes as f64 / (1u64 << 30) as f64 / write_elapsed.as_secs_f64()
    );

    // ---- Wait for MetaServer registrations ----
    eprintln!("[meta]  waiting for {num_blocks} hash registrations...");
    let meta_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        meta_store.run_pending_tasks().await;
        let count = meta_store.entry_count() as usize;
        if count >= num_blocks {
            eprintln!("[meta]  {count} hashes registered");
            break;
        }
        assert!(
            Instant::now() < meta_deadline,
            "MetaServer timeout: {count}/{num_blocks} registered"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // ---- Requester ----
    let req_rdma = init_rdma(&args.nic, args.requester_rdma_port, "requester");
    let req_engine = create_engine(
        pool_bytes,
        &meta_addr,
        "127.0.0.1:0", // requester doesn't serve blocks
        args.block_size,
        num_blocks + 64,
        Arc::clone(&req_rdma),
    );
    register_pool_regions(&req_rdma, &req_engine, "requester");

    // ---- Warmup: establish RDMA session before timing ----
    {
        eprintln!("\n[warmup] establishing RDMA session...");
        req_engine
            .register_context_layer(
                INSTANCE_ID,
                NAMESPACE,
                DEVICE_ID,
                LAYER_NAME.to_string(),
                req_gpu_ptr,
                total_bytes,
                num_blocks,
                args.block_size,
                0,
                1,
                0,
                1,
                1,
                1,
            )
            .expect("register_context_layer warmup");

        let warmup_hashes = &hashes[..1];
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            match req_engine
                .count_prefix_hit_blocks_with_prefetch(INSTANCE_ID, "warmup", warmup_hashes)
                .expect("warmup prefetch")
            {
                PrefetchStatus::Done { hit, .. } if hit >= 1 => break,
                _ => {}
            }
            assert!(Instant::now() < deadline, "warmup RDMA session timeout");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        eprintln!("[warmup] RDMA session ready");
    }

    // ---- Read phase ----
    eprintln!("\n[read]  fetching {} blocks via P2P RDMA...", num_blocks);
    let (rdma_elapsed, load_elapsed) = read_phase(
        &req_engine,
        req_gpu_ptr,
        total_bytes,
        num_blocks,
        args.block_size,
        hashes,
        args.skip_verify,
    )
    .await;
    eprintln!(
        "[read]  {:.3} s  |  {:.2} GB/s  (RDMA: MetaServer + AcquireLease + READ + ReleaseLease)",
        rdma_elapsed.as_secs_f64(),
        total_bytes as f64 / (1u64 << 30) as f64 / rdma_elapsed.as_secs_f64()
    );
    if let Some(load_dur) = load_elapsed {
        eprintln!(
            "[load]  {:.3} s  |  {:.2} GB/s  (pinned pool → GPU)",
            load_dur.as_secs_f64(),
            total_bytes as f64 / (1u64 << 30) as f64 / load_dur.as_secs_f64()
        );
    }

    // ---- Cleanup ----
    unregister_pool_regions(&owner_rdma, &owner_engine);
    unregister_pool_regions(&req_rdma, &req_engine);

    (write_elapsed, rdma_elapsed, load_elapsed)
}

// ============================================================================
// Entry point
// ============================================================================

fn main() {
    let args = Args::parse();
    pegaflow_core::logging::init_stdout_colored(&args.log_level);

    let total_bytes = (args.total_gb * (1u64 << 30) as f64) as usize;
    let num_blocks = total_bytes / args.block_size;
    assert!(num_blocks > 0, "--total-gb too small for --block-size");

    eprintln!(
        "\n=== P2P Bench ===\n  NIC:         {}\n  Block size:  {}\n  Blocks:      {}\n  Total:       {}\n  Verify:      {}\n",
        args.nic,
        fmt_bytes(args.block_size),
        num_blocks,
        fmt_bytes(total_bytes),
        !args.skip_verify,
    );

    // CUDA must be initialized on the main thread before Tokio spawns workers.
    let _ctx = CudaContext::new(DEVICE_ID as usize).expect("CUDA device 0");

    // Allocate + fill GPU buffers on the main thread (CUDA context is current here).
    let hashes = make_hashes(num_blocks);
    let mut host_data = vec![0u8; total_bytes];
    fill_test_pattern(&mut host_data, args.block_size);

    let owner_gpu = GpuBuffer::alloc(total_bytes);
    owner_gpu.copy_from_host(&host_data);

    // Requester GPU buffer: written by batch_load during the read phase.
    let req_gpu = GpuBuffer::alloc(total_bytes);

    let owner_ptr = owner_gpu.as_u64();
    let req_ptr = req_gpu.as_u64();

    // Multi-thread runtime: required because P2P backing store uses
    // `tokio::task::block_in_place` in the MetaServer query path.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime");

    let skip_verify = args.skip_verify;
    let (write_elapsed, rdma_elapsed, _load_elapsed) = runtime.block_on(async move {
        run_bench(&args, num_blocks, total_bytes, owner_ptr, req_ptr, &hashes).await
    });

    // Verify data integrity on the main thread (CUDA context still current).
    if !skip_verify {
        eprintln!("\n[verify] comparing requester GPU with original pattern...");
        let mut recv = vec![0u8; total_bytes];
        req_gpu.copy_to_host(&mut recv);

        // Report first mismatch if any, then count total errors.
        let mismatches: usize = recv.iter().zip(&host_data).filter(|(a, b)| a != b).count();
        if mismatches == 0 {
            eprintln!("[verify] PASS — {} bytes match", total_bytes);
        } else {
            let first = recv
                .iter()
                .zip(&host_data)
                .position(|(a, b)| a != b)
                .unwrap();
            eprintln!(
                "[verify] FAIL — {} mismatched bytes (first at offset {}: got {:#04x}, want {:#04x})",
                mismatches, first, recv[first], host_data[first],
            );
            std::process::exit(1);
        }
    }

    // ---- Summary ----
    let gib = (1u64 << 30) as f64;
    eprintln!(
        "\n=== Summary ===\n\
         Write (GPU→pool):   {:>8.3} s   {:>6.2} GB/s\n\
         P2P RDMA read:      {:>8.3} s   {:>6.2} GB/s\n",
        write_elapsed.as_secs_f64(),
        total_bytes as f64 / gib / write_elapsed.as_secs_f64(),
        rdma_elapsed.as_secs_f64(),
        total_bytes as f64 / gib / rdma_elapsed.as_secs_f64(),
    );
}
