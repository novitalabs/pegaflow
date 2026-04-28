//! P2P RDMA remote fetch integration test.
//!
//! Verifies the end-to-end flow:
//! Engine A saves blocks → MetaServer discovers them → Engine B fetches via RDMA READ
//! → data integrity verified.
//!
//! Run with: `cargo test -p pegaflow-server --test p2p_rdma -- --ignored`

use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use parking_lot::Mutex;
use pegaflow_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use pegaflow_core::*;
use pegaflow_metaserver::{BlockHashStore, GrpcMetaService};
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService};
use tokio::sync::Notify;
use tonic::transport::Server;

// ── GPU buffer (from pegaflow-core/tests/common/gpu_buffer.rs) ──────────────

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        assert!(len > 0);
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    fn as_u64(&self) -> u64 {
        self.ptr
    }

    fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }

    fn zero(&self) {
        check_cuda(
            unsafe { sys::cuMemsetD8_v2(self.ptr, 0, self.len) },
            "cuMemsetD8_v2",
        );
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            check_cuda(unsafe { sys::cuMemFree_v2(self.ptr) }, "cuMemFree_v2");
            self.ptr = 0;
        }
    }
}

fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

// ── Helpers (from pegaflow-core/tests/common/helpers.rs) ────────────────────

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = ((i % 251) + 1) as u8;
        block.fill(fill);
    }
}

fn make_block_ids(num_blocks: usize) -> Vec<i32> {
    (0..num_blocks).map(|i| i as i32).collect()
}

fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

// ── Infrastructure ──────────────────────────────────────────────────────────

fn get_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local_addr")
        .port()
}

async fn wait_for_grpc_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .is_ok()
        {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for gRPC on port {port}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn spawn_metaserver(port: u16) -> Arc<BlockHashStore> {
    let store = Arc::new(BlockHashStore::new());
    let service = GrpcMetaService::new(Arc::clone(&store));
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServerServer::new(service))
            .serve(addr)
            .await
            .expect("MetaServer gRPC serve");
    });
    wait_for_grpc_ready(port).await;
    store
}

async fn spawn_engine_server(engine: Arc<PegaEngine>, port: u16) {
    let registry = CudaTensorRegistry::new().expect("CudaTensorRegistry::new");
    let registry = Arc::new(Mutex::new(registry));
    let shutdown = Arc::new(Notify::new());
    let hll_tracker = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::HllTracker::new(
            Duration::from_secs(3600),
            Duration::from_secs(86400),
            14,
        ),
    ));
    let service = GrpcEngineService::new(engine, registry, shutdown, hll_tracker);
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(addr)
            .await
            .expect("Engine gRPC serve");
    });
    wait_for_grpc_ready(port).await;
}

async fn wait_for_cache(
    engine: &PegaEngine,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit_blocks: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let hit_blocks = engine
            .count_prefix_hit_blocks_with_prefetch(instance_id, "wait-for-cache", block_hashes)
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        if hit_blocks >= expected_hit_blocks {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected_hit_blocks} cached blocks (got {hit_blocks})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn wait_for_metaserver_registration(
    store: &BlockHashStore,
    namespace: &str,
    hashes: &[Vec<u8>],
    expected: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let found = store.query_prefix(namespace, hashes);
        if found.len() >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for MetaServer registration ({} / {})",
            found.len(),
            expected
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_prefetch_done(
    engine: &PegaEngine,
    instance_id: &str,
    req_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit_blocks: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let hit_blocks = engine
            .count_prefix_hit_blocks_with_prefetch(instance_id, req_id, block_hashes)
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        if hit_blocks >= expected_hit_blocks {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "prefetch done but hit_blocks={hit_blocks} missing_blocks={}, expected hit_blocks>={expected_hit_blocks}",
            block_hashes.len().saturating_sub(hit_blocks)
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

const NUM_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = 1024;
const TOTAL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;
const NAMESPACE: &str = "test-p2p";
const LAYER: &str = "layer_0";
const DEVICE_ID: i32 = 0;

#[tokio::test]
#[ignore] // Requires RDMA hardware (mlx5_1), CUDA GPU, and Python+torch
async fn p2p_rdma_remote_fetch_roundtrip() {
    pegaflow_common::logging::init_stdout_colored("debug");
    let _cuda_ctx = CudaContext::new(0).expect("CUDA init");

    // Allocate ephemeral ports
    let meta_port = get_free_port();
    let port_a = get_free_port();

    // ── 1. Start MetaServer ──
    let meta_store = spawn_metaserver(meta_port).await;

    // ── 2. Create Engine A (source of blocks) ──
    let config_a = StorageConfig {
        metaserver_addr: Some(format!("http://127.0.0.1:{meta_port}")),
        advertise_addr: Some(format!("127.0.0.1:{port_a}")),
        rdma_nic_names: Some(vec!["mlx5_1".into()]),
        ..StorageConfig::default()
    };
    let engine_a = Arc::new(PegaEngine::new_with_config(16 << 20, false, config_a));

    // ── 3. Start Engine A gRPC server ──
    spawn_engine_server(Arc::clone(&engine_a), port_a).await;

    // ── 4. Save blocks on Engine A ──
    let gpu_a = GpuBuffer::alloc(TOTAL_SIZE);
    let mut host_data = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut host_data, BLOCK_SIZE);
    gpu_a.copy_from_host(&host_data);

    engine_a
        .register_context_layer_batch(
            "inst-a",
            NAMESPACE,
            DEVICE_ID,
            0, // tp_rank
            1, // tp_size
            1, // world_size
            1, // num_layers
            &[LAYER.to_string()],
            &[gpu_a.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0], // kv_strides
            &[1], // segments
        )
        .expect("register layer on engine A");

    let block_ids = make_block_ids(NUM_BLOCKS);
    let block_hashes = make_block_hashes(NUM_BLOCKS, 42);

    engine_a
        .batch_save_kv_blocks_from_ipc(
            "inst-a",
            0,
            DEVICE_ID,
            vec![LayerSave {
                layer_name: LAYER.to_string(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.clone(),
            }],
        )
        .await
        .expect("save blocks on engine A");

    // ── 5. Wait for Engine A cache ──
    wait_for_cache(
        &engine_a,
        "inst-a",
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(5),
    )
    .await;

    // ── 6. Wait for MetaServer registration (fire-and-forget async) ──
    wait_for_metaserver_registration(
        &meta_store,
        NAMESPACE,
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(10),
    )
    .await;

    // ── 7. Create Engine B (fetcher) ──
    let port_b = get_free_port();
    let config_b = StorageConfig {
        metaserver_addr: Some(format!("http://127.0.0.1:{meta_port}")),
        advertise_addr: Some(format!("127.0.0.1:{port_b}")),
        rdma_nic_names: Some(vec!["mlx5_1".into()]),
        ..StorageConfig::default()
    };
    let engine_b = PegaEngine::new_with_config(16 << 20, false, config_b);

    let gpu_b = GpuBuffer::alloc(TOTAL_SIZE);
    gpu_b.zero();

    engine_b
        .register_context_layer_batch(
            "inst-b",
            NAMESPACE,
            DEVICE_ID,
            0,
            1,
            1,
            1,
            &[LAYER.to_string()],
            &[gpu_b.as_u64()],
            &[TOTAL_SIZE],
            &[NUM_BLOCKS],
            &[BLOCK_SIZE],
            &[0],
            &[1],
        )
        .expect("register layer on engine B");

    // ── 8. Remote fetch: Engine B discovers blocks via MetaServer → RDMA READ ──
    wait_for_prefetch_done(
        &engine_b,
        "inst-b",
        "req-1",
        &block_hashes,
        NUM_BLOCKS,
        Duration::from_secs(30),
    )
    .await;

    // ── 9. Load from Engine B cache → GPU ──
    let load_state = LoadState::new().expect("create LoadState");
    let shm_name = load_state.shm_name().to_string();

    engine_b
        .batch_load_kv_blocks_multi_layer(
            "inst-b",
            0,
            DEVICE_ID,
            &shm_name,
            &[LAYER],
            &block_ids,
            &block_hashes,
        )
        .expect("batch_load on engine B");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = load_state.get();
        if state == LOAD_STATE_SUCCESS {
            break;
        }
        assert!(state != LOAD_STATE_ERROR, "load reported ERROR");
        assert!(Instant::now() < deadline, "timed out waiting for load");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // ── 10. Verify data integrity ──
    let loaded = gpu_b.copy_to_host();
    assert_eq!(
        loaded, host_data,
        "GPU data mismatch: remote-fetched blocks differ from original"
    );
}
