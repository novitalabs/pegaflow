//! Shared harness for pegaflow-server `p2p_rdma` integration tests.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use pegaflow_common::NumaNode;
use pegaflow_core::{LayerSave, LoadState, PegaEngine, PrefetchStatus, QueryLeaseId, SealedBlock};
use pegaflow_metaserver::{BlockHashStore, GrpcMetaService};
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService, RegistryHandle};
use pegaflow_transfer_wire::TransferPlan;
use tokio::sync::Notify;
use tonic::transport::Server;

pub(crate) const P2P_NAMESPACE: &str = "test-p2p";

/// Model / workload shape for a p2p RDMA case.
#[derive(Debug, Clone)]
pub(crate) struct P2pShape {
    pub layers: usize,
    pub blocks: usize,
    pub block_size: usize,
    pub tp_size: usize,
    pub world_size: usize,
}

impl P2pShape {
    pub(crate) fn layer_bytes(&self) -> usize {
        self.blocks * self.block_size * 2
    }

    pub(crate) fn rank_bytes(&self) -> usize {
        self.layers * self.layer_bytes()
    }

    pub(crate) fn layer_names(&self) -> Vec<String> {
        (0..self.layers).map(|l| format!("layer_{l}")).collect()
    }
}

/// Class A cases: a single worker, every slot on one NUMA node. Multi-worker
/// NUMA shapes (TP, MLA replica) are driven directly in `p2p_rdma.rs` because
/// each needs its own register/save/load fan-out.
#[derive(Debug, Clone, Copy)]
pub(crate) enum P2pCase {
    SingleLayer,
    MultiLayerMultiBlock,
}

impl P2pCase {
    pub(crate) fn default_shape(self) -> P2pShape {
        match self {
            Self::SingleLayer => P2pShape {
                layers: 1,
                blocks: 4,
                block_size: 1024,
                tp_size: 1,
                world_size: 1,
            },
            Self::MultiLayerMultiBlock => P2pShape {
                layers: 8,
                blocks: 64,
                block_size: 512,
                tp_size: 1,
                world_size: 1,
            },
        }
    }
}

pub(crate) fn rdma_nic_names() -> Vec<String> {
    std::env::var("PEGAFLOW_P2P_NICS")
        .map(|s| {
            s.split(',')
                .map(|n| n.trim().to_string())
                .filter(|n| !n.is_empty())
                .collect()
        })
        .unwrap_or_else(|_| vec!["mlx5_1".into()])
}

pub(crate) fn cuda_device_count() -> usize {
    let mut count = 0i32;
    check_cuda(unsafe { sys::cuInit(0) }, "cuInit");
    check_cuda(
        unsafe { sys::cuDeviceGetCount(&raw mut count) },
        "cuDeviceGetCount",
    );
    count.max(0) as usize
}

pub(crate) fn make_block_hashes(blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

pub(crate) fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    fill_test_pattern_salted(host_data, block_size, 0);
}

/// Like [`fill_test_pattern`] but offsets the per-block byte by `salt`, so each
/// TP rank / replica gets a distinct, verifiable payload under the same hashes.
pub(crate) fn fill_test_pattern_salted(host_data: &mut [u8], block_size: usize, salt: usize) {
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = (((i + salt) % 251) + 1) as u8;
        block.fill(fill);
    }
}

pub(crate) fn get_free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral port")
        .local_addr()
        .expect("local_addr")
        .port()
}

pub(crate) async fn wait_for_grpc_ready(port: u16) {
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

pub(crate) async fn spawn_metaserver(port: u16) -> Arc<BlockHashStore> {
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

pub(crate) async fn spawn_engine_server(engine: Arc<PegaEngine>, port: u16) {
    let registry = RegistryHandle::spawn(CudaTensorRegistry::empty());
    let shutdown = Arc::new(Notify::new());
    let hll_tracker = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            vec![("24h".into(), Duration::from_secs(86400))],
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

pub(crate) struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    pub(crate) fn alloc(len: usize) -> Self {
        assert!(len > 0);
        let mut ptr: sys::CUdeviceptr = 0;
        check_cuda(
            unsafe { sys::cuMemAlloc_v2(&raw mut ptr, len) },
            "cuMemAlloc_v2",
        );
        Self { ptr, len }
    }

    pub(crate) fn as_u64(&self) -> u64 {
        self.ptr
    }

    pub(crate) fn copy_from_host(&self, data: &[u8]) {
        assert_eq!(data.len(), self.len);
        check_cuda(
            unsafe { sys::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, self.len) },
            "cuMemcpyHtoD_v2",
        );
    }

    pub(crate) fn copy_to_host(&self) -> Vec<u8> {
        let mut output = vec![0u8; self.len];
        check_cuda(
            unsafe { sys::cuMemcpyDtoH_v2(output.as_mut_ptr() as *mut c_void, self.ptr, self.len) },
            "cuMemcpyDtoH_v2",
        );
        output
    }

    pub(crate) fn zero(&self) {
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

pub(crate) fn check_cuda(result: sys::CUresult, op: &str) {
    assert!(
        result == sys::CUresult::CUDA_SUCCESS,
        "{op} failed with {result:?}"
    );
}

/// A [`GpuBuffer`] pinned to a specific CUDA device.
///
/// Multi-worker (TP / replica) cases place each worker on its own device, so
/// every device-touching op must bind that device's context first — otherwise
/// alloc/copy/free run against whatever context happens to be current. The
/// `Drop` impl rebinds before the inner [`GpuBuffer`] frees, mirroring the
/// single-GPU harness in `common/mod.rs`.
pub(crate) struct DeviceGpu {
    ctx: Arc<CudaContext>,
    gpu: GpuBuffer,
}

impl DeviceGpu {
    pub(crate) fn alloc(device_id: i32, len: usize) -> Self {
        let ctx = CudaContext::new(device_id as usize).expect("CUDA context for device");
        ctx.bind_to_thread().expect("bind CUDA context");
        let gpu = GpuBuffer::alloc(len);
        Self { ctx, gpu }
    }

    fn bind(&self) {
        self.ctx.bind_to_thread().expect("bind CUDA context");
    }

    pub(crate) fn as_u64(&self) -> u64 {
        self.gpu.as_u64()
    }

    pub(crate) fn copy_from_host(&self, data: &[u8]) {
        self.bind();
        self.gpu.copy_from_host(data);
    }

    pub(crate) fn copy_to_host(&self) -> Vec<u8> {
        self.bind();
        self.gpu.copy_to_host()
    }

    pub(crate) fn zero(&self) {
        self.bind();
        self.gpu.zero();
    }
}

impl Drop for DeviceGpu {
    fn drop(&mut self) {
        // Bind before the inner GpuBuffer's Drop frees on this device.
        let _ = self.ctx.bind_to_thread();
    }
}

pub(crate) async fn wait_for_cache(
    engine: &PegaEngine,
    instance_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit: usize,
    timeout: Duration,
) {
    let deadline = Instant::now() + timeout;
    loop {
        let status = engine
            .count_prefix_hit_blocks_with_prefetch(instance_id, "wait-for-cache", block_hashes)
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        let hit = match status {
            PrefetchStatus::Ready { blocks, .. } => blocks.len(),
            PrefetchStatus::Loading => 0,
        };
        if hit >= expected_hit {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected_hit} cached blocks (got {hit})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

pub(crate) async fn wait_for_metaserver_registration(
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

pub(crate) async fn wait_for_prefetch_done(
    engine: &PegaEngine,
    instance_id: &str,
    req_id: &str,
    block_hashes: &[Vec<u8>],
    expected_hit: usize,
    timeout: Duration,
) -> QueryLeaseId {
    let deadline = Instant::now() + timeout;
    loop {
        let status = engine
            .count_prefix_hit_blocks_with_prefetch(instance_id, req_id, block_hashes)
            .await
            .expect("count_prefix_hit_blocks_with_prefetch");
        match status {
            PrefetchStatus::Ready { blocks, .. } if blocks.len() >= expected_hit => {
                return engine
                    .create_query_lease(instance_id, blocks)
                    .expect("create query lease");
            }
            PrefetchStatus::Ready { blocks, missing } => {
                assert!(
                    Instant::now() < deadline,
                    "prefetch done but hit={} missing={missing}, expected hit>={expected_hit}",
                    blocks.len()
                );
            }
            PrefetchStatus::Loading => {}
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for prefetch done"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

pub(crate) fn decode_transfer_plan(bytes: &[u8]) -> TransferPlan {
    TransferPlan::decode_from_slice(bytes).expect("decode transfer plan")
}

pub(crate) fn assert_transfer_plan_numa(plan: &TransferPlan) {
    assert!(!plan.block_hashes.is_empty(), "plan must list blocks");
    assert!(
        !plan.remote_chunks.is_empty(),
        "plan must have remote chunks"
    );
    for chunk in &plan.remote_chunks {
        assert!(chunk.length > 0, "remote chunk length must be > 0");
        assert!(
            chunk.numa_node != NumaNode::UNKNOWN.0,
            "remote chunk must carry a valid NUMA node"
        );
    }
}

pub(crate) fn unique_chunk_numa_nodes(plan: &TransferPlan) -> BTreeSet<u32> {
    plan.remote_chunks.iter().map(|c| c.numa_node).collect()
}

pub(crate) fn holder_slot_numas_per_block(
    found: &[(pegaflow_core::BlockKey, Arc<pegaflow_core::SealedBlock>)],
) -> Vec<Vec<NumaNode>> {
    found
        .iter()
        .map(|(_, block)| block.slot_numas().to_vec())
        .collect()
}

/// Register one worker's KV layers at `(tp_rank, pp_rank)`. The engine seals the
/// instance topology once `world_size` workers have registered, so for
/// multi-worker cases every worker must register *before* the first save.
///
/// `tp_size` / `world_size` are the values the engine sees: for normal TP they
/// equal the deployment's TP size; for MLA they are the *effective* topology
/// (eff_tp_size = 1, world_size = replica count).
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors the engine's batched per-worker registration API"
)]
pub(crate) fn register_worker(
    engine: &PegaEngine,
    instance_id: &str,
    shape: &P2pShape,
    device_id: i32,
    tp_rank: usize,
    pp_rank: usize,
    tp_size: usize,
    world_size: usize,
    gpu_ptr: u64,
) {
    let layer_bytes = shape.layer_bytes();
    let ptrs: Vec<u64> = (0..shape.layers)
        .map(|l| gpu_ptr + (l * layer_bytes) as u64)
        .collect();

    engine
        .register_context_layer_batch(
            instance_id,
            P2P_NAMESPACE,
            device_id,
            tp_rank,
            pp_rank,
            tp_size,
            world_size,
            &shape.layer_names(),
            &ptrs,
            &vec![layer_bytes; shape.layers],
            &vec![shape.blocks; shape.layers],
            &vec![shape.block_size; shape.layers],
            &vec![shape.blocks * shape.block_size; shape.layers],
            &vec![2; shape.layers],
        )
        .expect("register_context_layer_batch");
}

/// Save every layer of `block_ids` from one worker, with an optional NUMA hint.
///
/// `numa_hint = None` reproduces the engine-direct path (slot NUMA = the GPU's
/// preferred NUMA), which is exactly how normal-TP saves land. `Some(node)`
/// reproduces the server's per-RPC NUMA round-robin used for MLA replicas.
#[allow(
    clippy::too_many_arguments,
    reason = "one save RPC's worth of routing + payload fields"
)]
pub(crate) async fn save_worker_blocks(
    engine: &PegaEngine,
    instance_id: &str,
    shape: &P2pShape,
    device_id: i32,
    tp_rank: usize,
    pp_rank: usize,
    block_ids: &[usize],
    block_hashes: &[Vec<u8>],
    numa_hint: Option<NumaNode>,
) {
    let saves: Vec<LayerSave> = shape
        .layer_names()
        .into_iter()
        .map(|layer_name| LayerSave {
            layer_name,
            block_ids: block_ids.to_vec(),
            block_hashes: block_hashes.to_vec(),
        })
        .collect();

    engine
        .batch_save_kv_blocks_from_ipc_with_numa_hint(
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            numa_hint,
        )
        .await
        .expect("batch_save_kv_blocks_from_ipc_with_numa_hint");
}

pub(crate) async fn holder_save_blocks(
    engine: &PegaEngine,
    instance_id: &str,
    shape: &P2pShape,
    device_id: i32,
    tp_rank: u32,
    block_hashes: &[Vec<u8>],
    gpu: &GpuBuffer,
) -> Vec<u8> {
    let rank_bytes = shape.rank_bytes();
    let mut host_data = vec![0u8; rank_bytes];
    fill_test_pattern(&mut host_data, shape.block_size);
    gpu.copy_from_host(&host_data);

    register_worker(
        engine,
        instance_id,
        shape,
        device_id,
        tp_rank as usize,
        0,
        shape.tp_size,
        shape.world_size,
        gpu.as_u64(),
    );

    let block_ids: Vec<usize> = (0..shape.blocks).collect();
    save_worker_blocks(
        engine,
        instance_id,
        shape,
        device_id,
        tp_rank as usize,
        0,
        &block_ids,
        block_hashes,
        None,
    )
    .await;

    host_data
}

pub(crate) async fn query_holder_transfer_plan(
    engine: &PegaEngine,
    block_hashes: &[Vec<u8>],
    requester_id: &str,
) -> TransferPlan {
    let (_, found) = engine.query_blocks_for_transfer(P2P_NAMESPACE, block_hashes, requester_id);
    assert!(
        !found.is_empty(),
        "holder must have blocks for transfer plan query"
    );
    let bytes = pegaflow_core::encode_transfer_plan_bytes(&found).expect("encode transfer plan");
    decode_transfer_plan(&bytes)
}

#[allow(
    clippy::too_many_arguments,
    reason = "single-worker load target + its expected payload"
)]
pub(crate) async fn requester_load_and_verify(
    engine: &PegaEngine,
    instance_id: &str,
    shape: &P2pShape,
    device_id: i32,
    tp_rank: usize,
    lease: QueryLeaseId,
    block_ids: &[usize],
    expected_host: &[u8],
    gpu: &GpuBuffer,
) {
    let load_state = LoadState::new().expect("create LoadState");
    let layer_names = shape.layer_names();
    let layer_name_refs: Vec<&str> = layer_names.iter().map(String::as_str).collect();

    engine
        .batch_load_kv_blocks_multi_layer(
            instance_id,
            tp_rank,
            device_id,
            load_state.shm_name(),
            &layer_name_refs,
            &[(lease, block_ids.to_vec())],
        )
        .expect("batch_load_kv_blocks_multi_layer");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let state = load_state.get();
        if state == pegaflow_core::sync_state::LOAD_STATE_SUCCESS {
            break;
        }
        assert!(
            state != pegaflow_core::sync_state::LOAD_STATE_ERROR,
            "load reported ERROR"
        );
        assert!(Instant::now() < deadline, "timed out waiting for load");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let loaded = gpu.copy_to_host();
    assert_eq!(
        loaded, expected_host,
        "GPU data mismatch after remote fetch + load"
    );
}

/// Load one worker's slice into its own device and verify it against the data
/// that worker's holder counterpart saved. The single `lease` is shared across
/// all workers — it is created with `refcount == world_size`, so each worker
/// consumes it exactly once (see `create_query_lease`).
#[allow(
    clippy::too_many_arguments,
    reason = "one worker's load target + its expected payload"
)]
pub(crate) async fn requester_load_worker_and_verify(
    engine: &PegaEngine,
    instance_id: &str,
    shape: &P2pShape,
    device_id: i32,
    tp_rank: usize,
    lease: QueryLeaseId,
    block_ids: &[usize],
    expected_host: &[u8],
    gpu: &DeviceGpu,
) {
    let load_state = LoadState::new().expect("create LoadState");
    let layer_names = shape.layer_names();
    let layer_name_refs: Vec<&str> = layer_names.iter().map(String::as_str).collect();

    engine
        .batch_load_kv_blocks_multi_layer(
            instance_id,
            tp_rank,
            device_id,
            load_state.shm_name(),
            &layer_name_refs,
            &[(lease, block_ids.to_vec())],
        )
        .expect("batch_load_kv_blocks_multi_layer");

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let state = load_state.get();
        if state == pegaflow_core::sync_state::LOAD_STATE_SUCCESS {
            break;
        }
        assert!(
            state != pegaflow_core::sync_state::LOAD_STATE_ERROR,
            "load reported ERROR (tp_rank={tp_rank})"
        );
        assert!(
            Instant::now() < deadline,
            "timed out waiting for load (tp_rank={tp_rank})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let loaded = gpu.copy_to_host();
    assert_eq!(
        loaded, expected_host,
        "GPU data mismatch after remote fetch + load (tp_rank={tp_rank})"
    );
}

/// Distinct NUMA nodes actually recorded across every slot of every block.
/// On a single-NUMA box this is `{0}`; on a 2-socket box spanning workers it is
/// the set the transfer plan's remote chunks must reproduce exactly.
pub(crate) fn distinct_slot_numas(
    found: &[(pegaflow_core::BlockKey, Arc<SealedBlock>)],
) -> BTreeSet<u32> {
    found
        .iter()
        .flat_map(|(_, block)| block.slot_numas().iter().map(|n| n.0))
        .collect()
}

/// True if any single block carries slots on more than one NUMA node (the
/// cross-slot shape produced by TP / layer-split across sockets).
pub(crate) fn any_block_spans_multiple_numa(
    found: &[(pegaflow_core::BlockKey, Arc<SealedBlock>)],
) -> bool {
    found.iter().any(|(_, block)| {
        let nodes: BTreeSet<u32> = block.slot_numas().iter().map(|n| n.0).collect();
        nodes.len() > 1
    })
}
