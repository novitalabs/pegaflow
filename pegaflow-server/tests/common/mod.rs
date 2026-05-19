use std::ffi::c_void;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use cudarc::driver::sys;
use parking_lot::Mutex;
use pegaflow_core::sync_state::{LOAD_STATE_ERROR, LOAD_STATE_SUCCESS};
use pegaflow_core::{LoadState, PegaEngine, StorageConfig};
use pegaflow_server::proto::engine::engine_client::EngineClient;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::proto::engine::{
    LeaseLoad, LoadRequest, LoadResponse, QueryRequest, QueryResponse, ReleaseRequest,
    ReleaseResponse, SaveLayer, SaveRequest, SaveResponse, SessionEvent, SessionRequest,
};
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService};
use tokio::sync::Notify;
use tonic::transport::{Channel, Server};
use tonic::{Status, Streaming};

pub(crate) const INSTANCE_ID: &str = "mock-vllm-rpc-e2e";
pub(crate) const SECOND_INSTANCE_ID: &str = "mock-vllm-rpc-e2e-second";
pub(crate) const NAMESPACE: &str = "mock-vllm";
pub(crate) const LAYER_NAME: &str = "layer_0";
pub(crate) const BLOCK_COUNT: usize = 4;
pub(crate) const BYTES_PER_BLOCK: usize = 1024;

const LOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
pub(crate) struct WorkerRegistration {
    pub(crate) device_id: i32,
    pub(crate) tp_rank: u32,
}

pub(crate) struct MockVllmRpcHarness {
    _ctxs: Vec<Arc<CudaContext>>,
    _engine: Arc<PegaEngine>,
    workers: Vec<WorkerRegistration>,
    server: tokio::task::JoinHandle<()>,
    pub(crate) gpus: Vec<TestGpuData>,
    scheduler: EngineClient<Channel>,
    worker: EngineClient<Channel>,
    extra_gpus: Vec<TestGpuData>,
}

pub(crate) struct RpcExchange<Request, Response> {
    pub(crate) request: Request,
    pub(crate) response: Response,
}

pub(crate) struct RpcError<Request> {
    pub(crate) request: Request,
    pub(crate) status: Status,
}

pub(crate) struct SessionRpcExchange {
    pub(crate) request: SessionRequest,
    pub(crate) stream: Streaming<SessionEvent>,
}

pub(crate) struct LoadRpcExchange {
    pub(crate) request: LoadRequest,
    pub(crate) response: LoadResponse,
    pub(crate) state: LoadState,
}

pub(crate) struct LoadRpcError {
    pub(crate) request: LoadRequest,
    pub(crate) status: Status,
    pub(crate) state: LoadState,
}

pub(crate) struct ReleaseRpcError {
    pub(crate) request: ReleaseRequest,
    pub(crate) status: Status,
}

impl MockVllmRpcHarness {
    pub(crate) async fn new() -> Self {
        Self::with_workers(
            1,
            1,
            vec![WorkerRegistration {
                device_id: 0,
                tp_rank: 0,
            }],
        )
        .await
    }

    pub(crate) async fn naive_tp2() -> Self {
        Self::with_workers(
            2,
            2,
            vec![
                WorkerRegistration {
                    device_id: 0,
                    tp_rank: 0,
                },
                WorkerRegistration {
                    device_id: 1,
                    tp_rank: 1,
                },
            ],
        )
        .await
    }

    async fn with_workers(
        tp_size: usize,
        world_size: usize,
        workers: Vec<WorkerRegistration>,
    ) -> Self {
        let mut ctxs = Vec::with_capacity(workers.len());
        let mut gpus = Vec::with_capacity(workers.len());
        for worker in &workers {
            let ctx = CudaContext::new(worker.device_id as usize).expect("CUDA init");
            let gpu = TestGpuData::new(Arc::clone(&ctx), BLOCK_COUNT, BYTES_PER_BLOCK);
            ctxs.push(ctx);
            gpus.push(gpu);
        }

        let engine = Arc::new(test_engine());
        register_test_layers(&engine, &workers, &gpus, tp_size, world_size);

        let (scheduler, worker, server) = spawn_engine_server(Arc::clone(&engine)).await;

        Self {
            _ctxs: ctxs,
            _engine: engine,
            workers,
            server,
            gpus,
            scheduler,
            worker,
            extra_gpus: Vec::new(),
        }
    }

    pub(crate) fn register_second_instance(&mut self) {
        let worker = self.workers[0];
        let ctx = CudaContext::new(worker.device_id as usize).expect("CUDA init");
        let gpu = TestGpuData::new(Arc::clone(&ctx), BLOCK_COUNT, BYTES_PER_BLOCK);
        register_instance_layers(
            &self._engine,
            SECOND_INSTANCE_ID,
            &[worker],
            std::slice::from_ref(&gpu),
            1,
            1,
        );
        self._ctxs.push(ctx);
        self.extra_gpus.push(gpu);
    }

    pub(crate) async fn save_blocks(
        &mut self,
        hashes: &[Vec<u8>],
    ) -> RpcExchange<SaveRequest, SaveResponse> {
        self.save_blocks_for_worker(0, hashes).await
    }

    pub(crate) async fn save_blocks_for_worker(
        &mut self,
        worker_index: usize,
        hashes: &[Vec<u8>],
    ) -> RpcExchange<SaveRequest, SaveResponse> {
        match self.try_save_blocks_for_worker(worker_index, hashes).await {
            Ok(exchange) => exchange,
            Err(err) => panic!("save rpc failed unexpectedly: {}", err.status),
        }
    }

    pub(crate) async fn try_save_blocks_for_worker(
        &mut self,
        worker_index: usize,
        hashes: &[Vec<u8>],
    ) -> Result<RpcExchange<SaveRequest, SaveResponse>, RpcError<SaveRequest>> {
        let worker = self.workers[worker_index];
        let request = SaveRequest {
            instance_id: INSTANCE_ID.to_string(),
            tp_rank: worker.tp_rank,
            device_id: worker.device_id,
            saves: vec![SaveLayer {
                layer_name: LAYER_NAME.to_string(),
                block_ids: (0..hashes.len() as i32).collect(),
                block_hashes: hashes.to_vec(),
            }],
            pp_rank: 0,
        };
        match self.worker.save(request.clone()).await {
            Ok(response) => {
                let response = response.into_inner();
                if response.status.as_ref().is_some_and(|status| status.ok) {
                    self._engine.flush_saves().await;
                }
                Ok(RpcExchange { request, response })
            }
            Err(status) => Err(RpcError { request, status }),
        }
    }

    pub(crate) async fn query_prefetch(
        &mut self,
        req_id: &str,
        hashes: &[Vec<u8>],
    ) -> RpcExchange<QueryRequest, QueryResponse> {
        self.query_prefetch_for_instance(INSTANCE_ID, req_id, hashes)
            .await
    }

    pub(crate) async fn query_prefetch_for_instance(
        &mut self,
        instance_id: &str,
        req_id: &str,
        hashes: &[Vec<u8>],
    ) -> RpcExchange<QueryRequest, QueryResponse> {
        match self
            .try_query_prefetch_for_instance(instance_id, req_id, hashes)
            .await
        {
            Ok(exchange) => exchange,
            Err(err) => panic!("query_prefetch rpc failed unexpectedly: {}", err.status),
        }
    }

    pub(crate) async fn try_query_prefetch(
        &mut self,
        req_id: &str,
        hashes: &[Vec<u8>],
    ) -> Result<RpcExchange<QueryRequest, QueryResponse>, RpcError<QueryRequest>> {
        self.try_query_prefetch_for_instance(INSTANCE_ID, req_id, hashes)
            .await
    }

    pub(crate) async fn try_query_prefetch_for_instance(
        &mut self,
        instance_id: &str,
        req_id: &str,
        hashes: &[Vec<u8>],
    ) -> Result<RpcExchange<QueryRequest, QueryResponse>, RpcError<QueryRequest>> {
        let request = QueryRequest {
            instance_id: instance_id.to_string(),
            block_hashes: hashes.to_vec(),
            req_id: req_id.to_string(),
        };
        match self.scheduler.query_prefetch(request.clone()).await {
            Ok(response) => Ok(RpcExchange {
                request,
                response: response.into_inner(),
            }),
            Err(status) => Err(RpcError { request, status }),
        }
    }

    pub(crate) async fn open_session(&mut self) -> SessionRpcExchange {
        let request = SessionRequest {
            instance_id: INSTANCE_ID.to_string(),
            namespace: NAMESPACE.to_string(),
            tp_size: self.workers.len() as u32,
            world_size: self.workers.len() as u32,
        };
        let stream = self
            .scheduler
            .session(request.clone())
            .await
            .expect("session rpc")
            .into_inner();
        SessionRpcExchange { request, stream }
    }

    pub(crate) async fn wait_for_instance_cleanup(
        &mut self,
        hashes: &[Vec<u8>],
    ) -> RpcError<QueryRequest> {
        let deadline = Instant::now() + LOAD_WAIT_TIMEOUT;
        loop {
            match self
                .try_query_prefetch("mock-vllm-session-cleanup-probe", hashes)
                .await
            {
                Ok(_) if Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Ok(_) => panic!("timed out waiting for session cleanup"),
                Err(err)
                    if err.status.code() == tonic::Code::FailedPrecondition
                        && err.status.message().contains("instance") =>
                {
                    return err;
                }
                Err(err) => panic!("unexpected cleanup probe error: {}", err.status),
            }
        }
    }

    pub(crate) async fn try_release_lease(
        &mut self,
        lease: Vec<u8>,
    ) -> Result<RpcExchange<ReleaseRequest, ReleaseResponse>, ReleaseRpcError> {
        let request = ReleaseRequest { lease };
        match self.scheduler.release(request.clone()).await {
            Ok(response) => Ok(RpcExchange {
                request,
                response: response.into_inner(),
            }),
            Err(status) => Err(ReleaseRpcError { request, status }),
        }
    }

    pub(crate) async fn submit_load(
        &mut self,
        lease: Vec<u8>,
        block_count: usize,
    ) -> LoadRpcExchange {
        self.submit_load_for_worker(0, lease, block_count).await
    }

    pub(crate) async fn submit_load_for_worker(
        &mut self,
        worker_index: usize,
        lease: Vec<u8>,
        block_count: usize,
    ) -> LoadRpcExchange {
        match self
            .try_submit_load_for_worker(worker_index, lease, block_count)
            .await
        {
            Ok(exchange) => exchange,
            Err(err) => panic!("load rpc failed unexpectedly: {}", err.status),
        }
    }

    pub(crate) async fn try_submit_load_for_worker(
        &mut self,
        worker_index: usize,
        lease: Vec<u8>,
        block_count: usize,
    ) -> Result<LoadRpcExchange, LoadRpcError> {
        self.try_submit_load_for_instance_worker(INSTANCE_ID, worker_index, lease, block_count)
            .await
    }

    pub(crate) async fn try_submit_load_for_instance_worker(
        &mut self,
        instance_id: &str,
        worker_index: usize,
        lease: Vec<u8>,
        block_count: usize,
    ) -> Result<LoadRpcExchange, LoadRpcError> {
        let worker = self.workers[worker_index];
        let state = LoadState::new().expect("create LoadState");
        let request = LoadRequest {
            instance_id: instance_id.to_string(),
            tp_rank: worker.tp_rank,
            device_id: worker.device_id,
            load_state_shm: state.shm_name().to_string(),
            layer_names: vec![LAYER_NAME.to_string()],
            loads: vec![LeaseLoad {
                lease,
                block_ids: (0..block_count as i32).collect(),
            }],
        };
        match self.worker.load(request.clone()).await {
            Ok(response) => Ok(LoadRpcExchange {
                request,
                response: response.into_inner(),
                state,
            }),
            Err(status) => Err(LoadRpcError {
                request,
                status,
                state,
            }),
        }
    }

    pub(crate) async fn wait_for_load(&self, load_state: &LoadState) {
        wait_for_load(load_state).await;
    }
}

impl Drop for MockVllmRpcHarness {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn spawn_engine_server(
    engine: Arc<PegaEngine>,
) -> (
    EngineClient<Channel>,
    EngineClient<Channel>,
    tokio::task::JoinHandle<()>,
) {
    let port = unused_local_port();
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let registry = Arc::new(Mutex::new(CudaTensorRegistry::empty()));
    let shutdown = Arc::new(Notify::new());
    let hll_tracker = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            vec![("24h".into(), Duration::from_secs(86400))],
            14,
        ),
    ));
    let service = GrpcEngineService::new(engine, registry, shutdown, hll_tracker);

    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(addr)
            .await
            .expect("Engine gRPC serve");
    });

    let endpoint = format!("http://127.0.0.1:{port}");
    let scheduler = connect_client(&endpoint).await;
    let worker = connect_client(&endpoint).await;
    (scheduler, worker, handle)
}

async fn connect_client(endpoint: &str) -> EngineClient<Channel> {
    let deadline = Instant::now() + LOAD_WAIT_TIMEOUT;
    loop {
        match EngineClient::connect(endpoint.to_string()).await {
            Ok(client) => return client,
            Err(err) if Instant::now() < deadline => {
                let _ = err;
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Err(err) => panic!("timed out connecting to test server: {err}"),
        }
    }
}

async fn wait_for_load(load_state: &LoadState) {
    let deadline = Instant::now() + LOAD_WAIT_TIMEOUT;
    loop {
        let state = load_state.get();
        if state == LOAD_STATE_SUCCESS {
            return;
        }
        assert!(state != LOAD_STATE_ERROR, "load reported ERROR");
        assert!(Instant::now() < deadline, "timed out waiting for load");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn test_engine() -> PegaEngine {
    PegaEngine::new_with_config(
        16 << 20,
        false,
        StorageConfig {
            enable_lfu_admission: false,
            ..StorageConfig::default()
        },
    )
    .expect("test engine should start")
}

fn register_test_layers(
    engine: &PegaEngine,
    workers: &[WorkerRegistration],
    gpus: &[TestGpuData],
    tp_size: usize,
    world_size: usize,
) {
    register_instance_layers(engine, INSTANCE_ID, workers, gpus, tp_size, world_size);
}

fn register_instance_layers(
    engine: &PegaEngine,
    instance_id: &str,
    workers: &[WorkerRegistration],
    gpus: &[TestGpuData],
    tp_size: usize,
    world_size: usize,
) {
    for (worker, gpu) in workers.iter().zip(gpus.iter()) {
        engine
            .register_context_layer_batch(
                instance_id,
                NAMESPACE,
                worker.device_id,
                worker.tp_rank as usize,
                0,
                tp_size,
                world_size,
                1,
                &[LAYER_NAME.to_string()],
                &[gpu.ptr()],
                &[gpu.total_size()],
                &[BLOCK_COUNT],
                &[BYTES_PER_BLOCK],
                &[0],
                &[1],
            )
            .expect("register_context_layer_batch");
    }
}

fn unused_local_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
    listener.local_addr().expect("local addr").port()
}

pub(crate) fn make_block_hashes(num_blocks: usize, salt: u8) -> Vec<Vec<u8>> {
    (0..num_blocks)
        .map(|idx| {
            let mut hash = Vec::with_capacity(5);
            hash.push(salt);
            hash.extend_from_slice(&(idx as u32).to_le_bytes());
            hash
        })
        .collect()
}

pub(crate) fn cuda_device_count() -> usize {
    let mut count = 0;
    check_cuda(unsafe { sys::cuInit(0) }, "cuInit");
    check_cuda(
        unsafe { sys::cuDeviceGetCount(&raw mut count) },
        "cuDeviceGetCount",
    );
    count as usize
}

pub(crate) struct TestGpuData {
    gpu: GpuBuffer,
    ctx: Arc<CudaContext>,
    expected: Vec<u8>,
    block_size: usize,
}

impl TestGpuData {
    fn new(ctx: Arc<CudaContext>, num_blocks: usize, block_size: usize) -> Self {
        ctx.bind_to_thread().expect("bind CUDA context");
        let total = num_blocks * block_size;
        let gpu = GpuBuffer::alloc(total);
        let mut expected = vec![0u8; total];
        fill_test_pattern(&mut expected, block_size);
        gpu.copy_from_host(&expected);
        Self {
            gpu,
            ctx,
            expected,
            block_size,
        }
    }

    fn ptr(&self) -> u64 {
        self.gpu.as_u64()
    }

    fn total_size(&self) -> usize {
        self.expected.len()
    }

    pub(crate) fn zero(&self) {
        self.ctx.bind_to_thread().expect("bind CUDA context");
        self.gpu.zero();
    }

    pub(crate) fn assert_matches_expected(&self) {
        self.ctx.bind_to_thread().expect("bind CUDA context");
        assert_eq!(self.gpu.copy_to_host(), self.expected, "GPU data mismatch");
    }

    pub(crate) fn assert_prefix_loaded_and_suffix_zero(&self, loaded_blocks: usize) {
        self.ctx.bind_to_thread().expect("bind CUDA context");
        let actual = self.gpu.copy_to_host();
        let loaded_bytes = loaded_blocks * self.block_size;
        assert_eq!(&actual[..loaded_bytes], &self.expected[..loaded_bytes]);
        assert!(
            actual[loaded_bytes..].iter().all(|byte| *byte == 0),
            "GPU suffix after loaded prefix should stay zero"
        );
    }
}

impl Drop for TestGpuData {
    fn drop(&mut self) {
        let _ = self.ctx.bind_to_thread();
    }
}

fn fill_test_pattern(host_data: &mut [u8], block_size: usize) {
    assert_eq!(
        host_data.len() % block_size,
        0,
        "host_data must contain full blocks"
    );
    for (i, block) in host_data.chunks_exact_mut(block_size).enumerate() {
        let fill = ((i % 251) + 1) as u8;
        block.fill(fill);
    }
}

struct GpuBuffer {
    ptr: sys::CUdeviceptr,
    len: usize,
}

impl GpuBuffer {
    fn alloc(len: usize) -> Self {
        assert!(len > 0, "GpuBuffer::alloc: len must be > 0");
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
