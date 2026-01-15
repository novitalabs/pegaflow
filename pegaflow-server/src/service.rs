use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    HealthRequest, HealthResponse, LoadRequest, LoadResponse, PrefetchState, QueryRequest,
    QueryResponse, RegisterContextRequest, RegisterContextResponse, ResponseStatus, SaveRequest,
    SaveResponse, ShutdownRequest, ShutdownResponse, UnregisterRequest, UnregisterResponse,
};
use crate::registry::{CudaTensorRegistry, TensorMetadata};
use parking_lot::Mutex;
use pegaflow_core::{EngineError, PegaEngine, PrefetchStatus};
use pyo3::{PyErr, Python};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;
use tonic::{async_trait, Request, Response, Status};
use tracing::{info, instrument, warn, Span};

#[derive(Clone)]
pub struct GrpcEngineService {
    engine: Arc<PegaEngine>,
    registry: Arc<Mutex<CudaTensorRegistry>>,
    shutdown: Arc<Notify>,
}

impl GrpcEngineService {
    pub fn new(
        engine: Arc<PegaEngine>,
        registry: Arc<Mutex<CudaTensorRegistry>>,
        shutdown: Arc<Notify>,
    ) -> Self {
        Self {
            engine,
            registry,
            shutdown,
        }
    }

    fn context_key(instance_id: &str, tp_rank: u32, device_id: i32) -> String {
        format!("{instance_id}:tp{tp_rank}:dev{device_id}")
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn map_engine_error(err: EngineError) -> Status {
        match err {
            EngineError::InvalidArgument(_) => Status::invalid_argument(err.to_string()),
            EngineError::InstanceMissing(_) | EngineError::WorkerMissing(_, _) => {
                Status::failed_precondition(err.to_string())
            }
            EngineError::TopologyMismatch(_) => Status::failed_precondition(err.to_string()),
            EngineError::CudaInit(_) | EngineError::Storage(_) | EngineError::Poisoned(_) => {
                Status::internal(err.to_string())
            }
        }
    }

    fn map_py_error(operation: &str, err: PyErr) -> Status {
        let message = Python::attach(|py| err.value(py).to_string());
        Status::internal(format!("{operation} failed: {message}"))
    }

    fn usize_from_u64(value: u64, field: &str) -> Result<usize, Status> {
        usize::try_from(value).map_err(|_| {
            Status::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    fn usize_from_u32(value: u32, field: &str) -> Result<usize, Status> {
        usize::try_from(value).map_err(|_| {
            Status::invalid_argument(format!("{field}={value} does not fit into usize"))
        })
    }

    fn build_register_context_response() -> RegisterContextResponse {
        RegisterContextResponse {
            status: Some(Self::ok_status()),
        }
    }

    fn build_simple_response() -> ResponseStatus {
        Self::ok_status()
    }

    fn handle_tensor_registration(
        &self,
        req: &RegisterContextRequest,
    ) -> Result<TensorMetadata, Status> {
        let mut registry = self.registry.lock();
        registry
            .register_layer(
                &Self::context_key(&req.instance_id, req.tp_rank, req.device_id),
                &req.layer_name,
                req.device_id,
                &req.wrapper_bytes,
            )
            .map_err(|err| Self::map_py_error("register tensor", err))
    }
}

#[async_trait]
impl Engine for GrpcEngineService {
    #[instrument(
        level = "info",
        skip(self, request),
        fields(instance=%request.get_ref().instance_id, tp_rank=%request.get_ref().tp_rank, device=%request.get_ref().device_id, layer=%request.get_ref().layer_name)
    )]
    async fn register_context(
        &self,
        request: Request<RegisterContextRequest>,
    ) -> Result<Response<RegisterContextResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<RegisterContextResponse>, Status> = async {
            let req = request.into_inner();
            let metadata = self.handle_tensor_registration(&req)?;

            let num_blocks = Self::usize_from_u64(req.num_blocks, "num_blocks")?;
            let bytes_per_block = Self::usize_from_u64(req.bytes_per_block, "bytes_per_block")?;
            let kv_stride_bytes = Self::usize_from_u64(req.kv_stride_bytes, "kv_stride_bytes")?;
            let segments = Self::usize_from_u32(req.segments, "segments")?;
            let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
            let tp_size = Self::usize_from_u32(req.tp_size, "tp_size")?;
            let num_layers = Self::usize_from_u32(req.num_layers, "num_layers")?;

            self.engine
                .register_context_layer(
                    &req.instance_id,
                    &req.namespace,
                    metadata.device_id,
                    req.layer_name.clone(),
                    metadata.data_ptr,
                    metadata.size_bytes,
                    num_blocks,
                    bytes_per_block,
                    kv_stride_bytes,
                    segments,
                    tp_rank,
                    tp_size,
                    num_layers,
                )
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(Self::build_register_context_response()))
        }
        .await;

        record_rpc_result("register_context", &result, start);
        result
    }

    #[instrument(
        level = "info",
        skip(self, request),
        fields(instance=%request.get_ref().instance_id, tp_rank=%request.get_ref().tp_rank, device=%request.get_ref().device_id, layers=%request.get_ref().saves.len(), blocks = tracing::field::Empty)
    )]
    async fn save(&self, request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<SaveResponse>, Status> = async {
            let req = request.into_inner();
            let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
            let total_blocks: usize = req.saves.first().map(|l| l.block_ids.len()).unwrap_or(0);

            let saves = req
                .saves
                .into_iter()
                .map(|layer| (layer.layer_name, layer.block_ids, layer.block_hashes))
                .collect();

            Span::current().record("blocks", total_blocks);
            self.engine
                .batch_save_kv_blocks_from_ipc(&req.instance_id, tp_rank, req.device_id, saves)
                .await
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(SaveResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        record_rpc_result("save", &result, start);
        result
    }

    #[instrument(
        level = "info",
        skip(self, request),
        fields(instance=%request.get_ref().instance_id, tp_rank=%request.get_ref().tp_rank, device=%request.get_ref().device_id, layers=%request.get_ref().layer_names.len(), blocks=%request.get_ref().block_ids.len())
    )]
    async fn load(&self, request: Request<LoadRequest>) -> Result<Response<LoadResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<LoadResponse>, Status> = async {
            let req = request.into_inner();
            let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
            let layer_names = req.layer_names;
            let layer_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();
            let block_ids = req.block_ids;
            let block_hashes = req.block_hashes;

            self.engine
                .batch_load_kv_blocks_multi_layer(
                    &req.instance_id,
                    tp_rank,
                    req.device_id,
                    &req.load_state_shm,
                    &layer_refs,
                    &block_ids,
                    &block_hashes,
                )
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(LoadResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        record_rpc_result("load", &result, start);
        result
    }

    #[instrument(
        level = "info",
        skip(self, request),
        fields(instance=%request.get_ref().instance_id, blocks=%request.get_ref().block_hashes.len()),
        ret
    )]
    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<QueryResponse>, Status> = async {
            let req = request.into_inner();

            // Use SSD prefetch-aware query
            let status = self
                .engine
                .count_prefix_hit_blocks_with_prefetch(&req.instance_id, &req.block_hashes)
                .map_err(Self::map_engine_error)?;

            let (prefetch_state, hit_blocks, loading_blocks, missing_blocks) = match status {
                PrefetchStatus::Done { hit, missing } => {
                    (PrefetchState::PrefetchDone, hit as u64, 0, missing as u64)
                }
                PrefetchStatus::Loading { hit, loading } => (
                    PrefetchState::PrefetchLoading,
                    hit as u64,
                    loading as u64,
                    0,
                ),
            };

            Ok(Response::new(QueryResponse {
                status: Some(Self::build_simple_response()),
                hit_blocks,
                prefetch_state: prefetch_state.into(),
                loading_blocks,
                missing_blocks,
            }))
        }
        .await;

        record_rpc_result("query", &result, start);
        result
    }

    #[instrument(
        level = "info",
        skip(self, request),
        fields(instance=%request.get_ref().instance_id)
    )]
    async fn unregister_context(
        &self,
        request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<UnregisterResponse>, Status> = async {
            let req = request.into_inner();
            let removed = {
                let mut registry = self.registry.lock();
                registry.drop_instance(&req.instance_id)
            };
            if removed > 0 {
                info!(
                    "Dropped {} CUDA tensors for instance {}",
                    removed, req.instance_id
                );
            }

            self.engine
                .unregister_instance(&req.instance_id)
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(UnregisterResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        record_rpc_result("unregister_context", &result, start);
        result
    }

    #[instrument(level = "info", skip(self, _request))]
    async fn shutdown(
        &self,
        _request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<ShutdownResponse>, Status> = async {
            {
                let mut registry = self.registry.lock();
                registry.clear();
            }
            warn!("Shutdown requested via RPC");
            self.shutdown.notify_waiters();

            Ok(Response::new(ShutdownResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        record_rpc_result("shutdown", &result, start);
        result
    }

    #[instrument(level = "info", skip(self, _request))]
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<HealthResponse>, Status> = async {
            Ok(Response::new(HealthResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        record_rpc_result("health", &result, start);
        result
    }
}
