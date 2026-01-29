use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    HealthRequest, HealthResponse, LoadRequest, LoadResponse, PrefetchState, QueryRequest,
    QueryResponse, RegisterContextRequest, RegisterContextResponse, ResponseStatus, SaveRequest,
    SaveResponse, ShutdownRequest, ShutdownResponse, UnpinRequest, UnpinResponse,
    UnregisterRequest, UnregisterResponse,
};
use crate::registry::{CudaTensorRegistry, TensorMetadata};
use log::{debug, info, warn};
use parking_lot::Mutex;
use pegaflow_core::{EngineError, PegaEngine, PrefetchStatus};
use pyo3::{PyErr, Python};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;
use tonic::{Request, Response, Status, async_trait};

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
    async fn register_context(
        &self,
        request: Request<RegisterContextRequest>,
    ) -> Result<Response<RegisterContextResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<RegisterContextResponse>, Status> = async {
            let req = request.into_inner();
            debug!(
                "RPC [register_context]: instance_id={} namespace={} layer_name={} device_id={} tp_rank={} tp_size={} world_size={} num_layers={} num_blocks={} bytes_per_block={} kv_stride_bytes={} segments={} wrapper_bytes={}",
                req.instance_id,
                req.namespace,
                req.layer_name,
                req.device_id,
                req.tp_rank,
                req.tp_size,
                req.world_size,
                req.num_layers,
                req.num_blocks,
                req.bytes_per_block,
                req.kv_stride_bytes,
                req.segments,
                req.wrapper_bytes.len()
            );
            let metadata = self.handle_tensor_registration(&req)?;

            let num_blocks = Self::usize_from_u64(req.num_blocks, "num_blocks")?;
            let bytes_per_block = Self::usize_from_u64(req.bytes_per_block, "bytes_per_block")?;
            let kv_stride_bytes = Self::usize_from_u64(req.kv_stride_bytes, "kv_stride_bytes")?;
            let segments = Self::usize_from_u32(req.segments, "segments")?;
            let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
            let tp_size = Self::usize_from_u32(req.tp_size, "tp_size")?;
            let world_size = Self::usize_from_u32(req.world_size, "world_size")?;
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
                    world_size,
                    num_layers,
                )
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(Self::build_register_context_response()))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [register_context] completed: ok elapsed_ms={:.2}",
                elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [register_context] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("register_context", &result, start);
        result
    }

    async fn save(&self, request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
        let start = Instant::now();

        let req = request.into_inner();
        let layer_count = req.saves.len();
        let (total_blocks, total_hashes) =
            req.saves.iter().fold((0usize, 0usize), |(b, h), layer| {
                (b + layer.block_ids.len(), h + layer.block_hashes.len())
            });

        let result: Result<Response<SaveResponse>, Status> = async {
            let SaveRequest {
                instance_id,
                tp_rank,
                device_id,
                saves,
                ..
            } = req;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;

            let saves: Vec<_> = saves
                .into_iter()
                .map(|layer| (layer.layer_name, layer.block_ids, layer.block_hashes))
                .collect();

            debug!(
                "RPC [save]: instance_id={} tp_rank={} device_id={} layers={} blocks={} hashes={}",
                instance_id, tp_rank, device_id, layer_count, total_blocks, total_hashes
            );

            self.engine
                .batch_save_kv_blocks_from_ipc(&instance_id, tp_rank, device_id, saves)
                .await
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(SaveResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [save] completed: ok layers={} blocks={} hashes={} elapsed_ms={:.2}",
                layer_count, total_blocks, total_hashes, elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [save] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("save", &result, start);
        result
    }

    async fn load(&self, request: Request<LoadRequest>) -> Result<Response<LoadResponse>, Status> {
        let start = Instant::now();

        let req = request.into_inner();
        let layer_count = req.layer_names.len();
        let block_count = req.block_ids.len();
        let hash_count = req.block_hashes.len();

        let result: Result<Response<LoadResponse>, Status> = async {
            let LoadRequest {
                instance_id,
                tp_rank,
                device_id,
                layer_names,
                block_ids,
                block_hashes,
                load_state_shm,
                ..
            } = req;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;
            debug!(
                "RPC [load]: instance_id={} tp_rank={} device_id={} layers={} block_ids={} block_hashes={} load_state_shm_len={}",
                instance_id,
                tp_rank,
                device_id,
                layer_count,
                block_count,
                hash_count,
                load_state_shm.len()
            );
            let layer_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            self.engine
                .batch_load_kv_blocks_multi_layer(
                    &instance_id,
                    tp_rank,
                    device_id,
                    &load_state_shm,
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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [load] completed: ok layers={} blocks={} elapsed_ms={:.2}",
                layer_count, block_count, elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [load] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("load", &result, start);
        result
    }

    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<QueryResponse>, Status> = async {
            let req = request.into_inner();
            debug!(
                "RPC [query]: instance_id={} block_hashes={}",
                req.instance_id,
                req.block_hashes.len()
            );

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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => {
                let resp = response.get_ref();
                let state = PrefetchState::try_from(resp.prefetch_state)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|_| format!("Unknown({})", resp.prefetch_state));
                debug!(
                    "RPC [query] completed: ok hit={} loading={} missing={} state={} elapsed_ms={:.2}",
                    resp.hit_blocks, resp.loading_blocks, resp.missing_blocks, state, elapsed_ms
                )
            }
            Err(status) => warn!(
                "RPC [query] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("query", &result, start);
        result
    }

    async fn unpin(
        &self,
        request: Request<UnpinRequest>,
    ) -> Result<Response<UnpinResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let hash_count = req.block_hashes.len();

        let result: Result<Response<UnpinResponse>, Status> = async {
            debug!(
                "RPC [unpin]: instance_id={} block_hashes={}",
                req.instance_id, hash_count
            );

            self.engine
                .unpin_blocks(&req.instance_id, &req.block_hashes)
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(UnpinResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [unpin] completed: ok blocks={} elapsed_ms={:.2}",
                hash_count, elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [unpin] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("unpin", &result, start);
        result
    }

    async fn unregister_context(
        &self,
        request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<UnregisterResponse>, Status> = async {
            let req = request.into_inner();
            debug!("RPC [unregister_context]: instance_id={}", req.instance_id);
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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [unregister_context] completed: ok elapsed_ms={:.2}",
                elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [unregister_context] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("unregister_context", &result, start);
        result
    }

    async fn shutdown(
        &self,
        _request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<ShutdownResponse>, Status> = async {
            debug!("RPC [shutdown] requested");
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

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!("RPC [shutdown] completed: ok elapsed_ms={:.2}", elapsed_ms),
            Err(status) => warn!(
                "RPC [shutdown] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("shutdown", &result, start);
        result
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<HealthResponse>, Status> = async {
            debug!("RPC [health]");
            Ok(Response::new(HealthResponse {
                status: Some(Self::build_simple_response()),
            }))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!("RPC [health] completed: ok elapsed_ms={:.2}", elapsed_ms),
            Err(status) => warn!(
                "RPC [health] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("health", &result, start);
        result
    }
}
