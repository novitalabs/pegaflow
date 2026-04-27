use pegaflow_core::{trace_in_span, trace_root};

use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    GetPdReceiveDescriptorRequest, GetPdReceiveDescriptorResponse, HealthRequest, HealthResponse,
    LoadRequest, LoadResponse, PdReceiveDescriptorState,
    PdReceiveLayerLayout as ProtoPdReceiveLayerLayout, PdReceiveRank, PdReceiveSlab,
    PrepareLoadRequest, PrepareLoadResponse, QueryBlocksForTransferRequest,
    QueryBlocksForTransferResponse, RdmaHandshakeRequest, RdmaHandshakeResponse,
    RegisterContextRequest, RegisterContextResponse, ReleaseTransferLockRequest,
    ReleaseTransferLockResponse, ResponseStatus, SaveRequest, SaveResponse, SessionEvent,
    SessionRequest, ShutdownRequest, ShutdownResponse, TransferBlockInfo, TransferSlotInfo,
    UnpinRequest, UnpinResponse, UnregisterRequest, UnregisterResponse,
};
use crate::registry::CudaTensorRegistry;
use crate::session::SessionRegistry;
use log::{debug, info, warn};
use parking_lot::Mutex;
use pegaflow_core::{
    EngineError, LayerSave, PdReceiveDescriptorLookup, PegaEngine, PrepareLoadOutcome,
    PrepareLoadRequest as CorePrepareLoadRequest, PrepareLoadState,
    PreparedLoadItem as CorePreparedLoadItem,
};
use pyo3::{PyErr, Python};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, async_trait};

#[derive(Clone)]
pub struct GrpcEngineService {
    engine: Arc<PegaEngine>,
    registry: Arc<Mutex<CudaTensorRegistry>>,
    shutdown: Arc<Notify>,
    hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::HllTracker>>,
    session_registry: Arc<SessionRegistry>,
}

impl GrpcEngineService {
    pub fn new(
        engine: Arc<PegaEngine>,
        registry: Arc<Mutex<CudaTensorRegistry>>,
        shutdown: Arc<Notify>,
        hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::HllTracker>>,
    ) -> Self {
        Self {
            engine,
            registry,
            shutdown,
            hll_tracker,
            session_registry: SessionRegistry::new(),
        }
    }

    /// Drop CUDA IPC tensors and engine-side instance state for `instance_id`.
    /// Idempotent — safe to call after the instance is already gone.
    fn cleanup_instance(
        engine: &PegaEngine,
        registry: &Mutex<CudaTensorRegistry>,
        instance_id: &str,
        reason: &'static str,
    ) {
        let removed = {
            let mut registry = registry.lock();
            registry.drop_instance(instance_id)
        };
        if removed > 0 {
            info!(
                "Session cleanup ({}): dropped {} CUDA tensors for instance {}",
                reason, removed, instance_id
            );
        }
        if let Err(err) = engine.unregister_instance(instance_id) {
            // `InstanceMissing` is normal if the instance was never registered
            // (vllm died before any register_context_batch). Log at debug.
            debug!(
                "Session cleanup ({}): engine.unregister_instance({}) returned {}",
                reason, instance_id, err
            );
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

    fn u32_from_usize(value: usize, field: &str) -> Result<u32, Status> {
        u32::try_from(value)
            .map_err(|_| Status::internal(format!("{field}={value} does not fit into u32")))
    }

    fn build_register_context_response() -> RegisterContextResponse {
        RegisterContextResponse {
            status: Some(Self::ok_status()),
        }
    }

    fn build_simple_response() -> ResponseStatus {
        Self::ok_status()
    }

    fn validate_prepared_load_items(items: &[CorePreparedLoadItem]) -> Result<(), Status> {
        for (index, item) in items.iter().enumerate() {
            if item.plan_id == 0 {
                return Err(Status::invalid_argument(format!(
                    "items[{index}].plan_id must be non-zero"
                )));
            }
            if item.block_ids.is_empty() {
                return Err(Status::invalid_argument(format!(
                    "items[{index}].block_ids must not be empty"
                )));
            }
        }
        Ok(())
    }

    fn validate_prepare_load_request(request: &PrepareLoadRequest) -> Result<(), Status> {
        if request.prepare_state_shm.is_empty() {
            return Err(Status::invalid_argument(
                "prepare_state_shm must not be empty",
            ));
        }
        if request.virtual_block_size == 0 {
            return Err(Status::invalid_argument(
                "virtual_block_size must be greater than zero",
            ));
        }
        if request.request_id.is_empty() {
            return Err(Status::invalid_argument("request_id must not be empty"));
        }
        Ok(())
    }

    fn build_transfer_slot_info(
        raw_block: &Arc<pegaflow_core::RawBlock>,
        numa_node: pegaflow_common::NumaNode,
    ) -> TransferSlotInfo {
        let layer_block = pegaflow_core::LayerBlock::new(Arc::clone(raw_block));
        if let Some(v_ptr) = layer_block.v_ptr() {
            TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: v_ptr as u64,
                v_size: layer_block.v_size().unwrap_or(0) as u64,
                numa_node: numa_node.0,
            }
        } else {
            TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: 0,
                v_size: 0,
                numa_node: numa_node.0,
            }
        }
    }
}

#[async_trait]
impl Engine for GrpcEngineService {
    async fn register_context_batch(
        &self,
        request: Request<RegisterContextRequest>,
    ) -> Result<Response<RegisterContextResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<RegisterContextResponse>, Status> = async {
            let req = request.into_inner();
            debug!(
                "RPC [register_context_batch]: instance_id={} namespace={} device_id={} tp_rank={} tp_size={} world_size={} num_layers={} layer_names={:?} num_blocks={:?} bytes_per_block={:?} kv_stride_bytes={:?} segments={:?} wrapper_bytes_lens={:?}",
                req.instance_id,
                req.namespace,
                req.device_id,
                req.tp_rank,
                req.tp_size,
                req.world_size,
                req.num_layers,
                req.layer_names,
                req.num_blocks,
                req.bytes_per_block,
                req.kv_stride_bytes,
                req.segments,
                req.wrapper_bytes.iter().map(|b| b.len()).collect::<Vec<_>>()
            );

            let num_layers = Self::usize_from_u32(req.num_layers, "num_layers")?;

            // Validate array lengths are consistent with each other.
            // Note: num_layers is the *instance-wide* total (used for topology),
            // which may exceed the local batch size when pipeline parallelism
            // splits layers across ranks.
            let batch_len = req.layer_names.len();
            if batch_len == 0
                || req.wrapper_bytes.len() != batch_len
                || req.num_blocks.len() != batch_len
                || req.bytes_per_block.len() != batch_len
                || req.kv_stride_bytes.len() != batch_len
                || req.segments.len() != batch_len
            {
                return Err(Status::invalid_argument(format!(
                    "all layer arrays must have the same non-zero length (got layer_names={batch_len})"
                )));
            }
            if batch_len > num_layers {
                return Err(Status::invalid_argument(format!(
                    "layer batch size {batch_len} exceeds instance num_layers {num_layers}"
                )));
            }

            // Materialize tensors and collect data_ptr/size_bytes
            let context_key = Self::context_key(&req.instance_id, req.tp_rank, req.device_id);
            let mut data_ptrs = Vec::with_capacity(num_layers);
            let mut size_bytes_list = Vec::with_capacity(num_layers);
            {
                let mut registry = self.registry.lock();
                for (layer_name, wrapper_bytes) in
                    req.layer_names.iter().zip(req.wrapper_bytes.iter())
                {
                    let metadata = registry
                        .register_layer(&context_key, layer_name, req.device_id, wrapper_bytes)
                        .map_err(|err| Self::map_py_error("register tensor", err))?;
                    data_ptrs.push(metadata.data_ptr);
                    size_bytes_list.push(metadata.size_bytes);
                }
            }

            let num_blocks_list: Vec<usize> = req
                .num_blocks
                .into_iter()
                .map(|v| Self::usize_from_u64(v, "num_blocks"))
                .collect::<Result<_, _>>()?;
            let bytes_per_block_list: Vec<usize> = req
                .bytes_per_block
                .into_iter()
                .map(|v| Self::usize_from_u64(v, "bytes_per_block"))
                .collect::<Result<_, _>>()?;
            let kv_stride_bytes_list: Vec<usize> = req
                .kv_stride_bytes
                .into_iter()
                .map(|v| Self::usize_from_u64(v, "kv_stride_bytes"))
                .collect::<Result<_, _>>()?;
            let segments_list: Vec<usize> = req
                .segments
                .into_iter()
                .map(|v| Self::usize_from_u32(v, "segments"))
                .collect::<Result<_, _>>()?;

            let tp_rank = Self::usize_from_u32(req.tp_rank, "tp_rank")?;
            let tp_size = Self::usize_from_u32(req.tp_size, "tp_size")?;
            let world_size = Self::usize_from_u32(req.world_size, "world_size")?;

            // Call engine batch registration
            self.engine
                .register_context_layer_batch(
                    &req.instance_id,
                    &req.namespace,
                    req.device_id,
                    tp_rank,
                    tp_size,
                    world_size,
                    num_layers,
                    &req.layer_names,
                    &data_ptrs,
                    &size_bytes_list,
                    &num_blocks_list,
                    &bytes_per_block_list,
                    &kv_stride_bytes_list,
                    &segments_list,
                )
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(Self::build_register_context_response()))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [register_context_batch] completed: ok elapsed_ms={:.2}",
                elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [register_context_batch] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("register_context_batch", &result, start);
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

        trace_root!("rpc.save", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("layers", layer_count.to_string()),
                ("blocks", total_blocks.to_string()),
            ]
        });

        let fut = async {
            let SaveRequest {
                instance_id,
                tp_rank,
                device_id,
                saves,
                ..
            } = req;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;

            let saves: Vec<LayerSave> = saves
                .into_iter()
                .map(|layer| LayerSave {
                    layer_name: layer.layer_name,
                    block_ids: layer.block_ids,
                    block_hashes: layer.block_hashes,
                })
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
        };

        let result: Result<Response<SaveResponse>, Status> = trace_in_span!(root, fut).await;

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
        let item_count = req.items.len();
        let block_count: usize = req.items.iter().map(|item| item.block_ids.len()).sum();

        trace_root!("rpc.load", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("layers", layer_count.to_string()),
                ("items", item_count.to_string()),
                ("blocks", block_count.to_string()),
            ]
        });

        let fut = async {
            let LoadRequest {
                instance_id,
                tp_rank,
                device_id,
                load_state_shm,
                layer_names,
                items,
                ..
            } = req;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;
            let items: Vec<CorePreparedLoadItem> = items
                .into_iter()
                .map(|item| CorePreparedLoadItem {
                    plan_id: item.plan_id,
                    block_ids: item.block_ids,
                })
                .collect();
            Self::validate_prepared_load_items(&items)?;
            debug!(
                "RPC [load]: instance_id={} tp_rank={} device_id={} layers={} items={} blocks={} load_state_shm_len={}",
                instance_id,
                tp_rank,
                device_id,
                layer_count,
                item_count,
                block_count,
                load_state_shm.len()
            );
            let layer_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();

            self.engine
                .load_prepared_kv_blocks_multi_layer(
                    &instance_id,
                    tp_rank,
                    device_id,
                    &load_state_shm,
                    &layer_refs,
                    &items,
                )
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(LoadResponse {
                status: Some(Self::build_simple_response()),
            }))
        };

        let result: Result<Response<LoadResponse>, Status> = trace_in_span!(root, fut).await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [load] completed: ok layers={} items={} blocks={} elapsed_ms={:.2}",
                layer_count, item_count, block_count, elapsed_ms
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

    async fn prepare_load(
        &self,
        request: Request<PrepareLoadRequest>,
    ) -> Result<Response<PrepareLoadResponse>, Status> {
        let req = request.into_inner();
        trace_root!("rpc.prepare_load", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("block_hashes", req.block_hashes.len().to_string()),
            ]
        });

        let start = Instant::now();
        let engine = Arc::clone(&self.engine);
        let hll_tracker = Arc::clone(&self.hll_tracker);
        let fut = async {
            Self::validate_prepare_load_request(&req)?;
            let state = PrepareLoadState::attach(&req.prepare_state_shm)
                .map_err(|err| Status::invalid_argument(format!("invalid prepare shm: {err}")))?;
            let core_request = CorePrepareLoadRequest {
                instance_id: req.instance_id,
                request_id: req.request_id,
                block_hashes: req.block_hashes,
                num_prompt_tokens: req.num_prompt_tokens,
                num_computed_tokens: req.num_computed_tokens,
                virtual_block_size: req.virtual_block_size,
                decode_request_id: if req.decode_request_id.is_empty() {
                    None
                } else {
                    Some(req.decode_request_id)
                },
                decode_expected_writes: Self::usize_from_u32(
                    req.decode_expected_writes,
                    "decode_expected_writes",
                )?,
            };
            debug!(
                "RPC [prepare_load]: instance_id={} request_id={} hashes={} computed={} prompt={} vbs={} decode={} shm_len={}",
                core_request.instance_id,
                core_request.request_id,
                core_request.block_hashes.len(),
                core_request.num_computed_tokens,
                core_request.num_prompt_tokens,
                core_request.virtual_block_size,
                core_request.decode_request_id.as_deref().unwrap_or(""),
                req.prepare_state_shm.len()
            );

            tokio::spawn(async move {
                loop {
                    match engine.prepare_load_step(&core_request).await {
                        Ok(PrepareLoadOutcome::Pending) => {
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        Ok(PrepareLoadOutcome::NoPlan) => {
                            if let Ok(mut tracker) = hll_tracker.lock() {
                                tracker.record_hashes(&core_request.block_hashes);
                            }
                            state.set_ready_no_plan();
                            break;
                        }
                        Ok(PrepareLoadOutcome::Plan {
                            plan_id,
                            num_tokens,
                        }) => {
                            if let Ok(mut tracker) = hll_tracker.lock() {
                                tracker.record_hashes(&core_request.block_hashes);
                            }
                            state.set_ready_plan(num_tokens, plan_id);
                            break;
                        }
                        Err(err) => {
                            warn!(
                                "prepare_load task failed: instance={} request={} error={}",
                                core_request.instance_id, core_request.request_id, err
                            );
                            state.set_error();
                            break;
                        }
                    }
                }
            });

            Ok(Response::new(PrepareLoadResponse {
                status: Some(Self::build_simple_response()),
            }))
        };

        let result: Result<Response<PrepareLoadResponse>, Status> = trace_in_span!(root, fut).await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!("RPC [prepare_load] accepted: elapsed_ms={:.2}", elapsed_ms),
            Err(status) => warn!(
                "RPC [prepare_load] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("prepare_load", &result, start);
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

    async fn query_blocks_for_transfer(
        &self,
        request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let hash_count = req.block_hashes.len();

        debug!(
            "RPC [query_blocks_for_transfer]: namespace={} hashes={} requester={}",
            req.namespace, hash_count, req.requester_id,
        );

        if !self.engine.has_rdma_transport() {
            return Err(Status::failed_precondition(
                "RDMA transfer engine is not configured",
            ));
        }

        let result: Result<Response<QueryBlocksForTransferResponse>, Status> = async {
            let (session_id, found_blocks) = self.engine.query_blocks_for_transfer(
                &req.namespace,
                &req.block_hashes,
                &req.requester_id,
            );

            let blocks: Vec<TransferBlockInfo> = found_blocks
                .iter()
                .map(|(key, block)| {
                    let slots: Vec<TransferSlotInfo> = block
                        .slots()
                        .iter()
                        .zip(block.slot_numas())
                        .map(|(raw, &numa)| Self::build_transfer_slot_info(raw, numa))
                        .collect();
                    TransferBlockInfo {
                        block_hash: key.hash.clone(),
                        slots,
                    }
                })
                .collect();

            Ok(Response::new(QueryBlocksForTransferResponse {
                status: Some(Self::build_simple_response()),
                blocks,
                transfer_session_id: session_id,
                lock_timeout_secs: self.engine.transfer_lock_timeout().as_secs() as u32,
            }))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => debug!(
                "RPC [query_blocks_for_transfer] completed: ok found={} session={} elapsed_ms={:.2}",
                response.get_ref().blocks.len(),
                response.get_ref().transfer_session_id,
                elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [query_blocks_for_transfer] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("query_blocks_for_transfer", &result, start);
        result
    }

    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!(
            "RPC [release_transfer_lock]: session={}",
            req.transfer_session_id
        );

        let released = self.engine.release_transfer_lock(&req.transfer_session_id);

        let result = Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(Self::build_simple_response()),
            released_blocks: released as u64,
        }));

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "RPC [release_transfer_lock] completed: ok released={} elapsed_ms={:.2}",
            released, elapsed_ms
        );
        record_rpc_result("release_transfer_lock", &result, start);
        result
    }

    async fn get_pd_receive_descriptor(
        &self,
        request: Request<GetPdReceiveDescriptorRequest>,
    ) -> Result<Response<GetPdReceiveDescriptorResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!(
            "RPC [get_pd_receive_descriptor]: dst_instance_id={} request_id={} receive_rank={} handle={}",
            req.dst_instance_id, req.request_id, req.receive_rank, req.handle
        );

        let result: Result<Response<GetPdReceiveDescriptorResponse>, Status> = async {
            if req.dst_instance_id.is_empty() {
                return Err(Status::invalid_argument(
                    "dst_instance_id must not be empty",
                ));
            }
            if req.request_id.is_empty() {
                return Err(Status::invalid_argument("request_id must not be empty"));
            }
            let receive_rank = if req.receive_rank < 0 {
                None
            } else {
                Some(usize::try_from(req.receive_rank).map_err(|_| {
                    Status::invalid_argument("receive_rank does not fit into usize")
                })?)
            };

            let lookup = self.engine.get_pd_receive_descriptor(
                &req.dst_instance_id,
                &req.request_id,
                receive_rank,
                Some(req.handle.as_str()),
            );

            let mut response = GetPdReceiveDescriptorResponse {
                status: Some(Self::build_simple_response()),
                state: PdReceiveDescriptorState::PdDescriptorPending.into(),
                handle: String::new(),
                slabs: Vec::new(),
                layers: Vec::new(),
                block_hashes: Vec::new(),
                imm_data: 0,
                expires_at_ms: 0,
                data_ready: false,
                ranks: Vec::new(),
            };

            match lookup {
                PdReceiveDescriptorLookup::Pending => {}
                PdReceiveDescriptorLookup::Failed => {
                    response.state = PdReceiveDescriptorState::PdDescriptorFailed.into();
                }
                PdReceiveDescriptorLookup::Expired => {
                    response.state = PdReceiveDescriptorState::PdDescriptorExpired.into();
                }
                PdReceiveDescriptorLookup::Ready(descriptor) => {
                    response.state = PdReceiveDescriptorState::PdDescriptorReady.into();
                    response.handle = descriptor.handle;
                    response.imm_data = descriptor.imm_data;
                    response.expires_at_ms = descriptor.expires_at_ms;
                    response.data_ready = descriptor.data_ready;
                    response.block_hashes = descriptor.block_hashes;
                    response.ranks = descriptor
                        .ranks
                        .into_iter()
                        .map(|rank| {
                            Ok(PdReceiveRank {
                                receive_rank: Self::u32_from_usize(
                                    rank.receive_rank,
                                    "receive_rank",
                                )?,
                                device_id: rank.device_id,
                                tp_rank: Self::u32_from_usize(rank.tp_rank, "tp_rank")?,
                                slab_index: Self::u32_from_usize(rank.slab_index, "slab_index")?,
                                numa_node: rank.numa_node.0,
                            })
                        })
                        .collect::<Result<Vec<_>, Status>>()?;
                    response.slabs = descriptor
                        .slabs
                        .into_iter()
                        .map(|slab| PdReceiveSlab {
                            base_ptr: slab.base_ptr,
                            size: slab.size,
                            numa_node: slab.numa_node.0,
                        })
                        .collect();
                    response.layers = descriptor
                        .layers
                        .into_iter()
                        .map(|layer| {
                            Ok(ProtoPdReceiveLayerLayout {
                                layer_name: layer.layer_name,
                                slab_index: Self::u32_from_usize(layer.slab_index, "slab_index")?,
                                layer_offset: layer.layer_offset,
                                block_stride: layer.block_stride,
                                segment_count: Self::u32_from_usize(
                                    layer.segment_count,
                                    "segment_count",
                                )?,
                                segment_size: layer.segment_size,
                                padded_segment_stride: layer.padded_segment_stride,
                                num_blocks: layer.num_blocks as u64,
                                slot_id: Self::u32_from_usize(layer.slot_id, "slot_id")?,
                                receive_rank: Self::u32_from_usize(
                                    layer.receive_rank,
                                    "receive_rank",
                                )?,
                            })
                        })
                        .collect::<Result<Vec<_>, Status>>()?;
                }
            }

            Ok(Response::new(response))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => {
                let resp = response.get_ref();
                let state = PdReceiveDescriptorState::try_from(resp.state)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|_| format!("Unknown({})", resp.state));
                debug!(
                    "RPC [get_pd_receive_descriptor] completed: ok state={} ranks={} slabs={} layers={} elapsed_ms={:.2}",
                    state,
                    resp.ranks.len(),
                    resp.slabs.len(),
                    resp.layers.len(),
                    elapsed_ms
                );
            }
            Err(status) => warn!(
                "RPC [get_pd_receive_descriptor] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("get_pd_receive_descriptor", &result, start);
        result
    }

    async fn rdma_handshake(
        &self,
        request: Request<RdmaHandshakeRequest>,
    ) -> Result<Response<RdmaHandshakeResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        debug!("RPC [rdma_handshake]: requester={}", req.requester_id,);

        if !self.engine.has_rdma_transport() {
            return Err(Status::failed_precondition(
                "RDMA transfer engine is not configured",
            ));
        }

        let result: Result<Response<RdmaHandshakeResponse>, Status> = async {
            let server_meta = self
                .engine
                .rdma_accept_handshake(&req.requester_id, &req.handshake_metadata)
                .map_err(|e| Status::internal(format!("RDMA handshake failed: {e}")))?;

            Ok(Response::new(RdmaHandshakeResponse {
                status: Some(Self::build_simple_response()),
                handshake_metadata: server_meta,
            }))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [rdma_handshake] completed: ok requester={} elapsed_ms={:.2}",
                req.requester_id, elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [rdma_handshake] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("rdma_handshake", &result, start);
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

    type SessionStream = ReceiverStream<Result<SessionEvent, Status>>;

    async fn session(
        &self,
        request: Request<SessionRequest>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let req = request.into_inner();
        let instance_id = req.instance_id;
        if instance_id.is_empty() {
            return Err(Status::invalid_argument("instance_id must not be empty"));
        }

        let token = self.session_registry.install(instance_id.clone());
        info!(
            "Session opened: instance_id={} namespace={} tp_size={} world_size={} token={}",
            instance_id, req.namespace, req.tp_size, req.world_size, token
        );

        // Channel capacity is 1: we don't emit events yet, and a tight capacity
        // makes backpressure explicit if we ever do.
        let (tx, rx) = mpsc::channel::<Result<SessionEvent, Status>>(1);

        // Hand a clone to the watcher; it observes `closed()` when the tonic
        // runtime drops the receiver side after client disconnect. We keep the
        // original `tx` alive until this function returns `Response`; once the
        // handler future is dropped by tonic on client cancel, both `tx` and
        // `rx` are dropped → `cleanup_tx.closed()` resolves.
        let cleanup_tx = tx.clone();

        let session_registry = Arc::clone(&self.session_registry);
        let engine = Arc::clone(&self.engine);
        let cuda_registry = Arc::clone(&self.registry);
        let id_for_watcher = instance_id.clone();

        tokio::spawn(async move {
            cleanup_tx.closed().await;
            if session_registry.take(&id_for_watcher, token) {
                info!(
                    "Session closed: instance_id={} token={} — running cleanup",
                    id_for_watcher, token
                );
                Self::cleanup_instance(&engine, &cuda_registry, &id_for_watcher, "stream closed");
            } else {
                debug!(
                    "Session closed: instance_id={} token={} superseded — skip cleanup",
                    id_for_watcher, token
                );
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    fn valid_prepare_load_request() -> PrepareLoadRequest {
        PrepareLoadRequest {
            instance_id: "instance".to_string(),
            request_id: "request".to_string(),
            block_hashes: Vec::new(),
            num_prompt_tokens: 0,
            num_computed_tokens: 0,
            virtual_block_size: 16,
            prepare_state_shm: "prepare-state".to_string(),
            decode_request_id: String::new(),
            decode_expected_writes: 0,
        }
    }

    #[tokio::test]
    async fn load_rejects_zero_plan_id() {
        let status = GrpcEngineService::validate_prepared_load_items(&[CorePreparedLoadItem {
            plan_id: 0,
            block_ids: vec![1],
        }])
        .expect_err("zero plan_id should fail");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("plan_id"));
    }

    #[tokio::test]
    async fn load_rejects_empty_prepared_block_ids() {
        let status = GrpcEngineService::validate_prepared_load_items(&[CorePreparedLoadItem {
            plan_id: 1,
            block_ids: vec![],
        }])
        .expect_err("empty prepared block_ids should fail");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("block_ids"));
    }

    #[tokio::test]
    async fn prepare_load_rejects_zero_virtual_block_size() {
        let mut request = valid_prepare_load_request();
        request.virtual_block_size = 0;

        let status = GrpcEngineService::validate_prepare_load_request(&request)
            .expect_err("zero virtual_block_size should fail");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("virtual_block_size"));
    }

    #[tokio::test]
    async fn prepare_load_rejects_empty_request_id() {
        let mut request = valid_prepare_load_request();
        request.request_id.clear();

        let status = GrpcEngineService::validate_prepare_load_request(&request)
            .expect_err("empty request_id should fail");

        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("request_id"));
    }
}
