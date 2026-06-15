use pegaflow_core::{trace_in_span, trace_root};

use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    HealthRequest, HealthResponse, LoadRequest, LoadResponse, QueryBlocksForTransferRequest,
    QueryBlocksForTransferResponse, QueryLoading, QueryReady, QueryRequest, QueryResponse,
    RdmaHandshakeRequest, RdmaHandshakeResponse, RegisterContextRequest, RegisterContextResponse,
    ReleaseRequest, ReleaseResponse, ReleaseTransferLockRequest, ReleaseTransferLockResponse,
    ResponseStatus, SaveRequest, SaveResponse, SessionEvent, SessionRequest, ShutdownRequest,
    ShutdownResponse, TransferBlockInfo, TransferSlotInfo, UnregisterRequest, UnregisterResponse,
    query_response,
};
use crate::registry::RegistryHandle;
use crate::session::SessionRegistry;
use log::{debug, info, warn};
use pegaflow_common::NumaNode;
use pegaflow_core::{EngineError, LayerSave, PegaEngine, PrefetchStatus, QueryLeaseId};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, async_trait};

#[derive(Clone)]
pub struct GrpcEngineService {
    engine: Arc<PegaEngine>,
    registry: RegistryHandle,
    shutdown: Arc<Notify>,
    hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::MultiWindowHllTracker>>,
    session_registry: Arc<SessionRegistry>,
}

impl GrpcEngineService {
    pub fn new(
        engine: Arc<PegaEngine>,
        registry: RegistryHandle,
        shutdown: Arc<Notify>,
        hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::MultiWindowHllTracker>>,
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
    async fn cleanup_instance(
        engine: &PegaEngine,
        registry: &RegistryHandle,
        instance_id: &str,
        reason: &'static str,
    ) {
        let removed = registry.drop_instance(instance_id.to_string()).await;
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

    fn context_key(instance_id: &str, tp_rank: u32, pp_rank: u32, device_id: i32) -> String {
        format!("{instance_id}:tp{tp_rank}:pp{pp_rank}:dev{device_id}")
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

    fn validate_device_id(device_id: i32) -> Result<(), Status> {
        if device_id < 0 {
            return Err(Status::invalid_argument(format!(
                "device_id {device_id} must be >= 0"
            )));
        }
        Ok(())
    }

    fn validate_register_context_request(req: &RegisterContextRequest) -> Result<(), Status> {
        let server_version = env!("CARGO_PKG_VERSION");
        if req.client_version != server_version {
            return Err(Status::failed_precondition(format!(
                "PegaFlow version mismatch: client={} server={server_version}",
                if req.client_version.is_empty() {
                    "<missing>"
                } else {
                    &req.client_version
                }
            )));
        }
        Self::validate_device_id(req.device_id)?;
        if req.tp_size == 0 {
            return Err(Status::invalid_argument("tp_size must be > 0"));
        }
        if req.world_size == 0 {
            return Err(Status::invalid_argument("world_size must be > 0"));
        }
        if req.tp_rank >= req.tp_size {
            return Err(Status::invalid_argument(format!(
                "tp_rank {} out of range (tp_size {})",
                req.tp_rank, req.tp_size
            )));
        }
        Ok(())
    }

    fn validate_save_layers(saves: &[crate::proto::engine::SaveLayer]) -> Result<(), Status> {
        for layer in saves {
            if layer.block_ids.len() != layer.block_hashes.len() {
                return Err(Status::invalid_argument(format!(
                    "block_ids length {} does not match block_hashes {} for layer {}",
                    layer.block_ids.len(),
                    layer.block_hashes.len(),
                    layer.layer_name
                )));
            }
        }
        Ok(())
    }

    fn validate_query_prefetch_request(req: &QueryRequest) -> Result<(), Status> {
        if req.req_id.is_empty() {
            return Err(Status::invalid_argument("req_id must not be empty"));
        }
        Ok(())
    }

    fn build_register_context_response() -> RegisterContextResponse {
        RegisterContextResponse {
            status: Some(Self::ok_status()),
        }
    }

    fn build_simple_response() -> ResponseStatus {
        Self::ok_status()
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

    fn save_numa_hint(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
    ) -> Option<NumaNode> {
        if tp_rank != 0 {
            return None;
        }

        match self.engine.instance_tp_size(instance_id) {
            Ok(1) => {}
            Ok(_) => return None,
            Err(err) => {
                debug!(
                    "save NUMA hint skipped: instance={} tp_rank={} could not read instance topology: {}",
                    instance_id, tp_rank, err
                );
                return None;
            }
        }

        let candidates = match self.engine.registered_numa_nodes_for_save_group(
            instance_id,
            tp_rank,
            pp_rank,
        ) {
            Ok(candidates) => candidates,
            Err(err) => {
                debug!(
                    "save NUMA hint skipped: instance={} tp_rank={} pp_rank={} could not read shard NUMA nodes: {}",
                    instance_id, tp_rank, pp_rank, err
                );
                return None;
            }
        };

        let hint =
            self.session_registry
                .next_save_numa_hint(instance_id, tp_rank, pp_rank, &candidates);
        if let Some(numa) = hint {
            debug!(
                "save NUMA hint selected: instance={} tp_rank={} pp_rank={} session_tp_size={} candidates={:?} hint={}",
                instance_id, tp_rank, pp_rank, numa.session_tp_size, candidates, numa.numa_node
            );
        }
        hint.map(|hint| hint.numa_node)
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
                "RPC [register_context_batch]: instance_id={} namespace={} device_id={} tp_rank={} pp_rank={} tp_size={} world_size={} layer_names={:?} num_blocks={:?} bytes_per_block={:?} kv_stride_bytes={:?} segments={:?} wrapper_bytes_lens={:?}",
                req.instance_id,
                req.namespace,
                req.device_id,
                req.tp_rank,
                req.pp_rank,
                req.tp_size,
                req.world_size,
                req.layer_names,
                req.num_blocks,
                req.bytes_per_block,
                req.kv_stride_bytes,
                req.segments,
                req.wrapper_bytes.iter().map(|b| b.len()).collect::<Vec<_>>()
            );

            Self::validate_register_context_request(&req)?;

            // Validate array lengths are consistent with each other.
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
            let pp_rank = Self::usize_from_u32(req.pp_rank, "pp_rank")?;
            let tp_size = Self::usize_from_u32(req.tp_size, "tp_size")?;
            let world_size = Self::usize_from_u32(req.world_size, "world_size")?;

            // Materialize tensors and collect data_ptr/size_bytes
            let context_key =
                Self::context_key(&req.instance_id, req.tp_rank, req.pp_rank, req.device_id);
            // Materialize on the dedicated registry thread (GIL + CUDA IPC) and
            // await the result, so this RPC never blocks an async worker. Move
            // the (large) wrapper bytes over; clone the layer names since the
            // engine call below still needs them.
            let layers: Vec<(String, Vec<u8>)> = req
                .layer_names
                .iter()
                .cloned()
                .zip(req.wrapper_bytes)
                .collect();
            let metadatas = self
                .registry
                .register_layers(context_key.clone(), req.device_id, layers)
                .await
                .map_err(|message| Status::internal(format!("register tensor failed: {message}")))?;
            let mut data_ptrs = Vec::with_capacity(batch_len);
            let mut size_bytes_list = Vec::with_capacity(batch_len);
            for metadata in &metadatas {
                data_ptrs.push(metadata.data_ptr);
                size_bytes_list.push(metadata.size_bytes);
            }

            // Call engine batch registration
            if let Err(err) = self.engine.register_context_layer_batch(
                &req.instance_id,
                &req.namespace,
                req.device_id,
                tp_rank,
                pp_rank,
                tp_size,
                world_size,
                &req.layer_names,
                &data_ptrs,
                &size_bytes_list,
                &num_blocks_list,
                &bytes_per_block_list,
                &kv_stride_bytes_list,
                &segments_list,
            ) {
                let status = Self::map_engine_error(err);
                let removed = self.registry.drop_context(context_key.clone()).await;
                if removed > 0 {
                    warn!(
                        "Rolled back {} CUDA tensor(s) for failed register_context_batch context {}",
                        removed, context_key
                    );
                }
                return Err(status);
            }

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
                pp_rank,
                device_id,
                saves,
                ..
            } = req;
            Self::validate_device_id(device_id)?;
            Self::validate_save_layers(&saves)?;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;
            let pp_rank = Self::usize_from_u32(pp_rank, "pp_rank")?;

            let saves: Vec<LayerSave> = saves
                .into_iter()
                .map(|layer| {
                    let block_ids = layer
                        .block_ids
                        .into_iter()
                        .map(|id| {
                            usize::try_from(id).map_err(|_| {
                                Status::invalid_argument(format!(
                                    "negative block_id {id} in layer {}",
                                    layer.layer_name
                                ))
                            })
                        })
                        .collect::<Result<Vec<usize>, _>>()?;
                    Ok::<_, Status>(LayerSave {
                        layer_name: layer.layer_name,
                        block_ids,
                        block_hashes: layer.block_hashes,
                    })
                })
                .collect::<Result<_, _>>()?;

            debug!(
                "RPC [save]: instance_id={} tp_rank={} pp_rank={} device_id={} layers={} blocks={} hashes={}",
                instance_id, tp_rank, pp_rank, device_id, layer_count, total_blocks, total_hashes
            );

            let numa_hint = self.save_numa_hint(&instance_id, tp_rank, pp_rank);

            self.engine
                .batch_save_kv_blocks_from_ipc_with_numa_hint(
                    &instance_id,
                    tp_rank,
                    pp_rank,
                    device_id,
                    saves,
                    numa_hint,
                )
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
        let load_count = req.loads.len();
        let block_count: usize = req.loads.iter().map(|load| load.block_ids.len()).sum();

        trace_root!("rpc.load", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("layers", layer_count.to_string()),
                ("blocks", block_count.to_string()),
            ]
        });

        let fut = async {
            let LoadRequest {
                instance_id,
                tp_rank,
                device_id,
                layer_names,
                loads,
                load_state_shm,
                ..
            } = req;
            Self::validate_device_id(device_id)?;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;
            debug!(
                "RPC [load]: instance_id={} tp_rank={} device_id={} layers={} loads={} blocks={} load_state_shm_len={}",
                instance_id,
                tp_rank,
                device_id,
                layer_count,
                load_count,
                block_count,
                load_state_shm.len()
            );
            let layer_refs: Vec<&str> = layer_names.iter().map(|s| s.as_str()).collect();
            let loads: Vec<(QueryLeaseId, Vec<usize>)> = loads
                .into_iter()
                .map(|load| {
                    let lease =
                        QueryLeaseId::from_bytes(&load.lease).map_err(Status::invalid_argument)?;
                    let block_ids = load
                        .block_ids
                        .into_iter()
                        .map(|id| {
                            usize::try_from(id).map_err(|_| {
                                Status::invalid_argument(format!(
                                    "negative destination block_id {id}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<usize>, _>>()?;
                    Ok::<_, Status>((lease, block_ids))
                })
                .collect::<Result<_, _>>()?;

            self.engine
                .batch_load_kv_blocks_multi_layer(
                    &instance_id,
                    tp_rank,
                    device_id,
                    &load_state_shm,
                    &layer_refs,
                    &loads,
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

    async fn query_prefetch(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        trace_root!("rpc.query_prefetch", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("block_hashes", req.block_hashes.len().to_string()),
            ]
        });

        let start = Instant::now();
        let fut = async {
            Self::validate_query_prefetch_request(&req)?;
            debug!(
                "RPC [query_prefetch]: instance_id={} block_hashes={}",
                req.instance_id,
                req.block_hashes.len()
            );

            // SSD prefetch-aware query
            let status = self
                .engine
                .count_prefix_hit_blocks_with_prefetch(
                    &req.instance_id,
                    &req.req_id,
                    &req.block_hashes,
                )
                .await
                .map_err(Self::map_engine_error)?;

            let outcome = match status {
                PrefetchStatus::Ready { blocks, missing } => {
                    let hit = blocks.len();
                    if let Ok(mut t) = self.hll_tracker.lock() {
                        t.record_hashes(&req.block_hashes);
                    }
                    let lease = if hit == 0 {
                        Vec::new()
                    } else {
                        self.engine
                            .create_query_lease(&req.instance_id, blocks)
                            .map_err(Self::map_engine_error)?
                            .to_bytes()
                            .to_vec()
                    };
                    debug!(
                        "RPC [query_prefetch] ready: instance_id={} hit={} missing={} lease={}",
                        req.instance_id,
                        hit,
                        missing,
                        !lease.is_empty()
                    );
                    query_response::Outcome::Ready(QueryReady {
                        num_hit_blocks: hit as u64,
                        lease,
                    })
                }
                PrefetchStatus::Loading => query_response::Outcome::Loading(QueryLoading {}),
            };

            Ok(Response::new(QueryResponse {
                outcome: Some(outcome),
            }))
        };

        let result: Result<Response<QueryResponse>, Status> = trace_in_span!(root, fut).await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => {
                let resp = response.get_ref();
                match &resp.outcome {
                    Some(query_response::Outcome::Ready(ready)) => debug!(
                        "RPC [query_prefetch] completed: ready hit={} lease={} elapsed_ms={:.2}",
                        ready.num_hit_blocks,
                        !ready.lease.is_empty(),
                        elapsed_ms
                    ),
                    Some(query_response::Outcome::Loading(_)) => debug!(
                        "RPC [query_prefetch] completed: loading elapsed_ms={:.2}",
                        elapsed_ms
                    ),
                    None => warn!(
                        "RPC [query_prefetch] completed without outcome elapsed_ms={:.2}",
                        elapsed_ms
                    ),
                }
            }
            Err(status) => warn!(
                "RPC [query_prefetch] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("query_prefetch", &result, start);
        result
    }

    async fn release(
        &self,
        request: Request<ReleaseRequest>,
    ) -> Result<Response<ReleaseResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();
        let lease_len = req.lease.len();

        let result: Result<Response<ReleaseResponse>, Status> = async {
            debug!("RPC [release]: lease_len={}", lease_len);

            let lease = QueryLeaseId::from_bytes(&req.lease).map_err(Status::invalid_argument)?;
            if !self.engine.release_query_lease(&lease) {
                return Err(Status::failed_precondition(
                    "query lease is unknown or expired",
                ));
            }

            Ok(Response::new(ReleaseResponse {}))
        }
        .await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(_) => debug!(
                "RPC [release] completed: ok lease_len={} elapsed_ms={:.2}",
                lease_len, elapsed_ms
            ),
            Err(status) => warn!(
                "RPC [release] failed: code={} message={} elapsed_ms={:.2}",
                status.code(),
                status.message(),
                elapsed_ms
            ),
        }
        record_rpc_result("release", &result, start);
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
            let removed = self.registry.drop_instance(req.instance_id.clone()).await;
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
            // Deliberately do NOT release CUDA tensors here. The process is on
            // its way out and the OS reclaims everything; meanwhile a
            // `torch.cuda.empty_cache()` on a wedged GPU would block forever and
            // stall the very shutdown path meant to let us escape. Just signal.
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

        let token = self.session_registry.install(
            instance_id.clone(),
            req.namespace.clone(),
            req.tp_size,
            req.world_size,
        );
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
        let cuda_registry = self.registry.clone();
        let id_for_watcher = instance_id.clone();

        tokio::spawn(async move {
            cleanup_tx.closed().await;
            if session_registry.take(&id_for_watcher, token) {
                info!(
                    "Session closed: instance_id={} token={} — running cleanup",
                    id_for_watcher, token
                );
                Self::cleanup_instance(&engine, &cuda_registry, &id_for_watcher, "stream closed")
                    .await;
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
    use crate::proto::engine::SaveLayer;
    use tonic::Code;

    #[test]
    fn validate_query_prefetch_rejects_empty_req_id() {
        let err = GrpcEngineService::validate_query_prefetch_request(&QueryRequest {
            instance_id: "instance".to_string(),
            block_hashes: Vec::new(),
            req_id: String::new(),
        })
        .expect_err("empty req_id must be rejected before engine lookup");

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("req_id"));
    }

    #[test]
    fn validate_query_prefetch_allows_empty_hashes() {
        GrpcEngineService::validate_query_prefetch_request(&QueryRequest {
            instance_id: "instance".to_string(),
            block_hashes: Vec::new(),
            req_id: "request".to_string(),
        })
        .expect("empty block_hashes are a valid zero-hit query");
    }

    #[test]
    fn validate_register_context_rejects_pure_argument_errors() {
        let err = GrpcEngineService::validate_register_context_request(&RegisterContextRequest {
            instance_id: "instance".to_string(),
            namespace: "namespace".to_string(),
            client_version: env!("CARGO_PKG_VERSION").to_string(),
            tp_rank: 1,
            tp_size: 1,
            world_size: 1,
            device_id: 0,
            layer_names: Vec::new(),
            wrapper_bytes: Vec::new(),
            num_blocks: Vec::new(),
            bytes_per_block: Vec::new(),
            kv_stride_bytes: Vec::new(),
            segments: Vec::new(),
            pp_rank: 0,
        })
        .expect_err("tp_rank outside tp_size must be rejected at RPC boundary");

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("tp_rank"));
    }

    #[test]
    fn validate_register_context_rejects_client_version_mismatch() {
        let err = GrpcEngineService::validate_register_context_request(&RegisterContextRequest {
            instance_id: "instance".to_string(),
            namespace: "namespace".to_string(),
            client_version: "0.0.0-test-mismatch".to_string(),
            tp_rank: 0,
            tp_size: 1,
            world_size: 1,
            device_id: 0,
            layer_names: Vec::new(),
            wrapper_bytes: Vec::new(),
            num_blocks: Vec::new(),
            bytes_per_block: Vec::new(),
            kv_stride_bytes: Vec::new(),
            segments: Vec::new(),
            pp_rank: 0,
        })
        .expect_err("client/server version mismatch must be rejected before registration");

        assert_eq!(err.code(), Code::FailedPrecondition);
        assert!(err.message().contains("PegaFlow version mismatch"));
        assert!(err.message().contains("client=0.0.0-test-mismatch"));
        assert!(
            err.message()
                .contains(concat!("server=", env!("CARGO_PKG_VERSION")))
        );
    }

    #[test]
    fn validate_save_layers_rejects_mismatched_block_shapes() {
        let err = GrpcEngineService::validate_save_layers(&[SaveLayer {
            layer_name: "layer_0".to_string(),
            block_ids: vec![0, 1],
            block_hashes: vec![vec![1]],
        }])
        .expect_err("service must reject malformed save shape");

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("does not match"));
    }

    #[test]
    fn validate_device_id_rejects_negative_values() {
        let err = GrpcEngineService::validate_device_id(-1)
            .expect_err("negative device_id must be rejected at RPC boundary");

        assert_eq!(err.code(), Code::InvalidArgument);
        assert!(err.message().contains("device_id"));
    }
}
