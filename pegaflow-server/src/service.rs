use pegaflow_core::{trace_in_span, trace_root};

use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    HealthRequest, HealthResponse, LoadRequest, LoadResponse, PrefetchState,
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, QueryRequest, QueryResponse,
    RegisterContextRequest, RegisterContextResponse, ReleaseTransferLockRequest,
    ReleaseTransferLockResponse, ResponseStatus, SaveRequest, SaveResponse, ShutdownRequest,
    ShutdownResponse, TransferBlockInfo, TransferSlotInfo, UnpinRequest, UnpinResponse,
    UnregisterRequest, UnregisterResponse,
};
use crate::registry::CudaTensorRegistry;
use log::{debug, info, warn};
use parking_lot::Mutex;
use pegaflow_core::{EngineError, LayerSave, PegaEngine, PrefetchStatus};
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
    rdma_session_id: Option<Vec<u8>>,
}

impl GrpcEngineService {
    pub fn new(
        engine: Arc<PegaEngine>,
        registry: Arc<Mutex<CudaTensorRegistry>>,
        shutdown: Arc<Notify>,
        rdma_session_id: Option<Vec<u8>>,
    ) -> Self {
        Self {
            engine,
            registry,
            shutdown,
            rdma_session_id,
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

    fn validate_load_request(block_ids: &[i32], block_hashes: &[Vec<u8>]) -> Result<(), Status> {
        if block_ids.len() != block_hashes.len() {
            return Err(Status::invalid_argument(format!(
                "block_ids and block_hashes must have the same length (got {} and {})",
                block_ids.len(),
                block_hashes.len()
            )));
        }
        Ok(())
    }

    fn build_transfer_slot_info(
        raw_block: &Arc<pegaflow_core::RawBlock>,
    ) -> Result<TransferSlotInfo, Status> {
        let layer_block = pegaflow_core::LayerBlock::new(Arc::clone(raw_block));
        if let Some(v_ptr) = layer_block.v_ptr() {
            Ok(TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: v_ptr as u64,
                v_size: layer_block.v_size().unwrap_or(0) as u64,
            })
        } else {
            Ok(TransferSlotInfo {
                k_ptr: layer_block.k_ptr() as u64,
                k_size: layer_block.k_size() as u64,
                v_ptr: 0,
                v_size: 0,
            })
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

            // Validate array lengths match num_layers
            if req.layer_names.len() != num_layers
                || req.wrapper_bytes.len() != num_layers
                || req.num_blocks.len() != num_layers
                || req.bytes_per_block.len() != num_layers
                || req.kv_stride_bytes.len() != num_layers
                || req.segments.len() != num_layers
            {
                return Err(Status::invalid_argument(format!(
                    "all layer arrays must have length {num_layers}"
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
        let block_count = req.block_ids.len();
        let hash_count = req.block_hashes.len();

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
            Self::validate_load_request(&block_ids, &block_hashes)?;
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

    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();
        trace_root!("rpc.query", root, || {
            [
                ("instance_id", req.instance_id.clone()),
                ("block_hashes", req.block_hashes.len().to_string()),
            ]
        });

        let start = Instant::now();
        let fut = async {
            debug!(
                "RPC [query]: instance_id={} block_hashes={}",
                req.instance_id,
                req.block_hashes.len()
            );

            // Pure memory-only query (no SSD prefetch)
            let (hit, missing) = self
                .engine
                .count_prefix_hit_blocks(&req.instance_id, &req.block_hashes)
                .map_err(Self::map_engine_error)?;

            Ok(Response::new(QueryResponse {
                status: Some(Self::build_simple_response()),
                hit_blocks: hit as u64,
                prefetch_state: PrefetchState::PrefetchDone.into(),
                loading_blocks: 0,
                missing_blocks: missing as u64,
            }))
        };

        let result: Result<Response<QueryResponse>, Status> = trace_in_span!(root, fut).await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => {
                let resp = response.get_ref();
                debug!(
                    "RPC [query] completed: ok hit={} missing={} elapsed_ms={:.2}",
                    resp.hit_blocks, resp.missing_blocks, elapsed_ms
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
        };

        let result: Result<Response<QueryResponse>, Status> = trace_in_span!(root, fut).await;

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(response) => {
                let resp = response.get_ref();
                let state = PrefetchState::try_from(resp.prefetch_state)
                    .map(|s| format!("{:?}", s))
                    .unwrap_or_else(|_| format!("Unknown({})", resp.prefetch_state));
                debug!(
                    "RPC [query_prefetch] completed: ok hit={} loading={} missing={} state={} elapsed_ms={:.2}",
                    resp.hit_blocks, resp.loading_blocks, resp.missing_blocks, state, elapsed_ms
                )
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
            req.namespace, hash_count, req.requester_id
        );

        const MAX_TRANSFER_BLOCK_HASHES: usize = 1024;
        if hash_count > MAX_TRANSFER_BLOCK_HASHES {
            return Err(Status::invalid_argument(format!(
                "block_hashes count {hash_count} exceeds maximum {MAX_TRANSFER_BLOCK_HASHES}"
            )));
        }

        let result: Result<Response<QueryBlocksForTransferResponse>, Status> = async {
            let rdma_session_id = self.rdma_session_id.clone().ok_or_else(|| {
                Status::failed_precondition("RDMA transfer engine is not configured")
            })?;
            let (session_id, found_blocks) = self.engine.query_blocks_for_transfer(
                &req.namespace,
                &req.block_hashes,
                &req.requester_id,
            );

            let blocks: Vec<TransferBlockInfo> = found_blocks
                .iter()
                .map(|(key, block)| -> Result<TransferBlockInfo, Status> {
                    let slots: Vec<TransferSlotInfo> = block
                        .slots()
                        .iter()
                        .map(Self::build_transfer_slot_info)
                        .collect::<Result<_, _>>()?;
                    Ok(TransferBlockInfo {
                        block_hash: key.hash.clone(),
                        slots,
                        rkey: 0, // TODO: set from RDMA engine when wired
                    })
                })
                .collect::<Result<_, _>>()?;

            Ok(Response::new(QueryBlocksForTransferResponse {
                status: Some(Self::build_simple_response()),
                blocks,
                transfer_session_id: session_id,
                rdma_session_id,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    #[tokio::test]
    async fn load_rejects_mismatched_block_ids_and_hashes() {
        let status = GrpcEngineService::validate_load_request(&[1, 2], &[vec![1]])
            .expect_err("mismatched load request should fail");
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("block_ids"));
        assert!(status.message().contains("block_hashes"));
    }
}
