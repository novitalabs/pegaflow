use pegaflow_core::{trace_in_span, trace_root};

use crate::fd_channel::RegistrationKey;
use crate::metric::record_rpc_result;
use crate::proto::engine::engine_server::Engine;
use crate::proto::engine::{
    FlushRequest, FlushResponse, HealthRequest, HealthResponse, LoadRequest, LoadResponse,
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, QueryLoading, QueryReady,
    QueryRequest, QueryResponse, RdmaHandshakeRequest, RdmaHandshakeResponse,
    RegisterContextRequest, RegisterContextResponse, ReleaseRequest, ReleaseResponse,
    ReleaseTransferLockRequest, ReleaseTransferLockResponse, SaveRequest, SaveResponse,
    SessionEvent, SessionRequest, ShutdownRequest, ShutdownResponse, TransferBlockInfo,
    TransferMode as ProtoTransferMode, TransferSlotInfo, UnregisterRequest, UnregisterResponse,
    query_response,
};
use crate::registry::{RegistryHandle, TensorRegistration};
use crate::session::SessionRegistry;
use log::{debug, info, warn};
use pegaflow_core::{LayerSave, PegaEngine, PrefetchStatus, QueryLeaseId};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Notify, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, async_trait};

mod helpers;
mod validation;

#[derive(Clone)]
pub struct GrpcEngineService {
    engine: Arc<PegaEngine>,
    registry: RegistryHandle,
    shutdown: Arc<Notify>,
    hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::MultiWindowHllTracker>>,
    session_registry: Arc<SessionRegistry>,
    fd_channel: Option<crate::fd_channel::FdChannel>,
}

impl GrpcEngineService {
    pub fn new(
        engine: Arc<PegaEngine>,
        registry: RegistryHandle,
        shutdown: Arc<Notify>,
        hll_tracker: Arc<std::sync::Mutex<pegaflow_common::hll::MultiWindowHllTracker>>,
        fd_channel: Option<crate::fd_channel::FdChannel>,
    ) -> Self {
        Self {
            engine,
            registry,
            shutdown,
            hll_tracker,
            session_registry: SessionRegistry::new(),
            fd_channel,
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

            validation::register_context(&req)?;

            // The connector picks the H2D/D2H backend per model and sends it
            // here. Read it before `req` is partially moved below.
            let transfer_mode = match req.transfer_mode() {
                ProtoTransferMode::Direct => pegaflow_core::TransferMode::Direct,
                ProtoTransferMode::Kernel => pegaflow_core::TransferMode::Kernel,
            };

            // Validate array lengths are consistent with each other.
            let batch_len = req.layer_names.len();
            if batch_len == 0
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

            let context_key =
                Self::context_key(&req.instance_id, req.tp_rank, req.pp_rank, req.device_id);
            let native = !req.native_kv_tensors.is_empty();
            let mut block_stride_bytes = Vec::with_capacity(batch_len);
            let layers = if native {
                req.layer_names
                    .iter()
                    .cloned()
                    .zip(req.native_kv_tensors)
                    .map(|(name, tensor)| {
                        block_stride_bytes.push(Self::usize_from_u64(
                            tensor.block_stride_bytes,
                            "block_stride_bytes",
                        )?);
                        Ok((
                            name,
                            TensorRegistration::Native {
                                offset_bytes: tensor.offset_bytes,
                                size_bytes: tensor.size_bytes,
                            },
                        ))
                    })
                    .collect::<Result<Vec<_>, Status>>()?
            } else {
                req.layer_names
                    .iter()
                    .cloned()
                    .zip(req.wrapper_bytes)
                    .map(|(name, bytes)| (name, TensorRegistration::Python(bytes)))
                    .collect()
            };

            let native_fd = if native {
                let channel = self.fd_channel.as_ref().ok_or_else(|| {
                    Status::failed_precondition(
                        "native VMM registration requires --fd-socket-path on the server",
                    )
                })?;
                let alloc_size = Self::usize_from_u64(req.native_alloc_size, "native_alloc_size")?;
                if alloc_size == 0 {
                    return Err(Status::invalid_argument(
                        "native registration requires a non-zero native_alloc_size",
                    ));
                }
                let fd = channel
                    .take(
                        RegistrationKey {
                            instance_id: req.instance_id.clone(),
                            device_id: req.device_id,
                        },
                        std::time::Duration::from_secs(60),
                    )
                    .await
                    .ok_or_else(|| {
                        Status::failed_precondition(format!(
                            "no allocation fd received on the side-channel for instance {}",
                            req.instance_id
                        ))
                    })?;
                Some((fd, alloc_size))
            } else {
                None
            };

            let registry = self.registry.clone();
            let engine = Arc::clone(&self.engine);
            // Finish registry + engine publication, or rollback both, even if
            // tonic cancels the request after the actor accepted registration.
            tokio::spawn(async move {
                let (metadatas, registration_guard, registration_permit) = registry
                    .register_layers(
                        context_key.clone(),
                        req.instance_id.clone(),
                        req.device_id,
                        layers,
                        native_fd,
                    )
                    .await
                    .map_err(|message| {
                        Status::internal(format!("register tensor failed: {message}"))
                    })?;
                let mut data_ptrs = Vec::with_capacity(batch_len);
                let mut size_bytes_list = Vec::with_capacity(batch_len);
                for metadata in &metadatas {
                    data_ptrs.push(metadata.data_ptr);
                    size_bytes_list.push(metadata.size_bytes);
                }

                if let Err(err) = engine.register_context_layer_batch_strided(
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
                    native.then_some(block_stride_bytes.as_slice()),
                    transfer_mode,
                    req.page_first,
                ) {
                    let status = Self::map_engine_error(err);
                    drop(registration_guard);
                    let cleanup = registry.drop_context(context_key.clone()).await;
                    if cleanup.tensor_count() > 0 {
                        warn!(
                            "Rolled back {} CUDA tensor(s) for failed register_context_batch context {}",
                            cleanup.tensor_count(),
                            context_key
                        );
                    }
                    registry.finish_cleanup(cleanup).await;
                    drop(registration_permit);
                    return Err(status);
                }

                drop(registration_permit);
                Ok::<(), Status>(())
            })
            .await
            .map_err(|err| Status::internal(format!("registration task failed: {err}")))??;
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
            let registration = self
                .registry
                .acquire_registration(instance_id.clone(), device_id)
                .await
                .map_err(Status::failed_precondition)?;

            let saves: Vec<LayerSave> = saves
                .into_iter()
                .map(|layer| LayerSave {
                    layer_name: layer.layer_name,
                    block_ids: layer.block_ids.into_iter().map(|id| id as usize).collect(),
                    block_hashes: layer.block_hashes,
                })
                .collect();

            debug!(
                "RPC [save]: instance_id={} tp_rank={} pp_rank={} device_id={} layers={} blocks={} hashes={}",
                instance_id, tp_rank, pp_rank, device_id, layer_count, total_blocks, total_hashes
            );

            let engine = Arc::clone(&self.engine);
            tokio::spawn(async move {
                let _registration = registration;
                engine
                    .batch_save_kv_blocks_from_ipc(&instance_id, tp_rank, pp_rank, device_id, saves)
                    .await
                    .map_err(Self::map_engine_error)
            })
            .await
            .map_err(|err| Status::internal(format!("save operation task failed: {err}")))??;

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
                wait_for_completion,
                ..
            } = req;
            Self::validate_device_id(device_id)?;
            let tp_rank = Self::usize_from_u32(tp_rank, "tp_rank")?;
            let registration = self
                .registry
                .acquire_registration(instance_id.clone(), device_id)
                .await
                .map_err(Status::failed_precondition)?;
            if registration.is_some() && !wait_for_completion {
                return Err(Status::invalid_argument(
                    "native VMM loads require wait_for_completion",
                ));
            }
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
            let loads: Vec<(QueryLeaseId, Vec<usize>)> = loads
                .into_iter()
                .map(|load| {
                    let lease =
                        QueryLeaseId::from_bytes(&load.lease).map_err(Status::invalid_argument)?;
                    let block_ids = load.block_ids.into_iter().map(|id| id as usize).collect();
                    Ok::<_, Status>((lease, block_ids))
                })
                .collect::<Result<_, _>>()?;

            if wait_for_completion {
                if !load_state_shm.is_empty() {
                    return Err(Status::invalid_argument(
                        "synchronous load must not include load_state_shm",
                    ));
                }
                let engine = Arc::clone(&self.engine);
                tokio::spawn(async move {
                    let _registration = registration;
                    let layer_refs: Vec<&str> = layer_names.iter().map(String::as_str).collect();
                    engine
                        .batch_load_kv_blocks_multi_layer_inproc(
                            &instance_id,
                            tp_rank,
                            device_id,
                            &layer_refs,
                            &loads,
                        )
                        .map_err(Self::map_engine_error)?
                        .await
                        .map_err(|_| Status::internal("load worker dropped completion"))?
                        .map_err(Self::map_engine_error)
                })
                .await
                .map_err(|err| Status::internal(format!("load operation task failed: {err}")))??;
            } else {
                let layer_refs: Vec<&str> = layer_names.iter().map(String::as_str).collect();
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
            }

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
                    req.wait_for_full_prefix,
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

    async fn flush(
        &self,
        _request: Request<FlushRequest>,
    ) -> Result<Response<FlushResponse>, Status> {
        self.engine.flush_saves_and_registrations().await;
        Ok(Response::new(FlushResponse {
            status: Some(Self::build_simple_response()),
        }))
    }

    async fn unregister_context(
        &self,
        request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        let start = Instant::now();
        let result: Result<Response<UnregisterResponse>, Status> = async {
            let req = request.into_inner();
            debug!("RPC [unregister_context]: instance_id={}", req.instance_id);
            let engine = Arc::clone(&self.engine);
            let registry = self.registry.clone();
            let instance_id = req.instance_id.clone();
            tokio::spawn(async move {
                let cleanup = registry.drop_instance(instance_id.clone()).await;
                if cleanup.tensor_count() > 0 {
                    info!(
                        "Dropped {} CUDA tensors for instance {}",
                        cleanup.tensor_count(),
                        instance_id
                    );
                }
                let unregister = engine.unregister_instance(&instance_id);
                registry.finish_cleanup(cleanup).await;
                unregister.map_err(Self::map_engine_error)
            })
            .await
            .map_err(|err| Status::internal(format!("unregister task failed: {err}")))??;

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
mod tests;
