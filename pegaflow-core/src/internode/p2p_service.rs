//! Embeddable P2P transfer gRPC service.
//!
//! A node that *serves* cross-node RDMA fetches must expose three RPCs to its
//! peers: `RdmaHandshake` (QP bring-up), `QueryBlocksForTransfer` (pin blocks +
//! return pinned-memory descriptors), and `ReleaseTransferLock`. The full
//! `pegaflow-server` binary provides them alongside the CUDA-IPC registration
//! surface — which an in-process Rust embedder (one that registers raw device
//! pointers and drives saves/loads through `PegaEngine` directly) has no use
//! for. This service is that minimal serving surface: the three transfer RPCs
//! plus `Health`, everything else answered with `unimplemented`.
//!
//! The embedder remains responsible for periodic GC of expired transfer locks
//! (`PegaEngine::gc_expired_transfer_locks`), mirroring pegaflow-server's
//! background GC task — a crashed peer must not pin blocks forever.

use std::sync::Arc;

use log::{debug, info, warn};
use tonic::{Request, Response, Status, async_trait};

use pegaflow_proto::proto::engine::engine_server::{Engine, EngineServer};
use pegaflow_proto::proto::engine::{
    HealthRequest, HealthResponse, LoadRequest, LoadResponse, QueryBlocksForTransferRequest,
    QueryBlocksForTransferResponse, QueryRequest, QueryResponse, RdmaHandshakeRequest,
    RdmaHandshakeResponse, RegisterContextRequest, RegisterContextResponse, ReleaseRequest,
    ReleaseResponse, ReleaseTransferLockRequest, ReleaseTransferLockResponse, ResponseStatus,
    SaveRequest, SaveResponse, SessionEvent, SessionRequest, SetColdBlocksRequest,
    SetColdBlocksResponse, ShutdownRequest, ShutdownResponse, TransferBlockInfo, TransferSlotInfo,
    UnregisterRequest, UnregisterResponse,
};

use crate::{LayerBlock, PegaEngine};

/// Match pegaflow-server's cap: a `QueryBlocksForTransfer` response carries
/// per-slot descriptors for every requested block, which overflows tonic's
/// default 4 MiB limit on large batches.
const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// The minimal gRPC surface a `PegaEngine` embedder exposes so RDMA peers can
/// fetch blocks from it. See the module docs for scope.
pub struct P2pTransferService {
    engine: Arc<PegaEngine>,
}

impl P2pTransferService {
    pub fn new(engine: Arc<PegaEngine>) -> Self {
        Self { engine }
    }

    /// Serve on `addr` until `shutdown` resolves. Must run inside a tokio
    /// runtime. The address must be the engine's routable advertise address —
    /// peers discover it through the MetaServer and dial it for handshakes.
    pub async fn serve(
        engine: Arc<PegaEngine>,
        addr: std::net::SocketAddr,
        shutdown: impl std::future::Future<Output = ()> + Send,
    ) -> Result<(), tonic::transport::Error> {
        info!("P2P transfer service listening on {addr}");
        let service = EngineServer::new(Self::new(engine))
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_shutdown(addr, shutdown)
            .await
    }

    /// [`Self::serve`] over a caller-bound listener stream. Binding first lets
    /// the embedder fail loud on a taken port before reporting itself ready.
    pub async fn serve_with_incoming<I, IO, IE>(
        engine: Arc<PegaEngine>,
        incoming: I,
        shutdown: impl std::future::Future<Output = ()> + Send,
    ) -> Result<(), tonic::transport::Error>
    where
        I: futures::Stream<Item = Result<IO, IE>>,
        IO: tonic::transport::server::Connected
            + tokio::io::AsyncRead
            + tokio::io::AsyncWrite
            + Send
            + Unpin
            + 'static,
        IE: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        let service = EngineServer::new(Self::new(engine))
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, shutdown)
            .await
    }

    fn ok_status() -> ResponseStatus {
        ResponseStatus {
            ok: true,
            message: String::new(),
        }
    }

    fn build_transfer_slot_info(
        raw_block: &crate::RawBlock,
        numa_node: pegaflow_common::NumaNode,
    ) -> TransferSlotInfo {
        let layer_block = LayerBlock::new(raw_block);
        TransferSlotInfo {
            k_ptr: layer_block.k_ptr() as u64,
            k_size: layer_block.k_size() as u64,
            v_ptr: layer_block.v_ptr().map(|p| p as u64).unwrap_or(0),
            v_size: layer_block.v_size().unwrap_or(0) as u64,
            numa_node: numa_node.0,
        }
    }

    fn not_served<T>(rpc: &str) -> Result<T, Status> {
        Err(Status::unimplemented(format!(
            "{rpc} is not served by the embedded P2P transfer service; \
             the embedder drives the engine in-process"
        )))
    }
}

#[async_trait]
impl Engine for P2pTransferService {
    async fn query_blocks_for_transfer(
        &self,
        request: Request<QueryBlocksForTransferRequest>,
    ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
        let req = request.into_inner();

        if !self.engine.has_rdma_transport() {
            return Err(Status::failed_precondition(
                "RDMA transfer engine is not configured",
            ));
        }

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

        debug!(
            "P2P query_blocks_for_transfer: requester={} requested={} found={} session={}",
            req.requester_id,
            req.block_hashes.len(),
            blocks.len(),
            session_id,
        );

        Ok(Response::new(QueryBlocksForTransferResponse {
            status: Some(Self::ok_status()),
            blocks,
            transfer_session_id: session_id,
            lock_timeout_secs: self.engine.transfer_lock_timeout().as_secs() as u32,
        }))
    }

    async fn release_transfer_lock(
        &self,
        request: Request<ReleaseTransferLockRequest>,
    ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
        let req = request.into_inner();
        let released = self.engine.release_transfer_lock(&req.transfer_session_id);
        debug!(
            "P2P release_transfer_lock: session={} released={released}",
            req.transfer_session_id
        );
        Ok(Response::new(ReleaseTransferLockResponse {
            status: Some(Self::ok_status()),
            released_blocks: released as u64,
        }))
    }

    async fn rdma_handshake(
        &self,
        request: Request<RdmaHandshakeRequest>,
    ) -> Result<Response<RdmaHandshakeResponse>, Status> {
        let req = request.into_inner();

        if !self.engine.has_rdma_transport() {
            return Err(Status::failed_precondition(
                "RDMA transfer engine is not configured",
            ));
        }

        let server_meta = self
            .engine
            .rdma_accept_handshake(&req.requester_id, &req.handshake_metadata)
            .map_err(|e| {
                warn!(
                    "P2P rdma_handshake failed: requester={} {e}",
                    req.requester_id
                );
                Status::internal(format!("RDMA handshake failed: {e}"))
            })?;

        info!(
            "P2P rdma_handshake accepted: requester={}",
            req.requester_id
        );
        Ok(Response::new(RdmaHandshakeResponse {
            status: Some(Self::ok_status()),
            handshake_metadata: server_meta,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: Some(Self::ok_status()),
        }))
    }

    // ── In-process embedders drive these through `PegaEngine` directly ──

    async fn register_context_batch(
        &self,
        _request: Request<RegisterContextRequest>,
    ) -> Result<Response<RegisterContextResponse>, Status> {
        Self::not_served("register_context_batch")
    }

    async fn save(&self, _request: Request<SaveRequest>) -> Result<Response<SaveResponse>, Status> {
        Self::not_served("save")
    }

    async fn load(&self, _request: Request<LoadRequest>) -> Result<Response<LoadResponse>, Status> {
        Self::not_served("load")
    }

    async fn query_prefetch(
        &self,
        _request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        Self::not_served("query_prefetch")
    }

    async fn release(
        &self,
        _request: Request<ReleaseRequest>,
    ) -> Result<Response<ReleaseResponse>, Status> {
        Self::not_served("release")
    }

    async fn unregister_context(
        &self,
        _request: Request<UnregisterRequest>,
    ) -> Result<Response<UnregisterResponse>, Status> {
        Self::not_served("unregister_context")
    }

    async fn set_cold_blocks(
        &self,
        _request: Request<SetColdBlocksRequest>,
    ) -> Result<Response<SetColdBlocksResponse>, Status> {
        Self::not_served("set_cold_blocks")
    }

    async fn shutdown(
        &self,
        _request: Request<ShutdownRequest>,
    ) -> Result<Response<ShutdownResponse>, Status> {
        Self::not_served("shutdown")
    }

    type SessionStream = futures::stream::Empty<Result<SessionEvent, Status>>;

    async fn session(
        &self,
        _request: Request<SessionRequest>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        Self::not_served("session")
    }
}
