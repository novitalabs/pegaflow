use log::debug;
use pegaflow_core::{BlockKey, LeaseError, PegaEngine, SealedBlock};
use pegaflow_transfer::MooncakeTransferEngine;
use std::sync::Arc;
use tonic::{Request, Response, Status, async_trait};
use uuid::Uuid;

use crate::proto::engine::{
    AcquireLeaseRequest, AcquireLeaseResponse, ReleaseLeaseRequest, ReleaseLeaseResponse,
    RemoteBlockDescriptor, RenewLeaseRequest, RenewLeaseResponse, SlotMemory,
    rdma_transfer_server::RdmaTransfer,
};

pub struct GrpcRdmaTransferService {
    engine: Arc<PegaEngine>,
    /// Owner's RDMA transfer engine. When present, `owner_domain_address` is
    /// populated in `AcquireLeaseResponse` so the requester can target the
    /// correct RDMA session for READ operations.
    transfer_engine: Option<Arc<MooncakeTransferEngine>>,
}

impl GrpcRdmaTransferService {
    pub fn new(engine: Arc<PegaEngine>) -> Self {
        Self {
            engine,
            transfer_engine: None,
        }
    }

    /// Create service with an associated RDMA engine.
    ///
    /// The engine's session ID is returned as `owner_domain_address` in every
    /// `AcquireLeaseResponse`, allowing requester nodes to issue RDMA READs
    /// directly into the owner's registered memory.
    pub fn new_with_rdma(
        engine: Arc<PegaEngine>,
        transfer_engine: Arc<MooncakeTransferEngine>,
    ) -> Self {
        Self {
            engine,
            transfer_engine: Some(transfer_engine),
        }
    }

    fn map_lease_error(err: LeaseError) -> Status {
        match &err {
            LeaseError::ResourceExhausted { .. } => Status::resource_exhausted(err.to_string()),
            LeaseError::NotFound(_) => Status::not_found(err.to_string()),
            LeaseError::PermissionDenied { .. } => Status::permission_denied(err.to_string()),
        }
    }
}

#[async_trait]
impl RdmaTransfer for GrpcRdmaTransferService {
    async fn acquire_lease(
        &self,
        request: Request<AcquireLeaseRequest>,
    ) -> Result<Response<AcquireLeaseResponse>, Status> {
        let req = request.into_inner();

        if req.namespace.is_empty() {
            return Err(Status::invalid_argument("namespace must not be empty"));
        }
        if req.block_hashes.is_empty() {
            return Err(Status::invalid_argument("block_hashes must not be empty"));
        }

        debug!(
            "RPC [acquire_lease]: requester={} ns={} blocks={}",
            req.requester_node_id,
            req.namespace,
            req.block_hashes.len(),
        );

        // Look up blocks in the local cache
        let (found_blocks, missing_hashes) = self
            .engine
            .get_blocks_for_lease(&req.namespace, &req.block_hashes);

        // Acquire lease (holds Arc<SealedBlock> to prevent eviction)
        let grant = self
            .engine
            .acquire_lease(
                &req.requester_node_id,
                &req.namespace,
                found_blocks,
                missing_hashes,
                req.lease_duration_secs,
            )
            .map_err(Self::map_lease_error)?;

        // Build block descriptors: for each leased block, expose the virtual
        // addresses of each slot so the requestor can issue RDMA READs.
        let blocks: Vec<RemoteBlockDescriptor> = grant
            .blocks
            .iter()
            .map(|(key, sealed)| block_to_descriptor(key, sealed))
            .collect();

        let missing_hashes: Vec<Vec<u8>> = grant.missing_hashes;

        debug!(
            "RPC [acquire_lease] done: lease_id={} found={} missing={}",
            grant.lease_id,
            blocks.len(),
            missing_hashes.len(),
        );

        let owner_domain_address = self
            .transfer_engine
            .as_ref()
            .map(|e| e.get_session_id().to_bytes().to_vec())
            .unwrap_or_default();

        Ok(Response::new(AcquireLeaseResponse {
            lease_id: grant.lease_id.to_string(),
            blocks,
            missing_hashes,
            expires_at_unix_ms: grant.expires_at_unix_ms,
            owner_domain_address,
        }))
    }

    async fn renew_lease(
        &self,
        request: Request<RenewLeaseRequest>,
    ) -> Result<Response<RenewLeaseResponse>, Status> {
        let req = request.into_inner();

        let lease_id = parse_lease_id(&req.lease_id)?;

        let new_expires_at = self
            .engine
            .renew_lease(lease_id, &req.requester_node_id, req.extend_duration_secs)
            .map_err(Self::map_lease_error)?;

        debug!(
            "RPC [renew_lease]: lease_id={} new_expires_at={new_expires_at}",
            req.lease_id
        );

        Ok(Response::new(RenewLeaseResponse {
            new_expires_at_unix_ms: new_expires_at,
        }))
    }

    async fn release_lease(
        &self,
        request: Request<ReleaseLeaseRequest>,
    ) -> Result<Response<ReleaseLeaseResponse>, Status> {
        let req = request.into_inner();

        let lease_id = parse_lease_id(&req.lease_id)?;

        self.engine
            .release_lease(lease_id, &req.requester_node_id)
            .map_err(Self::map_lease_error)?;

        debug!("RPC [release_lease]: lease_id={}", req.lease_id);

        Ok(Response::new(ReleaseLeaseResponse {}))
    }
}

fn parse_lease_id(s: &str) -> Result<Uuid, Status> {
    Uuid::parse_str(s).map_err(|_| Status::invalid_argument(format!("invalid lease_id: {s}")))
}

/// Build a `RemoteBlockDescriptor` from a sealed block's slot addresses.
///
/// Each slot maps to one TP rank. The addresses are virtual addresses within
/// the registered RDMA memory region (the pinned pool). The rkey is
/// exchanged out-of-band during the pegaflow-transfer UD handshake.
fn block_to_descriptor(key: &BlockKey, sealed: &SealedBlock) -> RemoteBlockDescriptor {
    let slots: Vec<SlotMemory> = sealed
        .rdma_slot_addrs()
        .into_iter()
        .map(|(k_addr, k_size, v_addr, v_size)| SlotMemory {
            k_addr,
            k_size,
            v_addr,
            v_size,
        })
        .collect();

    RemoteBlockDescriptor {
        block_hash: key.hash.clone(),
        slots,
    }
}
