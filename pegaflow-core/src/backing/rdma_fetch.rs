// RDMA remote block fetch: MetaServer query → gRPC QueryBlocksForTransfer → RDMA READ.
//
// Follows the same submit/oneshot pattern as SsdBackingStore::submit_prefix so that
// PrefetchScheduler can treat remote fetch the same way it treats SSD prefetch.

use std::ptr::NonNull;
use std::sync::Arc;

use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    QueryBlocksForTransferRequest, ReleaseTransferLockRequest, TransferBlockInfo,
};
use pegaflow_transfer::{HandshakeMetadata, TransferDesc, TransferOp};
use tokio::sync::oneshot;
use tonic::transport::{Channel, Endpoint};

use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::MetaServerClient;

/// RDMA remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries MetaServer for their
/// location, picks the best remote node, and uses gRPC + RDMA READ to fetch them.
pub(crate) struct RdmaFetchStore {
    metaserver_client: Arc<MetaServerClient>,
    rdma_transport: Arc<RdmaTransport>,
    allocate_fn: AllocateFn,
    advertise_addr: String,
}

impl RdmaFetchStore {
    pub(crate) fn new(
        metaserver_client: Arc<MetaServerClient>,
        rdma_transport: Arc<RdmaTransport>,
        allocate_fn: AllocateFn,
        advertise_addr: String,
    ) -> Self {
        info!("RDMA remote fetch enabled (advertise={})", advertise_addr);
        Self {
            metaserver_client,
            rdma_transport,
            allocate_fn,
            advertise_addr,
        }
    }

    /// Submit a remote fetch for the given block keys.
    ///
    /// Returns `(found_count, done_rx)` where `done_rx` delivers fetched blocks.
    /// Returns `(0, done_rx)` immediately if MetaServer has no hits.
    pub(crate) async fn submit_remote_fetch(
        &self,
        namespace: &str,
        keys: Vec<BlockKey>,
    ) -> (usize, oneshot::Receiver<PrefetchResult>) {
        let (done_tx, done_rx) = oneshot::channel();

        if keys.is_empty() {
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        }

        let hashes: Vec<Vec<u8>> = keys.iter().map(|k| k.hash.clone()).collect();

        // Query MetaServer for block locations
        let node_blocks = match self.metaserver_client.query(namespace, &hashes).await {
            Ok(nb) => nb,
            Err(e) => {
                warn!("MetaServer query failed for remote fetch: {e}");
                let _ = done_tx.send(Vec::new());
                return (0, done_rx);
            }
        };

        if node_blocks.is_empty() {
            debug!("MetaServer returned no remote hits for namespace={namespace}");
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        }

        // Pick the node with the most blocks
        let best = node_blocks
            .iter()
            .max_by_key(|nb| nb.block_hashes.len())
            .unwrap();

        let remote_addr = best.node.clone();
        let remote_hashes: Vec<Vec<u8>> = best.block_hashes.clone();
        let found = remote_hashes.len();

        debug!(
            "Remote fetch: namespace={namespace} best_node={remote_addr} blocks={found}/{}",
            hashes.len()
        );

        // Spawn the async RDMA fetch task
        let rdma = Arc::clone(&self.rdma_transport);
        let alloc_fn = Arc::clone(&self.allocate_fn);
        let advertise = self.advertise_addr.clone();
        let ns = namespace.to_string();

        tokio::spawn(async move {
            let result = rdma_fetch_task(
                &rdma,
                &alloc_fn,
                &remote_addr,
                &advertise,
                &ns,
                &remote_hashes,
            )
            .await;
            let _ = done_tx.send(result);
        });

        (found, done_rx)
    }
}

/// Execute RDMA fetch against a single remote node.
///
/// 1. gRPC connect to remote node
/// 2. Prepare RDMA handshake (local QPs)
/// 3. QueryBlocksForTransfer RPC (exchanges handshake metadata)
/// 4. RDMA READ all block segments
/// 5. Build SealedBlocks from fetched memory
/// 6. ReleaseTransferLock (fire-and-forget)
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    remote_addr: &str,
    advertise_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> PrefetchResult {
    // 1. Connect to remote node
    let mut client = match connect_remote(remote_addr).await {
        Ok(c) => c,
        Err(e) => {
            error!("RDMA fetch: failed to connect to {remote_addr}: {e}");
            return Vec::new();
        }
    };

    // 2. Prepare local RDMA handshake
    let local_meta = match rdma.engine().prepare_handshake() {
        Ok(m) => m,
        Err(e) => {
            error!("RDMA fetch: prepare_handshake failed: {e}");
            return Vec::new();
        }
    };

    // 3. QueryBlocksForTransfer RPC
    let request = QueryBlocksForTransferRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.to_vec(),
        requester_id: advertise_addr.to_string(),
        rdma_handshake: local_meta.to_bytes(),
    };

    let response = match client.query_blocks_for_transfer(request).await {
        Ok(r) => r.into_inner(),
        Err(e) => {
            error!("RDMA fetch: QueryBlocksForTransfer RPC failed: {e}");
            return Vec::new();
        }
    };

    if let Some(st) = &response.status
        && !st.ok
    {
        error!("RDMA fetch: remote returned error: {}", st.message);
        return Vec::new();
    }

    let transfer_session_id = response.transfer_session_id.clone();

    // Parse remote RDMA handshake
    let remote_meta = if response.rdma_session_id.is_empty() {
        error!("RDMA fetch: remote did not return RDMA handshake metadata");
        fire_and_forget_release(&mut client, &transfer_session_id).await;
        return Vec::new();
    } else {
        match HandshakeMetadata::from_bytes(&response.rdma_session_id) {
            Ok(m) => m,
            Err(e) => {
                error!("RDMA fetch: invalid remote handshake metadata: {e}");
                fire_and_forget_release(&mut client, &transfer_session_id).await;
                return Vec::new();
            }
        }
    };

    // Connect local QPs to remote peer (initiator side: lazy connect on first transfer)
    // The TransferEngine handles this internally on batch_transfer_async.

    // 4. RDMA READ all blocks
    let blocks = response.blocks;
    let result = match fetch_blocks_via_rdma(rdma, allocate_fn, namespace, &remote_meta, &blocks) {
        Ok(r) => r,
        Err(e) => {
            error!("RDMA fetch: block transfer failed: {e}");
            fire_and_forget_release(&mut client, &transfer_session_id).await;
            return Vec::new();
        }
    };

    // 6. Release transfer lock
    fire_and_forget_release(&mut client, &transfer_session_id).await;

    debug!(
        "RDMA fetch completed: {} blocks from {}",
        result.len(),
        remote_addr
    );
    result
}

/// Allocate local memory, build TransferDescs, execute RDMA READ, build SealedBlocks.
fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_meta: &HandshakeMetadata,
    blocks: &[TransferBlockInfo],
) -> Result<PrefetchResult, String> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    // For each block, allocate local memory and build transfer descriptors.
    // Each TransferBlockInfo has slots (one per TP rank), each slot has K (and optionally V).
    let mut all_descs: Vec<TransferDesc> = Vec::new();
    // (block_hash, Vec<(slot_segments)>) — for building SealedBlock afterwards
    let mut block_allocs: Vec<(Vec<u8>, Vec<Vec<SegmentAlloc>>)> = Vec::new();

    for block_info in blocks {
        let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

        for slot in &block_info.slots {
            let mut segments = Vec::new();

            // K segment
            if slot.k_size > 0 {
                let alloc = allocate_fn(slot.k_size, None).ok_or_else(|| {
                    format!(
                        "RDMA fetch: failed to allocate {} bytes for K segment",
                        slot.k_size
                    )
                })?;
                let local_ptr = alloc.as_non_null();
                let remote_ptr = NonNull::new(slot.k_ptr as *mut u8)
                    .ok_or_else(|| "RDMA fetch: remote K ptr is null".to_string())?;
                all_descs.push(TransferDesc {
                    local_ptr,
                    remote_ptr,
                    len: slot.k_size as usize,
                });
                segments.push(SegmentAlloc {
                    alloc,
                    size: slot.k_size as usize,
                });
            }

            // V segment (split KV)
            if slot.v_size > 0 && slot.v_ptr != 0 {
                let alloc = allocate_fn(slot.v_size, None).ok_or_else(|| {
                    format!(
                        "RDMA fetch: failed to allocate {} bytes for V segment",
                        slot.v_size
                    )
                })?;
                let local_ptr = alloc.as_non_null();
                let remote_ptr = NonNull::new(slot.v_ptr as *mut u8)
                    .ok_or_else(|| "RDMA fetch: remote V ptr is null".to_string())?;
                all_descs.push(TransferDesc {
                    local_ptr,
                    remote_ptr,
                    len: slot.v_size as usize,
                });
                segments.push(SegmentAlloc {
                    alloc,
                    size: slot.v_size as usize,
                });
            }

            slot_allocs.push(segments);
        }

        block_allocs.push((block_info.block_hash.clone(), slot_allocs));
    }

    if all_descs.is_empty() {
        return Ok(Vec::new());
    }

    // Execute RDMA READ
    let rx = rdma
        .engine()
        .batch_transfer_async(TransferOp::Read, remote_meta, &all_descs)
        .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?;

    // Wait for completion (blocking on std::sync::mpsc::Receiver)
    let _bytes = rx
        .recv()
        .map_err(|_| "RDMA transfer channel closed unexpectedly".to_string())?
        .map_err(|e| format!("RDMA transfer failed: {e}"))?;

    // Build SealedBlocks from allocated memory
    let mut result: PrefetchResult = Vec::with_capacity(block_allocs.len());
    for (hash, slot_allocs) in block_allocs {
        let key = BlockKey::new(namespace.to_string(), hash);
        let slots: Vec<Arc<RawBlock>> = slot_allocs
            .into_iter()
            .map(|segs| {
                let segments: Vec<Segment> = segs
                    .into_iter()
                    .map(|sa| Segment::new(sa.alloc.as_non_null(), sa.size, sa.alloc))
                    .collect();
                Arc::new(RawBlock::new(segments))
            })
            .collect();
        let sealed = Arc::new(SealedBlock::from_slots(slots));
        result.push((key, sealed));
    }

    Ok(result)
}

struct SegmentAlloc {
    alloc: Arc<crate::pinned_pool::PinnedAllocation>,
    size: usize,
}

async fn connect_remote(addr: &str) -> Result<EngineClient<Channel>, String> {
    let endpoint = Endpoint::from_shared(addr.to_string())
        .map_err(|e| format!("invalid remote address: {e}"))?
        .connect_timeout(std::time::Duration::from_secs(5));

    let channel = endpoint
        .connect()
        .await
        .map_err(|e| format!("gRPC connect failed: {e}"))?;

    Ok(EngineClient::new(channel))
}

async fn fire_and_forget_release(client: &mut EngineClient<Channel>, transfer_session_id: &str) {
    if transfer_session_id.is_empty() {
        return;
    }
    let req = ReleaseTransferLockRequest {
        transfer_session_id: transfer_session_id.to_string(),
    };
    if let Err(e) = client.release_transfer_lock(req).await {
        warn!("RDMA fetch: ReleaseTransferLock failed: {e}");
    }
}
