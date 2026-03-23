// RDMA remote block fetch: MetaServer query → gRPC QueryBlocksForTransfer → RDMA READ.
//
// Follows the same submit/oneshot pattern as SsdBackingStore::submit_prefix so that
// PrefetchScheduler can treat remote fetch the same way it treats SSD prefetch.

use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use log::{debug, error, info, warn};
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, ReleaseTransferLockRequest,
    TransferBlockInfo,
};
use pegaflow_transfer::{ConnectionStatus, HandshakeMetadata, TransferDesc, TransferOp};
use tokio::sync::oneshot;
use tonic::transport::{Channel, Endpoint};

use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::MetaServerClient;

/// Timeout for a single RDMA batch transfer. If the NIC hangs or a remote peer
/// disappears mid-transfer, we give up after this duration instead of blocking
/// a `spawn_blocking` thread forever.
const RDMA_TRANSFER_TIMEOUT: Duration = Duration::from_secs(30);

/// RDMA remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries MetaServer for their
/// location, picks the best remote node, and uses gRPC + RDMA READ to fetch them.
pub(crate) struct RdmaFetchStore {
    metaserver_client: Arc<MetaServerClient>,
    rdma_transport: Arc<RdmaTransport>,
    allocate_fn: AllocateFn,
    advertise_addr: String,
    /// Lazy gRPC channel cache keyed by remote address. Tonic channels multiplex
    /// requests over a single HTTP/2 connection; cloning is cheap.
    grpc_channels: Arc<DashMap<String, EngineClient<Channel>>>,
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
            grpc_channels: Arc::new(DashMap::new()),
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
        let channels = Arc::clone(&self.grpc_channels);
        let advertise = self.advertise_addr.clone();
        let ns = namespace.to_string();

        tokio::spawn(async move {
            let result = rdma_fetch_task(
                &rdma,
                &alloc_fn,
                &channels,
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
/// 1. Check/prepare RDMA connection (local, fail fast)
/// 2. gRPC channel + QueryBlocksForTransfer RPC
/// 3. Complete RDMA handshake if new connection
/// 4. RDMA READ all block segments + build SealedBlocks
/// 5. ReleaseTransferLock (fire-and-forget, non-blocking)
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    advertise_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> PrefetchResult {
    // 1. Check/prepare RDMA connection (local operation — fail fast before gRPC)
    let local_meta = match rdma.engine().get_or_prepare(remote_addr) {
        Ok(ConnectionStatus::Existing) => None,
        Ok(ConnectionStatus::Prepared(m)) => Some(m),
        Err(e) => {
            error!("RDMA prepare connection to {remote_addr} failed: {e}");
            return Vec::new();
        }
    };

    // 2. gRPC channel + QueryBlocksForTransfer RPC
    let (client, response) = match query_remote_blocks(
        grpc_channels,
        remote_addr,
        namespace,
        block_hashes,
        advertise_addr,
        &local_meta,
    )
    .await
    {
        Ok(cr) => cr,
        Err(e) => {
            error!("Remote query to {remote_addr} failed: {e}");
            if let Some(m) = &local_meta {
                rdma.engine().abort_handshake(m);
            }
            return Vec::new();
        }
    };

    let transfer_session_id = response.transfer_session_id.clone();

    // 3. Complete RDMA handshake if we prepared a new connection
    if let Some(local_meta) = &local_meta {
        if let Err(e) = finish_handshake(rdma, remote_addr, local_meta, &response.rdma_session_id) {
            error!("RDMA handshake to {remote_addr} failed: {e}");
            rdma.engine().abort_handshake(local_meta);
            spawn_release_lock(client.clone(), transfer_session_id.clone());
            return Vec::new();
        }
    }

    // 4. RDMA READ all blocks + build SealedBlocks
    let blocks = response.blocks;
    let result =
        match fetch_blocks_via_rdma(rdma, allocate_fn, namespace, remote_addr, &blocks).await {
            Ok(r) => r,
            Err(e) => {
                error!("RDMA transfer from {remote_addr} failed: {e}");
                rdma.engine().invalidate_connection(remote_addr);
                spawn_release_lock(client, transfer_session_id);
                return Vec::new();
            }
        };

    // 5. Release transfer lock (fire-and-forget: spawns a detached task)
    spawn_release_lock(client, transfer_session_id);

    debug!(
        "RDMA transfer from {remote_addr} completed: {} blocks",
        result.len(),
    );
    result
}

/// Allocate local memory, build TransferDescs, execute RDMA READ, build SealedBlocks.
async fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_addr: &str,
    blocks: &[TransferBlockInfo],
) -> Result<PrefetchResult, String> {
    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    // (block_hash, Vec<(slot_segments)>) — for building SealedBlock afterwards
    let mut block_allocs: Vec<(Vec<u8>, Vec<Vec<SegmentAlloc>>)> = Vec::new();

    // Build TransferDescs and submit RDMA READ inside a sync block so that
    // all_descs (which contains NonNull<u8>, !Send) is dropped before any .await.
    let rx = {
        let mut all_descs: Vec<TransferDesc> = Vec::new();

        for block_info in blocks {
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

            for slot in &block_info.slots {
                let mut segments = Vec::new();

                // K segment
                if slot.k_size > 0 {
                    let alloc = allocate_fn(slot.k_size, None).ok_or_else(|| {
                        format!("failed to allocate {} bytes for K segment", slot.k_size)
                    })?;
                    let local_ptr = alloc.as_non_null();
                    let remote_ptr = NonNull::new(slot.k_ptr as *mut u8)
                        .ok_or_else(|| "remote K ptr is null".to_string())?;
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
                        format!("failed to allocate {} bytes for V segment", slot.v_size)
                    })?;
                    let local_ptr = alloc.as_non_null();
                    let remote_ptr = NonNull::new(slot.v_ptr as *mut u8)
                        .ok_or_else(|| "remote V ptr is null".to_string())?;
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

        // Submit RDMA READ; all_descs is dropped at the end of this block.
        rdma.engine()
            .batch_transfer_async(TransferOp::Read, remote_addr, &all_descs)
            .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?
    };

    // Offload blocking recv() to a dedicated thread to avoid blocking the async runtime.
    let _bytes = tokio::task::spawn_blocking(move || rx.recv_timeout(RDMA_TRANSFER_TIMEOUT))
        .await
        .map_err(|e| format!("RDMA transfer task panicked: {e}"))?
        .map_err(|e| format!("RDMA transfer timed out or channel closed: {e}"))?
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

fn get_or_create_channel(
    cache: &DashMap<String, EngineClient<Channel>>,
    addr: &str,
) -> Result<EngineClient<Channel>, String> {
    if let Some(client) = cache.get(addr) {
        return Ok(client.clone());
    }
    let url = if addr.starts_with("http://") || addr.starts_with("https://") {
        addr.to_string()
    } else {
        format!("http://{addr}")
    };
    let channel = Endpoint::from_shared(url)
        .map_err(|e| format!("invalid remote address: {e}"))?
        .connect_timeout(Duration::from_secs(5))
        .connect_lazy();
    let client = EngineClient::new(channel);
    cache.insert(addr.to_string(), client.clone());
    Ok(client)
}

/// Get/create gRPC channel and call QueryBlocksForTransfer.
async fn query_remote_blocks(
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
    advertise_addr: &str,
    local_meta: &Option<HandshakeMetadata>,
) -> Result<(EngineClient<Channel>, QueryBlocksForTransferResponse), String> {
    let mut client = get_or_create_channel(grpc_channels, remote_addr)?;

    let request = QueryBlocksForTransferRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.to_vec(),
        requester_id: advertise_addr.to_string(),
        rdma_handshake: local_meta
            .as_ref()
            .map(|m| m.to_bytes())
            .unwrap_or_default(),
    };

    let response = client
        .query_blocks_for_transfer(request)
        .await
        .map_err(|e| format!("QueryBlocksForTransfer RPC failed: {e}"))?
        .into_inner();

    if let Some(st) = &response.status
        && !st.ok
    {
        return Err(format!("remote returned error: {}", st.message));
    }

    Ok((client, response))
}

/// Decode remote handshake metadata and complete the RDMA connection.
fn finish_handshake(
    rdma: &RdmaTransport,
    remote_addr: &str,
    local_meta: &HandshakeMetadata,
    remote_bytes: &[u8],
) -> Result<(), String> {
    if remote_bytes.is_empty() {
        return Err("server returned empty rdma_session_id".into());
    }
    let remote_meta = HandshakeMetadata::from_bytes(remote_bytes)
        .map_err(|e| format!("invalid metadata: {e}"))?;
    rdma.engine()
        .complete_handshake(remote_addr, local_meta, &remote_meta)
        .map_err(|e| format!("{e}"))
}

/// Release a transfer lock in a detached task. Does not block the caller.
fn spawn_release_lock(mut client: EngineClient<Channel>, transfer_session_id: String) {
    if transfer_session_id.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let req = ReleaseTransferLockRequest {
            transfer_session_id: transfer_session_id.clone(),
        };
        if let Err(e) = client.release_transfer_lock(req).await {
            warn!("ReleaseTransferLock failed for session {transfer_session_id}: {e}");
        }
    });
}
