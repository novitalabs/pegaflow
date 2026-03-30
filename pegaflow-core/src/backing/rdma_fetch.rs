// RDMA remote block fetch: MetaServer query → gRPC QueryBlocksForTransfer → RDMA READ.
//
// Follows the same submit/oneshot pattern as SsdBackingStore::submit_prefix so that
// PrefetchScheduler can treat remote fetch the same way it treats SSD prefetch.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use log::{debug, info, warn};
use mea::oneshot;
use mea::singleflight::Group;
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    QueryBlocksForTransferRequest, QueryBlocksForTransferResponse, RdmaHandshakeRequest,
    ReleaseTransferLockRequest, TransferBlockInfo,
};
use pegaflow_transfer::{ConnectionStatus, HandshakeMetadata, TransferDesc, TransferOp};
use tonic::transport::{Channel, Endpoint};

use pegaflow_common::NumaNode;

use opentelemetry::KeyValue;

use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

/// Minimum usable transfer timeout. If the server's lock timeout minus the
/// safety margin falls below this, we use this floor to avoid instant timeouts.
const MIN_TRANSFER_TIMEOUT: Duration = Duration::from_secs(10);

/// Safety margin subtracted from the server's lock timeout. The client must
/// finish the RDMA transfer before the server releases the lock.
const LOCK_TIMEOUT_MARGIN: Duration = Duration::from_secs(60);

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
    /// Singleflight group to deduplicate concurrent RDMA handshakes to the
    /// same remote address. Without this, N concurrent fetches to the same
    /// peer would each create QPs and race on the server, causing all but
    /// the last handshake's QPs to be invalidated.
    connect_group: Arc<Group<String, ()>>,
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
            connect_group: Arc::new(Group::new()),
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

        // Pick the node with the most blocks, excluding self.
        let best = node_blocks
            .iter()
            .filter(|nb| nb.node != self.advertise_addr)
            .max_by_key(|nb| nb.block_hashes.len());

        let Some(best) = best else {
            debug!(
                "MetaServer returned only self-node hits for namespace={namespace}, skipping RDMA fetch"
            );
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        };

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
        let connect_group = Arc::clone(&self.connect_group);
        let advertise = self.advertise_addr.clone();
        let ns = namespace.to_string();

        tokio::spawn(async move {
            let result = rdma_fetch_task(
                &rdma,
                &alloc_fn,
                &channels,
                &connect_group,
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
/// 1. Ensure RDMA connection (singleflight per remote_addr)
/// 2. gRPC QueryBlocksForTransfer RPC (connection reuse, no handshake)
/// 3. RDMA READ all block segments + build SealedBlocks
/// 4. ReleaseTransferLock (fire-and-forget, non-blocking)
#[allow(clippy::too_many_arguments)]
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    connect_group: &Group<String, ()>,
    remote_addr: &str,
    advertise_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> PrefetchResult {
    let t0 = Instant::now();

    // 1. Ensure RDMA connection (singleflight: at most one handshake per remote_addr)
    let connect_start = Instant::now();
    if let Err(e) = ensure_connected(
        connect_group,
        rdma,
        grpc_channels,
        remote_addr,
        advertise_addr,
    )
    .await
    {
        warn!("RDMA connect to {remote_addr} failed: {e}");
        core_metrics()
            .rdma_fetch_total
            .add(1, &[KeyValue::new("status", "error")]);
        return Vec::new();
    }
    let connect_elapsed = connect_start.elapsed();

    // 2. gRPC QueryBlocksForTransfer (connection already established)
    let query_start = Instant::now();
    let (client, response) = match query_remote_blocks(
        grpc_channels,
        remote_addr,
        namespace,
        block_hashes,
        advertise_addr,
    )
    .await
    {
        Ok(cr) => cr,
        Err(e) => {
            warn!("Remote query to {remote_addr} failed: {e}");
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };
    let query_elapsed = query_start.elapsed();

    let transfer_session_id = response.transfer_session_id.clone();

    // 3. RDMA READ all blocks + build SealedBlocks
    let transfer_timeout = transfer_timeout_from_server(response.lock_timeout_secs);
    let blocks = response.blocks;
    let total_bytes: u64 = blocks
        .iter()
        .flat_map(|b| &b.slots)
        .map(|s| s.k_size + s.v_size)
        .sum();
    let slot_count: usize = blocks.iter().map(|b| b.slots.len()).sum();
    let mut k_segments = 0usize;
    let mut v_segments = 0usize;
    let mut slot_numa_counts: HashMap<NumaNode, usize> = HashMap::new();
    for block in &blocks {
        for slot in &block.slots {
            if slot.k_size > 0 {
                k_segments += 1;
            }
            if slot.v_size > 0 && slot.v_ptr != 0 {
                v_segments += 1;
            }
            *slot_numa_counts
                .entry(NumaNode(slot.numa_node))
                .or_default() += 1;
        }
    }
    let mut slot_numa_items: Vec<_> = slot_numa_counts.into_iter().collect();
    slot_numa_items.sort_unstable_by_key(|(node, _)| *node);
    let slot_numa_summary = slot_numa_items
        .into_iter()
        .map(|(node, count)| format!("{node}:{count}"))
        .collect::<Vec<_>>()
        .join(", ");
    let total_segments = k_segments + v_segments;
    let fetch = match fetch_blocks_via_rdma(
        rdma,
        allocate_fn,
        namespace,
        remote_addr,
        &blocks,
        transfer_timeout,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("RDMA transfer from {remote_addr} failed: {e}");
            rdma.engine().invalidate_connection(remote_addr);
            spawn_release_lock(client, transfer_session_id);
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };
    let result = fetch.result;

    // 4. Release transfer lock (fire-and-forget: spawns a detached task)
    spawn_release_lock(client, transfer_session_id);

    let elapsed = t0.elapsed();
    let accounted = connect_elapsed
        + query_elapsed
        + fetch.prepare_elapsed
        + fetch.rdma_wait_elapsed
        + fetch.rebuild_elapsed;
    let other_elapsed = elapsed.saturating_sub(accounted);
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    debug!(
        "RDMA fetch from {remote_addr}: {}/{} blocks, {mb:.1} MiB in {elapsed:.1?} ({:.0} MiB/s) \
         [connect={connect_elapsed:.1?} query={query_elapsed:.1?} prepare={:.1?} rdma_wait={:.1?} rebuild={:.1?} other={other_elapsed:.1?} \
         descs={} slots={} segs={} (k={},v={}) avg_desc={} avg_slot_segs={:.1} slot_numa={}]",
        result.len(),
        block_hashes.len(),
        if elapsed.as_secs_f64() > 0.0 {
            mb / elapsed.as_secs_f64()
        } else {
            0.0
        },
        fetch.prepare_elapsed,
        fetch.rdma_wait_elapsed,
        fetch.rebuild_elapsed,
        fetch.desc_count,
        slot_count,
        total_segments,
        k_segments,
        v_segments,
        if fetch.desc_count > 0 {
            format!(
                "{:.1} KiB",
                total_bytes as f64 / fetch.desc_count as f64 / 1024.0
            )
        } else {
            "0.0 KiB".to_string()
        },
        if slot_count > 0 {
            total_segments as f64 / slot_count as f64
        } else {
            0.0
        },
        slot_numa_summary,
    );
    let m = core_metrics();
    let ok = &[KeyValue::new("status", "ok")];
    m.rdma_fetch_total.add(1, ok);
    m.rdma_fetch_duration_seconds
        .record(elapsed.as_secs_f64(), ok);
    m.rdma_fetch_bytes.add(total_bytes, ok);
    result
}

/// Ensure an RDMA connection to `remote_addr` exists, using singleflight to
/// deduplicate concurrent handshakes to the same peer.
///
/// On the first call for a given remote, one task performs the full handshake
/// (prepare QPs → gRPC metadata exchange → complete connection). Concurrent
/// callers wait for that handshake to finish and then reuse the connection.
/// If the handshake fails, the error is returned to the leader and waiting
/// callers retry independently (mea try_work semantics).
async fn ensure_connected(
    connect_group: &Group<String, ()>,
    rdma: &RdmaTransport,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    advertise_addr: &str,
) -> Result<(), String> {
    connect_group
        .try_work(remote_addr.to_string(), async || {
            // Fast path: already connected
            let local_meta = match rdma.engine().get_or_prepare(remote_addr) {
                Ok(ConnectionStatus::Existing) => return Ok(()),
                Ok(ConnectionStatus::Connecting) => {
                    return Err("handshake to this peer already in progress".into());
                }
                Ok(ConnectionStatus::Prepared(m)) => m,
                Err(e) => return Err(format!("RDMA prepare: {e}")),
            };

            // Exchange handshake metadata via the dedicated RdmaHandshake RPC.
            let mut client = get_or_create_channel(grpc_channels, remote_addr)
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?;

            let request = RdmaHandshakeRequest {
                requester_id: advertise_addr.to_string(),
                handshake_metadata: local_meta.to_bytes(),
            };
            let response = client
                .rdma_handshake(request)
                .await
                .map_err(|e| format!("RdmaHandshake RPC failed: {e}"))
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?
                .into_inner();

            // Complete the RDMA connection with the server's QP info.
            finish_handshake(rdma, remote_addr, &local_meta, &response.handshake_metadata)
                .inspect_err(|_| rdma.engine().abort_handshake(remote_addr, &local_meta))?;

            Ok(())
        })
        .await
}

/// Allocate local memory, build TransferDescs, execute RDMA READ, build SealedBlocks.
async fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_addr: &str,
    blocks: &[TransferBlockInfo],
    transfer_timeout: Duration,
) -> Result<FetchBlocksViaRdmaResult, String> {
    if blocks.is_empty() {
        return Ok(FetchBlocksViaRdmaResult::default());
    }

    let prepare_start = Instant::now();

    // (block_hash, Vec<(slot_segments)>) — for building SealedBlock afterwards
    let mut block_allocs: Vec<(Vec<u8>, Vec<Vec<SegmentAlloc>>)> = Vec::new();

    // Build TransferDescs and submit RDMA READ inside a sync block so that
    // all_descs (which contains NonNull<u8>, !Send) is dropped before any .await.
    let (receivers, desc_count) = {
        let mut all_descs: Vec<TransferDesc> = Vec::new();

        for block_info in blocks {
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

            for slot in &block_info.slots {
                let mut segments = Vec::new();
                let numa = NumaNode(slot.numa_node);

                // K segment
                if slot.k_size > 0 {
                    let alloc = allocate_fn(slot.k_size, Some(numa)).ok_or_else(|| {
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
                    let alloc = allocate_fn(slot.v_size, Some(numa)).ok_or_else(|| {
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
            return Ok(FetchBlocksViaRdmaResult::default());
        }

        // Submit RDMA READ; all_descs is dropped at the end of this block.
        let desc_count = all_descs.len();
        let receivers = rdma
            .engine()
            .batch_transfer_async(TransferOp::Read, remote_addr, &all_descs)
            .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?;
        (receivers, desc_count)
    };
    let prepare_elapsed = prepare_start.elapsed();

    let rdma_wait_start = Instant::now();
    tokio::time::timeout(transfer_timeout, async {
        for rx in receivers {
            rx.await
                .map_err(|_| "RDMA transfer channel closed".to_string())?
                .map_err(|e| format!("RDMA transfer failed: {e}"))?;
        }
        Ok::<(), String>(())
    })
    .await
    .map_err(|_| "RDMA transfer timed out".to_string())??;
    let rdma_wait_elapsed = rdma_wait_start.elapsed();

    // Build SealedBlocks from allocated memory
    let rebuild_start = Instant::now();
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
    let rebuild_elapsed = rebuild_start.elapsed();

    Ok(FetchBlocksViaRdmaResult {
        result,
        prepare_elapsed,
        rdma_wait_elapsed,
        rebuild_elapsed,
        desc_count,
    })
}

#[derive(Default)]
struct FetchBlocksViaRdmaResult {
    result: PrefetchResult,
    prepare_elapsed: Duration,
    rdma_wait_elapsed: Duration,
    rebuild_elapsed: Duration,
    desc_count: usize,
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
) -> Result<(EngineClient<Channel>, QueryBlocksForTransferResponse), String> {
    let mut client = get_or_create_channel(grpc_channels, remote_addr)?;

    let request = QueryBlocksForTransferRequest {
        namespace: namespace.to_string(),
        block_hashes: block_hashes.to_vec(),
        requester_id: advertise_addr.to_string(),
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
        return Err("server returned empty handshake_metadata".into());
    }
    let remote_meta = HandshakeMetadata::from_bytes(remote_bytes)
        .map_err(|e| format!("invalid metadata: {e}"))?;
    rdma.engine()
        .complete_handshake(remote_addr, local_meta, &remote_meta)
        .map_err(|e| format!("{e}"))
}

/// Compute client-side transfer timeout from server's lock timeout.
/// Returns `max(server_timeout - 60s, 10s)` so the client always finishes
/// before the server force-releases the lock.
fn transfer_timeout_from_server(lock_timeout_secs: u32) -> Duration {
    let server = Duration::from_secs(lock_timeout_secs as u64);
    server
        .saturating_sub(LOCK_TIMEOUT_MARGIN)
        .max(MIN_TRANSFER_TIMEOUT)
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
