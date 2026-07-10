// RDMA remote block fetch: MetaServer query -> gRPC QueryBlocksForTransfer -> RDMA READ.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use log::{debug, info, warn};
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

/// Upper bound for a single pinned-pool allocation while staging an RDMA fetch.
/// LRU reclaim must carve a contiguous hole of the requested size, so a
/// whole-prefix slab can force eviction of far more bytes than the fetch needs.
const FETCH_CHUNK_BYTES: u64 = 256 * 1024 * 1024;

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

    /// Query MetaServer for the best remote node that holds a prefix of `hashes`.
    /// Returns `(node_addr, prefix_len)`, or `None` if no remote node has any.
    pub(crate) async fn query_prefix(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Option<(String, usize)> {
        if hashes.is_empty() {
            return None;
        }

        let nodes = match self.metaserver_client.query_prefix(namespace, hashes).await {
            Ok(n) => n,
            Err(e) => {
                warn!("MetaServer query failed for remote fetch: {e}");
                return None;
            }
        };

        let best = nodes
            .iter()
            .filter(|n| n.node != self.advertise_addr)
            .max_by_key(|n| n.prefix_len)?;

        let prefix_len = best.prefix_len as usize;
        if prefix_len == 0 {
            return None;
        }

        debug!(
            "Remote prefix query: namespace={namespace} best_node={} prefix={prefix_len}/{}",
            best.node,
            hashes.len()
        );

        Some((best.node.clone(), prefix_len))
    }

    /// Fetch `hashes` from `remote_addr`.
    pub(crate) async fn fetch_blocks(
        &self,
        remote_addr: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchResult {
        rdma_fetch_task(
            &self.rdma_transport,
            &self.allocate_fn,
            &self.grpc_channels,
            &self.connect_group,
            remote_addr,
            req_id,
            &self.advertise_addr,
            namespace,
            hashes,
        )
        .await
    }
}

/// Execute RDMA fetch against a single remote node.
///
/// 1. Ensure RDMA connection (singleflight per remote_addr)
/// 2. gRPC QueryBlocksForTransfer RPC (connection reuse, no handshake)
/// 3. RDMA READ all block segments + build SealedBlocks
/// 4. ReleaseTransferLock (fire-and-forget, non-blocking)
#[allow(
    clippy::too_many_arguments,
    reason = "RDMA task arguments are the per-fetch context passed from the scheduler"
)]
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    connect_group: &Group<String, ()>,
    remote_addr: &str,
    req_id: &str,
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
    let (result, transfer_timing) = match fetch_blocks_via_rdma(
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

    // 4. Release transfer lock (fire-and-forget: spawns a detached task)
    spawn_release_lock(client, transfer_session_id);

    let elapsed = t0.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
        mb / elapsed.as_secs_f64()
    } else {
        0.0
    };
    info!(
        "RDMA fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} slots={} descs={} slabs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
        result.len(),
        block_hashes.len(),
        transfer_timing.slot_count,
        transfer_timing.transfer_desc_count,
        transfer_timing.numa_slab_count,
    );
    info!(
        "RDMA fetch stages: req_id={req_id} remote={remote_addr} connect_ms={:.2} query_ms={:.2} build_transfer_tasks_ms={:.2} submit_transfer_ms={:.2} rdma_wait_ms={:.2} rebuild_ms={:.2}",
        connect_elapsed.as_secs_f64() * 1000.0,
        query_elapsed.as_secs_f64() * 1000.0,
        transfer_timing.build_transfer_tasks.as_secs_f64() * 1000.0,
        transfer_timing.submit_transfer.as_secs_f64() * 1000.0,
        transfer_timing.rdma_wait.as_secs_f64() * 1000.0,
        transfer_timing.rebuild.as_secs_f64() * 1000.0,
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

/// One fetched slot: its RDMA-staged segments plus the NUMA node they sit on.
/// The NUMA travels with the slot so a re-served block advertises real topology.
type StagedSlot = (Vec<SegmentAlloc>, NumaNode);
/// A staged block awaiting SealedBlock rebuild: its hash and per-slot allocations.
type StagedBlock = (Vec<u8>, Vec<StagedSlot>);

/// Allocate local memory, build TransferDescs, execute RDMA READ, build SealedBlocks.
async fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_addr: &str,
    blocks: &[TransferBlockInfo],
    transfer_timeout: Duration,
) -> Result<(PrefetchResult, TransferTiming), String> {
    if blocks.is_empty() {
        return Ok((Vec::new(), TransferTiming::default()));
    }

    let mut slabs = ChunkedSlabs::new(allocate_fn, FETCH_CHUNK_BYTES);

    // (block_hash, Vec<(slot_segments, slot_numa)>) — for building SealedBlock afterwards.
    // The per-slot NUMA is preserved so a re-served fetched block advertises real topology.
    let mut block_allocs: Vec<StagedBlock> = Vec::new();
    let mut slot_count = 0usize;
    let build_start = Instant::now();

    // Build TransferDescs and submit RDMA READ inside a sync block so that
    // all_descs (which contains NonNull<u8>, !Send) is dropped before any .await.
    let (receivers, mut timing) = {
        let mut all_descs: Vec<TransferDesc> = Vec::new();

        for block_info in blocks {
            slot_count += block_info.slots.len();
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());

            for slot in &block_info.slots {
                let mut segments = Vec::new();
                let numa = NumaNode(slot.numa_node);

                // K segment
                if slot.k_size > 0 {
                    let len = usize::try_from(slot.k_size)
                        .map_err(|_| format!("K size exceeds usize: {}", slot.k_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "K")?;
                    let remote_ptr = NonNull::new(slot.k_ptr as *mut u8)
                        .ok_or_else(|| "remote K ptr is null".to_string())?;
                    all_descs.push(TransferDesc {
                        local_ptr,
                        remote_ptr,
                        len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                // V segment (split KV)
                if slot.v_size > 0 && slot.v_ptr != 0 {
                    let len = usize::try_from(slot.v_size)
                        .map_err(|_| format!("V size exceeds usize: {}", slot.v_size))?;
                    let (local_ptr, alloc) = slabs.alloc_segment(numa, len, "V")?;
                    let remote_ptr = NonNull::new(slot.v_ptr as *mut u8)
                        .ok_or_else(|| "remote V ptr is null".to_string())?;
                    all_descs.push(TransferDesc {
                        local_ptr,
                        remote_ptr,
                        len,
                    });
                    segments.push(SegmentAlloc {
                        ptr_addr: local_ptr.as_ptr() as u64,
                        alloc,
                        size: len,
                    });
                }

                slot_allocs.push((segments, numa));
            }

            block_allocs.push((block_info.block_hash.clone(), slot_allocs));
        }

        if all_descs.is_empty() {
            let timing = TransferTiming {
                build_transfer_tasks: build_start.elapsed(),
                slot_count,
                numa_slab_count: slabs.chunk_count,
                ..TransferTiming::default()
            };
            return Ok((Vec::new(), timing));
        }

        let transfer_desc_count = all_descs.len();

        // Submit RDMA READ; all_descs is dropped at the end of this block.
        let submit_start = Instant::now();
        let receivers = rdma
            .engine()
            .batch_transfer_async(TransferOp::Read, remote_addr, &all_descs)
            .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?;
        let submit_transfer = submit_start.elapsed();

        let timing = TransferTiming {
            build_transfer_tasks: build_start.elapsed().saturating_sub(submit_transfer),
            submit_transfer,
            transfer_desc_count,
            slot_count,
            numa_slab_count: slabs.chunk_count,
            ..TransferTiming::default()
        };
        (receivers, timing)
    };

    let wait_start = Instant::now();
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
    timing.rdma_wait = wait_start.elapsed();

    // Build SealedBlocks from allocated memory
    let rebuild_start = Instant::now();
    let mut result: PrefetchResult = Vec::with_capacity(block_allocs.len());
    for (hash, slot_allocs) in block_allocs {
        let key = BlockKey::new(namespace.to_string(), hash);
        let slots: Vec<(RawBlock, NumaNode)> = slot_allocs
            .into_iter()
            .map(|(segs, numa)| {
                let segments: Vec<Segment> = segs
                    .into_iter()
                    .map(|sa| {
                        let ptr = NonNull::new(sa.ptr_addr as *mut u8)
                            .expect("slab segment pointer must be non-null");
                        Segment::new(ptr, sa.size, sa.alloc)
                    })
                    .collect();
                (RawBlock::new(segments), numa)
            })
            .collect();
        let sealed = Arc::new(SealedBlock::from_slots(slots));
        result.push((key, sealed));
    }
    timing.rebuild = rebuild_start.elapsed();

    Ok((result, timing))
}

/// Bump allocator over bounded pinned chunks, one active chunk per NUMA node.
/// Each staged segment holds an Arc to its own chunk, so starting a fresh
/// chunk never invalidates previously staged segments, and fetched blocks are
/// freed chunk-by-chunk on eviction instead of all-or-nothing per fetch.
struct ChunkedSlabs<'a> {
    allocate_fn: &'a AllocateFn,
    chunk_bytes: u64,
    current: HashMap<NumaNode, NumaSlab>,
    chunk_count: usize,
}

impl<'a> ChunkedSlabs<'a> {
    fn new(allocate_fn: &'a AllocateFn, chunk_bytes: u64) -> Self {
        Self {
            allocate_fn,
            chunk_bytes,
            current: HashMap::new(),
            chunk_count: 0,
        }
    }

    fn alloc_segment(
        &mut self,
        numa: NumaNode,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::pinned_pool::PinnedAllocation>), String> {
        if let Some(slab) = self.current.get_mut(&numa)
            && let Ok(seg) = slab.allocate(len, segment_kind)
        {
            return Ok(seg);
        }

        // No chunk on this NUMA yet, or the current one can't fit the segment.
        let chunk = self.chunk_bytes.max(len as u64);
        let allocation = (self.allocate_fn)(chunk, Some(numa)).ok_or_else(|| {
            format!("failed to allocate fetch chunk ({chunk} bytes) on {numa} for {segment_kind}")
        })?;
        let capacity = usize::try_from(chunk)
            .map_err(|_| format!("fetch chunk size exceeds usize: {chunk}"))?;
        self.chunk_count += 1;
        self.current.insert(
            numa,
            NumaSlab {
                allocation,
                next_offset: 0,
                capacity,
            },
        );
        self.current
            .get_mut(&numa)
            .expect("chunk just inserted")
            .allocate(len, segment_kind)
    }
}

struct NumaSlab {
    allocation: Arc<crate::pinned_pool::PinnedAllocation>,
    next_offset: usize,
    capacity: usize,
}

impl NumaSlab {
    fn allocate(
        &mut self,
        len: usize,
        segment_kind: &str,
    ) -> Result<(NonNull<u8>, Arc<crate::pinned_pool::PinnedAllocation>), String> {
        let end = self.next_offset.checked_add(len).ok_or_else(|| {
            format!(
                "slab offset overflow while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            )
        })?;
        if end > self.capacity {
            return Err(format!(
                "slab exhausted while allocating {segment_kind}: offset={} len={len} capacity={}",
                self.next_offset, self.capacity
            ));
        }

        let ptr = unsafe { self.allocation.as_non_null().as_ptr().add(self.next_offset) };
        self.next_offset = end;
        let ptr = NonNull::new(ptr).ok_or_else(|| "slab pointer is null".to_string())?;
        Ok((ptr, Arc::clone(&self.allocation)))
    }
}

struct SegmentAlloc {
    ptr_addr: u64,
    alloc: Arc<crate::pinned_pool::PinnedAllocation>,
    size: usize,
}

#[derive(Default)]
struct TransferTiming {
    build_transfer_tasks: Duration,
    submit_transfer: Duration,
    rdma_wait: Duration,
    rebuild: Duration,
    transfer_desc_count: usize,
    slot_count: usize,
    numa_slab_count: usize,
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
    // Match the engine server's 64 MiB message cap: a QueryBlocksForTransfer
    // response carries per-slot transfer descriptors, so a large block batch
    // overflows tonic's default 4 MiB decode limit.
    const MAX_GRPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;
    let client = EngineClient::new(channel)
        .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
        .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_allocate_fn(calls: Arc<AtomicUsize>) -> AllocateFn {
        let allocator = Arc::new(crate::pinned_pool::PinnedAllocator::new_global(
            32 * 1024 * 1024,
            1,
            false,
            false,
            None,
        ));
        Arc::new(move |size, _numa| {
            calls.fetch_add(1, Ordering::Relaxed);
            allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN)
        })
    }

    #[test]
    fn chunked_slabs_bump_within_chunk_then_refill() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));
        let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024);

        let (p1, a1) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("first");
        let (p2, _a2) = slabs.alloc_segment(NumaNode(0), 512, "V").expect("second");
        assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 512);
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        // Third segment exceeds the current chunk: a fresh chunk is allocated
        // while earlier segments stay valid through their own chunk Arc.
        let (_p3, a3) = slabs.alloc_segment(NumaNode(0), 512, "K").expect("third");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(slabs.chunk_count, 2);
        assert!(!Arc::ptr_eq(&a1, &a3));
    }

    #[test]
    fn chunked_slabs_oversized_segment_gets_dedicated_chunk() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));
        let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024);

        slabs
            .alloc_segment(NumaNode(0), 4096, "K")
            .expect("oversized segment");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(slabs.chunk_count, 1);
    }

    #[test]
    fn chunked_slabs_allocation_failure_is_an_error() {
        let allocate_fn: AllocateFn = Arc::new(|_, _| None);
        let mut slabs = ChunkedSlabs::new(&allocate_fn, 1024);

        let err = match slabs.alloc_segment(NumaNode(0), 512, "K") {
            Ok(_) => panic!("allocation should fail"),
            Err(err) => err,
        };
        assert!(err.contains("failed to allocate fetch chunk"));
    }
}
