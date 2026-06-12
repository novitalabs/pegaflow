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
    ReleaseTransferLockRequest, TransferSlotInfo,
};
use pegaflow_transfer::{ConnectionStatus, HandshakeMetadata, TransferDesc, TransferOp};
use tonic::transport::{Channel, Endpoint};

use pegaflow_common::NumaNode;

use opentelemetry::KeyValue;

use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;
use crate::transfer_runs::{SegmentRunTable, TransferPlan, plan_transfer};

/// Minimum usable transfer timeout. If the server's lock timeout minus the
/// safety margin falls below this, we use this floor to avoid instant timeouts.
const MIN_TRANSFER_TIMEOUT: Duration = Duration::from_secs(10);

/// Safety margin subtracted from the server's lock timeout. The client must
/// finish the RDMA transfer before the server releases the lock.
const LOCK_TIMEOUT_MARGIN: Duration = Duration::from_secs(60);

/// Upper bound for a single RDMA READ work request. Dense regions larger
/// than this are split so each READ stays well under the HCA max message
/// size (1 GiB on mlx5) and large transfers stripe across QPs/NICs.
const MAX_RDMA_READ_BYTES: u64 = 32 * 1024 * 1024;

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
    let block_count = response.block_count as usize;
    if block_count > block_hashes.len() {
        warn!(
            "Remote {remote_addr} returned {block_count} blocks for {} requested hashes",
            block_hashes.len()
        );
        spawn_release_lock(client, transfer_session_id);
        core_metrics()
            .rdma_fetch_total
            .add(1, &[KeyValue::new("status", "error")]);
        return Vec::new();
    }
    let template = response.slot_template;
    let segment_runs = SegmentRunTable {
        runs_per_lane: response.runs_per_lane,
        base: response.run_base,
        stride: response.run_stride,
        count: response.run_count,
    };
    let (result, transfer_timing) = match fetch_blocks_via_rdma(
        rdma,
        allocate_fn,
        namespace,
        remote_addr,
        &block_hashes[..block_count],
        &template,
        &segment_runs,
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
    let total_bytes = transfer_timing.total_bytes;
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

/// Plan the transfer, allocate per-NUMA slabs, execute RDMA READ, rebuild SealedBlocks.
///
/// `block_hashes` is the homogeneous prefix granted by the holder; every
/// block follows `template`, and `segment_runs` carries the remote addresses
/// as per-lane affine runs. Planning merges runs whose chunks tile densely,
/// so the common bump-allocated holder layout needs a handful of READs
/// instead of one per segment.
#[allow(
    clippy::too_many_arguments,
    reason = "arguments mirror the QueryBlocksForTransfer wire format"
)]
async fn fetch_blocks_via_rdma(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    namespace: &str,
    remote_addr: &str,
    block_hashes: &[Vec<u8>],
    template: &[TransferSlotInfo],
    segment_runs: &SegmentRunTable,
    transfer_timeout: Duration,
) -> Result<(PrefetchResult, TransferTiming), String> {
    if block_hashes.is_empty() {
        return Ok((Vec::new(), TransferTiming::default()));
    }

    let build_start = Instant::now();
    let plan = plan_transfer(template, block_hashes.len(), segment_runs)?;
    if plan.descs.is_empty() {
        return Err(format!(
            "transfer plan produced no descriptors for {} blocks",
            block_hashes.len()
        ));
    }
    let numa_slabs = allocate_numa_slabs(allocate_fn, &plan.bytes_per_numa)?;
    let slot_count = block_hashes.len() * template.len();

    // Build TransferDescs and submit RDMA READ inside a sync block so that
    // all_descs (which contains NonNull<u8>, !Send) is dropped before any .await.
    let (receivers, mut timing) = {
        let mut all_descs: Vec<TransferDesc> = Vec::with_capacity(plan.descs.len());
        for desc in &plan.descs {
            let slab = numa_slabs
                .get(&desc.numa)
                .ok_or_else(|| format!("missing slab for {}", desc.numa))?;
            // Split dense regions: a single READ must stay under the HCA
            // max message size (1 GiB on mlx5), and multiple descriptors let
            // the transfer engine stripe a large region across QPs/NICs.
            let mut offset = 0u64;
            while offset < desc.len {
                let chunk = (desc.len - offset).min(MAX_RDMA_READ_BYTES);
                let len = usize::try_from(chunk)
                    .map_err(|_| format!("transfer desc length exceeds usize: {chunk}"))?;
                let remote = desc.remote_base + offset;
                all_descs.push(TransferDesc {
                    local_ptr: slab.ptr_at(desc.local_offset + offset, len)?,
                    remote_ptr: NonNull::new(remote as *mut u8)
                        .ok_or_else(|| "remote desc ptr is null".to_string())?,
                    len,
                });
                offset += chunk;
            }
        }

        let submit_start = Instant::now();
        let receivers = rdma
            .engine()
            .batch_transfer_async(TransferOp::Read, remote_addr, &all_descs)
            .map_err(|e| format!("RDMA batch_transfer_async failed: {e}"))?;
        let submit_transfer = submit_start.elapsed();

        let timing = TransferTiming {
            build_transfer_tasks: build_start.elapsed().saturating_sub(submit_transfer),
            submit_transfer,
            transfer_desc_count: all_descs.len(),
            slot_count,
            numa_slab_count: numa_slabs.len(),
            total_bytes: plan.bytes_per_numa.values().sum(),
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

    let rebuild_start = Instant::now();
    let result = rebuild_sealed_blocks(namespace, block_hashes, template, &plan, &numa_slabs)?;
    timing.rebuild = rebuild_start.elapsed();

    Ok((result, timing))
}

/// Rebuild SealedBlocks from the local mirror: every segment's address is
/// derived arithmetically from its lane run's local placement.
fn rebuild_sealed_blocks(
    namespace: &str,
    block_hashes: &[Vec<u8>],
    template: &[TransferSlotInfo],
    plan: &TransferPlan,
    numa_slabs: &HashMap<NumaNode, NumaSlab>,
) -> Result<PrefetchResult, String> {
    let mut cursors = vec![0usize; plan.lanes.len()];
    let mut result: PrefetchResult = Vec::with_capacity(block_hashes.len());
    for (block, hash) in block_hashes.iter().enumerate() {
        let mut slots: Vec<Arc<RawBlock>> = Vec::with_capacity(template.len());
        let mut lane_idx = 0;
        for slot in template {
            let seg_count = usize::from(slot.k_size > 0) + usize::from(slot.v_size > 0);
            let mut segments: Vec<Segment> = Vec::with_capacity(seg_count);
            for _ in 0..seg_count {
                let lane = &plan.lanes[lane_idx];
                let cursor = &mut cursors[lane_idx];
                let mut run = &lane.runs[*cursor];
                while block >= run.start_block as usize + run.count as usize {
                    *cursor += 1;
                    run = lane.runs.get(*cursor).ok_or_else(|| {
                        format!("lane {lane_idx} runs exhausted at block {block}")
                    })?;
                }
                let slab = numa_slabs
                    .get(&lane.numa)
                    .ok_or_else(|| format!("missing slab for {} while rebuilding", lane.numa))?;
                let offset =
                    run.local_offset + (block as u64 - run.start_block as u64) * run.local_stride;
                let len = usize::try_from(lane.seg_size)
                    .map_err(|_| format!("segment size exceeds usize: {}", lane.seg_size))?;
                let ptr = slab.ptr_at(offset, len)?;
                segments.push(Segment::new(ptr, len, Arc::clone(&slab.allocation)));
                lane_idx += 1;
            }
            slots.push(Arc::new(RawBlock::new(segments)));
        }
        let key = BlockKey::new(namespace.to_string(), hash.clone());
        result.push((key, Arc::new(SealedBlock::from_slots(slots))));
    }
    Ok(result)
}

fn allocate_numa_slabs(
    allocate_fn: &AllocateFn,
    bytes_per_numa: &HashMap<NumaNode, u64>,
) -> Result<HashMap<NumaNode, NumaSlab>, String> {
    let mut numa_slabs: HashMap<NumaNode, NumaSlab> = HashMap::new();
    for (&numa, &total_bytes) in bytes_per_numa {
        if total_bytes == 0 {
            continue;
        }
        let allocation = allocate_fn(total_bytes, Some(numa))
            .ok_or_else(|| format!("failed to allocate slab ({total_bytes} bytes) for {numa}"))?;
        let capacity = usize::try_from(total_bytes)
            .map_err(|_| format!("slab size exceeds usize for {numa}: {total_bytes}"))?;
        numa_slabs.insert(
            numa,
            NumaSlab {
                allocation,
                capacity,
            },
        );
    }
    Ok(numa_slabs)
}

/// One pinned allocation per NUMA node; the transfer plan addresses it by
/// byte offset.
struct NumaSlab {
    allocation: Arc<crate::pinned_pool::PinnedAllocation>,
    capacity: usize,
}

impl NumaSlab {
    fn ptr_at(&self, offset: u64, len: usize) -> Result<NonNull<u8>, String> {
        let offset =
            usize::try_from(offset).map_err(|_| format!("slab offset exceeds usize: {offset}"))?;
        let in_bounds = offset
            .checked_add(len)
            .is_some_and(|end| end <= self.capacity);
        if !in_bounds {
            return Err(format!(
                "slab range out of bounds: offset={offset} len={len} capacity={}",
                self.capacity
            ));
        }
        let ptr = unsafe { self.allocation.as_non_null().as_ptr().add(offset) };
        NonNull::new(ptr).ok_or_else(|| "slab pointer is null".to_string())
    }
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
    /// Payload bytes, summed from the validated transfer plan.
    total_bytes: u64,
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
    // QueryBlocksForTransfer responses are multi-MB; the HTTP/2 spec-default
    // 64 KiB flow-control window would stall the holder every 64 KiB to wait
    // for a WINDOW_UPDATE round trip.
    const H2_WINDOW_SIZE: u32 = 16 * 1024 * 1024;
    let channel = Endpoint::from_shared(url)
        .map_err(|e| format!("invalid remote address: {e}"))?
        .connect_timeout(Duration::from_secs(5))
        .initial_stream_window_size(Some(H2_WINDOW_SIZE))
        .initial_connection_window_size(Some(H2_WINDOW_SIZE))
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
    fn allocate_numa_slabs_calls_allocator_once_per_numa() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));

        let bytes_per_numa = HashMap::from([(NumaNode(0), 1024_u64), (NumaNode(1), 2048_u64)]);
        let slabs = allocate_numa_slabs(&allocate_fn, &bytes_per_numa).expect("allocate slabs");

        assert_eq!(slabs.len(), 2);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn slab_offsets_are_relative_and_bounded() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));

        let slabs = allocate_numa_slabs(&allocate_fn, &HashMap::from([(NumaNode(0), 1024_u64)]))
            .expect("allocate slabs");
        let slab = slabs.get(&NumaNode(0)).expect("slab for numa 0");

        let p0 = slab.ptr_at(0, 256).expect("offset 0");
        let p256 = slab.ptr_at(256, 128).expect("offset 256");
        assert_eq!(p256.as_ptr() as usize - p0.as_ptr() as usize, 256);

        assert!(slab.ptr_at(1024, 1).is_err());
        assert!(slab.ptr_at(512, 1024).is_err());
    }
}
