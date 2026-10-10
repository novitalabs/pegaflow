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
    FetchSegment, QueryBlocksForTransferRequest, QueryBlocksForTransferResponse,
    RdmaHandshakeRequest, TransferBlockInfo, TransferSlotInfo,
};
use pegaflow_transfer::{ConnectionStatus, HandshakeMetadata, TransferDesc, TransferOp};
use tonic::transport::{Channel, Endpoint};

use pegaflow_common::NumaNode;

use opentelemetry::KeyValue;

use super::transfer_lock_guard::TransferLockGuard;
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

/// Slots at least this large are staged in their own pinned allocation.
///
/// Fetched memory must match the pool's resident allocation sizes: a fetch
/// that asks for one large contiguous slab makes LRU reclaim evict blocks
/// until a hole that big happens to open, which on a fragmented pool flushes
/// most of the cache. Page-first saves allocate one page per slot, so staging
/// per slot keeps every allocation the same size and owned by one block.
/// Smaller (per-layer) slots are coalesced per (block, NUMA) instead, which is
/// still fixed-size per model and keeps the allocation count bounded.
const PER_SLOT_ALLOC_MIN_BYTES: u64 = 256 * 1024;

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct FetchPlanSegment {
    node: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FetchPlan {
    segments: Vec<FetchPlanSegment>,
    block_count: usize,
}

impl FetchPlan {
    pub(crate) fn block_count(&self) -> usize {
        self.block_count
    }

    fn segment_blocks_summary(&self) -> String {
        self.segments
            .iter()
            .map(|segment| (segment.end - segment.start).to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn validate_fetch_plan(
    segments: Vec<FetchSegment>,
    hash_count: usize,
    exclude_node: &str,
) -> Result<Option<FetchPlan>, String> {
    if segments.is_empty() {
        return Ok(None);
    }

    let mut validated = Vec::with_capacity(segments.len());
    let mut offset = 0usize;
    for (index, segment) in segments.into_iter().enumerate() {
        if segment.node.is_empty() {
            return Err(format!("segment {index} has an empty node"));
        }
        if segment.node == exclude_node {
            return Err(format!("segment {index} selects the excluded requester"));
        }
        if segment.block_count == 0 {
            return Err(format!("segment {index} has zero blocks"));
        }
        if validated
            .last()
            .is_some_and(|previous: &FetchPlanSegment| previous.node == segment.node)
        {
            return Err(format!(
                "segment {index} repeats the previous node instead of merging"
            ));
        }

        let count = usize::try_from(segment.block_count)
            .map_err(|_| format!("segment {index} block count exceeds usize"))?;
        let end = offset
            .checked_add(count)
            .ok_or_else(|| format!("segment {index} block count overflows"))?;
        if end > hash_count {
            return Err(format!(
                "segment {index} ends at block {end}, beyond request length {hash_count}"
            ));
        }
        validated.push(FetchPlanSegment {
            node: segment.node,
            start: offset,
            end,
        });
        offset = end;
    }

    Ok(Some(FetchPlan {
        segments: validated,
        block_count: offset,
    }))
}

#[tonic::async_trait]
trait SegmentFetcher {
    async fn fetch_segment(&self, remote_addr: &str, hashes: &[Vec<u8>]) -> PrefetchResult;
}

struct RdmaSegmentFetcher<'a> {
    store: &'a RdmaFetchStore,
    req_id: &'a str,
    namespace: &'a str,
}

#[tonic::async_trait]
impl SegmentFetcher for RdmaSegmentFetcher<'_> {
    async fn fetch_segment(&self, remote_addr: &str, hashes: &[Vec<u8>]) -> PrefetchResult {
        self.store
            .fetch_blocks(remote_addr, self.req_id, self.namespace, hashes)
            .await
    }
}

async fn execute_fetch_plan<F: SegmentFetcher>(
    fetcher: &F,
    plan: &FetchPlan,
    hashes: &[Vec<u8>],
) -> (PrefetchResult, usize, Option<(usize, usize, usize)>) {
    let mut fetched = Vec::with_capacity(plan.block_count);
    let mut completed_segments = 0usize;
    let mut failed_segment = None;

    for (index, segment) in plan.segments.iter().enumerate() {
        let expected = &hashes[segment.start..segment.end];
        // The holder answers with `prefix_only`, so `returned` is a prefix of
        // `expected`; a short segment ends the contiguous remote prefix.
        let returned = fetcher.fetch_segment(&segment.node, expected).await;
        let returned_count = returned.len();
        fetched.extend(returned);

        if returned_count != expected.len() {
            failed_segment = Some((index, expected.len(), returned_count));
            break;
        }
        completed_segments += 1;
    }

    (fetched, completed_segments, failed_segment)
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

    /// Query MetaServer for a validated ordered plan covering a prefix of `hashes`.
    pub(crate) async fn query_plan(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> Option<FetchPlan> {
        if hashes.is_empty() {
            return None;
        }

        let segments = match self
            .metaserver_client
            .query_plan(namespace, hashes, &self.advertise_addr)
            .await
        {
            Ok(segments) => segments,
            Err(e) => {
                warn!("MetaServer query failed for remote fetch: {e}");
                return None;
            }
        };

        let plan = match validate_fetch_plan(segments, hashes.len(), &self.advertise_addr) {
            Ok(plan) => plan?,
            Err(error) => {
                warn!("MetaServer returned invalid remote fetch plan: {error}");
                return None;
            }
        };

        debug!(
            "Remote prefix query: segments={} prefix={}/{}",
            plan.segments.len(),
            plan.block_count,
            hashes.len(),
        );

        Some(plan)
    }

    pub(crate) async fn fetch_plan(
        &self,
        plan: &FetchPlan,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchResult {
        let started_at = Instant::now();
        let fetcher = RdmaSegmentFetcher {
            store: self,
            req_id,
            namespace,
        };
        let (fetched, completed_segments, failure) =
            execute_fetch_plan(&fetcher, plan, hashes).await;
        let metrics = core_metrics();
        metrics
            .rdma_fetch_plan_segments
            .record(plan.segments.len() as u64, &[]);
        metrics
            .rdma_fetch_plan_completed_segments
            .record(completed_segments as u64, &[]);
        let (failed_segment, failed_planned_blocks, failed_returned_blocks) = failure
            .map(|(index, planned, returned)| {
                (index.to_string(), planned.to_string(), returned.to_string())
            })
            .unwrap_or_else(|| ("none".into(), "none".into(), "none".into()));

        info!(
            "RDMA multi-node fetch plan summary: req_id={} planned_segments={} completed_segments={} planned_blocks={} segment_blocks={} fetched_blocks={} failed_segment={} failed_segment_planned_blocks={} failed_segment_returned_blocks={} total_ms={:.2}",
            req_id,
            plan.segments.len(),
            completed_segments,
            plan.block_count,
            plan.segment_blocks_summary(),
            fetched.len(),
            failed_segment,
            failed_planned_blocks,
            failed_returned_blocks,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );

        fetched
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
    let (client, mut response) = match query_remote_blocks(
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

    // The holder pinned the blocks when the query created this session, so
    // every exit from here — completion, error, panic, or this future being
    // dropped — must send ReleaseTransferLock. The guard's Drop covers the
    // paths no explicit call can reach.
    let lock_guard = TransferLockGuard::new(
        client,
        std::mem::take(&mut response.transfer_session_id),
        remote_addr,
        req_id,
    );

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
            lock_guard.release(false);
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };

    // 4. Release transfer lock (fire-and-forget: spawns a detached task) and
    // let the holder demote its now-replicated source copies.
    lock_guard.release(true);

    let elapsed = t0.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
        mb / elapsed.as_secs_f64()
    } else {
        0.0
    };
    info!(
        "RDMA fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} slots={} descs={} allocs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
        result.len(),
        block_hashes.len(),
        transfer_timing.slot_count,
        transfer_timing.transfer_desc_count,
        transfer_timing.alloc_count,
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

    // (block_hash, Vec<(slot_segments, slot_numa)>) — for building SealedBlock afterwards.
    // The per-slot NUMA is preserved so a re-served fetched block advertises real topology.
    let mut block_allocs: Vec<StagedBlock> = Vec::new();
    let mut slot_count = 0usize;
    let mut alloc_count = 0usize;
    let build_start = Instant::now();

    // Build TransferDescs and submit RDMA READ inside a sync block so that
    // all_descs (which contains NonNull<u8>, !Send) is dropped before any .await.
    let (receivers, mut timing) = {
        let mut all_descs: Vec<TransferDesc> = Vec::new();

        for block_info in blocks {
            slot_count += block_info.slots.len();
            let mut slot_allocs: Vec<Option<StagedSlot>> =
                (0..block_info.slots.len()).map(|_| None).collect();

            for group in plan_block_allocations(&block_info.slots)? {
                let allocation = (allocate_fn)(group.bytes, Some(group.numa)).ok_or_else(|| {
                    format!(
                        "failed to allocate fetch staging ({} bytes) on {}",
                        group.bytes, group.numa
                    )
                })?;
                alloc_count += 1;
                let base = allocation.as_non_null();
                let mut offset = 0usize;

                for slot_idx in group.slots {
                    let slot = &block_info.slots[slot_idx];
                    let mut segments = Vec::new();
                    for (kind, remote, size) in slot_segments(slot) {
                        let len = usize::try_from(size)
                            .map_err(|_| format!("{kind} size exceeds usize: {size}"))?;
                        let remote_ptr = NonNull::new(remote as *mut u8)
                            .ok_or_else(|| format!("remote {kind} ptr is null"))?;
                        // SAFETY: the group allocation is sized to the sum of its
                        // slots' segments, so `offset + len` stays in bounds.
                        let local_ptr = unsafe { base.add(offset) };
                        offset += len;
                        all_descs.push(TransferDesc {
                            local_ptr,
                            remote_ptr,
                            len,
                        });
                        segments.push(SegmentAlloc {
                            ptr_addr: local_ptr.as_ptr() as u64,
                            alloc: Arc::clone(&allocation),
                            size: len,
                        });
                    }
                    slot_allocs[slot_idx] = Some((segments, group.numa));
                }
            }

            // Slots with nothing to read keep an empty segment list.
            let slot_allocs = slot_allocs
                .into_iter()
                .zip(&block_info.slots)
                .map(|(staged, slot)| {
                    staged.unwrap_or_else(|| (Vec::new(), NumaNode(slot.numa_node)))
                })
                .collect();
            block_allocs.push((block_info.block_hash.clone(), slot_allocs));
        }

        if all_descs.is_empty() {
            let timing = TransferTiming {
                build_transfer_tasks: build_start.elapsed(),
                slot_count,
                alloc_count,
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
            alloc_count,
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

/// Remote segments of one slot that need an RDMA READ: `(kind, remote_ptr, bytes)`.
fn slot_segments(slot: &TransferSlotInfo) -> impl Iterator<Item = (&'static str, u64, u64)> {
    let k = (slot.k_size > 0).then_some(("K", slot.k_ptr, slot.k_size));
    let v = (slot.v_size > 0 && slot.v_ptr != 0).then_some(("V", slot.v_ptr, slot.v_size));
    k.into_iter().chain(v)
}

/// One pinned allocation backing some of a block's slots on a single NUMA node.
#[derive(Debug, PartialEq, Eq)]
struct SlotGroup {
    numa: NumaNode,
    bytes: u64,
    slots: Vec<usize>,
}

/// Decide how a fetched block's slots map onto pinned allocations.
///
/// Every allocation belongs to exactly one block, so evicting the block frees
/// its memory. Slots of at least `PER_SLOT_ALLOC_MIN_BYTES` (page-first pages)
/// get one allocation each, matching the save path's page size. Smaller slots
/// are coalesced into one allocation per NUMA node. Slots with nothing to read
/// are left out.
fn plan_block_allocations(slots: &[TransferSlotInfo]) -> Result<Vec<SlotGroup>, String> {
    let mut groups: Vec<SlotGroup> = Vec::new();
    // Index into `groups` of the coalesced small-slot group per NUMA node.
    let mut small: HashMap<NumaNode, usize> = HashMap::new();

    for (slot_idx, slot) in slots.iter().enumerate() {
        let bytes = slot_segments(slot).try_fold(0u64, |total, (kind, _, size)| {
            total
                .checked_add(size)
                .ok_or_else(|| format!("slot {slot_idx} {kind} size overflows"))
        })?;
        if bytes == 0 {
            continue;
        }
        let numa = NumaNode(slot.numa_node);
        if bytes >= PER_SLOT_ALLOC_MIN_BYTES {
            groups.push(SlotGroup {
                numa,
                bytes,
                slots: vec![slot_idx],
            });
            continue;
        }
        let idx = *small.entry(numa).or_insert_with(|| {
            groups.push(SlotGroup {
                numa,
                bytes: 0,
                slots: Vec::new(),
            });
            groups.len() - 1
        });
        let group = &mut groups[idx];
        group.bytes = group
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| format!("block staging bytes overflow on {numa}"))?;
        group.slots.push(slot_idx);
    }
    Ok(groups)
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
    alloc_count: usize,
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
        // Prefix fetch keeps only the contiguous prefix; never lock blocks past a gap.
        prefix_only: true,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    fn slot(numa: u32, k_size: u64, v_size: u64) -> TransferSlotInfo {
        TransferSlotInfo {
            k_ptr: if k_size > 0 { 0x1000 } else { 0 },
            k_size,
            v_ptr: if v_size > 0 { 0x2000 } else { 0 },
            v_size,
            numa_node: numa,
        }
    }

    fn group(numa: u32, bytes: u64, slots: &[usize]) -> SlotGroup {
        SlotGroup {
            numa: NumaNode(numa),
            bytes,
            slots: slots.to_vec(),
        }
    }

    fn segment(node: &str, block_count: u32) -> FetchSegment {
        FetchSegment {
            node: node.to_string(),
            block_count,
        }
    }

    fn fetched_block(hash: u8) -> (BlockKey, Arc<SealedBlock>) {
        (
            BlockKey::new("ns".to_string(), vec![hash]),
            Arc::new(SealedBlock::from_slots(Vec::new())),
        )
    }

    #[derive(Default)]
    struct FakeSegmentFetcher {
        calls: Mutex<Vec<(String, Vec<Vec<u8>>)>>,
        responses: Mutex<VecDeque<PrefetchResult>>,
    }

    #[tonic::async_trait]
    impl SegmentFetcher for FakeSegmentFetcher {
        async fn fetch_segment(&self, remote_addr: &str, hashes: &[Vec<u8>]) -> PrefetchResult {
            self.calls
                .lock()
                .unwrap()
                .push((remote_addr.to_string(), hashes.to_vec()));
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default()
        }
    }

    #[test]
    fn validates_ordered_fetch_plan_offsets() {
        let plan = validate_fetch_plan(
            vec![segment("node-a", 2), segment("node-b", 1)],
            3,
            "requester",
        )
        .expect("plan should be valid")
        .expect("plan should be non-empty");

        assert_eq!(plan.block_count, 3);
        assert_eq!(plan.segment_blocks_summary(), "2,1");
        assert_eq!(
            plan.segments,
            vec![
                FetchPlanSegment {
                    node: "node-a".into(),
                    start: 0,
                    end: 2,
                },
                FetchPlanSegment {
                    node: "node-b".into(),
                    start: 2,
                    end: 3,
                },
            ]
        );
    }

    #[test]
    fn rejects_invalid_fetch_plans() {
        for (segments, expected) in [
            (vec![segment("", 1)], "empty node"),
            (vec![segment("node-a", 0)], "zero blocks"),
            (vec![segment("requester", 1)], "excluded requester"),
            (vec![segment("node-a", 2)], "beyond request length"),
            (
                vec![segment("node-a", 1), segment("node-a", 1)],
                "repeats the previous node",
            ),
        ] {
            let error =
                validate_fetch_plan(segments, 1, "requester").expect_err("plan should be rejected");
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[tokio::test]
    async fn fetch_plan_executes_segments_in_order() {
        let plan = validate_fetch_plan(
            vec![segment("node-a", 2), segment("node-b", 1)],
            3,
            "requester",
        )
        .unwrap()
        .unwrap();
        let fetcher = FakeSegmentFetcher {
            calls: Mutex::new(Vec::new()),
            responses: Mutex::new(VecDeque::from([
                vec![fetched_block(1), fetched_block(2)],
                vec![fetched_block(3)],
            ])),
        };
        let hashes = vec![vec![1], vec![2], vec![3]];

        let (fetched, completed, failure) = execute_fetch_plan(&fetcher, &plan, &hashes).await;

        assert_eq!(fetched.len(), 3);
        assert_eq!(completed, 2);
        assert_eq!(failure, None);
        assert_eq!(
            *fetcher.calls.lock().unwrap(),
            vec![
                ("node-a".into(), vec![vec![1], vec![2]]),
                ("node-b".into(), vec![vec![3]]),
            ]
        );
    }

    #[tokio::test]
    async fn fetch_plan_stops_after_first_short_segment() {
        let plan = validate_fetch_plan(
            vec![
                segment("node-a", 1),
                segment("node-b", 1),
                segment("node-c", 1),
            ],
            3,
            "requester",
        )
        .unwrap()
        .unwrap();
        let fetcher = FakeSegmentFetcher {
            calls: Mutex::new(Vec::new()),
            responses: Mutex::new(VecDeque::from([
                vec![fetched_block(1)],
                Vec::new(),
                vec![fetched_block(3)],
            ])),
        };
        let hashes = vec![vec![1], vec![2], vec![3]];

        let (fetched, completed, failure) = execute_fetch_plan(&fetcher, &plan, &hashes).await;

        assert_eq!(fetched.len(), 1);
        assert_eq!(completed, 1);
        assert_eq!(failure, Some((1, 1, 0)));
        assert_eq!(fetcher.calls.lock().unwrap().len(), 2);
    }

    #[test]
    fn plans_block_staging_allocations() {
        let page = PER_SLOT_ALLOC_MIN_BYTES;
        for (name, slots, expected) in [
            (
                "page-first pages get one allocation each",
                vec![slot(0, page, 0), slot(1, page, 0), slot(0, page, 0)],
                vec![
                    group(0, page, &[0]),
                    group(1, page, &[1]),
                    group(0, page, &[2]),
                ],
            ),
            (
                "small split-K/V slots coalesce per NUMA",
                vec![slot(0, 512, 512), slot(1, 256, 0), slot(0, 512, 512)],
                vec![group(0, 2048, &[0, 2]), group(1, 256, &[1])],
            ),
            (
                "mixed sizes keep pages separate from the coalesced group",
                vec![slot(0, 1024, 0), slot(0, page, 0), slot(0, 1024, 0)],
                vec![group(0, 2048, &[0, 2]), group(0, page, &[1])],
            ),
            (
                "empty slots are skipped",
                vec![slot(0, 0, 0), slot(0, 512, 0)],
                vec![group(0, 512, &[1])],
            ),
        ] {
            assert_eq!(
                plan_block_allocations(&slots).expect(name),
                expected,
                "{name}"
            );
        }
    }

    #[test]
    fn split_slot_counts_both_segments_but_skips_null_v() {
        let mut no_v_ptr = slot(0, 512, 512);
        no_v_ptr.v_ptr = 0;
        let plan = plan_block_allocations(&[slot(0, 512, 512), no_v_ptr]).unwrap();
        assert_eq!(plan, vec![group(0, 1536, &[0, 1])]);
    }
}
