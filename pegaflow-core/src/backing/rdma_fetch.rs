// RDMA remote block fetch: MetaServer query -> gRPC QueryBlocksForTransfer
// (locks blocks, returns sizes) -> gRPC PushBlocks (holder RDMA-WRITEs the
// blocks into our pinned memory; the RPC response is the completion signal).

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use log::{debug, info, warn};
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use pegaflow_proto::proto::engine::{
    PushBlockDst, PushBlocksRequest, PushSegmentDst, PushSlotDst, QueryBlocksForTransferRequest,
    QueryBlocksForTransferResponse, RemoteMemoryRegion, TransferBlockInfo,
};
use tonic::transport::{Channel, Endpoint};

use pegaflow_common::NumaNode;

use opentelemetry::KeyValue;

use super::rdma::mr_desc_to_proto;
use super::{AllocateFn, PrefetchResult, RdmaTransport};
use crate::block::{BlockKey, RawBlock, SealedBlock, Segment};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

/// Hard ceiling on one PushBlocks RPC. If this fires, the holder may still
/// have RDMA WRITEs in flight targeting our slabs, so the slabs are leaked
/// instead of recycled (see `leak_allocations`).
const PUSH_TIMEOUT: Duration = Duration::from_secs(60);

/// RDMA remote block fetch backing store.
///
/// When all requested blocks are missing locally, queries MetaServer for their
/// location, picks the best remote node, and asks it to push the blocks into
/// local pinned memory via RDMA WRITE.
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
            remote_addr,
            req_id,
            &self.advertise_addr,
            namespace,
            hashes,
        )
        .await
    }
}

/// Execute a push fetch against a single remote node.
///
/// 1. gRPC QueryBlocksForTransfer (locks blocks remotely, returns sizes)
/// 2. Allocate local NUMA slabs and build the push request
/// 3. gRPC PushBlocks — the holder RDMA-WRITEs into our slabs and releases
///    its transfer lock before responding
/// 4. Build SealedBlocks from the now-filled slabs
#[allow(
    clippy::too_many_arguments,
    reason = "RDMA task arguments are the per-fetch context passed from the scheduler"
)]
async fn rdma_fetch_task(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    grpc_channels: &DashMap<String, EngineClient<Channel>>,
    remote_addr: &str,
    req_id: &str,
    advertise_addr: &str,
    namespace: &str,
    block_hashes: &[Vec<u8>],
) -> PrefetchResult {
    let t0 = Instant::now();

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

    let blocks = response.blocks;
    let total_bytes: u64 = blocks
        .iter()
        .flat_map(|b| &b.slots)
        .map(|s| s.k_size + s.v_size)
        .sum();
    let (result, timing) = match fetch_blocks_via_push(
        rdma,
        allocate_fn,
        client,
        namespace,
        response.transfer_session_id,
        &blocks,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("RDMA push fetch from {remote_addr} failed: {e}");
            core_metrics()
                .rdma_fetch_total
                .add(1, &[KeyValue::new("status", "error")]);
            return Vec::new();
        }
    };

    let elapsed = t0.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let throughput_mib_s = if elapsed.as_secs_f64() > 0.0 {
        mb / elapsed.as_secs_f64()
    } else {
        0.0
    };
    info!(
        "RDMA fetch summary: req_id={req_id} remote={remote_addr} blocks={}/{} slots={} segments={} slabs={} bytes_mib={mb:.1} total_ms={elapsed_ms:.2} tp_mib_s={throughput_mib_s:.0}",
        result.len(),
        block_hashes.len(),
        timing.slot_count,
        timing.segment_count,
        timing.numa_slab_count,
    );
    info!(
        "RDMA fetch stages: req_id={req_id} remote={remote_addr} query_ms={:.2} build_ms={:.2} push_ms={:.2} rebuild_ms={:.2}",
        query_elapsed.as_secs_f64() * 1000.0,
        timing.build.as_secs_f64() * 1000.0,
        timing.push.as_secs_f64() * 1000.0,
        timing.rebuild.as_secs_f64() * 1000.0,
    );
    let m = core_metrics();
    let ok = &[KeyValue::new("status", "ok")];
    m.rdma_fetch_total.add(1, ok);
    m.rdma_fetch_duration_seconds
        .record(elapsed.as_secs_f64(), ok);
    m.rdma_fetch_bytes.add(total_bytes, ok);
    result
}

/// Allocate local memory, send PushBlocks, build SealedBlocks.
async fn fetch_blocks_via_push(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    mut client: EngineClient<Channel>,
    namespace: &str,
    transfer_session_id: String,
    blocks: &[TransferBlockInfo],
) -> Result<(PrefetchResult, FetchTiming), String> {
    if blocks.is_empty() {
        return Ok((Vec::new(), FetchTiming::default()));
    }

    let build_start = Instant::now();
    let bytes_per_numa = sum_segment_bytes_by_numa(blocks)?;
    let mut numa_slabs = allocate_numa_slabs(allocate_fn, bytes_per_numa)?;
    let numa_slab_count = numa_slabs.len();

    // (block_hash, Vec<slot segments>) — for building SealedBlocks afterwards
    let mut block_allocs: Vec<(Vec<u8>, Vec<Vec<SegmentAlloc>>)> = Vec::new();
    let mut slot_count = 0usize;
    let mut segment_count = 0usize;

    // Wire MR table, deduplicated by region base pointer.
    let mut mr_table: Vec<RemoteMemoryRegion> = Vec::new();
    let mut mr_index_by_base: HashMap<u64, u32> = HashMap::new();
    let mut push_blocks: Vec<PushBlockDst> = Vec::new();

    // Allocation and request assembly stay in a sync block so that NonNull
    // pointers (not Send) are dropped before any .await.
    {
        let mut segment_dst = |slabs: &mut HashMap<NumaNode, NumaSlab>,
                               numa: NumaNode,
                               len: usize,
                               kind: &str|
         -> Result<(PushSegmentDst, SegmentAlloc), String> {
            let (local_ptr, alloc) = alloc_segment_from_slab(slabs, numa, len, kind)?;
            let addr = local_ptr.as_ptr() as u64;
            let region = rdma.region_for(addr, len as u64).ok_or_else(|| {
                format!("slab segment 0x{addr:x}+{len} is not in a registered RDMA region")
            })?;
            let desc = region.descriptor();
            let mr_index = *mr_index_by_base.entry(desc.ptr).or_insert_with(|| {
                mr_table.push(mr_desc_to_proto(desc));
                (mr_table.len() - 1) as u32
            });
            Ok((
                PushSegmentDst {
                    mr_index,
                    dst_addr: addr,
                },
                SegmentAlloc {
                    ptr_addr: addr,
                    alloc,
                    size: len,
                },
            ))
        };

        for block_info in blocks {
            slot_count += block_info.slots.len();
            let mut slot_allocs = Vec::with_capacity(block_info.slots.len());
            let mut slot_dsts = Vec::with_capacity(block_info.slots.len());

            for slot in &block_info.slots {
                let mut segments = Vec::new();
                let numa = NumaNode(slot.numa_node);

                let k_dst = if slot.k_size > 0 {
                    let len = usize::try_from(slot.k_size)
                        .map_err(|_| format!("K size exceeds usize: {}", slot.k_size))?;
                    let (dst, alloc) = segment_dst(&mut numa_slabs, numa, len, "K")?;
                    segments.push(alloc);
                    segment_count += 1;
                    Some(dst)
                } else {
                    None
                };

                let v_dst = if slot.v_size > 0 {
                    let len = usize::try_from(slot.v_size)
                        .map_err(|_| format!("V size exceeds usize: {}", slot.v_size))?;
                    let (dst, alloc) = segment_dst(&mut numa_slabs, numa, len, "V")?;
                    segments.push(alloc);
                    segment_count += 1;
                    Some(dst)
                } else {
                    None
                };

                slot_dsts.push(PushSlotDst { k: k_dst, v: v_dst });
                slot_allocs.push(segments);
            }

            block_allocs.push((block_info.block_hash.clone(), slot_allocs));
            push_blocks.push(PushBlockDst {
                block_hash: block_info.block_hash.clone(),
                slots: slot_dsts,
            });
        }
    }
    let build = build_start.elapsed();

    if segment_count == 0 {
        let timing = FetchTiming {
            build,
            slot_count,
            numa_slab_count,
            ..FetchTiming::default()
        };
        return Ok((Vec::new(), timing));
    }

    let push_start = Instant::now();
    let request = PushBlocksRequest {
        transfer_session_id,
        memory_regions: mr_table,
        blocks: push_blocks,
    };
    let push_result = tokio::time::timeout(PUSH_TIMEOUT, client.push_blocks(request)).await;
    let response = match push_result {
        Ok(Ok(response)) => response.into_inner(),
        Ok(Err(status)) => {
            // FAILED_PRECONDITION is the holder's contract for "rejected
            // before submitting any WRITE": the slabs were never touched and
            // dropping them recycles the memory. Any other failure leaves
            // the WRITE state unknown and the slabs must be leaked.
            if status.code() == tonic::Code::FailedPrecondition {
                return Err(format!("PushBlocks rejected by holder: {status}"));
            }
            leak_allocations(numa_slabs, block_allocs);
            return Err(format!("PushBlocks RPC failed: {status}"));
        }
        Err(_) => {
            leak_allocations(numa_slabs, block_allocs);
            return Err(format!("PushBlocks timed out after {PUSH_TIMEOUT:?}"));
        }
    };
    if let Some(st) = &response.status
        && !st.ok
    {
        leak_allocations(numa_slabs, block_allocs);
        return Err(format!("PushBlocks rejected by holder: {}", st.message));
    }
    let push = push_start.elapsed();

    // Build SealedBlocks from the now-filled memory.
    let rebuild_start = Instant::now();
    let mut result: PrefetchResult = Vec::with_capacity(block_allocs.len());
    for (hash, slot_allocs) in block_allocs {
        let key = BlockKey::new(namespace.to_string(), hash);
        let slots: Vec<Arc<RawBlock>> = slot_allocs
            .into_iter()
            .map(|segs| {
                let segments: Vec<Segment> = segs
                    .into_iter()
                    .map(|sa| {
                        let ptr = NonNull::new(sa.ptr_addr as *mut u8)
                            .expect("slab segment pointer must be non-null");
                        Segment::new(ptr, sa.size, sa.alloc)
                    })
                    .collect();
                Arc::new(RawBlock::new(segments))
            })
            .collect();
        let sealed = Arc::new(SealedBlock::from_slots(slots));
        result.push((key, sealed));
    }

    let timing = FetchTiming {
        build,
        push,
        rebuild: rebuild_start.elapsed(),
        segment_count,
        slot_count,
        numa_slab_count,
    };
    Ok((result, timing))
}

/// Leak the fetch's pinned allocations after a failed push.
///
/// After a push failure we cannot know whether the holder still has RDMA
/// WRITEs in flight targeting these slabs (the gRPC path and the RDMA fabric
/// can fail independently). Returning the memory to the pool could let a
/// late WRITE corrupt an unrelated allocation, so we deliberately leak it.
#[allow(
    clippy::mem_forget,
    reason = "leak is the safety mechanism: freed memory could be corrupted by in-flight WRITEs"
)]
fn leak_allocations(
    numa_slabs: HashMap<NumaNode, NumaSlab>,
    block_allocs: Vec<(Vec<u8>, Vec<Vec<SegmentAlloc>>)>,
) {
    let leaked: usize = numa_slabs.values().map(|s| s.capacity).sum();
    log::error!(
        "Leaking {leaked} bytes of pinned memory across {} slab(s): push failed and remote WRITEs may still be in flight",
        numa_slabs.len()
    );
    core_metrics()
        .rdma_fetch_leaked_bytes
        .add(leaked as u64, &[]);
    std::mem::forget(numa_slabs);
    std::mem::forget(block_allocs);
}

fn sum_segment_bytes_by_numa(
    blocks: &[TransferBlockInfo],
) -> Result<HashMap<NumaNode, u64>, String> {
    let mut bytes_per_numa: HashMap<NumaNode, u64> = HashMap::new();
    for block_info in blocks {
        for slot in &block_info.slots {
            let numa = NumaNode(slot.numa_node);
            if slot.k_size > 0 {
                let total = bytes_per_numa.entry(numa).or_insert(0);
                *total = total.checked_add(slot.k_size).ok_or_else(|| {
                    format!("numa bytes overflow while summing K segments on {numa}")
                })?;
            }
            if slot.v_size > 0 {
                let total = bytes_per_numa.entry(numa).or_insert(0);
                *total = total.checked_add(slot.v_size).ok_or_else(|| {
                    format!("numa bytes overflow while summing V segments on {numa}")
                })?;
            }
        }
    }
    Ok(bytes_per_numa)
}

fn allocate_numa_slabs(
    allocate_fn: &AllocateFn,
    bytes_per_numa: HashMap<NumaNode, u64>,
) -> Result<HashMap<NumaNode, NumaSlab>, String> {
    let mut numa_slabs: HashMap<NumaNode, NumaSlab> = HashMap::new();
    for (numa, total_bytes) in bytes_per_numa {
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
                next_offset: 0,
                capacity,
            },
        );
    }
    Ok(numa_slabs)
}

fn alloc_segment_from_slab(
    slabs: &mut HashMap<NumaNode, NumaSlab>,
    numa: NumaNode,
    len: usize,
    segment_kind: &str,
) -> Result<(NonNull<u8>, Arc<crate::pinned_pool::PinnedAllocation>), String> {
    let slab = slabs
        .get_mut(&numa)
        .ok_or_else(|| format!("missing slab for {numa} while allocating {segment_kind}"))?;
    slab.allocate(len, segment_kind)
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
struct FetchTiming {
    build: Duration,
    push: Duration,
    rebuild: Duration,
    segment_count: usize,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use pegaflow_proto::proto::engine::TransferSlotInfo;

    fn slot(k_size: u64, v_size: u64, numa: u32) -> TransferSlotInfo {
        TransferSlotInfo {
            k_size,
            v_size,
            numa_node: numa,
        }
    }

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
    fn sum_segment_bytes_by_numa_aggregates_k_and_v() {
        let blocks = vec![
            TransferBlockInfo {
                block_hash: vec![1],
                slots: vec![
                    slot(100, 0, 0),   // contiguous
                    slot(200, 300, 0), // split KV
                ],
            },
            TransferBlockInfo {
                block_hash: vec![2],
                slots: vec![slot(400, 500, 1)],
            },
        ];

        let totals = sum_segment_bytes_by_numa(&blocks).expect("sum bytes");
        assert_eq!(totals.get(&NumaNode(0)), Some(&600)); // 100 + (200 + 300)
        assert_eq!(totals.get(&NumaNode(1)), Some(&900)); // 400 + 500
    }

    #[test]
    fn allocate_numa_slabs_calls_allocator_once_per_numa() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));

        let bytes_per_numa = HashMap::from([(NumaNode(0), 1024_u64), (NumaNode(1), 2048_u64)]);
        let slabs = allocate_numa_slabs(&allocate_fn, bytes_per_numa).expect("allocate slabs");

        assert_eq!(slabs.len(), 2);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn slab_allocation_is_contiguous_and_bounded() {
        let calls = Arc::new(AtomicUsize::new(0));
        let allocate_fn = test_allocate_fn(Arc::clone(&calls));

        let mut slabs = allocate_numa_slabs(&allocate_fn, HashMap::from([(NumaNode(0), 1024_u64)]))
            .expect("allocate slabs");
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let (p1, _a1) =
            alloc_segment_from_slab(&mut slabs, NumaNode(0), 256, "K").expect("alloc p1");
        let (p2, _a2) =
            alloc_segment_from_slab(&mut slabs, NumaNode(0), 128, "V").expect("alloc p2");
        assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 256);

        let overflow = alloc_segment_from_slab(&mut slabs, NumaNode(0), 1024, "K");
        assert!(overflow.is_err());
    }
}
