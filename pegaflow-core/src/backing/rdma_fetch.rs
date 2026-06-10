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
    PushBlocksRequest, PushSlabDst, QueryBlocksForTransferRequest, QueryBlocksForTransferResponse,
    RemoteMemoryRegion, TransferSlotInfo,
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

    let block_count = response.block_count as usize;
    if block_count > block_hashes.len() {
        warn!(
            "Remote {remote_addr} locked {block_count} blocks but only {} were requested",
            block_hashes.len()
        );
        core_metrics()
            .rdma_fetch_total
            .add(1, &[KeyValue::new("status", "error")]);
        return Vec::new();
    }
    let template = response.slot_template;
    // Remote-controlled values; saturate instead of trusting them — the
    // checked path in sum_segment_bytes_by_numa is the real gate.
    let per_block_bytes = template.iter().fold(0u64, |acc, s| {
        acc.saturating_add(s.k_size).saturating_add(s.v_size)
    });
    let total_bytes = per_block_bytes.saturating_mul(block_count as u64);
    let (result, timing) = match fetch_blocks_via_push(
        rdma,
        allocate_fn,
        client,
        namespace,
        response.transfer_session_id,
        &block_hashes[..block_count],
        &template,
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
///
/// `hashes` is the locked prefix in request order and `template` the
/// per-slot layout shared by every block (both from QueryBlocksForTransfer).
/// Destinations are assigned by slot-major bump allocation over the
/// template; the wire request carries only the slab bases, and the holder
/// replays the identical allocation to derive every address.
async fn fetch_blocks_via_push(
    rdma: &RdmaTransport,
    allocate_fn: &AllocateFn,
    mut client: EngineClient<Channel>,
    namespace: &str,
    transfer_session_id: String,
    hashes: &[Vec<u8>],
    template: &[TransferSlotInfo],
) -> Result<(PrefetchResult, FetchTiming), String> {
    if hashes.is_empty() {
        return Ok((Vec::new(), FetchTiming::default()));
    }

    let build_start = Instant::now();
    let bytes_per_numa = sum_segment_bytes_by_numa(template, hashes.len())?;
    let mut numa_slabs = allocate_numa_slabs(allocate_fn, bytes_per_numa)?;
    let numa_slab_count = numa_slabs.len();
    let slab_alloc_elapsed = build_start.elapsed();

    let slot_count = hashes.len() * template.len();
    let mut segment_count = 0usize;

    // Wire MR table, deduplicated by region base pointer.
    let mut mr_table: Vec<RemoteMemoryRegion> = Vec::new();
    let mut mr_index_by_base: HashMap<u64, u32> = HashMap::new();
    let mut push_slabs: Vec<PushSlabDst> = Vec::new();

    // Wire slab table: one entry per NUMA slab.
    for (numa, slab) in &numa_slabs {
        let base = slab.allocation.as_non_null().as_ptr() as u64;
        let capacity = slab.capacity as u64;
        let region = rdma.region_for(base, capacity).ok_or_else(|| {
            format!("slab 0x{base:x}+{capacity} is not in a registered RDMA region")
        })?;
        let desc = region.descriptor();
        let mr_index = *mr_index_by_base.entry(desc.ptr).or_insert_with(|| {
            mr_table.push(mr_desc_to_proto(desc));
            (mr_table.len() - 1) as u32
        });
        push_slabs.push(PushSlabDst {
            numa_node: numa.0,
            mr_index,
            base_addr: base,
            capacity,
        });
    }

    // Slot-major assignment, K run before V run: the holder stores each
    // layer's K and V in separate allocations with consecutive blocks
    // adjacent inside, so visiting (slot, K) across all blocks gives the
    // holder source-contiguous runs it can coalesce into single WRITEs.
    // The holder replays this exact order to derive the addresses.
    //
    // One slab allocation covers a whole (slot, K/V) run — the per-segment
    // addresses inside it are `base + block_idx * len`, exactly the
    // per-segment bump the holder replays.
    let mut runs: Vec<(Option<RunDesc>, Option<RunDesc>)> = Vec::with_capacity(template.len());
    for slot in template {
        let numa = NumaNode(slot.numa_node);
        let k = alloc_run(&mut numa_slabs, numa, slot.k_size, hashes.len(), "K")?;
        let v = alloc_run(&mut numa_slabs, numa, slot.v_size, hashes.len(), "V")?;
        if k.is_none() && v.is_none() {
            return Err(format!(
                "slot template entry has zero-size K and V on {numa}"
            ));
        }
        segment_count += (k.is_some() as usize + v.is_some() as usize) * hashes.len();
        runs.push((k, v));
    }

    if segment_count == 0 {
        let timing = FetchTiming {
            build: build_start.elapsed(),
            slot_count,
            numa_slab_count,
            ..FetchTiming::default()
        };
        return Ok((Vec::new(), timing));
    }

    // Materialize per-block SealedBlocks from the run table, blocks fanned
    // out across threads (~600k Segment/Arc constructions otherwise dominate
    // the fetch's CPU time). Only the cache insert waits for the push.
    let sealed_blocks: Vec<Arc<SealedBlock>> = {
        let runs = &runs;
        let build_block = move |block_idx: usize| -> Arc<SealedBlock> {
            let slots: Vec<Arc<RawBlock>> = runs
                .iter()
                .map(|(k, v)| {
                    let seg = |r: &RunDesc| r.segment(block_idx);
                    Arc::new(match (k, v) {
                        (Some(k), Some(v)) => RawBlock::two_segments(seg(k), seg(v)),
                        (Some(s), None) | (None, Some(s)) => RawBlock::single_segment(seg(s)),
                        (None, None) => unreachable!("rejected above"),
                    })
                })
                .collect();
            Arc::new(SealedBlock::from_slots(slots))
        };
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 8)
            .min(hashes.len());
        let chunk = hashes.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|t| {
                    let lo = t * chunk;
                    let hi = ((t + 1) * chunk).min(hashes.len());
                    scope.spawn(move || (lo..hi).map(build_block).collect::<Vec<_>>())
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|w| w.join().expect("block build worker panicked"))
                .collect()
        })
    };
    let build = build_start.elapsed();
    debug!(
        "RDMA fetch build phases: slab_alloc_ms={:.2} total_ms={:.2}",
        slab_alloc_elapsed.as_secs_f64() * 1000.0,
        build.as_secs_f64() * 1000.0,
    );

    let push_start = Instant::now();
    let request = PushBlocksRequest {
        transfer_session_id,
        memory_regions: mr_table,
        slabs: push_slabs,
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
            leak_allocations(numa_slabs, sealed_blocks);
            return Err(format!("PushBlocks RPC failed: {status}"));
        }
        Err(_) => {
            leak_allocations(numa_slabs, sealed_blocks);
            return Err(format!("PushBlocks timed out after {PUSH_TIMEOUT:?}"));
        }
    };
    if let Some(st) = &response.status
        && !st.ok
    {
        leak_allocations(numa_slabs, sealed_blocks);
        return Err(format!("PushBlocks rejected by holder: {}", st.message));
    }
    let push = push_start.elapsed();

    // The memory is filled; key the prebuilt SealedBlocks for cache insert.
    let rebuild_start = Instant::now();
    let result: PrefetchResult = hashes
        .iter()
        .zip(sealed_blocks)
        .map(|(hash, sealed)| (BlockKey::new(namespace.to_string(), hash.clone()), sealed))
        .collect();

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
fn leak_allocations(numa_slabs: HashMap<NumaNode, NumaSlab>, sealed_blocks: Vec<Arc<SealedBlock>>) {
    let leaked: usize = numa_slabs.values().map(|s| s.capacity).sum();
    log::error!(
        "Leaking {leaked} bytes of pinned memory across {} slab(s): push failed and remote WRITEs may still be in flight",
        numa_slabs.len()
    );
    core_metrics()
        .rdma_fetch_leaked_bytes
        .add(leaked as u64, &[]);
    std::mem::forget(numa_slabs);
    std::mem::forget(sealed_blocks);
}

/// One (slot, K/V) run of equal-size segments for all blocks, carved from a
/// NUMA slab in a single bump allocation.
struct RunDesc {
    base: u64,
    len: usize,
    alloc: Arc<crate::pinned_pool::PinnedAllocation>,
}

impl RunDesc {
    fn segment(&self, block_idx: usize) -> Segment {
        let addr = self.base + (block_idx * self.len) as u64;
        let ptr = NonNull::new(addr as *mut u8).expect("run segment pointer must be non-null");
        Segment::new(ptr, self.len, Arc::clone(&self.alloc))
    }
}

/// Allocate one slot's run (`count` segments of `size` bytes) from the slab
/// on `numa`. Returns `None` for absent segments (`size == 0`).
fn alloc_run(
    slabs: &mut HashMap<NumaNode, NumaSlab>,
    numa: NumaNode,
    size: u64,
    count: usize,
    kind: &str,
) -> Result<Option<RunDesc>, String> {
    if size == 0 {
        return Ok(None);
    }
    let len = usize::try_from(size).map_err(|_| format!("{kind} size exceeds usize: {size}"))?;
    let total = len
        .checked_mul(count)
        .ok_or_else(|| format!("{kind} run size overflows: {len} x {count}"))?;
    let slab = slabs
        .get_mut(&numa)
        .ok_or_else(|| format!("missing slab for {numa} while allocating {kind}"))?;
    let (ptr, alloc) = slab.allocate(total, kind)?;
    Ok(Some(RunDesc {
        base: ptr.as_ptr() as u64,
        len,
        alloc,
    }))
}

fn sum_segment_bytes_by_numa(
    template: &[TransferSlotInfo],
    block_count: usize,
) -> Result<HashMap<NumaNode, u64>, String> {
    let mut bytes_per_numa: HashMap<NumaNode, u64> = HashMap::new();
    for slot in template {
        let numa = NumaNode(slot.numa_node);
        let per_block = slot
            .k_size
            .checked_add(slot.v_size)
            .ok_or_else(|| format!("slot bytes overflow on {numa}"))?;
        let slot_total = per_block
            .checked_mul(block_count as u64)
            .ok_or_else(|| format!("numa bytes overflow on {numa}"))?;
        let total = bytes_per_numa.entry(numa).or_insert(0);
        *total = total
            .checked_add(slot_total)
            .ok_or_else(|| format!("numa bytes overflow on {numa}"))?;
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
        let template = vec![
            slot(100, 0, 0),   // contiguous
            slot(200, 300, 0), // split KV
            slot(400, 500, 1),
        ];

        let totals = sum_segment_bytes_by_numa(&template, 2).expect("sum bytes");
        assert_eq!(totals.get(&NumaNode(0)), Some(&1200)); // (100 + 200 + 300) * 2 blocks
        assert_eq!(totals.get(&NumaNode(1)), Some(&1800)); // (400 + 500) * 2 blocks
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

        let slab = slabs.get_mut(&NumaNode(0)).expect("slab for numa 0");
        let (p1, _a1) = slab.allocate(256, "K").expect("alloc p1");
        let (p2, _a2) = slab.allocate(128, "V").expect("alloc p2");
        assert_eq!(p2.as_ptr() as usize - p1.as_ptr() as usize, 256);

        let overflow = slab.allocate(1024, "K");
        assert!(overflow.is_err());
    }
}
