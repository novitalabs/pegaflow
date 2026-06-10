// RDMA transport for cross-node KV transfer, built on the v2 transfer engine.
//
// One engine per NUMA node that has RDMA NICs; each engine drives that node's
// NICs from a worker thread pinned to a NUMA-local CPU. All pinned-pool
// regions are registered with remote access at startup, so per-fetch slab
// allocations never touch MR registration. Connections to peers are
// established lazily by the engine's UD control plane — no out-of-band
// handshake is needed; peers only exchange MR descriptors over gRPC.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use log::{info, warn};
use pegaflow_common::NumaNode;
use pegaflow_transfer::{
    AsyncTransferEngine as _, Device, DomainAddress, GroupTransferRouting, MemoryRegionDescriptor,
    MemoryRegionHandle, MemoryRegionRemoteKey, RdmaEngine as _, ScatterTarget,
    ScatterTransferRequest, SmallVec, TransferEngine, TransferEngineBuilder, TransferRequest,
    detect_host_topology,
};

use crate::PushBlocksError;
use crate::pinned_pool::PinnedAllocator;

/// One v2 transfer engine driving the RDMA NICs of a single NUMA node.
struct NumaEngine {
    numa: NumaNode,
    engine: TransferEngine,
    num_domains: usize,
}

/// A pinned-pool region registered for RDMA (local + remote access).
pub(crate) struct RegisteredRegion {
    base: u64,
    len: usize,
    engine_idx: usize,
    handle: MemoryRegionHandle,
    desc: MemoryRegionDescriptor,
}

impl RegisteredRegion {
    fn contains(&self, addr: u64, len: u64) -> bool {
        addr >= self.base && addr.saturating_add(len) <= self.base + self.len as u64
    }

    /// MR descriptor advertised to remote peers: base pointer plus one
    /// (domain address, rkey) pair per NIC of the owning engine.
    pub(crate) fn descriptor(&self) -> &MemoryRegionDescriptor {
        &self.desc
    }
}

/// One segment of a push transfer: holder-local source bytes and the
/// requester-side destination (index into the request's MR table + absolute
/// address). Indexing instead of embedding the descriptor keeps the build
/// loop free of per-segment rkey-list clones.
pub(crate) struct PushSegment {
    pub(crate) src_addr: u64,
    pub(crate) len: u64,
    pub(crate) dst_mr_index: u32,
    pub(crate) dst_addr: u64,
}

pub(crate) struct RdmaTransport {
    engines: Vec<NumaEngine>,
    regions: Vec<RegisteredRegion>,
}

impl RdmaTransport {
    /// Create per-NUMA transfer engines and register all pinned memory regions.
    fn new(nic_names: &[String], allocator: &PinnedAllocator) -> Result<Self, String> {
        let t0 = Instant::now();

        let groups = detect_host_topology().map_err(|e| e.to_string())?;
        let requested: Vec<&str> = nic_names.iter().map(String::as_str).collect();
        let available: Vec<String> = groups
            .iter()
            .flat_map(|g| g.domains.iter().map(|d| d.name().into_owned()))
            .collect();
        for name in &requested {
            if !available.iter().any(|a| a == name) {
                return Err(format!(
                    "RDMA NIC {name} not found; available NICs: {available:?}"
                ));
            }
        }

        let mut engines = Vec::new();
        for group in groups {
            let domains: Vec<_> = group
                .domains
                .iter()
                .filter(|d| requested.is_empty() || requested.iter().any(|n| *n == d.name()))
                .cloned()
                .collect();
            if domains.is_empty() {
                continue;
            }
            let num_domains = domains.len();
            // Pin the engine's polling worker to the last CPU of the NUMA
            // group; low CPU indices are favoured by other pinned threads.
            let pin_cpu = group.cpus.last().copied();
            let engine =
                TransferEngineBuilder::build_host(domains, pin_cpu).map_err(|e| e.to_string())?;
            engines.push(NumaEngine {
                numa: NumaNode(group.numa as u32),
                engine,
                num_domains,
            });
        }
        if engines.is_empty() {
            return Err(format!(
                "no RDMA NICs selected (requested={requested:?}, available={available:?})"
            ));
        }

        let mut regions = Vec::new();
        for (ptr, len, numa) in allocator.memory_regions_with_numa() {
            let engine_idx = engines
                .iter()
                .position(|e| e.numa == numa)
                .unwrap_or_else(|| {
                    warn!(
                        "pinned region on {numa} has no NUMA-local RDMA engine; using {}",
                        engines[0].numa
                    );
                    0
                });
            let (handle, desc) = engines[engine_idx]
                .engine
                .register_memory_allow_remote(ptr.cast(), len, Device::Host)
                .map_err(|e| format!("RDMA register {len} bytes on {numa}: {e}"))?;
            regions.push(RegisteredRegion {
                base: ptr.as_ptr() as u64,
                len,
                engine_idx,
                handle,
                desc,
            });
        }

        info!(
            "RDMA transport initialised: engines={} ({}), regions={}, elapsed={:?}",
            engines.len(),
            engines
                .iter()
                .map(|e| format!("{}x{}", e.numa, e.num_domains))
                .collect::<Vec<_>>()
                .join(","),
            regions.len(),
            t0.elapsed(),
        );

        Ok(Self { engines, regions })
    }

    /// Total RDMA domains (NICs) across all engines.
    pub(crate) fn num_domains(&self) -> usize {
        self.engines.iter().map(|e| e.num_domains).sum()
    }

    /// Find the registered region containing `[addr, addr+len)`.
    pub(crate) fn region_for(&self, addr: u64, len: u64) -> Option<&RegisteredRegion> {
        self.regions.iter().find(|r| r.contains(addr, len))
    }

    /// RDMA-WRITE `segments` into remote memory. Resolves each segment's
    /// source region, groups segments per engine/region, and submits one
    /// scatter per group sharded across that engine's NICs. Returns the total
    /// bytes written once every WRITE has completed (RC completion implies
    /// remote placement).
    ///
    /// Adjacent segments whose source and destination are both contiguous are
    /// coalesced into one WRITE: TP-sliced KV blocks arrive as ~4 KiB
    /// segments and push throughput is bounded by WR posting rate, not
    /// bandwidth. The requester assigns destinations from a bump allocator in
    /// the same iteration order, so coalescing only depends on holder-side
    /// source adjacency.
    ///
    /// All segments are validated before anything is submitted, so an error
    /// from the build phase is a clean [`PushBlocksError::Rejected`].
    pub(crate) async fn push_segments(
        &self,
        mr_descs: &[MemoryRegionDescriptor],
        segments: Vec<PushSegment>,
    ) -> Result<u64, PushBlocksError> {
        let mut groups: HashMap<usize, Vec<(u32, ScatterTarget)>> = HashMap::new();
        let mut total_bytes = 0u64;
        let raw_segments = segments.len();
        // Adjacency diagnostics: how many consecutive segment pairs are
        // contiguous on each side. Coalescing needs both.
        let mut src_adjacent = 0usize;
        let mut dst_adjacent = 0usize;
        let mut prev_ends: Option<(u64, u64)> = None;
        // Consecutive segments almost always share a region (slot-major
        // order makes sources adjacent), so try the previous hit first.
        let mut last_region_idx = 0usize;
        for seg in segments {
            if let Some((src_end, dst_end)) = prev_ends {
                src_adjacent += usize::from(src_end == seg.src_addr);
                dst_adjacent += usize::from(dst_end == seg.dst_addr);
            }
            prev_ends = Some((seg.src_addr + seg.len, seg.dst_addr + seg.len));
            let region_idx = if self
                .regions
                .get(last_region_idx)
                .is_some_and(|r| r.contains(seg.src_addr, seg.len))
            {
                last_region_idx
            } else {
                self.regions
                    .iter()
                    .position(|r| r.contains(seg.src_addr, seg.len))
                    .ok_or_else(|| {
                        PushBlocksError::Rejected(format!(
                            "push source 0x{:x}+{} is not in any registered region",
                            seg.src_addr, seg.len
                        ))
                    })?
            };
            last_region_idx = region_idx;
            let region = &self.regions[region_idx];
            let dst_mr = mr_descs.get(seg.dst_mr_index as usize).ok_or_else(|| {
                PushBlocksError::Rejected(format!(
                    "mr_index {} out of bounds ({} regions in request)",
                    seg.dst_mr_index,
                    mr_descs.len()
                ))
            })?;
            if self.engines[region.engine_idx].num_domains > dst_mr.addr_rkey_list.len() {
                return Err(PushBlocksError::Rejected(format!(
                    "requester MR has {} domain rkeys but local engine has {} NICs; \
                     NIC counts must match 1:1 per NUMA group",
                    dst_mr.addr_rkey_list.len(),
                    self.engines[region.engine_idx].num_domains
                )));
            }
            let dst_offset = seg.dst_addr.checked_sub(dst_mr.ptr).ok_or_else(|| {
                PushBlocksError::Rejected(format!(
                    "push destination 0x{:x} is below its MR base 0x{:x}",
                    seg.dst_addr, dst_mr.ptr
                ))
            })?;
            total_bytes += seg.len;
            let src_offset = seg.src_addr - region.base;
            let targets = groups.entry(region_idx).or_default();
            match targets.last_mut() {
                Some((last_mr_index, last))
                    if *last_mr_index == seg.dst_mr_index
                        && last.src_offset + last.length == src_offset
                        && last.dst_offset + last.length == dst_offset =>
                {
                    last.length += seg.len;
                }
                _ => targets.push((
                    seg.dst_mr_index,
                    ScatterTarget {
                        dst_mr: dst_mr.clone(),
                        length: seg.len,
                        src_offset,
                        dst_offset,
                    },
                )),
            }
        }

        let coalesced: usize = groups.values().map(Vec::len).sum();
        info!(
            "RDMA push: segments={raw_segments} coalesced={coalesced} bytes={total_bytes} groups={} src_adjacent={src_adjacent} dst_adjacent={dst_adjacent}",
            groups.len()
        );

        let transfers = groups.into_iter().map(|(region_idx, targets)| {
            let targets: Vec<ScatterTarget> = targets.into_iter().map(|(_, t)| t).collect();
            let region = &self.regions[region_idx];
            let engine = &self.engines[region.engine_idx].engine;
            // Coalesced targets are ~MB-sized runs, one per (slot, K/V); whole
            // targets round-robined across NICs beat byte-sharding each one
            // into N sub-MTU-pipeline writes.
            engine.submit_transfer_async(TransferRequest::Scatter(ScatterTransferRequest {
                src_mr: region.handle,
                dst_handle: None,
                dsts: Arc::new(targets),
                imm_data: None,
                domain: GroupTransferRouting::AllDomainsShardPeers,
            }))
        });
        futures::future::try_join_all(transfers)
            .await
            .map_err(|e| PushBlocksError::Failed(format!("RDMA push failed: {e}")))?;
        Ok(total_bytes)
    }
}

impl Drop for RdmaTransport {
    fn drop(&mut self) {
        for region in &self.regions {
            let engine = &self.engines[region.engine_idx].engine;
            if let Err(e) = engine.unregister_memory(region.handle.ptr) {
                warn!(
                    "Failed to unregister RDMA region at 0x{:x}: {e}",
                    region.base
                );
            }
        }
        for e in &self.engines {
            e.engine.stop();
        }
    }
}

/// Convert a wire `RemoteMemoryRegion` into an engine MR descriptor.
pub(crate) fn mr_desc_from_proto(
    mr: &pegaflow_proto::proto::engine::RemoteMemoryRegion,
) -> MemoryRegionDescriptor {
    let mut addr_rkey_list = SmallVec::new();
    for rkey in &mr.rkeys {
        addr_rkey_list.push((
            DomainAddress(bytes::Bytes::from(rkey.domain_address.clone())),
            MemoryRegionRemoteKey(rkey.rkey),
        ));
    }
    MemoryRegionDescriptor {
        ptr: mr.base_ptr,
        addr_rkey_list,
    }
}

/// Convert a local MR descriptor into its wire form for the push request.
pub(crate) fn mr_desc_to_proto(
    desc: &MemoryRegionDescriptor,
) -> pegaflow_proto::proto::engine::RemoteMemoryRegion {
    pegaflow_proto::proto::engine::RemoteMemoryRegion {
        base_ptr: desc.ptr,
        rkeys: desc
            .addr_rkey_list
            .iter()
            .map(
                |(addr, rkey)| pegaflow_proto::proto::engine::RemoteDomainRkey {
                    domain_address: addr.0.to_vec(),
                    rkey: rkey.0,
                },
            )
            .collect(),
    }
}

/// Create an [`RdmaTransport`].
pub(crate) fn new_rdma(
    nic_names: &[String],
    allocator: &PinnedAllocator,
) -> Result<Arc<RdmaTransport>, String> {
    RdmaTransport::new(nic_names, allocator)
        .map(Arc::new)
        .map_err(|e| format!("Failed to initialise RDMA transport (nics={nic_names:?}): {e}"))
}
