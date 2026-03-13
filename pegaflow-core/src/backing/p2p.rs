// ============================================================================
// P2P Backing Store with RDMA Read Path
//
// Implements BackingStore for cross-node block transfer:
// - ingest_batch: fire-and-forget hash registration to MetaServer
// - submit_prefix: MetaServer query → AcquireLease → RDMA READ → ReleaseLease
// ============================================================================

use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};
use tokio::runtime::Handle;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;
use tonic::transport::{Channel, Endpoint};

use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use pegaflow_proto::proto::engine::rdma_transfer_client::RdmaTransferClient;
use pegaflow_proto::proto::engine::{
    AcquireLeaseRequest, InsertBlockHashesRequest, QueryBlockHashesRequest, ReleaseLeaseRequest,
};
use pegaflow_transfer::MooncakeTransferEngine;

use crate::block::{BlockKey, SealedBlock};
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;
use crate::seal_offload::SlotMeta;

use super::{AllocateFn, BackingStore, BackingStoreKind, BakingStoreConfig, PrefetchResult};

/// Timeout for a single MetaServer query RPC.
const QUERY_TIMEOUT: Duration = Duration::from_millis(500);

/// Timeout for AcquireLease/ReleaseLease RPCs.
const LEASE_RPC_TIMEOUT: Duration = Duration::from_secs(5);

/// Default lease duration requested (seconds).
const DEFAULT_LEASE_DURATION_SECS: u32 = 60;

// ============================================================================
// Insert actor command
// ============================================================================

struct InsertCmd {
    namespace: String,
    block_hashes: Vec<Vec<u8>>,
}

// ============================================================================
// Per-node RDMA transfer context
// ============================================================================

/// Blocks owned by a single remote node, ready for RDMA transfer.
struct NodeTransferBatch {
    /// gRPC address of the owning node (e.g. "http://10.0.0.1:50055").
    node_addr: String,
    /// Block hashes on this node, in prefix order.
    block_hashes: Vec<Vec<u8>>,
    /// Corresponding BlockKeys (parallel to block_hashes).
    block_keys: Vec<BlockKey>,
}

// ============================================================================
// P2pBakingStore
// ============================================================================

pub(crate) struct P2pBakingStore {
    insert_tx: UnboundedSender<InsertCmd>,
    /// gRPC client for MetaServer queries. Channel connects lazily on first RPC.
    query_client: MetaServerClient<Channel>,
    /// RDMA transfer engine for cross-node block reads.
    transfer_engine: Arc<MooncakeTransferEngine>,
    /// Allocator for local pinned memory (RDMA read destination buffers).
    allocate_fn: AllocateFn,
    /// This node's UUID string for lease requester identity.
    node_id: String,
}

impl P2pBakingStore {
    fn create(
        config: BakingStoreConfig,
        allocate_fn: AllocateFn,
        transfer_engine: Arc<MooncakeTransferEngine>,
    ) -> Option<Arc<dyn BackingStore>> {
        let handle = match Handle::try_current() {
            Ok(handle) => handle,
            Err(err) => {
                error!(
                    "Failed to initialize P2P baking store: no Tokio runtime available: {}",
                    err
                );
                return None;
            }
        };

        let (insert_tx, insert_rx) = mpsc::unbounded_channel();

        let coordinator = config.p2p_coordinator_addr.clone();
        let node = config.p2p_node_addr.clone();
        handle.spawn(insert_actor(insert_rx, coordinator.clone(), node));

        // Lazy channel: no TCP connection until the first RPC call.
        let channel = Endpoint::from_shared(coordinator)
            .expect("invalid coordinator address")
            .connect_lazy();
        let query_client = MetaServerClient::new(channel);

        info!(
            "P2P baking store configured (coordinator={}, node={})",
            config.p2p_coordinator_addr, config.p2p_node_addr
        );

        Some(Arc::new(Self {
            insert_tx,
            query_client,
            transfer_engine,
            allocate_fn,
            node_id: config.node_id,
        }))
    }
}

impl BackingStore for P2pBakingStore {
    fn kind(&self) -> BackingStoreKind {
        BackingStoreKind::P2p
    }

    fn ingest_batch(&self, blocks: Vec<(BlockKey, Weak<SealedBlock>)>) {
        if blocks.is_empty() {
            return;
        }
        let mut grouped: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for (key, _) in blocks {
            grouped.entry(key.namespace).or_default().push(key.hash);
        }
        for (namespace, block_hashes) in grouped {
            let _ = self.insert_tx.send(InsertCmd {
                namespace,
                block_hashes,
            });
        }
    }

    fn submit_prefix(&self, keys: Vec<BlockKey>) -> (usize, oneshot::Receiver<PrefetchResult>) {
        let (done_tx, done_rx) = oneshot::channel();

        if keys.is_empty() {
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        }

        // Extract namespace (all keys in a prefix share the same namespace).
        let namespace = keys[0].namespace.clone();
        let block_hashes: Vec<Vec<u8>> = keys.iter().map(|k| k.hash.clone()).collect();

        // Query MetaServer to discover which remote nodes own these blocks.
        let node_batches = match self.query_metaserver_prefix(&namespace, &block_hashes, &keys) {
            Some(batches) if !batches.is_empty() => batches,
            _ => {
                // No remote blocks found or query failed.
                let _ = done_tx.send(Vec::new());
                return (0, done_rx);
            }
        };

        // Count how many prefix blocks were found on remote nodes.
        let found: usize = node_batches.iter().map(|b| b.block_hashes.len()).sum();

        // Spawn async task for RDMA transfer.
        let engine = Arc::clone(&self.transfer_engine);
        let allocate_fn = self.allocate_fn.clone();
        let node_id = self.node_id.clone();

        tokio::spawn(async move {
            let results =
                rdma_transfer_task(engine, allocate_fn, node_id, namespace, node_batches).await;
            let _ = done_tx.send(results);
        });

        (found, done_rx)
    }
}

// ============================================================================
// MetaServer prefix query (synchronous, called from submit_prefix)
// ============================================================================

impl P2pBakingStore {
    /// Query MetaServer for block ownership, returning batches grouped by owning node.
    ///
    /// Preserves prefix order: stops at the first hash NOT found on any remote node.
    fn query_metaserver_prefix(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        keys: &[BlockKey],
    ) -> Option<Vec<NodeTransferBatch>> {
        tokio::task::block_in_place(|| {
            Handle::current()
                .block_on(async { self.do_prefix_query(namespace, block_hashes, keys).await })
        })
    }

    async fn do_prefix_query(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        keys: &[BlockKey],
    ) -> Option<Vec<NodeTransferBatch>> {
        let mut client = self.query_client.clone();

        let req = QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: block_hashes.to_vec(),
        };
        let t = Instant::now();
        let resp = match tokio::time::timeout(QUERY_TIMEOUT, client.query_block_hashes(req)).await {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(err)) => {
                warn!("p2p prefix query rpc failed: {}", err);
                return None;
            }
            Err(_) => {
                warn!(
                    "p2p prefix query timed out: ns={} queried={}",
                    namespace,
                    block_hashes.len()
                );
                return None;
            }
        };

        if resp.found_count == 0 || resp.node_blocks.is_empty() {
            debug!(
                "p2p prefix query: ns={} queried={} found=0 rpc={:.1}ms",
                namespace,
                block_hashes.len(),
                t.elapsed().as_secs_f64() * 1000.0,
            );
            return None;
        }

        // Build a lookup: hash → node_addr from response.
        let mut hash_to_node: HashMap<Vec<u8>, String> = HashMap::new();
        for nb in &resp.node_blocks {
            for hash in &nb.block_hashes {
                hash_to_node.insert(hash.clone(), nb.node.clone());
            }
        }

        // Walk the prefix in order, stop at first miss.
        let mut prefix_len = 0usize;
        for hash in block_hashes {
            if hash_to_node.contains_key(hash) {
                prefix_len += 1;
            } else {
                break;
            }
        }

        if prefix_len == 0 {
            return None;
        }

        // Group the prefix hits by owning node, preserving per-node order.
        let mut node_map: HashMap<String, Vec<(Vec<u8>, BlockKey)>> = HashMap::new();
        for i in 0..prefix_len {
            let hash = &block_hashes[i];
            let node = hash_to_node.get(hash).unwrap().clone();
            node_map
                .entry(node)
                .or_default()
                .push((hash.clone(), keys[i].clone()));
        }

        let batches: Vec<NodeTransferBatch> = node_map
            .into_iter()
            .map(|(node_addr, entries)| {
                let (hashes, bkeys): (Vec<_>, Vec<_>) = entries.into_iter().unzip();
                NodeTransferBatch {
                    node_addr,
                    block_hashes: hashes,
                    block_keys: bkeys,
                }
            })
            .collect();

        info!(
            "p2p prefix query: ns={} queried={} prefix_hit={} nodes={} rpc={:.1}ms",
            namespace,
            block_hashes.len(),
            prefix_len,
            batches.len(),
            t.elapsed().as_secs_f64() * 1000.0,
        );

        Some(batches)
    }
}

// ============================================================================
// RDMA Transfer Task (async, spawned from submit_prefix)
// ============================================================================

/// Execute RDMA reads for all node batches, returning successfully transferred blocks.
async fn rdma_transfer_task(
    engine: Arc<MooncakeTransferEngine>,
    allocate_fn: AllocateFn,
    node_id: String,
    namespace: String,
    node_batches: Vec<NodeTransferBatch>,
) -> PrefetchResult {
    let mut all_results: PrefetchResult = Vec::new();

    for batch in node_batches {
        match rdma_transfer_one_node(&engine, &allocate_fn, &node_id, &namespace, batch).await {
            Ok(blocks) => {
                all_results.extend(blocks);
            }
            Err(err) => {
                warn!("p2p RDMA transfer failed for node: {}", err);
            }
        }
    }

    all_results
}

/// Transfer blocks from a single remote node via RDMA.
///
/// Flow: connect to remote → AcquireLease → allocate local memory →
///       RDMA READ → ReleaseLease → rebuild SealedBlocks
async fn rdma_transfer_one_node(
    engine: &MooncakeTransferEngine,
    allocate_fn: &AllocateFn,
    node_id: &str,
    namespace: &str,
    batch: NodeTransferBatch,
) -> Result<PrefetchResult, String> {
    let node_addr = &batch.node_addr;
    let t = Instant::now();

    // 1. Connect to remote node's RdmaTransfer gRPC service.
    let remote_endpoint = format!("http://{}", node_addr);
    let channel = Endpoint::from_shared(remote_endpoint.clone())
        .map_err(|e| format!("invalid endpoint {}: {}", node_addr, e))?
        .connect()
        .await
        .map_err(|e| format!("connect to {} failed: {}", node_addr, e))?;
    let mut lease_client = RdmaTransferClient::new(channel);

    // 2. AcquireLease
    let acquire_req = AcquireLeaseRequest {
        requester_node_id: node_id.to_string(),
        namespace: namespace.to_string(),
        block_hashes: batch.block_hashes.clone(),
        lease_duration_secs: DEFAULT_LEASE_DURATION_SECS,
    };

    let acquire_resp =
        tokio::time::timeout(LEASE_RPC_TIMEOUT, lease_client.acquire_lease(acquire_req))
            .await
            .map_err(|_| format!("AcquireLease timed out for {}", node_addr))?
            .map_err(|e| format!("AcquireLease failed for {}: {}", node_addr, e))?
            .into_inner();

    let lease_id = acquire_resp.lease_id.clone();

    if acquire_resp.blocks.is_empty() {
        debug!(
            "p2p AcquireLease: node={} all blocks missing, lease_id={}",
            node_addr, lease_id
        );
        let release_req = ReleaseLeaseRequest {
            lease_id: lease_id.clone(),
            requester_node_id: node_id.to_string(),
        };
        let _ =
            tokio::time::timeout(LEASE_RPC_TIMEOUT, lease_client.release_lease(release_req)).await;
        return Ok(Vec::new());
    }

    debug!(
        "p2p AcquireLease: node={} lease_id={} found={} missing={} rpc={:.1}ms",
        node_addr,
        lease_id,
        acquire_resp.blocks.len(),
        acquire_resp.missing_hashes.len(),
        t.elapsed().as_secs_f64() * 1000.0,
    );

    // Parse the owner's RDMA domain address for session targeting.
    let owner_domain_addr = if !acquire_resp.owner_domain_address.is_empty() {
        pegaflow_transfer::DomainAddress::from_bytes(&acquire_resp.owner_domain_address)
    } else {
        None
    };

    // 3. Allocate local pinned memory and perform RDMA READs.
    let rdma_result = tokio::task::block_in_place(|| {
        rdma_read_blocks(
            engine,
            allocate_fn,
            &acquire_resp,
            &batch,
            owner_domain_addr.as_ref(),
        )
    });

    // 4. ReleaseLease (always attempt, even if RDMA failed).
    let release_req = ReleaseLeaseRequest {
        lease_id: lease_id.clone(),
        requester_node_id: node_id.to_string(),
    };

    if let Err(e) =
        tokio::time::timeout(LEASE_RPC_TIMEOUT, lease_client.release_lease(release_req)).await
    {
        warn!(
            "p2p ReleaseLease timeout for node={} lease_id={}: {}",
            node_addr, lease_id, e
        );
    }

    let blocks = rdma_result?;

    info!(
        "p2p RDMA transfer complete: node={} blocks={} total={:.1}ms",
        node_addr,
        blocks.len(),
        t.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(blocks)
}

// ============================================================================
// RDMA Read + Block Rebuild (blocking, called via block_in_place)
// ============================================================================

/// Perform RDMA READs for leased blocks and rebuild SealedBlocks from local memory.
fn rdma_read_blocks(
    engine: &MooncakeTransferEngine,
    allocate_fn: &AllocateFn,
    acquire_resp: &pegaflow_proto::proto::engine::AcquireLeaseResponse,
    batch: &NodeTransferBatch,
    owner_domain_addr: Option<&pegaflow_transfer::DomainAddress>,
) -> Result<PrefetchResult, String> {
    let Some(domain_addr) = owner_domain_addr else {
        return Err("no owner_domain_address in AcquireLeaseResponse".into());
    };

    // Build per-block transfer descriptors.
    // Each RemoteBlockDescriptor has slots with (k_addr, k_size, v_addr, v_size).
    // We allocate a single contiguous pinned buffer per block, then issue RDMA READs
    // for all slots in a batch.

    let mut all_local_ptrs: Vec<u64> = Vec::new();
    let mut all_remote_ptrs: Vec<u64> = Vec::new();
    let mut all_lens: Vec<usize> = Vec::new();

    // Per-block allocation info for rebuild.
    struct BlockAlloc {
        key: BlockKey,
        allocation: Arc<PinnedAllocation>,
        slot_metas: Vec<SlotMeta>,
        total_size: u64,
    }

    let mut block_allocs: Vec<BlockAlloc> = Vec::new();

    // Map from block_hash to the BlockKey from our batch.
    let hash_to_key: HashMap<&[u8], &BlockKey> = batch
        .block_hashes
        .iter()
        .zip(&batch.block_keys)
        .map(|(h, k)| (h.as_slice(), k))
        .collect();

    for descriptor in &acquire_resp.blocks {
        let Some(key) = hash_to_key.get(descriptor.block_hash.as_slice()) else {
            warn!("p2p RDMA: unexpected block hash in lease response, skipping");
            continue;
        };

        // Calculate total size for this block.
        let total_size: u64 = descriptor.slots.iter().map(|s| s.k_size + s.v_size).sum();

        if total_size == 0 {
            continue;
        }

        // Allocate local pinned memory.
        let allocation = match allocate_fn(total_size, None) {
            Some(alloc) => alloc,
            None => {
                warn!(
                    "p2p RDMA: failed to allocate {} bytes for block, skipping",
                    total_size
                );
                continue;
            }
        };

        let base_ptr = allocation.as_ptr() as u64;

        // Build RDMA read descriptors: for each slot, read K then V into contiguous local memory.
        let mut offset = 0u64;
        let mut slot_metas = Vec::with_capacity(descriptor.slots.len());

        for slot in &descriptor.slots {
            let slot_size = slot.k_size + slot.v_size;
            let is_split = slot.v_addr != 0 && slot.v_size > 0;

            // If K and V are contiguous in the remote address space, merge
            // them into a single RDMA READ to halve the number of operations.
            let kv_contiguous = is_split && slot.v_addr == slot.k_addr + slot.k_size;

            if kv_contiguous {
                all_local_ptrs.push(base_ptr + offset);
                all_remote_ptrs.push(slot.k_addr);
                all_lens.push(slot_size as usize);
                offset += slot_size;
            } else {
                // K segment
                all_local_ptrs.push(base_ptr + offset);
                all_remote_ptrs.push(slot.k_addr);
                all_lens.push(slot.k_size as usize);
                offset += slot.k_size;

                // V segment (if split storage)
                if is_split {
                    all_local_ptrs.push(base_ptr + offset);
                    all_remote_ptrs.push(slot.v_addr);
                    all_lens.push(slot.v_size as usize);
                    offset += slot.v_size;
                }
            }

            slot_metas.push(SlotMeta {
                is_split,
                size: slot_size,
                numa_node: NumaNode::UNKNOWN,
            });
        }

        block_allocs.push(BlockAlloc {
            key: (*key).clone(),
            allocation,
            slot_metas,
            total_size,
        });
    }

    if all_local_ptrs.is_empty() {
        return Ok(Vec::new());
    }

    // Issue batch RDMA READ.
    let rdma_start = Instant::now();
    let transferred = engine
        .batch_transfer_sync_read(domain_addr, &all_local_ptrs, &all_remote_ptrs, &all_lens)
        .map_err(|e| format!("RDMA batch_transfer_sync_read failed: {}", e))?;

    let rdma_elapsed = rdma_start.elapsed();
    let total_bytes: u64 = block_allocs.iter().map(|b| b.total_size).sum();
    debug!(
        "p2p RDMA READ: blocks={} bytes={} transferred={} elapsed={:.1}ms",
        block_allocs.len(),
        total_bytes,
        transferred,
        rdma_elapsed.as_secs_f64() * 1000.0,
    );

    // Rebuild SealedBlocks from local allocations.
    let mut results: PrefetchResult = Vec::with_capacity(block_allocs.len());

    for block_alloc in block_allocs {
        match rebuild_sealed_block(&block_alloc.allocation, &block_alloc.slot_metas) {
            Ok(sealed) => {
                results.push((block_alloc.key, Arc::new(sealed)));
            }
            Err(e) => {
                warn!("p2p RDMA: failed to rebuild sealed block: {}", e);
            }
        }
    }

    Ok(results)
}

/// Rebuild a SealedBlock from a single contiguous allocation with per-slot metadata.
fn rebuild_sealed_block(
    allocation: &Arc<PinnedAllocation>,
    slot_metas: &[SlotMeta],
) -> Result<SealedBlock, String> {
    let mut layer_blocks = Vec::with_capacity(slot_metas.len());
    let mut offset = 0usize;

    for meta in slot_metas {
        // Safety: allocation was sized to hold all slots contiguously.
        let lb = unsafe { meta.make_layer_block(Arc::clone(allocation), offset) };
        offset += meta.size as usize;
        layer_blocks.push(lb);
    }

    Ok(SealedBlock::from_slots(layer_blocks))
}

// ============================================================================
// Insert actor (fire-and-forget, batched) — unchanged from original
// ============================================================================

async fn insert_actor(
    mut rx: UnboundedReceiver<InsertCmd>,
    coordinator_addr: String,
    node_addr: String,
) {
    let mut client: Option<MetaServerClient<Channel>> = None;

    while let Some(cmd) = rx.recv().await {
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        let mut by_ns: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for cmd in cmds {
            by_ns
                .entry(cmd.namespace)
                .or_default()
                .extend(cmd.block_hashes);
        }

        for (namespace, block_hashes) in by_ns {
            if client.is_none() {
                client = connect_client(&coordinator_addr).await;
            }
            let Some(c) = client.as_mut() else {
                continue;
            };

            let count = block_hashes.len();
            let req = InsertBlockHashesRequest {
                namespace: namespace.clone(),
                block_hashes,
                node: node_addr.clone(),
                domain_addresses: vec![],
            };
            let t = Instant::now();
            match c.insert_block_hashes(req).await {
                Ok(resp) => {
                    debug!(
                        "p2p insert: ns={} sent={} inserted={} rpc={:.1}ms",
                        namespace,
                        count,
                        resp.into_inner().inserted_count,
                        t.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                Err(err) => {
                    warn!(
                        "p2p insert failed after {:.1}ms: {}",
                        t.elapsed().as_secs_f64() * 1000.0,
                        err,
                    );
                    client = None;
                }
            }
        }
    }

    info!("p2p insert actor shutting down");
}

// ============================================================================
// Shared helpers
// ============================================================================

async fn connect_client(addr: &str) -> Option<MetaServerClient<Channel>> {
    match MetaServerClient::connect(addr.to_string()).await {
        Ok(client) => Some(client),
        Err(err) => {
            warn!("Failed to connect P2P coordinator at {}: {}", addr, err);
            None
        }
    }
}

pub(super) fn new_p2p(
    config: BakingStoreConfig,
    allocate_fn: AllocateFn,
    transfer_engine: Arc<MooncakeTransferEngine>,
) -> Option<Arc<dyn BackingStore>> {
    P2pBakingStore::create(config, allocate_fn, transfer_engine)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinned_pool::PinnedAllocator;
    use std::num::NonZeroU64;

    fn make_allocator() -> Arc<PinnedAllocator> {
        Arc::new(PinnedAllocator::new_global(4 << 20, false, false, None))
    }

    /// `rebuild_sealed_block` with contiguous slots reconstructs the correct
    /// number of layer blocks and total memory footprint.
    #[test]
    fn rebuild_sealed_block_contiguous_correct_layout() {
        let allocator = make_allocator();
        let slot_size: u64 = 512;
        let num_slots = 3usize;
        let total = slot_size * num_slots as u64;

        let allocation = allocator
            .allocate(
                NonZeroU64::new(total).unwrap(),
                crate::numa::NumaNode::UNKNOWN,
            )
            .expect("allocate");

        let slot_metas: Vec<SlotMeta> = (0..num_slots)
            .map(|_| SlotMeta {
                is_split: false,
                size: slot_size,
                numa_node: crate::numa::NumaNode::UNKNOWN,
            })
            .collect();

        let sealed = rebuild_sealed_block(&allocation, &slot_metas).expect("rebuild_sealed_block");

        assert_eq!(sealed.slots().len(), num_slots);
        assert_eq!(sealed.memory_footprint(), total);
    }

    /// `rebuild_sealed_block` with split slots (separate K/V) works correctly.
    #[test]
    fn rebuild_sealed_block_split_correct_layout() {
        let allocator = make_allocator();
        let slot_size: u64 = 256;
        let num_slots = 2usize;
        let total = slot_size * num_slots as u64;

        let allocation = allocator
            .allocate(
                NonZeroU64::new(total).unwrap(),
                crate::numa::NumaNode::UNKNOWN,
            )
            .expect("allocate");

        let slot_metas: Vec<SlotMeta> = (0..num_slots)
            .map(|_| SlotMeta {
                is_split: true,
                size: slot_size,
                numa_node: crate::numa::NumaNode::UNKNOWN,
            })
            .collect();

        let sealed = rebuild_sealed_block(&allocation, &slot_metas).expect("rebuild_sealed_block");

        assert_eq!(sealed.slots().len(), num_slots);
        assert_eq!(sealed.memory_footprint(), total);
    }

    /// `rebuild_sealed_block` on an empty slot list yields a valid empty SealedBlock.
    #[test]
    fn rebuild_sealed_block_empty_slots() {
        let allocator = make_allocator();
        let allocation = allocator
            .allocate(NonZeroU64::new(64).unwrap(), crate::numa::NumaNode::UNKNOWN)
            .expect("allocate");

        let sealed = rebuild_sealed_block(&allocation, &[]).expect("rebuild_sealed_block");
        assert_eq!(sealed.slots().len(), 0);
        assert_eq!(sealed.memory_footprint(), 0);
    }

    /// `submit_prefix` with an empty key list returns `(0, rx)` where `rx`
    /// immediately delivers an empty result — no async work spawned.
    #[tokio::test]
    async fn submit_prefix_empty_keys_returns_zero() {
        let rt = tokio::runtime::Handle::current();
        let _ = rt; // ensure we're in a Tokio runtime

        let engine = Arc::new(MooncakeTransferEngine::new());
        let allocator = make_allocator();
        let allocate_fn: AllocateFn = Arc::new(move |size, node| {
            allocator.allocate(NonZeroU64::new(size)?, node.unwrap_or(NumaNode::UNKNOWN))
        });

        let store = P2pBakingStore::create(
            BakingStoreConfig {
                p2p_coordinator_addr: "http://127.0.0.1:1".to_string(),
                p2p_node_addr: "127.0.0.1:50055".to_string(),
                node_id: "test".to_string(),
            },
            allocate_fn,
            engine,
        )
        .expect("create p2p store");

        let (found, rx) = store.submit_prefix(vec![]);
        assert_eq!(found, 0);
        let result = rx.await.expect("receiver");
        assert!(result.is_empty());
    }
}
