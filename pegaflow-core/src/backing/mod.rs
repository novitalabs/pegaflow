//! Backing store implementations for sealed blocks.
//!
//! Two concrete backing stores:
//! - [`SsdBackingStore`]: local SSD cache via io_uring
//! - [`P2pBackingStore`]: cross-node RDMA transfer via MetaServer
//!
//! No shared trait — SSD and P2P have fundamentally different characteristics
//! (sync local disk vs async remote RDMA) and are used through concrete types.

pub(super) mod p2p;
pub(super) mod ssd;
pub(super) mod ssd_cache;
pub(super) mod uring;

use std::sync::Arc;

pub use ssd_cache::{
    DEFAULT_MAX_PREFETCH_BLOCKS, DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
    DEFAULT_SSD_WRITE_INFLIGHT, DEFAULT_SSD_WRITE_QUEUE_DEPTH, SSD_ALIGNMENT, SsdCacheConfig,
};

use crate::block::{BlockKey, SealedBlock};
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;

pub(crate) use p2p::P2pBackingStore;
pub(crate) use ssd::SsdBackingStore;

/// Batch of successfully-read blocks from the backing store.
pub(crate) type PrefetchResult = Vec<(BlockKey, Arc<SealedBlock>)>;

/// Allocator closure for pinned memory (shared by SSD and P2P backing stores).
pub(crate) type AllocateFn =
    Arc<dyn Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync>;

/// Configuration for the P2P backing store.
#[derive(Debug, Clone)]
pub struct P2pConfig {
    /// gRPC endpoint of the coordinator that tracks block ownership.
    pub p2p_coordinator_addr: String,
    /// Advertised node address reported as the block owner.
    pub p2p_node_addr: String,
    /// This node's UUID, used as requester identity in lease RPCs.
    pub node_id: String,
}

// ============================================================================
// Factory
// ============================================================================

/// Create a P2P backing store.
///
/// `allocate_fn` provides NUMA-aware pinned memory for RDMA read buffers.
/// `transfer_engine` is an already-initialized RDMA transfer engine.
pub(crate) fn new_p2p(
    config: P2pConfig,
    allocate_fn: AllocateFn,
    transfer_engine: Arc<pegaflow_transfer::TransferEngine>,
) -> Option<Arc<P2pBackingStore>> {
    p2p::new_p2p(config, allocate_fn, transfer_engine)
}

/// Create an SSD backing store.
///
/// `allocate_fn` must provide NUMA-aware pinned memory (with cache eviction).
/// Returns `None` and logs an error if the SSD cache cannot be initialised.
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: AllocateFn,
    is_numa: bool,
) -> Option<Arc<SsdBackingStore>> {
    ssd::new_ssd(config, allocate_fn, is_numa)
}
