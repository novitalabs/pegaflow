//! Backing store abstraction for sealed blocks.
//!
//! Exposes a single [`BackingStore`] trait that the storage engine uses
//! to write and read blocks from a secondary tier.  All implementation
//! details (ring buffer, io_uring, SSD workers) stay inside this module.

pub(super) mod p2p;
pub(super) mod ssd;
pub(super) mod ssd_cache;
pub(super) mod uring;

use std::sync::{Arc, Weak};

use tokio::sync::oneshot;

pub use ssd_cache::{
    DEFAULT_MAX_PREFETCH_BLOCKS, DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
    DEFAULT_SSD_WRITE_INFLIGHT, DEFAULT_SSD_WRITE_QUEUE_DEPTH, SSD_ALIGNMENT, SsdCacheConfig,
};

use crate::block::{BlockKey, SealedBlock};
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;

/// Batch of successfully-read blocks from the backing store.
pub(crate) type PrefetchResult = Vec<(BlockKey, Arc<SealedBlock>)>;

/// Allocator closure for pinned memory (shared by SSD and P2P backing stores).
pub(crate) type AllocateFn =
    Arc<dyn Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync>;

/// Configuration for the P2P baking store.
#[derive(Debug, Clone)]
pub struct BakingStoreConfig {
    /// gRPC endpoint of the coordinator that tracks block ownership.
    pub p2p_coordinator_addr: String,
    /// Advertised node address reported as the block owner.
    pub p2p_node_addr: String,
    /// This node's UUID, used as requester identity in lease RPCs.
    pub node_id: String,
}

/// Runtime kind of a backing store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackingStoreKind {
    P2p,
    Ssd,
}

// ============================================================================
// Public interface
// ============================================================================

/// Secondary storage tier for sealed blocks.
///
/// Implementations must be `Send + Sync + 'static` so they can be wrapped in
/// `Arc<dyn BackingStore>` and shared across async tasks and threads.
pub(crate) trait BackingStore: Send + Sync + 'static {
    /// Store kind, used for wiring decisions such as alignment requirements.
    fn kind(&self) -> BackingStoreKind;

    /// Fire-and-forget write.
    ///
    /// `blocks` holds `Weak` references so the backing store cannot prevent
    /// cache eviction from freeing the pinned memory before the write completes.
    fn ingest_batch(&self, blocks: Vec<(BlockKey, Weak<SealedBlock>)>);

    /// Submit prefix reads: scan `keys` in order, submit reads for consecutive hits, stop at first miss.
    ///
    /// Returns `(submitted, done_rx)` where `done_rx` delivers completed blocks.
    fn submit_prefix(&self, keys: Vec<BlockKey>) -> (usize, oneshot::Receiver<PrefetchResult>);
}

// ============================================================================
// Factory
// ============================================================================

/// Create a P2P baking-backed [`BackingStore`].
///
/// `allocate_fn` provides NUMA-aware pinned memory for RDMA read buffers.
/// `transfer_engine` is an already-initialized RDMA transfer engine.
pub(crate) fn new_p2p(
    config: BakingStoreConfig,
    allocate_fn: AllocateFn,
    transfer_engine: Arc<pegaflow_transfer::MooncakeTransferEngine>,
) -> Option<Arc<dyn BackingStore>> {
    p2p::new_p2p(config, allocate_fn, transfer_engine)
}

/// Create an SSD-backed [`BackingStore`].
///
/// `allocate_fn` must provide NUMA-aware pinned memory (with cache eviction).
/// Returns `None` and logs an error if the SSD cache cannot be initialised.
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: AllocateFn,
    is_numa: bool,
) -> Option<Arc<dyn BackingStore>> {
    ssd::new_ssd(config, allocate_fn, is_numa)
}
