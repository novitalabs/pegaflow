//! Backing store abstraction for sealed blocks.
//!
//! Exposes a single [`BackingStore`] trait that the storage engine uses
//! to write and read blocks from a secondary tier.  All implementation
//! details (ring buffer, io_uring, SSD workers) stay inside this module.

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

// ============================================================================
// Public interface
// ============================================================================

/// Secondary storage tier for sealed blocks.
///
/// Implementations must be `Send + Sync + 'static` so they can be wrapped in
/// `Arc<dyn BackingStore>` and shared across async tasks and threads.
pub(crate) trait BackingStore: Send + Sync + 'static {
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

/// Create an SSD-backed [`BackingStore`].
///
/// `allocate_fn` must provide NUMA-aware pinned memory (with cache eviction).
/// Returns `None` and logs an error if the SSD cache cannot be initialised.
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: impl Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync + 'static,
    is_numa: bool,
) -> Option<Arc<dyn BackingStore>> {
    ssd::new_ssd(config, allocate_fn, is_numa)
}
