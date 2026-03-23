pub(super) mod rdma;
pub(super) mod rdma_fetch;
pub(super) mod ssd;
pub(super) mod ssd_cache;
pub(super) mod uring;

use std::sync::Arc;

pub use ssd_cache::{
    DEFAULT_MAX_PREFETCH_BLOCKS, DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
    DEFAULT_SSD_WRITE_INFLIGHT, DEFAULT_SSD_WRITE_QUEUE_DEPTH, SSD_ALIGNMENT, SsdCacheConfig,
};

use crate::block::{BlockKey, SealedBlock};
use crate::pinned_pool::PinnedAllocation;
use pegaflow_common::NumaNode;

pub(crate) use rdma::{RdmaTransport, new_rdma};
pub(crate) use rdma_fetch::RdmaFetchStore;
pub(crate) use ssd::SsdBackingStore;
pub(crate) use ssd::new_ssd;

pub(crate) type PrefetchResult = Vec<(BlockKey, Arc<SealedBlock>)>;

/// Allocator closure for pinned memory, passed to the SSD backing store.
pub(crate) type AllocateFn =
    Arc<dyn Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync>;
