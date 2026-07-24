// Backing-tier seam: every tier answers `plan` (metadata-only candidate
// discovery) and `fetch` (data transfer for a planned prefix). Policy —
// tier order, budget backpressure — lives in the prefetch orchestrator.

use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "rdma")]
use crate::backing::RdmaFetchStore;
use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::BlockKey;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum TierSource {
    Ssd,
    Rdma,
}

#[cfg(feature = "rdma")]
#[derive(Clone)]
pub(super) struct RdmaFetch(Arc<RdmaFetchStore>);

#[cfg(not(feature = "rdma"))]
#[derive(Clone)]
pub(super) struct RdmaFetch;

#[cfg(feature = "rdma")]
impl RdmaFetch {
    pub(super) fn new(store: Arc<RdmaFetchStore>) -> Self {
        Self(store)
    }

    async fn plan(&self, query: &TierQuery<'_>) -> Option<TierPlan> {
        let (node, len) = self.0.query_prefix(query.namespace, query.hashes).await?;
        query.accepts(len).then_some(TierPlan::Rdma { node, len })
    }

    async fn fetch(
        &self,
        query: &TierQuery<'_>,
        plan: &TierPlan,
        take: usize,
    ) -> (usize, PrefetchResult) {
        let TierPlan::Rdma { node, .. } = plan else {
            unreachable!("BUG: RDMA tier received a foreign plan");
        };
        let blocks = self
            .0
            .fetch_blocks(node, query.req_id, query.namespace, &query.hashes[..take])
            .await;
        // Report the commitment, not `blocks.len()`: the remote may deliver
        // fewer (concurrent eviction) and the caller detects the shortfall.
        (take, blocks)
    }
}

#[cfg(not(feature = "rdma"))]
impl RdmaFetch {
    async fn plan(&self, _query: &TierQuery<'_>) -> Option<TierPlan> {
        None
    }

    async fn fetch(
        &self,
        _query: &TierQuery<'_>,
        _plan: &TierPlan,
        _take: usize,
    ) -> (usize, PrefetchResult) {
        unreachable!("BUG: RDMA fetch without the rdma feature (plan is always None)")
    }
}

/// Invariant: `keys[i]` and `hashes[i]` describe the same block.
pub(super) struct TierQuery<'a> {
    pub(super) req_id: &'a str,
    pub(super) namespace: &'a str,
    pub(super) keys: &'a [BlockKey],
    pub(super) hashes: &'a [Vec<u8>],
    pub(super) require_full: bool,
}

impl TierQuery<'_> {
    /// The single planned-length policy point: every tier's `plan` and the
    /// post-`fetch` acceptance check route their length gating through here.
    /// `require_full` callers cannot proceed on a partial prefix, so a length
    /// short of the whole miss suffix is worth nothing to them.
    pub(super) fn accepts(&self, len: usize) -> bool {
        len > 0 && (!self.require_full || len == self.keys.len())
    }
}

/// Consecutive blocks a tier can supply from the front of the query, plus
/// whatever its `fetch` needs to retrieve them.
pub(super) enum TierPlan {
    #[cfg(feature = "rdma")]
    Rdma {
        node: String,
        len: usize,
    },
    Ssd {
        len: usize,
    },
    #[cfg(test)]
    Fake {
        len: usize,
    },
}

impl TierPlan {
    pub(super) fn len(&self) -> usize {
        match self {
            #[cfg(feature = "rdma")]
            Self::Rdma { len, .. } => *len,
            Self::Ssd { len } => *len,
            #[cfg(test)]
            Self::Fake { len } => *len,
        }
    }
}

/// Enum rather than a trait object: the tier set is closed and plans carry
/// tier-specific data (RDMA's target node).
#[derive(Clone)]
pub(super) enum BackingTier {
    Rdma(RdmaFetch),
    Ssd(Arc<SsdBackingStore>),
    #[cfg(test)]
    Fake(Arc<FakeTier>),
}

impl BackingTier {
    pub(super) fn source(&self) -> TierSource {
        match self {
            Self::Rdma(_) => TierSource::Rdma,
            Self::Ssd(_) => TierSource::Ssd,
            #[cfg(test)]
            Self::Fake(fake) => fake.source,
        }
    }

    /// Metadata-only discovery: no data moves. `None` = no usable prefix.
    pub(super) async fn plan(&self, query: &TierQuery<'_>) -> Option<TierPlan> {
        match self {
            Self::Rdma(rdma) => rdma.plan(query).await,
            Self::Ssd(ssd) => {
                let len = ssd.prefix_len(query.keys);
                query.accepts(len).then_some(TierPlan::Ssd { len })
            }
            #[cfg(test)]
            Self::Fake(fake) => fake.plan(query),
        }
    }

    /// Transfer the first `take` planned blocks (`take <= plan.len()`).
    /// Returns the committed count — which may exceed the delivered blocks —
    /// and panics on a foreign `TierPlan`.
    pub(super) async fn fetch(
        &self,
        query: &TierQuery<'_>,
        plan: &TierPlan,
        take: usize,
    ) -> (usize, PrefetchResult) {
        match self {
            Self::Rdma(rdma) => rdma.fetch(query, plan, take).await,
            Self::Ssd(ssd) => {
                assert!(
                    matches!(plan, TierPlan::Ssd { .. }),
                    "BUG: SSD tier received a foreign plan"
                );
                ssd.prefetch_prefix(query.keys[..take].to_vec()).await
            }
            #[cfg(test)]
            Self::Fake(fake) => {
                assert!(
                    matches!(plan, TierPlan::Fake { .. }),
                    "BUG: fake tier received a foreign plan"
                );
                fake.fetch(take)
            }
        }
    }
}

#[cfg(test)]
pub(super) struct FakeTier {
    source: TierSource,
    plan_len: usize,
    blocks: PrefetchResult,
    plan_calls: AtomicUsize,
    fetch_calls: AtomicUsize,
}

#[cfg(test)]
impl FakeTier {
    pub(super) fn new(source: TierSource, plan_len: usize) -> Self {
        Self {
            source,
            plan_len,
            blocks: Vec::new(),
            plan_calls: AtomicUsize::new(0),
            fetch_calls: AtomicUsize::new(0),
        }
    }

    pub(super) fn with_blocks(mut self, blocks: PrefetchResult) -> Self {
        self.blocks = blocks;
        self
    }

    pub(super) fn plan_calls(&self) -> usize {
        self.plan_calls.load(Ordering::SeqCst)
    }

    pub(super) fn fetch_calls(&self) -> usize {
        self.fetch_calls.load(Ordering::SeqCst)
    }

    fn plan(&self, query: &TierQuery<'_>) -> Option<TierPlan> {
        self.plan_calls.fetch_add(1, Ordering::SeqCst);
        query
            .accepts(self.plan_len)
            .then_some(TierPlan::Fake { len: self.plan_len })
    }

    fn fetch(&self, take: usize) -> (usize, PrefetchResult) {
        self.fetch_calls.fetch_add(1, Ordering::SeqCst);
        (take, self.blocks.clone())
    }
}
