// Decision-level tier attribution for `query_prefetch`.
//
// Splits a single `query_prefetch` decision's block budget into four mutually
// exclusive tiers: RAM (resident-cache prefix hit), RDMA / SSD (backing tier
// selected to satisfy the remaining prefix), and MISS (everything no tier
// could satisfy this decision, including SSD backpressure and RDMA partial
// returns).
//
// `ram + rdma + ssd + miss == total` is enforced on construction.
//
// This is decision attribution: tiers report which path was *selected* by
// `full_prefix_scan`, not which path *eventually* succeeded. Backing
// completion / failure must be observed via the existing `rdma_fetch_total`
// and `ssd_prefetch_failures` counters.

/// The tier a block was attributed to for a single `query_prefetch` decision.
///
/// Only backing tiers (`Rdma`, `Ssd`) are represented here; the local RAM
/// prefix hit is conveyed through the `hit` argument to `classify` rather
/// than as a separate `AttributionSource` variant. Keeping the enum closed
/// over the two backing sources makes the call sites in `prefetch.rs` total
/// without a dead RAM branch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(super) enum AttributionSource {
    /// RDMA remote-fetch was selected for at least one remaining block.
    Rdma,
    /// SSD prefetch was selected for at least one remaining block.
    Ssd,
}

/// Per-decision block counts. Invariant: `ram + rdma + ssd + miss == total`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(super) struct TierAttribution {
    pub(super) ram: usize,
    pub(super) rdma: usize,
    pub(super) ssd: usize,
    pub(super) miss: usize,
}

impl TierAttribution {
    /// Build the per-decision attribution from the values `full_prefix_scan`
    /// already knows. `loading_source` is `Some` iff a backing tier was
    /// chosen for the remaining prefix.
    ///
    /// # Panics
    /// Debug-only assertion that `hit + loading + miss == total`.
    pub(super) fn classify(
        total: usize,
        hit: usize,
        loading: usize,
        loading_source: Option<AttributionSource>,
    ) -> Self {
        let miss = total
            .checked_sub(hit + loading)
            .expect("hit + loading must not exceed total");
        let (rdma, ssd) = match loading_source {
            Some(AttributionSource::Rdma) => (loading, 0),
            Some(AttributionSource::Ssd) => (0, loading),
            // `loading == 0` is the only valid shape when no backing tier was
            // selected; if a caller hands us `loading > 0` without a source it
            // is a programmer error, surfaced under debug builds.
            None => {
                debug_assert_eq!(loading, 0, "loading > 0 without a backing source");
                (0, 0)
            }
        };
        let attribution = Self {
            ram: hit,
            rdma,
            ssd,
            miss,
        };
        debug_assert_eq!(
            attribution.sum(),
            total,
            "tier attribution must sum to total"
        );
        attribution
    }

    pub(super) fn sum(&self) -> usize {
        self.ram + self.rdma + self.ssd + self.miss
    }
}

/// Record a tier attribution to the OTel counter. Skips zero-count tiers so
/// each decision adds at most four points (typically one to three).
///
/// The `total` parameter is used only for the debug invariant; release builds
/// strip it.
pub(super) fn record_cache_tier_block_requests(total: usize, attribution: TierAttribution) {
    debug_assert_eq!(
        attribution.sum(),
        total,
        "tier attribution must sum to total"
    );
    let metrics = crate::metrics::core_metrics();
    if attribution.ram > 0 {
        metrics
            .cache_tier_block_requests
            .add(attribution.ram as u64, &*crate::metrics::TIER_RAM);
    }
    if attribution.rdma > 0 {
        metrics
            .cache_tier_block_requests
            .add(attribution.rdma as u64, &*crate::metrics::TIER_RDMA);
    }
    if attribution.ssd > 0 {
        metrics
            .cache_tier_block_requests
            .add(attribution.ssd as u64, &*crate::metrics::TIER_SSD);
    }
    if attribution.miss > 0 {
        metrics
            .cache_tier_block_requests
            .add(attribution.miss as u64, &*crate::metrics::TIER_MISS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_only_hit_attributes_all_to_ram() {
        let a = TierAttribution::classify(8, 8, 0, None);
        assert_eq!(
            a,
            TierAttribution {
                ram: 8,
                rdma: 0,
                ssd: 0,
                miss: 0
            }
        );
        assert_eq!(a.sum(), 8);
    }

    #[test]
    fn rdma_loading_splits_ram_rdma_miss() {
        let a = TierAttribution::classify(10, 3, 5, Some(AttributionSource::Rdma));
        assert_eq!(
            a,
            TierAttribution {
                ram: 3,
                rdma: 5,
                ssd: 0,
                miss: 2
            }
        );
    }

    #[test]
    fn ssd_loading_splits_ram_ssd_miss() {
        let a = TierAttribution::classify(10, 4, 4, Some(AttributionSource::Ssd));
        assert_eq!(
            a,
            TierAttribution {
                ram: 4,
                rdma: 0,
                ssd: 4,
                miss: 2
            }
        );
    }

    #[test]
    fn no_backing_no_ram_attributes_all_to_miss() {
        let a = TierAttribution::classify(7, 0, 0, None);
        assert_eq!(
            a,
            TierAttribution {
                ram: 0,
                rdma: 0,
                ssd: 0,
                miss: 7
            }
        );
    }

    #[test]
    fn empty_query_yields_all_zeros() {
        let a = TierAttribution::classify(0, 0, 0, None);
        assert_eq!(a, TierAttribution::default());
        assert_eq!(a.sum(), 0);
    }

    #[test]
    fn ssd_backpressure_residual_falls_into_miss() {
        // 6 requested; 1 RAM hit; SSD was selected for 2 (rest skipped by
        // backpressure inside `limit_ssd_prefetch`, which surfaces here as a
        // smaller `loading`).
        let a = TierAttribution::classify(6, 1, 2, Some(AttributionSource::Ssd));
        assert_eq!(a.ram, 1);
        assert_eq!(a.ssd, 2);
        assert_eq!(a.miss, 3);
        assert_eq!(a.sum(), 6);
    }

    #[test]
    fn rdma_partial_residual_falls_into_miss() {
        // RDMA query promised 3 of the 4 remaining; the 4th counts as miss.
        let a = TierAttribution::classify(5, 1, 3, Some(AttributionSource::Rdma));
        assert_eq!(a.ram, 1);
        assert_eq!(a.rdma, 3);
        assert_eq!(a.miss, 1);
    }

    #[test]
    #[should_panic(expected = "hit + loading must not exceed total")]
    fn overcounting_panics_in_debug() {
        let _ = TierAttribution::classify(3, 2, 2, Some(AttributionSource::Ssd));
    }
}
