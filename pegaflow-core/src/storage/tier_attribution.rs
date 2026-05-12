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
#[derive(Copy, Clone, Debug)]
pub(super) struct TierAttribution {
    ram: usize,
    rdma: usize,
    ssd: usize,
    miss: usize,
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

    fn sum(&self) -> usize {
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
    crate::metrics::record_cache_tier_block_requests(
        attribution.ram,
        attribution.rdma,
        attribution.ssd,
        attribution.miss,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_backing_decisions_keep_selected_tier_and_residual_contract() {
        let cases = [
            // Done branch: local RAM prefix plus residual miss when no backing
            // tier is selected for this decision.
            (TierAttribution::classify(7, 3, 0, None), (3, 0, 0, 4, 7)),
            // RDMA found only part of the non-RAM prefix.
            (
                TierAttribution::classify(5, 1, 3, Some(AttributionSource::Rdma)),
                (1, 3, 0, 1, 5),
            ),
            // SSD prefetch accepted only part of the non-RAM prefix, for
            // example after backpressure trimming.
            (
                TierAttribution::classify(6, 1, 2, Some(AttributionSource::Ssd)),
                (1, 0, 2, 3, 6),
            ),
        ];

        for (attribution, expected) in cases {
            assert_eq!(
                (
                    attribution.ram,
                    attribution.rdma,
                    attribution.ssd,
                    attribution.miss,
                    attribution.sum()
                ),
                expected
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "loading > 0 without a backing source")]
    fn loading_without_source_panics_in_debug() {
        let _ = TierAttribution::classify(4, 1, 1, None);
    }

    #[test]
    #[should_panic(expected = "hit + loading must not exceed total")]
    fn overcounting_panics_in_debug() {
        let _ = TierAttribution::classify(3, 2, 2, Some(AttributionSource::Ssd));
    }
}
