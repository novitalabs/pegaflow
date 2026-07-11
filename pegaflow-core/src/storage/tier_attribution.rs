// Attribution for one `query_prefetch` decision: RAM prefix hit, blocks
// routed to the selected backing tier, residual miss. Records which tier was
// *selected*, not whether its transfer succeeded — completion/failure shows
// up in `rdma_fetch_total` / `ssd_prefetch_failures`.

use super::backing_tier::TierSource;

/// `source` is `Some` iff a backing tier was selected; `tier_blocks` counts
/// the blocks it committed to.
#[derive(Copy, Clone, Debug)]
pub(super) struct TierDecision {
    pub(super) total: usize,
    pub(super) hit: usize,
    pub(super) tier_blocks: usize,
    pub(super) source: Option<TierSource>,
}

impl TierDecision {
    pub(super) fn record(self) {
        if self.total == 0 {
            return;
        }
        let attribution = Attribution::classify(&self);
        crate::metrics::record_cache_tier_block_requests(
            attribution.ram,
            attribution.rdma,
            attribution.ssd,
            attribution.miss,
        );
    }
}

/// Invariant: `ram + rdma + ssd + miss == total`.
#[derive(Copy, Clone, Debug)]
pub(super) struct Attribution {
    ram: usize,
    rdma: usize,
    ssd: usize,
    miss: usize,
}

impl Attribution {
    /// # Panics
    /// When `hit + tier_blocks` exceeds `total` — in release builds too.
    /// `tier_blocks > 0` without a `source` is a debug-only assertion.
    pub(super) fn classify(decision: &TierDecision) -> Self {
        let TierDecision {
            total,
            hit,
            tier_blocks,
            source,
        } = *decision;
        let miss = total
            .checked_sub(hit + tier_blocks)
            .expect("hit + tier_blocks must not exceed total");
        let (rdma, ssd) = match source {
            Some(TierSource::Rdma) => (tier_blocks, 0),
            Some(TierSource::Ssd) => (0, tier_blocks),
            None => {
                debug_assert_eq!(tier_blocks, 0, "tier_blocks > 0 without a source");
                (0, 0)
            }
        };
        Self {
            ram: hit,
            rdma,
            ssd,
            miss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_tiers() {
        let decisions = [
            // Local RAM prefix plus residual miss, no backing tier selected.
            (
                TierDecision {
                    total: 7,
                    hit: 3,
                    tier_blocks: 0,
                    source: None,
                },
                (3, 0, 0, 4),
            ),
            // RDMA found only part of the non-RAM prefix.
            (
                TierDecision {
                    total: 5,
                    hit: 1,
                    tier_blocks: 3,
                    source: Some(TierSource::Rdma),
                },
                (1, 3, 0, 1),
            ),
            // SSD accepted only part, e.g. after backpressure trimming.
            (
                TierDecision {
                    total: 6,
                    hit: 1,
                    tier_blocks: 2,
                    source: Some(TierSource::Ssd),
                },
                (1, 0, 2, 3),
            ),
        ];

        for (decision, expected) in decisions {
            let a = Attribution::classify(&decision);
            assert_eq!((a.ram, a.rdma, a.ssd, a.miss), expected);
        }
    }

    #[test]
    #[should_panic(expected = "hit + tier_blocks must not exceed total")]
    fn overcount_panics() {
        let _ = Attribution::classify(&TierDecision {
            total: 3,
            hit: 2,
            tier_blocks: 2,
            source: Some(TierSource::Ssd),
        });
    }
}
