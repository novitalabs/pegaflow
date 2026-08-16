use std::sync::Arc;
use std::time::{Duration, Instant};

use log::info;

use crate::block::{BlockKey, PrefetchStatus, SealedBlock};

use super::super::backing_tier::{BackingTier, TierSource};
use super::super::read_cache::ReadCache;
use super::super::tier_attribution::TierDecision;
use super::task::{TaskDeps, TaskInput, run_task};
use super::{PrefixScan, Scheduler};

/// Production tier order: RDMA first, SSD fallback. Slots are positional;
/// a tier in the wrong slot would silently reorder production, so panic.
pub(super) fn tier_lineup(rdma: Option<BackingTier>, ssd: Option<BackingTier>) -> Vec<BackingTier> {
    assert!(
        rdma.as_ref()
            .is_none_or(|tier| tier.source() == TierSource::Rdma),
        "BUG: non-RDMA tier in the RDMA lineup slot"
    );
    assert!(
        ssd.as_ref()
            .is_none_or(|tier| tier.source() == TierSource::Ssd),
        "BUG: non-SSD tier in the SSD lineup slot"
    );
    [rdma, ssd].into_iter().flatten().collect()
}

struct ScanTiming<'a> {
    req_id: &'a str,
    total_keys: usize,
    hit: usize,
    key_build: Duration,
    cache_scan: Duration,
    schedule: Duration,
    elapsed: Duration,
}

fn log_scan(timing: &ScanTiming<'_>, task_started: bool) {
    let missing = timing.total_keys - timing.hit;
    if task_started {
        info!(
            "Prefetch scheduling timing: req_id={} total_keys={} hit={} remaining={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
            timing.req_id,
            timing.total_keys,
            timing.hit,
            missing,
            timing.key_build,
            timing.cache_scan,
            timing.schedule,
            timing.elapsed
        );
    } else {
        info!(
            "Prefetch local-hit timing: req_id={} total_keys={} hit={} missing={} key_build={:?} cache_scan={:?} task_schedule={:?} total={:?}",
            timing.req_id,
            timing.total_keys,
            timing.hit,
            missing,
            timing.key_build,
            timing.cache_scan,
            timing.schedule,
            timing.elapsed
        );
    }
}

fn local_ready(total: usize, prefix: Vec<Arc<SealedBlock>>) -> PrefetchStatus {
    let hit = prefix.len();
    TierDecision {
        total,
        hit,
        tier_blocks: 0,
        source: None,
    }
    .record();
    PrefetchStatus::Ready {
        blocks: prefix,
        missing: total - hit,
    }
}

impl Scheduler {
    pub(super) async fn scan_prefix(
        &self,
        read_cache: &ReadCache,
        scan: PrefixScan<'_>,
    ) -> PrefetchStatus {
        let start = Instant::now();
        let build_start = Instant::now();
        let keys: Vec<BlockKey> = scan
            .hashes
            .iter()
            .map(|hash| BlockKey::new(scan.namespace.to_string(), hash.clone()))
            .collect();
        let key_build = build_start.elapsed();

        let cache_start = Instant::now();
        let (hit, prefix) = read_cache.get_prefix_blocks(&keys);
        let cache_scan = cache_start.elapsed();

        let task_start = Instant::now();
        let task_started = hit < keys.len() && self.start_task(&scan, &keys[hit..], prefix.clone());

        log_scan(
            &ScanTiming {
                req_id: scan.req_id,
                total_keys: keys.len(),
                hit,
                key_build,
                cache_scan,
                schedule: task_start.elapsed(),
                elapsed: start.elapsed(),
            },
            task_started,
        );

        if task_started {
            PrefetchStatus::Loading
        } else {
            local_ready(keys.len(), prefix)
        }
    }

    fn start_task(
        &self,
        scan: &PrefixScan<'_>,
        remaining: &[BlockKey],
        prefix: Vec<Arc<SealedBlock>>,
    ) -> bool {
        let req_id = scan.req_id;
        // One guard spans check-then-spawn: duplicate suppression, failed-
        // remote filtering, and registration must not interleave.
        let mut state = self.state.lock();
        if state.has_active(req_id) {
            return true;
        }

        let skip_rdma = state.rdma_failed(req_id);
        let tiers: Vec<BackingTier> = self
            .tiers
            .iter()
            .filter(|tier| !(skip_rdma && tier.source() == TierSource::Rdma))
            .cloned()
            .collect();
        if tiers.is_empty() {
            return false;
        }

        let deps = TaskDeps {
            tiers,
            state: Arc::clone(&self.state),
            max_prefetch_blocks: self.max_prefetch_blocks,
        };
        let input = TaskInput {
            req_id: req_id.to_string(),
            miss_keys: remaining.to_vec(),
            prefix,
            require_full: scan.require_full,
        };

        let handle = tokio::spawn(async move { run_task(deps, input).await });
        state.insert_active(req_id.to_string(), handle);
        true
    }
}
