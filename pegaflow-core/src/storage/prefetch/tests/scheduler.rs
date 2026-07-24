use std::time::Duration;

use crate::block::PrefetchStatus;
use crate::storage::backing_tier::{BackingTier, FakeTier, TierSource};
use crate::storage::prefetch::scan::tier_lineup;
use crate::storage::prefetch::{Scheduler, TaskResult};

use super::{Harness, hashes, keyed_blocks, tracked, untracked};

#[test]
fn rdma_first() {
    let lineup: Vec<TierSource> = tier_lineup(
        Some(untracked(FakeTier::new(TierSource::Rdma, 0))),
        Some(untracked(FakeTier::new(TierSource::Ssd, 0))),
    )
    .iter()
    .map(BackingTier::source)
    .collect();
    assert_eq!(lineup, [TierSource::Rdma, TierSource::Ssd]);
}

#[tokio::test]
async fn rdma_shortfall() {
    // RDMA commits 2 blocks (plan = 2, fetch commits take) but delivers none.
    let (rdma, rdma_tier) = tracked(FakeTier::new(TierSource::Rdma, 2));
    let (ssd, ssd_tier) = tracked(FakeTier::new(TierSource::Ssd, 0));
    let harness = Harness::new(vec![rdma_tier, ssd_tier], 100);
    let query_hashes = hashes(2);

    let status = harness.await_ready("r1", &query_hashes).await;
    let PrefetchStatus::Ready { blocks, missing } = status else {
        panic!("expected Ready");
    };
    assert!(blocks.is_empty());
    assert_eq!(missing, 2);
    assert_eq!(
        ssd.plan_calls(),
        0,
        "a shortfall is terminal for the task: no same-task SSD fallback"
    );

    // The next lookup for the same request skips RDMA and consults SSD.
    let status = harness.await_ready("r1", &query_hashes).await;
    let PrefetchStatus::Ready { missing, .. } = status else {
        panic!("expected Ready");
    };
    assert_eq!(missing, 2);
    assert_eq!(
        rdma.plan_calls(),
        1,
        "failed-remote request must not re-trigger RDMA"
    );
    assert_eq!(ssd.plan_calls(), 1);
}

/// Feed a finished task carrying the given outcome through `poll_task` and
/// report whether the request came out RDMA-blacklisted.
async fn poll_marks_rdma(source: Option<TierSource>, committed: usize, delivered: u8) -> bool {
    let harness = Harness::new(Vec::new(), 100);
    let result = TaskResult {
        source,
        committed,
        inserts: keyed_blocks(delivered),
        ready_blocks: Vec::new(),
        missing: 0,
    };
    let handle = tokio::spawn(async move { result });
    while !handle.is_finished() {
        tokio::task::yield_now().await;
    }
    harness
        .scheduler
        .state
        .lock()
        .insert_active("req".to_string(), handle);

    let _ = harness.scheduler.poll_task(&harness.cache, "req").await;

    harness.scheduler.state.lock().rdma_failed("req")
}

/// `rdma_shortfall` covers the end-to-end shortfall path; this pins which
/// task outcomes count as a shortfall in the first place.
#[tokio::test]
async fn only_short_rdma_delivery_is_marked() {
    // (case, source, committed, delivered, marked)
    let cases = [
        ("rdma partial", Some(TierSource::Rdma), 3, 2, true),
        ("rdma empty", Some(TierSource::Rdma), 3, 0, true),
        ("rdma full", Some(TierSource::Rdma), 3, 3, false),
        ("ssd partial", Some(TierSource::Ssd), 3, 2, false),
        ("no tier", None, 0, 0, false),
    ];

    for (id, source, committed, delivered, marked) in cases {
        assert_eq!(
            poll_marks_rdma(source, committed, delivered).await,
            marked,
            "case: {id}"
        );
    }
}

#[test]
fn gc_sweeps_only_expired_rdma_marks() {
    let scheduler = Scheduler::with_tiers(Vec::new(), 100);
    scheduler.state.lock().mark_rdma("req");

    let (_, swept) = scheduler.gc_stale(Duration::ZERO, Duration::from_secs(60));
    assert_eq!(swept, 0, "a fresh mark must survive the sweep");

    let (_, swept) = scheduler.gc_stale(Duration::ZERO, Duration::ZERO);
    assert_eq!(swept, 1);
    assert!(!scheduler.state.lock().rdma_failed("req"));
}
