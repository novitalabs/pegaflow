use std::sync::Arc;

use parking_lot::Mutex;

use crate::storage::backing_tier::{BackingTier, FakeTier, TierSource};
use crate::storage::prefetch::state::{State, reserve_ssd};
use crate::storage::prefetch::task::{TaskDeps, TaskInput, run_task};

use super::{key, keyed_blocks, tracked, untracked};

fn deps(tiers: Vec<BackingTier>, max_prefetch_blocks: usize) -> TaskDeps {
    TaskDeps {
        tiers,
        state: Arc::new(Mutex::new(State::new())),
        max_prefetch_blocks,
    }
}

fn deps_state(
    tiers: Vec<BackingTier>,
    max_prefetch_blocks: usize,
) -> (Arc<Mutex<State>>, TaskDeps) {
    let deps = deps(tiers, max_prefetch_blocks);
    (Arc::clone(&deps.state), deps)
}

fn input(n_miss: u8) -> TaskInput {
    TaskInput {
        req_id: "req".to_string(),
        miss_keys: (0..n_miss).map(key).collect(),
        prefix: Vec::new(),
        require_full: false,
    }
}

fn full_input(n_miss: u8) -> TaskInput {
    TaskInput {
        require_full: true,
        ..input(n_miss)
    }
}

#[tokio::test]
async fn first_tier_wins() {
    let first = untracked(FakeTier::new(TierSource::Rdma, 3).with_blocks(keyed_blocks(3)));
    let (second, second_tier) = tracked(FakeTier::new(TierSource::Ssd, 3));

    let result = run_task(deps(vec![first, second_tier], 100), input(3)).await;

    assert_eq!(result.source, Some(TierSource::Rdma));
    assert_eq!(result.committed, 3);
    assert_eq!(second.plan_calls(), 0, "winning tier must short-circuit");
}

#[tokio::test]
async fn no_plan_falls() {
    let (rdma, rdma_tier) = tracked(FakeTier::new(TierSource::Rdma, 0));
    let ssd = untracked(FakeTier::new(TierSource::Ssd, 2).with_blocks(keyed_blocks(2)));

    let result = run_task(deps(vec![rdma_tier, ssd], 100), input(2)).await;

    assert_eq!(rdma.fetch_calls(), 0, "no plan means no fetch");
    assert_eq!(result.source, Some(TierSource::Ssd));
    assert_eq!(result.committed, 2);
}

#[tokio::test]
async fn ssd_budget_caps() {
    let ssd = untracked(FakeTier::new(TierSource::Ssd, 10).with_blocks(keyed_blocks(4)));
    let (state, deps) = deps_state(vec![ssd], 4);

    let result = run_task(deps, input(10)).await;

    assert_eq!(result.committed, 4, "budget must cap the fetch");
    let (regrant, _hold) =
        reserve_ssd(state, 4, 4, false).expect("budget must be released after the task");
    assert_eq!(regrant, 4);
}

#[tokio::test]
async fn no_budget_skips() {
    let (ssd, ssd_tier) = tracked(FakeTier::new(TierSource::Ssd, 5));
    let (state, deps) = deps_state(vec![ssd_tier], 8);
    let hold = reserve_ssd(Arc::clone(&state), 8, 8, false).expect("fill the budget");

    let result = run_task(deps, input(5)).await;

    assert_eq!(ssd.fetch_calls(), 0, "exhausted budget must skip the fetch");
    assert_eq!(result.committed, 0);
    drop(hold);
}

#[test]
fn strict_reservation_is_all_or_nothing() {
    let state = Arc::new(Mutex::new(State::new()));
    let (_granted, hold) =
        reserve_ssd(Arc::clone(&state), 10, 6, false).expect("reservation within capacity");
    // 4 of 10 slots remain.

    assert!(
        reserve_ssd(Arc::clone(&state), 10, 5, true).is_none(),
        "a strict request above availability must be denied"
    );
    let (regrant, hold2) = reserve_ssd(Arc::clone(&state), 10, 4, true)
        .expect("a strict request within availability must be granted in full");
    assert_eq!(regrant, 4, "denial must not have reserved anything");
    drop(hold2);

    let (partial, _hold3) =
        reserve_ssd(Arc::clone(&state), 10, 5, false).expect("partial reservation");
    assert_eq!(partial, 4, "non-strict still takes what it can get");
    drop(hold);
}

#[tokio::test]
async fn full_prefix_rejects_short_plan() {
    // The tier covers 2 of the 3 missing blocks: useless to an
    // all-or-nothing caller, so it must not fetch.
    let (ssd, ssd_tier) = tracked(FakeTier::new(TierSource::Ssd, 2));

    let result = run_task(deps(vec![ssd_tier], 100), full_input(3)).await;

    assert_eq!(ssd.fetch_calls(), 0, "a short plan must not transfer data");
    assert_eq!(result.source, None);
    assert_eq!(result.committed, 0);
    assert_eq!(result.missing, 3);
}

#[tokio::test]
async fn full_prefix_accepts_complete_plan() {
    let ssd = untracked(FakeTier::new(TierSource::Ssd, 3).with_blocks(keyed_blocks(3)));

    let result = run_task(deps(vec![ssd], 100), full_input(3)).await;

    assert_eq!(result.source, Some(TierSource::Ssd));
    assert_eq!(result.committed, 3);
    assert_eq!(result.missing, 0);
}

#[tokio::test]
async fn full_prefix_rejects_budget_capped_fetch() {
    // Plan covers the whole suffix but the budget can only fund part of it.
    let (ssd, ssd_tier) = tracked(FakeTier::new(TierSource::Ssd, 5).with_blocks(keyed_blocks(5)));
    let (state, deps) = deps_state(vec![ssd_tier], 8);
    let hold = reserve_ssd(Arc::clone(&state), 8, 5, false).expect("leave 3 of 8 slots");

    let result = run_task(deps, full_input(5)).await;

    assert_eq!(
        ssd.fetch_calls(),
        0,
        "a capped budget must be denied outright, not fetched partially"
    );
    assert_eq!(result.committed, 0);
    drop(hold);
}
