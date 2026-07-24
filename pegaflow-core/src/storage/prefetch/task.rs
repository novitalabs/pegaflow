use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::warn;
use parking_lot::Mutex;

use crate::backing::PrefetchResult;
use crate::block::{BlockKey, SealedBlock};

use super::super::backing_tier::{BackingTier, TierQuery, TierSource};
use super::super::tier_attribution::TierDecision;
use super::TaskResult;
use super::state::{State, reserve_ssd};

const REMOTE_WAIT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const REMOTE_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

pub(super) struct TaskDeps {
    pub(super) tiers: Vec<BackingTier>,
    pub(super) state: Arc<Mutex<State>>,
    pub(super) max_prefetch_blocks: usize,
}

/// `miss_keys` is never empty: `scan_prefix` only starts a task when the
/// cache scan left a miss suffix.
pub(super) struct TaskInput {
    pub(super) req_id: String,
    pub(super) miss_keys: Vec<BlockKey>,
    pub(super) prefix: Vec<Arc<SealedBlock>>,
    pub(super) require_full: bool,
}

pub(super) struct TierAttempt {
    pub(super) source: TierSource,
    pub(super) committed: usize,
    pub(super) blocks: PrefetchResult,
}

struct AttemptCtx<'a> {
    query: &'a TierQuery<'a>,
    state: &'a Arc<Mutex<State>>,
    max_prefetch_blocks: usize,
}

pub(super) async fn run_task(deps: TaskDeps, input: TaskInput) -> TaskResult {
    let hit = input.prefix.len();
    let total = hit + input.miss_keys.len();
    let miss_hashes: Vec<Vec<u8>> = input.miss_keys.iter().map(|k| k.hash.clone()).collect();
    let query = TierQuery {
        req_id: &input.req_id,
        namespace: &input.miss_keys[0].namespace,
        keys: &input.miss_keys,
        hashes: &miss_hashes,
        require_full: input.require_full,
    };
    let ctx = AttemptCtx {
        query: &query,
        state: &deps.state,
        max_prefetch_blocks: deps.max_prefetch_blocks,
    };

    let mut winner = None;
    for tier in &deps.tiers {
        if let Some(attempt) = try_tier(tier, &ctx).await {
            winner = Some(attempt);
            break;
        }
    }

    if winner.is_none() && input.require_full {
        winner = wait_for_remote(&deps.tiers, &ctx).await;
    }

    TierDecision {
        total,
        hit,
        tier_blocks: winner.as_ref().map_or(0, |attempt| attempt.committed),
        source: winner.as_ref().map(|attempt| attempt.source),
    }
    .record();

    assemble(input.prefix, input.miss_keys, winner)
}

/// `None` means "nothing usable from this tier, try the next one".
async fn try_tier(tier: &BackingTier, ctx: &AttemptCtx<'_>) -> Option<TierAttempt> {
    let plan = tier.plan(ctx.query).await?;
    let candidate = plan.len();

    let (take, _reservation) = if tier.source() == TierSource::Ssd {
        let (reserved, guard) = reserve_ssd(
            Arc::clone(ctx.state),
            ctx.max_prefetch_blocks,
            candidate,
            ctx.query.require_full,
        )?;
        (reserved, Some(guard))
    } else {
        (candidate, None)
    };

    let (committed, blocks) = tier.fetch(ctx.query, &plan, take).await;
    // Same length policy as `plan`: a budget-capped or short fetch is not a
    // usable answer for an all-or-nothing caller.
    if !ctx.query.accepts(committed) {
        return None;
    }
    Some(TierAttempt {
        source: tier.source(),
        committed,
        blocks,
    })
}

/// An all-or-nothing caller blocks on a prefix that a peer may still be
/// producing, so poll RDMA until it appears. SSD is not retried: its content
/// only grows through this node's own save path, which cannot complete while
/// the caller waits on this prefix.
async fn wait_for_remote(tiers: &[BackingTier], ctx: &AttemptCtx<'_>) -> Option<TierAttempt> {
    let rdma = tiers
        .iter()
        .find(|tier| tier.source() == TierSource::Rdma)?;

    let started_at = Instant::now();
    while started_at.elapsed() < REMOTE_WAIT_TIMEOUT {
        tokio::time::sleep(REMOTE_WAIT_POLL_INTERVAL).await;
        if let Some(attempt) = try_tier(rdma, ctx).await {
            return Some(attempt);
        }
    }

    warn!(
        "Timed out waiting for remote prefix: req_id={} timeout_secs={}",
        ctx.query.req_id,
        REMOTE_WAIT_TIMEOUT.as_secs()
    );
    None
}

/// Rebuild the ready prefix: local prefix blocks followed by fetched blocks
/// in requested-key order, stopping at the first gap.
pub(super) fn assemble(
    prefix: Vec<Arc<SealedBlock>>,
    miss_keys: Vec<BlockKey>,
    attempt: Option<TierAttempt>,
) -> TaskResult {
    let total = prefix.len() + miss_keys.len();
    let (source, committed, inserts) = match attempt {
        Some(attempt) => (Some(attempt.source), attempt.committed, attempt.blocks),
        None => (None, 0, Vec::new()),
    };

    let mut ready_blocks = prefix;
    {
        let by_key: HashMap<_, _> = inserts.iter().map(|(key, block)| (key, block)).collect();
        ready_blocks.extend(
            miss_keys[..committed]
                .iter()
                .map_while(|key| by_key.get(key).map(|block| Arc::clone(*block))),
        );
    }
    let missing = total.saturating_sub(ready_blocks.len());
    TaskResult {
        source,
        committed,
        inserts,
        ready_blocks,
        missing,
    }
}

/// RDMA-fetched blocks that survived cache admission are resident here and
/// get re-advertised to the MetaServer so peers can fetch from this node.
/// SSD prefetches never advertise: the save path already registered them,
/// and eviction explicitly unregisters.
pub(super) fn rdma_advert(resident: &[BlockKey]) -> Option<(String, Vec<Vec<u8>>)> {
    if resident.is_empty() {
        return None;
    }

    let namespace = resident[0].namespace.clone();
    let hashes = resident.iter().map(|key| key.hash.clone()).collect();
    Some((namespace, hashes))
}
