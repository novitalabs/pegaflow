// Decode backup: a node without inference instances pulls retained LRU-tail
// blocks from prefill nodes so they can drop them first under pressure, and
// serves them back through the regular P2P path.

use std::sync::{Arc, Weak};
use std::time::Duration;

use log::{debug, warn};

use super::ReadCache;
use crate::backing::{FetchPurpose, RdmaFetchStore};
use crate::block::BlockKey;
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

/// Upper bound on one pulled plan; matches the RDMA staging chunk size.
const BACKUP_PULL_MAX_BYTES: u64 = 256 * 1024 * 1024;
/// Sleep when this node is not a target or the MetaServer has nothing queued;
/// also the cap on a pacing retry hint.
const BACKUP_IDLE_SLEEP: Duration = Duration::from_secs(1);

/// Pull and copy backup plans while the MetaServer names this node a target.
///
/// Exits once the read cache is dropped with the storage engine.
pub(super) async fn backup_loop(
    read_cache: Weak<ReadCache>,
    store: Arc<RdmaFetchStore>,
    client: Arc<MetaServerClient>,
) {
    while read_cache.strong_count() > 0 {
        let pause = if client.is_backup_target() {
            pull_once(&read_cache, &store, &client).await
        } else {
            BACKUP_IDLE_SLEEP
        };
        if !pause.is_zero() {
            tokio::time::sleep(pause).await;
        }
    }
}

/// Copy one plan. Returns how long to wait before the next pull.
async fn pull_once(
    read_cache: &Weak<ReadCache>,
    store: &RdmaFetchStore,
    client: &MetaServerClient,
) -> Duration {
    let plan = match client.pull_backup_plan(BACKUP_PULL_MAX_BYTES).await {
        Ok(plan) => plan,
        Err(e) => {
            warn!("Decode backup plan pull failed: {e}");
            return BACKUP_IDLE_SLEEP;
        }
    };
    if plan.block_hashes.is_empty() {
        // Pacing tells us when the next plan is due; sleeping the full idle
        // period instead would cap backup far below the configured rate.
        return match plan.retry_after_ms {
            0 => BACKUP_IDLE_SLEEP,
            ms => Duration::from_millis(ms).min(BACKUP_IDLE_SLEEP),
        };
    }

    // A restarted MetaServer or an expired in-flight entry can re-plan blocks
    // this node already holds; skip them instead of copying twice.
    let wanted: Vec<Vec<u8>> = {
        let Some(cache) = read_cache.upgrade() else {
            return Duration::ZERO;
        };
        let keys: Vec<BlockKey> = plan
            .block_hashes
            .iter()
            .map(|hash| BlockKey::new(plan.namespace.clone(), hash.clone()))
            .collect();
        plan.block_hashes
            .into_iter()
            .zip(cache.contains_keys(&keys))
            .filter_map(|(hash, present)| (!present).then_some(hash))
            .collect()
    };
    if wanted.is_empty() {
        return Duration::ZERO;
    }

    let req_id = format!("backup:{}", plan.source_node);
    let fetched = store
        .fetch_blocks(
            &plan.source_node,
            &req_id,
            &plan.namespace,
            &wanted,
            FetchPurpose::Backup,
        )
        .await;
    let missing = wanted.len() - fetched.len();
    if missing > 0 {
        core_metrics()
            .backup_pull_missing_blocks
            .add(missing as u64, &[]);
    }

    let Some(cache) = read_cache.upgrade() else {
        return Duration::ZERO;
    };
    // Backups enter retained like local saves, so this node's own LRU decides
    // what it drops, and they are advertised so prefill nodes can recall them.
    let resident = cache.batch_insert_refs(&fetched);
    debug!(
        "Decode backup pulled: source={} namespace={} planned={} fetched={} resident={}",
        plan.source_node,
        plan.namespace,
        wanted.len(),
        fetched.len(),
        resident.len()
    );
    client.try_register_namespace_without_reclaim_hint(
        plan.namespace,
        resident.into_iter().map(|key| key.hash).collect(),
    );
    Duration::ZERO
}
