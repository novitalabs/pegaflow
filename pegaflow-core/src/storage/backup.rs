// Decode backup: a node without inference instances pulls retained LRU-tail
// blocks from prefill nodes so they can drop them first under pressure, and
// serves them back through the regular P2P path.

use std::sync::{Arc, Weak};
use std::time::Duration;

use bytesize::ByteSize;
use log::{debug, info, warn};
use tokio::time::Instant;

use super::ReadCache;
use crate::backing::{FetchPurpose, RdmaFetchStore};
use crate::block::BlockKey;
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;

/// Sleep when this node is not a target or the MetaServer has nothing queued.
const BACKUP_IDLE_SLEEP: Duration = Duration::from_secs(1);

/// How a backup target paces its pulls.
#[derive(Clone, Copy)]
pub(super) struct BackupPull {
    /// Most bytes one plan copies.
    pub(super) max_bytes: u64,
    /// Most bytes copied per second; `0` is unlimited.
    pub(super) bytes_per_sec: u64,
}

/// Pull and copy backup plans while the MetaServer names this node a target.
///
/// Exits once the read cache is dropped with the storage engine.
pub(super) async fn backup_loop(
    read_cache: Weak<ReadCache>,
    store: Arc<RdmaFetchStore>,
    client: Arc<MetaServerClient>,
    pull: BackupPull,
) {
    info!(
        "Decode backup pulls: max_bytes={} bandwidth={}",
        ByteSize(pull.max_bytes),
        match pull.bytes_per_sec {
            0 => "unlimited".to_string(),
            rate => format!("{}/s", ByteSize(rate)),
        }
    );
    while read_cache.strong_count() > 0 {
        if !client.is_backup_target() {
            tokio::time::sleep(BACKUP_IDLE_SLEEP).await;
            continue;
        }
        let start = Instant::now();
        match pull_once(&read_cache, &store, &client, pull.max_bytes).await {
            Some(bytes) => {
                tokio::time::sleep_until(start + pull_pause(bytes, pull.bytes_per_sec)).await;
            }
            None => tokio::time::sleep(BACKUP_IDLE_SLEEP).await,
        }
    }
}

/// Time `bytes` of copying cost against the bandwidth limit. The next pull
/// starts this long after the previous one started, so a fast pull waits out
/// the rest of its slot and idle time never banks a burst.
fn pull_pause(bytes: u64, bytes_per_sec: u64) -> Duration {
    if bytes_per_sec == 0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(bytes as f64 / bytes_per_sec as f64)
}

/// Copy one plan. Returns the bytes copied, or `None` when there was nothing
/// to copy and the loop should idle.
async fn pull_once(
    read_cache: &Weak<ReadCache>,
    store: &RdmaFetchStore,
    client: &MetaServerClient,
    max_bytes: u64,
) -> Option<u64> {
    let plan = match client.pull_backup_plan(max_bytes).await {
        Ok(plan) => plan,
        Err(e) => {
            warn!("Decode backup plan pull failed: {e}");
            return None;
        }
    };
    if plan.block_hashes.is_empty() {
        return None;
    }

    // A restarted MetaServer, a dropped registration, or a pairing change can
    // re-plan blocks this node already holds. Skip copying them, but register
    // them again: until the MetaServer sees this node as an owner it keeps
    // planning them and never tells the source they are backed up.
    let cache = read_cache.upgrade()?;
    let keys: Vec<BlockKey> = plan
        .block_hashes
        .iter()
        .map(|hash| BlockKey::new(plan.namespace.clone(), hash.clone()))
        .collect();
    // Hold the resident blocks until the registration is queued, like the
    // save and prefetch paths: pressure eviction skips referenced blocks, so
    // an eviction's removal cannot be queued ahead of this insert and leave a
    // stale owner on the MetaServer.
    let resident = cache.peek_blocks(&keys);
    drop(cache);
    let mut held = Vec::new();
    let mut wanted = Vec::new();
    for (hash, block) in plan.block_hashes.into_iter().zip(&resident) {
        if block.is_some() {
            held.push(hash);
        } else {
            wanted.push(hash);
        }
    }
    if !held.is_empty() {
        client.try_register_namespace_without_reclaim_hint(plan.namespace.clone(), held);
    }
    drop(resident);
    if wanted.is_empty() {
        return Some(0);
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
    if fetched.is_empty() {
        // Connect/RDMA failure or a source that evicted the whole batch.
        // Retrying at once would spin through the source's queue against an
        // unreachable peer.
        return None;
    }
    let bytes = fetched
        .iter()
        .map(|(_, block)| block.memory_footprint())
        .sum();

    let cache = read_cache.upgrade()?;
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
    Some(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pull_pause_spreads_bytes_over_the_limit() {
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;
        let us = Duration::from_micros;
        let cases = [
            ("unlimited", 64 * MIB, 0, Duration::ZERO),
            ("nothing copied", 0, GIB, Duration::ZERO),
            ("full plan", 64 * MIB, GIB, us(62_500)),
            ("partial plan", 16 * MIB, GIB, us(15_625)),
        ];
        for (name, bytes, rate, want) in cases {
            assert_eq!(pull_pause(bytes, rate), want, "{name}");
        }
    }
}
