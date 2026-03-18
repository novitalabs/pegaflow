// Per-request prefetch state machine. A single Mutex is sufficient because
// prefetch operations are per-query (low frequency, never a bottleneck).
//
// Handles both SSD prefetch and cross-node remote fetch (mutually exclusive).

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::backing::{PrefetchResult, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus};
use crate::metrics::core_metrics;
use crate::storage::remote_fetch::RemoteFetchFn;

use super::read_cache::ReadCache;

struct PrefetchEntry {
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    hit_count: usize,
    loading_count: usize,
    /// True when this entry was dispatched via remote RDMA fetch (for metrics).
    is_remote: bool,
}

struct PrefetchState {
    active: HashMap<String, PrefetchEntry>,
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
}

impl PrefetchState {
    fn remove_entry(&mut self, req_id: &str) -> Option<PrefetchEntry> {
        if let Some(entry) = self.active.remove(req_id) {
            self.inflight_count = self.inflight_count.saturating_sub(entry.loading_count);
            Some(entry)
        } else {
            None
        }
    }
}

pub(super) struct PrefetchScheduler {
    state: Mutex<PrefetchState>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    max_prefetch_blocks: usize,
    remote_fetch_fn: Option<RemoteFetchFn>,
    max_remote_fetch_blocks: usize,
}

impl PrefetchScheduler {
    pub(super) fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        max_prefetch_blocks: usize,
        remote_fetch: Option<(RemoteFetchFn, usize)>,
    ) -> Self {
        let (remote_fetch_fn, max_remote_fetch_blocks) =
            remote_fetch.map(|(f, m)| (Some(f), m)).unwrap_or((None, 0));
        Self {
            state: Mutex::new(PrefetchState {
                active: HashMap::new(),
                inflight_count: 0,
            }),
            ssd_store,
            max_prefetch_blocks,
            remote_fetch_fn,
            max_remote_fetch_blocks,
        }
    }

    pub(super) async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        if let Some(status) = self.poll_existing(read_cache, req_id) {
            match status {
                PollResult::StillLoading { hit, loading } => {
                    return PrefetchStatus::Loading { hit, loading };
                }
                PollResult::Completed => {
                    // Fall through to full scan
                }
            }
        }

        self.full_prefix_scan(
            read_cache,
            instance_id,
            req_id,
            namespace,
            hashes,
            num_workers,
        )
        .await
    }

    fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> Option<PollResult> {
        let mut state = self.state.lock();
        let entry = state.active.get_mut(req_id)?;

        match entry.blocks_rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => Some(PollResult::StillLoading {
                hit: entry.hit_count,
                loading: entry.loading_count,
            }),
            Ok(blocks) => {
                let is_remote = entry.is_remote;
                state.remove_entry(req_id);
                drop(state);
                if is_remote && !blocks.is_empty() {
                    core_metrics()
                        .remote_fetch_blocks_hit
                        .add(blocks.len() as u64, &[]);
                }
                read_cache.batch_insert(blocks);
                Some(PollResult::Completed)
            }
            Err(oneshot::error::TryRecvError::Closed) => {
                let source = if entry.is_remote {
                    "Remote fetch"
                } else {
                    "Backing prefetch"
                };
                warn!(
                    "{} sender dropped for req_id={}, falling back to re-scan",
                    source, req_id
                );
                state.remove_entry(req_id);
                Some(PollResult::Completed)
            }
        }
    }

    async fn full_prefix_scan(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();

        let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);

        let remaining = &keys[hit..];
        let mut loading = 0usize;
        let mut blocks_rx: Option<oneshot::Receiver<PrefetchResult>> = None;
        let mut is_remote_entry = false;

        let is_remote = self.remote_fetch_fn.is_some();
        let has_backing = self.ssd_store.is_some() || is_remote;
        let max_blocks = if is_remote {
            self.max_remote_fetch_blocks
        } else {
            self.max_prefetch_blocks
        };

        let check_keys = if !remaining.is_empty() && has_backing {
            let available = {
                let state = self.state.lock();
                max_blocks.saturating_sub(state.inflight_count)
            };
            if available > 0 {
                let check_limit = remaining.len().min(available);
                let backpressure_skipped = remaining.len() - check_limit;
                if backpressure_skipped > 0 {
                    if is_remote {
                        core_metrics()
                            .remote_fetch_backpressure_blocks
                            .add(backpressure_skipped as u64, &[]);
                    } else {
                        core_metrics()
                            .ssd_prefetch_backpressure_blocks
                            .add(backpressure_skipped as u64, &[]);
                    }
                }
                Some(remaining[..check_limit].to_vec())
            } else {
                if is_remote {
                    core_metrics()
                        .remote_fetch_backpressure_blocks
                        .add(remaining.len() as u64, &[]);
                } else {
                    core_metrics()
                        .ssd_prefetch_backpressure_blocks
                        .add(remaining.len() as u64, &[]);
                }
                None
            }
        } else {
            None
        };

        // SSD prefetch path
        if let Some(check_keys) = &check_keys
            && let Some(ssd) = &self.ssd_store
        {
            let (found, rx) = ssd.submit_prefix(check_keys.clone());
            if found > 0 {
                self.state.lock().inflight_count += found;
                loading = found;
                blocks_rx = Some(rx);
            }
        }
        // Remote fetch path (mutually exclusive with SSD)
        else if let Some(check_keys) = &check_keys
            && let Some(ref fetch_fn) = self.remote_fetch_fn
        {
            let (tx, rx) = oneshot::channel();
            (fetch_fn)(check_keys.clone(), tx);
            loading = check_keys.len();
            blocks_rx = Some(rx);
            self.state.lock().inflight_count += loading;
            core_metrics().remote_fetch_requests_total.add(1, &[]);
            is_remote_entry = true;
        }

        let missing = keys.len() - hit - loading;

        if loading > 0 {
            if let Some(rx) = blocks_rx {
                let mut state = self.state.lock();
                state.active.insert(
                    req_id.to_string(),
                    PrefetchEntry {
                        blocks_rx: rx,
                        hit_count: hit,
                        loading_count: loading,
                        is_remote: is_remote_entry,
                    },
                );
            }
            PrefetchStatus::Loading { hit, loading }
        } else {
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            PrefetchStatus::Done { hit, missing }
        }
    }
}

enum PollResult {
    StillLoading { hit: usize, loading: usize },
    Completed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::SealedBlock;

    fn make_read_cache() -> Arc<ReadCache> {
        Arc::new(ReadCache::new(1 << 20, false, None))
    }

    fn noop_remote_fetch_fn() -> RemoteFetchFn {
        Arc::new(|_keys, _tx| {
            // Don't send anything — simulates a fetch that never completes
        })
    }

    fn hanging_remote_fetch_fn() -> RemoteFetchFn {
        Arc::new(|_keys, tx| {
            tokio::spawn(async move {
                let _tx = tx;
                std::future::pending::<()>().await;
            });
        })
    }

    fn immediate_remote_fetch_fn() -> RemoteFetchFn {
        Arc::new(|keys, tx| {
            let blocks: Vec<(BlockKey, Arc<SealedBlock>)> = keys
                .into_iter()
                .map(|key| (key, Arc::new(SealedBlock::from_slots(Vec::new()))))
                .collect();
            let _ = tx.send(blocks);
        })
    }

    // ---- SSD-less, remote-less tests ----

    #[tokio::test]
    async fn all_hit_no_backing() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, None);

        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key, block)]);

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[tokio::test]
    async fn all_miss_no_backing() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, None);

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 0, missing: 1 }
        ));
    }

    // ---- Remote fetch tests ----

    #[tokio::test]
    async fn remote_fetch_dispatched_when_configured() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Loading { hit: 0, loading: 1 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_all_hit_returns_done() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        // Insert block into cache
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key, block)]);

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_immediate_completes_on_second_call() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((immediate_remote_fetch_fn(), 100)));

        // First call triggers fetch
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));

        // Second call polls completed fetch, inserts into cache, re-scans -> all hit
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_backpressure_limits_fetch() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 1)));

        // First fetch uses up all capacity
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));

        // Second fetch for different req_id — backpressure
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req2", "ns", &[vec![2]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 0, missing: 1 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_mixed_prefix_some_hit_some_remote() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        // Insert first block; second is missing
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let block1 = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key1, block1)]);

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1], vec![2]], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Loading { hit: 1, loading: 1 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_repoll_preserves_original_counts() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((hanging_remote_fetch_fn(), 100)));

        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key, block)]);

        let status = scheduler
            .check_and_prefetch(
                &cache,
                "inst",
                "req1",
                "ns",
                &[vec![1], vec![2], vec![3]],
                1,
            )
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Loading { hit: 1, loading: 2 }
        ));

        let status = scheduler
            .check_and_prefetch(
                &cache,
                "inst",
                "req1",
                "ns",
                &[vec![1], vec![2], vec![3]],
                1,
            )
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Loading { hit: 1, loading: 2 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_partial_backpressure() {
        let cache = make_read_cache();
        // Capacity for exactly 2 blocks
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 2)));

        let status = scheduler
            .check_and_prefetch(
                &cache,
                "inst",
                "req1",
                "ns",
                &[vec![1], vec![2], vec![3]],
                1,
            )
            .await;
        // All 3 are missing, but loading is capped at 2
        assert!(matches!(
            status,
            PrefetchStatus::Loading { hit: 0, loading: 2 }
        ));
    }

    #[tokio::test]
    async fn remote_fetch_sender_dropped_rescans() {
        let cache = make_read_cache();
        // noop drops the sender immediately
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        // First call dispatches; sender is already dropped
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));

        // Second call: poll sees Closed -> Completed -> re-scan -> still missing -> new fetch
        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));
    }

    #[tokio::test]
    async fn remote_fetch_different_req_ids_independent() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[vec![1]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req2", "ns", &[vec![2]], 1)
            .await;
        assert!(matches!(status, PrefetchStatus::Loading { .. }));
    }

    #[tokio::test]
    async fn remote_fetch_empty_hashes() {
        let cache = make_read_cache();
        let scheduler = PrefetchScheduler::new(None, 100, Some((noop_remote_fetch_fn(), 100)));

        let status = scheduler
            .check_and_prefetch(&cache, "inst", "req1", "ns", &[], 1)
            .await;
        assert!(matches!(
            status,
            PrefetchStatus::Done { hit: 0, missing: 0 }
        ));
    }
}
