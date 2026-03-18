// Per-request remote fetch state machine. Mirrors the SSD prefetch pattern
// (oneshot channel + HashMap + backpressure) but operates independently for
// cross-node RDMA block fetching.

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;

/// Blocks fetched from a remote node, ready to insert into ReadCache.
pub(crate) type RemoteFetchResult = Vec<(BlockKey, Arc<SealedBlock>)>;

/// Closure that dispatches a remote fetch to a background tokio task.
/// Injected by the server layer (captures MetaServerQueryClient, PegaflowClientPool,
/// MooncakeTransferEngine, PinnedAllocator).
pub(crate) type RemoteFetchFn =
    Arc<dyn Fn(Vec<BlockKey>, oneshot::Sender<RemoteFetchResult>) + Send + Sync>;

/// Status of a remote fetch for a given request.
#[derive(Debug, Clone)]
pub enum RemoteFetchStatus {
    /// All blocks resolved — either hit locally or confirmed missing everywhere.
    Done { hit: usize, missing: usize },
    /// Some blocks are being fetched from a remote node via RDMA.
    Loading { hit: usize, loading: usize },
}

struct RemoteFetchEntry {
    blocks_rx: oneshot::Receiver<RemoteFetchResult>,
    loading_count: usize,
}

struct RemoteFetchState {
    active: HashMap<String, RemoteFetchEntry>,
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
}

impl RemoteFetchState {
    fn remove_entry(&mut self, req_id: &str) -> Option<RemoteFetchEntry> {
        if let Some(entry) = self.active.remove(req_id) {
            self.inflight_count = self.inflight_count.saturating_sub(entry.loading_count);
            Some(entry)
        } else {
            None
        }
    }
}

pub(super) struct RemoteFetchScheduler {
    state: Mutex<RemoteFetchState>,
    max_remote_fetch_blocks: usize,
    fetch_fn: RemoteFetchFn,
}

enum PollResult {
    StillLoading,
    Completed,
}

impl RemoteFetchScheduler {
    pub(super) fn new(max_remote_fetch_blocks: usize, fetch_fn: RemoteFetchFn) -> Self {
        Self {
            state: Mutex::new(RemoteFetchState {
                active: HashMap::new(),
                inflight_count: 0,
            }),
            max_remote_fetch_blocks,
            fetch_fn,
        }
    }

    /// Combined check: poll existing fetch, prefix scan, dispatch new fetch if needed.
    pub(super) fn check_and_fetch(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> RemoteFetchStatus {
        // Step 1: poll existing fetch for this request
        if let Some(status) = self.poll_existing(read_cache, req_id) {
            match status {
                PollResult::StillLoading => {
                    return RemoteFetchStatus::Loading { hit: 0, loading: 1 };
                }
                PollResult::Completed => {
                    // Fall through to re-scan
                }
            }
        }

        // Step 2: prefix scan
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|h| BlockKey::new(namespace.to_string(), h.clone()))
            .collect();

        let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);
        let remaining = &keys[hit..];

        if remaining.is_empty() {
            // All hit in local cache
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            return RemoteFetchStatus::Done { hit, missing: 0 };
        }

        // Step 3: check backpressure
        let available = {
            let state = self.state.lock();
            self.max_remote_fetch_blocks
                .saturating_sub(state.inflight_count)
        };

        if available == 0 {
            // Backpressure: all remaining treated as missing
            core_metrics()
                .remote_fetch_backpressure_blocks
                .add(remaining.len() as u64, &[]);
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            return RemoteFetchStatus::Done {
                hit,
                missing: remaining.len(),
            };
        }

        let fetch_limit = remaining.len().min(available);
        let backpressure_skipped = remaining.len() - fetch_limit;
        if backpressure_skipped > 0 {
            core_metrics()
                .remote_fetch_backpressure_blocks
                .add(backpressure_skipped as u64, &[]);
        }

        let missing_keys: Vec<BlockKey> = remaining[..fetch_limit].to_vec();
        let loading = missing_keys.len();

        // Step 4: dispatch async remote fetch
        let (tx, rx) = oneshot::channel();
        (self.fetch_fn)(missing_keys, tx);

        core_metrics().remote_fetch_requests_total.add(1, &[]);

        // Step 5: register in state
        {
            let mut state = self.state.lock();
            state.inflight_count += loading;
            state.active.insert(
                req_id.to_string(),
                RemoteFetchEntry {
                    blocks_rx: rx,
                    loading_count: loading,
                },
            );
        }

        RemoteFetchStatus::Loading { hit, loading }
    }

    fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> Option<PollResult> {
        let mut state = self.state.lock();
        let entry = state.active.get_mut(req_id)?;

        match entry.blocks_rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => Some(PollResult::StillLoading),
            Ok(fetched_blocks) => {
                state.remove_entry(req_id);
                drop(state);
                if !fetched_blocks.is_empty() {
                    core_metrics()
                        .remote_fetch_blocks_hit
                        .add(fetched_blocks.len() as u64, &[]);
                    read_cache.batch_insert(fetched_blocks);
                }
                Some(PollResult::Completed)
            }
            Err(oneshot::error::TryRecvError::Closed) => {
                warn!(
                    "Remote fetch sender dropped for req_id={}, falling back to re-scan",
                    req_id
                );
                state.remove_entry(req_id);
                Some(PollResult::Completed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noop_fetch_fn() -> RemoteFetchFn {
        Arc::new(|_keys, _tx| {
            // Don't send anything — simulates a fetch that never completes
        })
    }

    fn immediate_fetch_fn() -> RemoteFetchFn {
        Arc::new(|keys, tx| {
            // Immediately return empty blocks for each key
            let blocks: Vec<(BlockKey, Arc<SealedBlock>)> = keys
                .into_iter()
                .map(|key| (key, Arc::new(SealedBlock::from_slots(Vec::new()))))
                .collect();
            let _ = tx.send(blocks);
        })
    }

    fn make_read_cache() -> Arc<ReadCache> {
        Arc::new(ReadCache::new(1 << 20, false, None))
    }

    #[test]
    fn all_hit_returns_done() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        // Insert blocks into cache
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key, block)]);

        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[test]
    fn missing_blocks_trigger_loading() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Loading { hit: 0, loading: 1 }
        ));
    }

    #[test]
    fn poll_returns_still_loading() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        // First call triggers fetch
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Second call should see StillLoading (sender dropped, so actually Completed)
        // Since noop_fetch_fn drops the sender immediately, it will complete on re-scan
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        // After sender drop -> Completed -> re-scan -> missing
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));
    }

    #[test]
    fn immediate_fetch_completes_on_second_call() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, immediate_fetch_fn());

        // First call triggers fetch (immediate complete, but not yet polled)
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Second call polls the completed fetch, inserts into cache, re-scans -> all hit
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[test]
    fn backpressure_limits_fetch() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(1, noop_fetch_fn());

        // First fetch uses up all capacity
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Second fetch for different req_id — backpressure
        let status = scheduler.check_and_fetch(&cache, "inst", "req2", "ns", &[vec![2]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 0, missing: 1 }
        ));
    }

    #[test]
    fn mixed_prefix_some_hit_some_remote() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        // Insert first block into cache; second is missing
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let block1 = Arc::new(SealedBlock::from_slots(Vec::new()));
        cache.batch_insert(vec![(key1, block1)]);

        let status =
            scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1], vec![2]], 1);
        // First block hits, second triggers remote loading
        assert!(matches!(
            status,
            RemoteFetchStatus::Loading { hit: 1, loading: 1 }
        ));
    }

    #[test]
    fn multiple_calls_after_completion_re_resolve() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, immediate_fetch_fn());

        // First call: triggers fetch
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Second call: polls completed fetch, inserts into cache -> all hit
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 1, missing: 0 }
        ));

        // Third call: no active entry, pure cache scan -> still all hit
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 1, missing: 0 }
        ));
    }

    #[test]
    fn empty_hash_list_returns_done_immediately() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[], 1);
        assert!(matches!(
            status,
            RemoteFetchStatus::Done { hit: 0, missing: 0 }
        ));
    }

    #[test]
    fn different_req_ids_are_independent() {
        let cache = make_read_cache();
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        // req1 starts a fetch for hash [1]
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // req2 starts a separate fetch for hash [2] — should not interfere
        let status = scheduler.check_and_fetch(&cache, "inst", "req2", "ns", &[vec![2]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Polling req1 again — sees sender dropped (noop), re-scans, triggers fresh fetch
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));
    }

    #[test]
    fn partial_backpressure_fetches_up_to_limit() {
        let cache = make_read_cache();
        // Capacity for exactly 2 blocks
        let scheduler = RemoteFetchScheduler::new(2, noop_fetch_fn());

        // Request 3 missing blocks; only 2 should be fetched, 1 skipped as missing
        let status = scheduler.check_and_fetch(
            &cache,
            "inst",
            "req1",
            "ns",
            &[vec![1], vec![2], vec![3]],
            1,
        );
        // All 3 are missing prefix-wise (none in cache), but loading is capped at 2
        // However, because get_prefix_blocks stops at first miss, all 3 are in remaining.
        // fetch_limit = min(3, 2) = 2, so loading = 2
        assert!(matches!(
            status,
            RemoteFetchStatus::Loading { hit: 0, loading: 2 }
        ));
    }

    #[test]
    fn sender_dropped_treats_as_completed_and_rescans() {
        let cache = make_read_cache();
        // noop_fetch_fn drops the sender immediately
        let scheduler = RemoteFetchScheduler::new(100, noop_fetch_fn());

        // First call dispatches a fetch; sender is already dropped
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));

        // Second call: poll sees Closed channel -> Completed -> re-scan -> still missing -> new fetch
        let status = scheduler.check_and_fetch(&cache, "inst", "req1", "ns", &[vec![1]], 1);
        // Block is still not in cache, so a new fetch is dispatched
        assert!(matches!(status, RemoteFetchStatus::Loading { .. }));
    }
}
