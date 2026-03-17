// ============================================================================
// PrefetchScheduler: backing-store prefetch with unified concurrency.
//
// Replaces the old DashMap<String, PrefetchEntry> + AtomicUsize combo
// with a single Mutex<PrefetchState>. Prefetch operations are low-frequency
// (once per query), so the lock is never a bottleneck.
//
// Entries are cleaned up when the same req_id is polled again.
// ============================================================================

use std::collections::HashMap;
use std::sync::Arc;

use log::warn;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::backing::{P2pBackingStore, PrefetchResult, SsdBackingStore};
use crate::block::{BlockKey, PrefetchStatus};
use crate::metrics::core_metrics;

use super::read_cache::ReadCache;

// ============================================================================
// PrefetchEntry (per-request state)
// ============================================================================

struct PrefetchEntry {
    /// Delivers batch of successfully-read blocks when all backing I/O finishes.
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    /// Number of blocks submitted to the backing store (for backpressure).
    loading_count: usize,
}

// ============================================================================
// PrefetchState (single lock)
// ============================================================================

struct PrefetchState {
    /// Active prefetch receivers, keyed by req_id.
    active: HashMap<String, PrefetchEntry>,
    /// Total blocks currently being read from backing store (backpressure).
    ///
    /// Invariant: `inflight_count == active.values().map(|e| e.loading_count).sum()`
    inflight_count: usize,
}

impl PrefetchState {
    /// Remove an entry and maintain the inflight_count invariant.
    fn remove_entry(&mut self, req_id: &str) -> Option<PrefetchEntry> {
        if let Some(entry) = self.active.remove(req_id) {
            self.inflight_count = self.inflight_count.saturating_sub(entry.loading_count);
            Some(entry)
        } else {
            None
        }
    }
}

// ============================================================================
// PrefetchScheduler
// ============================================================================

pub(crate) struct PrefetchScheduler {
    state: Mutex<PrefetchState>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    p2p_store: Option<Arc<P2pBackingStore>>,
    max_prefetch_blocks: usize,
}

impl PrefetchScheduler {
    pub fn new(
        ssd_store: Option<Arc<SsdBackingStore>>,
        p2p_store: Option<Arc<P2pBackingStore>>,
        max_prefetch_blocks: usize,
    ) -> Self {
        Self {
            state: Mutex::new(PrefetchState {
                active: HashMap::new(),
                inflight_count: 0,
            }),
            ssd_store,
            p2p_store,
            max_prefetch_blocks,
        }
    }

    /// Check prefix blocks and schedule backing-store reads if needed.
    ///
    /// Uses per-request state machine: first call does full scan + dispatches
    /// backing reads; retries poll a oneshot receiver. Pins hit blocks on Done.
    pub async fn check_and_prefetch(
        &self,
        read_cache: &ReadCache,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        // Fast path: poll existing prefetch for this req_id
        if let Some(status) = self.poll_existing(read_cache, req_id) {
            match status {
                PollResult::StillLoading => {
                    return PrefetchStatus::Loading { hit: 0, loading: 1 };
                }
                PollResult::Completed => {
                    // Fall through to full scan
                }
            }
        }

        // Full scan: cache → backing store → pin
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

    // ====================================================================
    // Internal
    // ====================================================================

    /// Poll an existing prefetch entry. Returns None if no entry exists.
    fn poll_existing(&self, read_cache: &ReadCache, req_id: &str) -> Option<PollResult> {
        let mut state = self.state.lock();
        let entry = state.active.get_mut(req_id)?;

        match entry.blocks_rx.try_recv() {
            Err(oneshot::error::TryRecvError::Empty) => Some(PollResult::StillLoading),
            Ok(ssd_blocks) => {
                state.remove_entry(req_id);
                drop(state);
                read_cache.batch_insert(ssd_blocks);
                Some(PollResult::Completed)
            }
            Err(oneshot::error::TryRecvError::Closed) => {
                warn!(
                    "Backing prefetch sender dropped for req_id={}, falling back to re-scan",
                    req_id
                );
                state.remove_entry(req_id);
                Some(PollResult::Completed)
            }
        }
    }

    /// Full prefix scan: cache → backing stores → pin.
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

        // Phase 1: cache prefix scan
        let (hit, blocks_to_pin) = read_cache.get_prefix_blocks(&keys);

        // Phase 2: backing store prefix scan for remaining keys
        let remaining = &keys[hit..];
        let mut loading = 0usize;
        let mut blocks_rx: Option<oneshot::Receiver<PrefetchResult>> = None;

        let has_backing = self.p2p_store.is_some() || self.ssd_store.is_some();

        // Compute check_keys under the lock, then release it before any async work.
        let check_keys = if !remaining.is_empty() && has_backing {
            let available = {
                let state = self.state.lock();
                self.max_prefetch_blocks
                    .saturating_sub(state.inflight_count)
            };
            if available > 0 {
                let check_limit = remaining.len().min(available);
                let backpressure_skipped = remaining.len() - check_limit;
                if backpressure_skipped > 0 {
                    core_metrics()
                        .ssd_prefetch_backpressure_blocks
                        .add(backpressure_skipped as u64, &[]);
                }
                Some(remaining[..check_limit].to_vec())
            } else {
                core_metrics()
                    .ssd_prefetch_backpressure_blocks
                    .add(remaining.len() as u64, &[]);
                None
            }
        } else {
            None
        };

        // Submit to backing stores (no lock held across await).
        if let Some(check_keys) = check_keys {
            // TODO: make backing store priority configurable
            // Try P2P first (remote RDMA, highest bandwidth)
            if loading == 0
                && let Some(p2p) = &self.p2p_store
            {
                let (found, rx) = p2p.submit_prefix(check_keys.clone()).await;
                if found > 0 {
                    self.state.lock().inflight_count += found;
                    loading = found;
                    blocks_rx = Some(rx);
                }
            }

            // Then SSD (local disk)
            if loading == 0
                && let Some(ssd) = &self.ssd_store
            {
                let (found, rx) = ssd.submit_prefix(check_keys);
                if found > 0 {
                    self.state.lock().inflight_count += found;
                    loading = found;
                    blocks_rx = Some(rx);
                }
            }
        }

        let missing = keys.len() - hit - loading;

        // Phase 3: pin or return
        if loading > 0 {
            if let Some(rx) = blocks_rx {
                let mut state = self.state.lock();
                state.active.insert(
                    req_id.to_string(),
                    PrefetchEntry {
                        blocks_rx: rx,
                        loading_count: loading,
                    },
                );
            }
            PrefetchStatus::Loading { hit, loading }
        } else {
            // All cache hits — pin.
            read_cache.pin_blocks(instance_id, num_workers, &blocks_to_pin);
            PrefetchStatus::Done { hit, missing }
        }
    }
}

enum PollResult {
    StillLoading,
    Completed,
}
