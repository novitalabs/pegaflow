// ============================================================================
// StorageEngine: Two-phase block storage with separate write and read paths.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Architecture:
// - Mutex<StorageInner>: pending_from_backing, cache, pinned state
// - Insert Worker (dedicated thread): owns inflight HashMap, receives
//   RawSaveBatch messages via channel, builds LayerBlocks and seals completed blocks
// - Allocator: PinnedMemoryPool for pinned memory allocation
// - Backing store (optional): secondary SSD tier, owned by Arc<dyn BackingStore>
// ============================================================================
use bytesize::ByteSize;
use dashmap::DashMap;
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Weak};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

use pegaflow_proto::proto::engine::InsertBlockHashesRequest;
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use tonic::transport::Channel;

use crate::backing::{BackingStore, DEFAULT_MAX_PREFETCH_BLOCKS, PrefetchResult, SsdCacheConfig};
use crate::block::{BlockKey, InflightBlock, PrefetchStatus, SealedBlock, SlotInsertResult};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::offload::InsertEntries;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};

// ============================================================================
// Constants
// ============================================================================

/// Number of LRU blocks to evict per iteration when reclaiming memory
const RECLAIM_BATCH_SIZE: usize = 64;

// ============================================================================
// Per-request prefetch state machine
// ============================================================================

/// Per-request prefetch tracking.
///
/// ```text
///   query(req_id)        all pending done         consumed / cancelled
///   ┌───────┐  backing  ┌─────────┐  re-scan  ┌──────┐
///   │ (new) │ ────────→ │ Loading │ ────────→ │ Done │ → remove from map
///   └───────┘           └─────────┘           └──────┘
///       │ all in cache       ↑ try_recv: fires when coordinator task completes
///       └──→ pin & return    └──────────────────────────
/// ```
struct PrefetchEntry {
    /// Delivers batch of successfully-read blocks when all backing I/O finishes.
    blocks_rx: oneshot::Receiver<PrefetchResult>,
    /// Number of blocks submitted to the backing store (for backpressure accounting).
    loading_count: usize,
}

// ============================================================================
// Metrics helpers (keep insert/evict logic together for easy audit)
// ============================================================================

/// Records metrics when bytes are added to inflight blocks.
fn record_inflight_bytes_added(bytes: u64) {
    if let Ok(v) = i64::try_from(bytes) {
        core_metrics().inflight_bytes.add(v, &[]);
    }
}

/// Records metrics when bytes are removed from inflight blocks (seal or gc).
fn record_inflight_bytes_removed(bytes: u64) {
    if let Ok(v) = i64::try_from(bytes) {
        core_metrics().inflight_bytes.add(-v, &[]);
    }
}

/// Records metrics for a new cache insertion.
fn record_cache_insert_new(footprint_bytes: u64) {
    let m = core_metrics();
    m.cache_block_insertions.add(1, &[]);
    if let Ok(v) = i64::try_from(footprint_bytes) {
        m.cache_resident_bytes.add(v, &[]);
    }
}

/// Records metrics for a cache eviction.
fn record_cache_eviction(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().cache_resident_bytes.add(-v, &[]);
    }
}

/// Records metrics when a new unique block is pinned.
fn record_pin_unique_added(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().pinned_for_load_unique_bytes.add(v, &[]);
    }
}

/// Records metrics when the last reference to a unique block is unpinned.
fn record_pin_unique_removed(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().pinned_for_load_unique_bytes.add(-v, &[]);
    }
}

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity)
    pub hint_value_size_bytes: Option<usize>,
    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch).
    /// ~15GB assuming 10MB per block.
    pub max_prefetch_blocks: usize,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
    /// Enable NUMA-aware memory allocation. When true on multi-NUMA systems,
    /// PegaEngine auto-detects topology and creates per-node pinned pools.
    pub enable_numa_affinity: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
            max_prefetch_blocks: DEFAULT_MAX_PREFETCH_BLOCKS,
            ssd_cache_config: None,
            enable_numa_affinity: true,
        }
    }
}

// A "slot" in this file refers to a specific position in the flattened logical storage,
// calculated as `layer_id * tp_size + tp_rank`.
// vLLM/Connectors report the total topology (layers * tp_size) via registration,
// and this count is immutable for the lifetime of the Instance.

// ============================================================================
// StorageEngine
// ============================================================================

/// Notification sent when a block is sealed (for SSD offload, etc.)
pub type SealNotification = (BlockKey, Weak<SealedBlock>);

// ============================================================================
// Insert Worker (actor model for inflight block management)
// ============================================================================

/// Command sent to the insert worker.
enum InsertWorkerCommand {
    /// Deferred save: build LayerBlocks + insert into inflight.
    RawInsert(crate::offload::RawSaveBatch),
    /// GC stale inflight blocks older than max_age.
    Gc {
        max_age: std::time::Duration,
        reply: oneshot::Sender<usize>,
    },
}

/// Inner state protected by a single mutex (no longer contains inflight)
struct StorageInner {
    /// Read path: sealed blocks available for lookup (TinyLFU admission + LRU eviction)
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    /// Pinned blocks between query and load (prevents eviction race)
    /// Key: (instance_id, block_key), Value: (block, ref_count)
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
    /// Aggregated pinned_for_load refcounts by block key (for attribution metrics).
    /// Value: (footprint_bytes, total_refcount)
    pinned_for_load_by_key: HashMap<BlockKey, (u64, usize)>,
}

pub struct StorageEngine {
    /// Unified pinned memory allocator (handles both global and NUMA modes)
    allocator: Arc<PinnedAllocator>,

    /// Mutable state under one lock (cache, pending_from_backing, pinned_for_load)
    inner: Mutex<StorageInner>,

    /// In-flight SSD prefetch receivers, keyed by req_id.
    /// Populated on first query; polled on retries until the batch arrives.
    ///
    /// TODO: entries are only cleaned up when the same req_id is polled again.
    /// If a client abandons a request (network drop, cancellation), the entry
    /// and its `loading_count` contribution to `inflight_prefetch` leak.
    /// Consider a periodic sweep or RAII guard if this proves problematic.
    active_prefetches: DashMap<String, PrefetchEntry>,

    /// Channel to the insert worker thread (owns inflight HashMap)
    insert_tx: Sender<InsertWorkerCommand>,

    /// Channel to notify consumers when blocks are sealed (for SSD offload)
    seal_notify_tx: Option<UnboundedSender<SealNotification>>,

    /// Optional secondary backing store (SSD tier).
    backing: Option<Arc<dyn BackingStore>>,

    /// Max blocks allowed in prefetching state (backpressure for backing reads)
    max_prefetch_blocks: usize,

    /// Count of blocks currently being read from backing store (for backpressure).
    inflight_prefetch: AtomicUsize,

    /// Channel to the metaserver insert worker (set by `PegaEngine::set_metaserver_client`)
    metaserver_tx: Mutex<Option<UnboundedSender<crate::MetaserverInsertCmd>>>,
}

impl StorageEngine {
    /// Create a new StorageEngine with optional seal notification channel.
    /// Returns (engine, receiver) where receiver gets notified of sealed blocks.
    ///
    /// # Arguments
    /// * `capacity_bytes` - Total pinned memory pool capacity
    /// * `use_hugepages` - Use 2MB huge pages for allocation
    /// * `config` - Storage behavior configuration
    /// * `numa_nodes` - NUMA nodes for per-node pools (empty = single global pool)
    pub(crate) fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: impl Into<StorageConfig>,
        numa_nodes: &[NumaNode],
    ) -> (Arc<Self>, UnboundedReceiver<SealNotification>) {
        let config = config.into();
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));
        let max_prefetch_blocks = config.max_prefetch_blocks;
        let ssd_cache_config = config.ssd_cache_config;

        // Create unified allocator based on NUMA configuration
        let allocator = if !numa_nodes.is_empty() {
            info!(
                "Creating NUMA-aware pinned pools for {} nodes",
                numa_nodes.len()
            );
            Arc::new(PinnedAllocator::new_numa(
                capacity_bytes,
                numa_nodes,
                use_hugepages,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                use_hugepages,
                unit_hint,
            ))
        };

        let cache = TinyLfuCache::new_unbounded(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
        );

        let inner = Mutex::new(StorageInner {
            cache,
            pinned_for_load: HashMap::new(),
            pinned_for_load_by_key: HashMap::new(),
        });

        // Create unbounded channel for seal notifications
        let (seal_notify_tx, seal_notify_rx) = mpsc::unbounded_channel();

        // Create insert worker channel (std::sync::mpsc — worker is a dedicated OS thread)
        let (insert_tx, insert_rx) = std::sync::mpsc::channel();

        let is_numa = allocator.is_numa();
        let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
            let backing = ssd_cache_config.and_then(|ssd_cfg| {
                let weak_engine = weak_engine.clone();
                crate::backing::new_ssd(
                    ssd_cfg,
                    move |size, numa_node| {
                        weak_engine
                            .upgrade()
                            .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
                    },
                    is_numa,
                )
            });

            Self {
                allocator,
                inner,
                active_prefetches: DashMap::new(),
                insert_tx,
                seal_notify_tx: Some(seal_notify_tx),
                backing,
                max_prefetch_blocks,
                inflight_prefetch: AtomicUsize::new(0),
                metaserver_tx: Mutex::new(None),
            }
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let weak_engine = Arc::downgrade(&engine);
            std::thread::Builder::new()
                .name("pegaflow-insert".into())
                .spawn(move || insert_worker_loop(insert_rx, weak_engine))
                .expect("failed to spawn insert worker thread");
        }

        (engine, seal_notify_rx)
    }

    /// Set the MetaServer client for cross-node block hash registry.
    ///
    /// Spawns a background worker that batches and sends insert requests.
    pub(crate) fn set_metaserver_client(
        &self,
        client: MetaServerClient<Channel>,
        node_url: String,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(metaserver_worker_loop(rx, client, node_url.clone()));
        *self.metaserver_tx.lock() = Some(tx);
        info!(
            "MetaServer client configured for block hash registry (node_url={})",
            node_url
        );
    }

    /// Send block hashes to the metaserver insert worker (fire-and-forget).
    fn send_metaserver_insert(&self, namespace: String, block_hashes: Vec<Vec<u8>>) {
        if let Some(tx) = self.metaserver_tx.lock().as_ref() {
            let _ = tx.send(crate::MetaserverInsertCmd {
                namespace,
                block_hashes,
            });
        }
    }

    /// Returns true if a secondary backing store (SSD) is enabled.
    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.backing.is_some()
    }

    /// Returns true if NUMA-aware allocation is enabled.
    pub(crate) fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    // ========================================================================
    // Allocation
    // ========================================================================

    /// Allocate pinned memory, optionally from a specific NUMA node's pool.
    ///
    /// If `numa_node` is `Some` and NUMA pools are configured, allocates from
    /// that NUMA node's pool. Otherwise uses the global pool.
    ///
    /// Returns `None` if the pool is exhausted after eviction attempts.
    pub(crate) fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        let requested_bytes = size.get();
        let node = numa_node.unwrap_or(NumaNode::UNKNOWN);

        loop {
            // Try to allocate from the unified allocator
            if let Some(alloc) = self.allocator.allocate(size, node) {
                return Some(alloc);
            }

            // Allocation failed, try to reclaim memory
            let (freed_blocks, freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes);

            if largest_free >= requested_bytes {
                continue;
            }

            // Still can't allocate, report error
            let (used, total) = self.allocator.usage();
            error!(
                "Pinned memory pool exhausted; cannot satisfy allocation: requested={} used={} total={} largest_free={} freed_blocks={} freed_bytes={} numa={:?}",
                ByteSize(requested_bytes),
                ByteSize(used),
                ByteSize(total),
                ByteSize(largest_free),
                freed_blocks,
                ByteSize(freed_bytes),
                numa_node
            );
            core_metrics().pool_alloc_failures.add(1, &[]);
            return None;
        }
    }

    /// Get aggregate pool usage: (used_bytes, total_bytes)
    fn pool_usage(&self) -> (u64, u64) {
        self.allocator.usage()
    }

    /// Get largest free allocation across all pools.
    fn largest_free_allocation(&self) -> u64 {
        self.allocator.largest_free_allocation()
    }

    // ========================================================================
    // Write path (inflight)
    // ========================================================================

    /// Fire-and-forget raw insert: send a deferred save batch to the insert worker.
    ///
    /// The worker builds `LayerBlock` objects, groups by hash, and inserts
    /// into inflight. The caller does NOT need to wait — once GPU→CPU copy
    /// is done the pinned memory is reference-counted.
    pub(crate) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    }

    /// In-place filter for hashes that are NOT already sealed in cache.
    ///
    /// After return, `hashes` only contains entries that need saving.
    /// Since cache membership is hash-based (not layer-specific), this only
    /// needs to be called once for all layers sharing the same namespace.
    pub(crate) fn filter_hashes_not_in_cache_inplace(
        &self,
        namespace: &str,
        hashes: &mut HashSet<Vec<u8>>,
    ) {
        let namespace = namespace.to_string();
        let inner = self.inner.lock();
        hashes.retain(|hash| {
            let key = BlockKey::new(namespace.clone(), hash.clone());
            !inner.cache.contains_key(&key)
        });
    }

    /// Forward sealed blocks to the backing store for async persistence.
    ///
    /// Converts `Arc<SealedBlock>` to `Weak` so the backing store cannot
    /// prevent cache eviction. Silently no-ops when no backing store is set.
    fn send_ssd_batch(&self, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
        let Some(backing) = self.backing.as_ref() else {
            return;
        };
        if blocks.is_empty() {
            return;
        }
        let weak_blocks: Vec<(BlockKey, Weak<SealedBlock>)> = blocks
            .iter()
            .map(|(k, b)| (k.clone(), Arc::downgrade(b)))
            .collect();
        backing.ingest_batch(weak_blocks);
    }

    // ========================================================================
    // Read path (cache)
    // ========================================================================

    /// Lookup multiple blocks for load operation.
    /// Consumes pinned blocks (removes from pinned_for_load).
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "storage.cache_lookup_many")
    )]
    pub(crate) fn cache_lookup_many(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .map(|key| (instance_id.to_string(), key.clone()))
            .collect();

        let mut inner = self.inner.lock();
        let mut result: Vec<Arc<SealedBlock>> = Vec::with_capacity(keys.len());

        for (idx, (key, pin_key)) in keys.into_iter().zip(pin_keys.into_iter()).enumerate() {
            // Consume pinned_for_load (ref_count -1, remove if 0)
            if let Entry::Occupied(mut entry) = inner.pinned_for_load.entry(pin_key) {
                let (block, count) = entry.get_mut();
                let cloned = Arc::clone(block);
                *count -= 1;

                if *count == 0 {
                    entry.remove();
                }

                // Track unique block removal
                let mut unique_bytes_to_remove: Option<u64> = None;
                if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                    *total = total.saturating_sub(1);
                    if *total == 0 {
                        unique_bytes_to_remove = Some(*bytes);
                    }
                } else {
                    error!(
                        "BUG: pinned_for_load_by_key missing key during consume: namespace={} hash_len={}",
                        key.namespace,
                        key.hash.len()
                    );
                }

                if let Some(bytes) = unique_bytes_to_remove {
                    inner.pinned_for_load_by_key.remove(&key);
                    record_pin_unique_removed(bytes);
                }

                result.push(cloned);
            } else {
                error!(
                    "missing pinned KV block: instance={} idx={} hash_len={}",
                    instance_id,
                    idx,
                    key.hash.len()
                );
                return Err(format!(
                    "missing pinned KV block at index {} (namespace={}, hash_len={})",
                    idx,
                    key.namespace,
                    key.hash.len()
                ));
            }
        }

        Ok(result)
    }

    /// Unpin blocks that were pinned during query.
    /// This decrements the ref_count and removes the entry when it reaches 0.
    /// Returns the number of blocks that were successfully unpinned.
    pub(crate) fn unpin_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> usize {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .map(|key| (instance_id.to_string(), key.clone()))
            .collect();

        let mut inner = self.inner.lock();
        let mut unpinned = 0usize;

        for (key, pin_key) in keys.into_iter().zip(pin_keys.into_iter()) {
            if let Some((_, count)) = inner.pinned_for_load.get_mut(&pin_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    inner.pinned_for_load.remove(&pin_key);
                }
                unpinned += 1;

                // Track unique block removal
                let mut unique_bytes_to_remove: Option<u64> = None;
                if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                    *total = total.saturating_sub(1);
                    if *total == 0 {
                        unique_bytes_to_remove = Some(*bytes);
                    }
                } else {
                    error!(
                        "BUG: pinned_for_load_by_key missing key during unpin: namespace={} hash_len={}",
                        key.namespace,
                        key.hash.len()
                    );
                }

                if let Some(bytes) = unique_bytes_to_remove {
                    inner.pinned_for_load_by_key.remove(&key);
                    record_pin_unique_removed(bytes);
                }
            }
        }

        unpinned
    }

    // ========================================================================
    // Eviction (cache only)
    // ========================================================================

    fn reclaim_until_allocator_can_allocate(&self, required_bytes: u64) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (0, 0, self.largest_free_allocation());
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.largest_free_allocation();

        while largest_free < required_bytes {
            let used_before = self.pool_usage().0;

            // Collect evicted blocks under lock, then drop outside lock
            let evicted: Vec<_> = {
                let mut inner = self.inner.lock();
                (0..RECLAIM_BATCH_SIZE)
                    .map_while(|_| inner.cache.remove_lru())
                    .collect()
            };

            if evicted.is_empty() {
                break;
            }

            let mut batch_bytes = 0u64;
            let mut still_referenced = 0u64;
            for (_key, block) in evicted.iter() {
                let b = block.memory_footprint();
                batch_bytes = batch_bytes.saturating_add(b);
                if Arc::strong_count(block) > 1 {
                    still_referenced += 1;
                }
                record_cache_eviction(b);
            }

            if still_referenced > 0 {
                core_metrics()
                    .cache_block_evictions_still_referenced
                    .add(still_referenced, &[]);
            }

            freed_bytes = freed_bytes.saturating_add(batch_bytes);
            freed_blocks += evicted.len();

            drop(evicted); // allow allocation drops to run before sampling allocator usage
            let used_after = self.pool_usage().0;
            let reclaimed = used_before.saturating_sub(used_after);
            if reclaimed > 0 {
                core_metrics()
                    .cache_eviction_reclaimed_bytes
                    .add(reclaimed, &[]);
            }

            largest_free = self.largest_free_allocation();
        }

        if freed_blocks > 0 {
            debug!(
                "Reclaimed cache blocks toward allocator request: freed_blocks={} freed_bytes={} largest_free={} required={}",
                freed_blocks,
                ByteSize(freed_bytes),
                ByteSize(largest_free),
                ByteSize(required_bytes)
            );
            core_metrics()
                .cache_block_evictions
                .add(freed_blocks as u64, &[]);
        }

        (freed_blocks, freed_bytes, largest_free)
    }

    /// Remove stale inflight blocks that have been stuck for longer than `max_age`.
    ///
    /// Sends a GC command to the insert worker, which owns the inflight HashMap.
    /// This is async because it waits for the worker's reply.
    ///
    /// Returns the number of cleaned blocks.
    pub(crate) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .insert_tx
            .send(InsertWorkerCommand::Gc {
                max_age,
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.await.unwrap_or(0)
    }

    /// Pure memory-only prefix check. Returns `(hit, missing)` counts.
    ///
    /// No backing-store prefetch, no pinning — suitable for lightweight query RPCs.
    pub(crate) fn check_prefix_memory_only(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> (usize, usize) {
        let mut hit = 0usize;

        {
            let mut inner = self.inner.lock();

            for hash in hashes {
                let key = BlockKey::new(namespace.to_string(), hash.clone());
                if inner.cache.get(&key).is_some() {
                    hit += 1;
                } else {
                    break;
                }
            }
        }

        let missing = hashes.len() - hit;
        (hit, missing)
    }

    /// Check prefix blocks and schedule backing-store reads if needed.
    ///
    /// Uses per-request state machine: first call does full scan + dispatches
    /// backing reads; retries poll a oneshot receiver (lock-free fast path).
    /// Pins hit blocks when returning Done.
    pub(crate) fn check_prefix_and_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        // This request has been seen before and backing reads are still in flight.
        if let Some(mut entry) = self.active_prefetches.get_mut(req_id) {
            match entry.blocks_rx.try_recv() {
                Err(oneshot::error::TryRecvError::Empty) => {
                    return PrefetchStatus::Loading { hit: 0, loading: 1 };
                }
                Ok(ssd_blocks) => {
                    let loading_count = entry.loading_count;
                    drop(entry);
                    self.active_prefetches.remove(req_id);
                    self.inflight_prefetch
                        .fetch_sub(loading_count, Ordering::Relaxed);
                    self.batch_insert_cache(ssd_blocks);
                    // Fall through to full_prefix_scan below.
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    warn!(
                        "SSD prefetch sender dropped for req_id={}, falling back to re-scan",
                        req_id
                    );
                    let loading_count = entry.loading_count;
                    drop(entry);
                    self.active_prefetches.remove(req_id);
                    self.inflight_prefetch
                        .fetch_sub(loading_count, Ordering::Relaxed);
                }
            }
        }

        // Full scan: cache → backing store → prefix break.
        self.full_prefix_scan(instance_id, req_id, namespace, hashes, num_workers)
    }

    /// Full prefix scan: cache → SSD → pin.
    ///
    /// Phase 1: scan cache prefix (first lock).
    /// Phase 2: pass remaining keys to SSD `read_prefix` (no engine lock held).
    /// Phase 3: if all hits, re-acquire lock and pin blocks.
    fn full_prefix_scan(
        &self,
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

        // Phase 1: cpu cache prefix scan
        let mut hit = 0usize;
        let mut blocks_to_pin: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();
        {
            let mut inner = self.inner.lock();
            for key in &keys {
                if let Some(block) = inner.cache.get(key) {
                    hit += 1;
                    blocks_to_pin.push((key.clone(), Arc::clone(&block)));
                } else {
                    break;
                }
            }
        }

        // Phase 2: SSD prefix scan for remaining keys
        let remaining = &keys[hit..];
        let mut loading = 0usize;
        let mut blocks_rx: Option<oneshot::Receiver<PrefetchResult>> = None;

        if !remaining.is_empty() {
            let inflight = self.inflight_prefetch.load(Ordering::Relaxed);
            let available = self.max_prefetch_blocks.saturating_sub(inflight);

            if available > 0 {
                if let Some(backing) = self.backing.as_ref() {
                    let check_limit = remaining.len().min(available);
                    let check_keys = remaining[..check_limit].to_vec();

                    let (found, rx) = backing.read_prefix(check_keys);

                    if found > 0 {
                        self.inflight_prefetch.fetch_add(found, Ordering::Relaxed);
                        loading = found;
                        blocks_rx = Some(rx);
                    }

                    let backpressure_skipped = remaining.len() - check_limit;
                    if backpressure_skipped > 0 {
                        core_metrics()
                            .ssd_prefetch_backpressure_blocks
                            .add(backpressure_skipped as u64, &[]);
                    }
                }
            } else {
                core_metrics()
                    .ssd_prefetch_backpressure_blocks
                    .add(remaining.len() as u64, &[]);
            }
        }

        let missing = keys.len() - hit - loading;

        // Phase 3: pin or return
        if loading > 0 {
            if let Some(blocks_rx) = blocks_rx {
                self.active_prefetches.insert(
                    req_id.to_string(),
                    PrefetchEntry {
                        blocks_rx,
                        loading_count: loading,
                    },
                );
            }
            PrefetchStatus::Loading { hit, loading }
        } else {
            // All cache hits — re-acquire lock and pin.
            self.pin_blocks_inner(instance_id, num_workers, &blocks_to_pin);
            PrefetchStatus::Done { hit, missing }
        }
    }

    /// Batch-insert SSD-read blocks into the in-memory cache.
    fn batch_insert_cache(&self, blocks: PrefetchResult) {
        let mut inner = self.inner.lock();
        for (key, block) in blocks {
            let footprint_bytes = block.memory_footprint();
            match inner.cache.insert(key, block) {
                CacheInsertOutcome::InsertedNew => record_cache_insert_new(footprint_bytes),
                CacheInsertOutcome::AlreadyExists => {}
                CacheInsertOutcome::Rejected => {
                    core_metrics().cache_block_admission_rejections.add(1, &[]);
                }
            }
        }
    }

    /// Pin blocks under a single lock acquisition.
    fn pin_blocks_inner(
        &self,
        instance_id: &str,
        num_workers: usize,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) {
        let mut inner = self.inner.lock();
        let instance_id_owned = instance_id.to_string();
        for (key, block) in blocks {
            let pin_key = (instance_id_owned.clone(), key.clone());
            let footprint = block.memory_footprint();

            match inner.pinned_for_load.entry(pin_key) {
                Entry::Occupied(mut o) => {
                    o.get_mut().1 += num_workers;
                }
                Entry::Vacant(v) => {
                    v.insert((Arc::clone(block), num_workers));
                }
            }

            match inner.pinned_for_load_by_key.entry(key.clone()) {
                Entry::Occupied(mut o) => {
                    o.get_mut().1 += num_workers;
                }
                Entry::Vacant(v) => {
                    v.insert((footprint, num_workers));
                    record_pin_unique_added(footprint);
                }
            }
        }
    }
}

// ============================================================================
// Insert Worker (dedicated thread, owns inflight HashMap)
// ============================================================================

/// Dedicated insert worker task. Owns the inflight HashMap exclusively,
/// eliminating lock contention on the hot insert path. Sealed blocks are
/// admitted to cache via brief `StorageInner` lock acquisitions.
fn insert_worker_loop(rx: Receiver<InsertWorkerCommand>, engine: Weak<StorageEngine>) {
    let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

    while let Ok(cmd) = rx.recv() {
        // Drain additional commands for batching
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        for cmd in cmds {
            match cmd {
                InsertWorkerCommand::RawInsert(batch) => {
                    process_raw_save_batch(&mut inflight, &engine, batch);
                }
                InsertWorkerCommand::Gc { max_age, reply } => {
                    let cleaned = gc_inflight(&mut inflight, max_age);
                    let _ = reply.send(cleaned);
                }
            }
        }
    }

    info!(
        "Insert worker shutting down, {} inflight blocks remaining",
        inflight.len()
    );
}

/// Process a deferred raw save batch: build LayerBlocks, then delegate to
/// `process_insert_batch` for inflight/seal/cache logic.
fn process_raw_save_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    engine: &Weak<StorageEngine>,
    batch: crate::offload::RawSaveBatch,
) {
    let phase4_start = std::time::Instant::now();
    let namespace = batch.namespace.clone();
    let numa_node = batch.numa_node;
    let total_slots = batch.total_slots;

    let (entries, _total_bytes, _total_blocks) = crate::offload::build_insert_entries(&batch);

    process_insert_batch(
        inflight,
        engine,
        entries,
        total_slots,
        numa_node,
        &namespace,
    );

    debug!(
        "insert_worker phase4: blocks={} bytes={} ms={:.2}",
        _total_blocks,
        _total_bytes,
        phase4_start.elapsed().as_secs_f64() * 1000.0,
    );
}

/// Process a single insert batch (fire-and-forget).
/// Inflight HashMap is owned exclusively by the worker (no lock needed).
/// Cache insertion + metaserver announcement + SSD offload handled internally.
fn process_insert_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    engine: &Weak<StorageEngine>,
    entries: InsertEntries,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
) {
    let mut sealed_blocks: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();
    let mut inflight_bytes_added: u64 = 0;
    let mut inflight_bytes_removed: u64 = 0;

    for (key, slots) in entries {
        // Get or create inflight block (no lock — worker-exclusive HashMap)
        let inflight_block = match inflight.entry(key.clone()) {
            Entry::Vacant(v) => v.insert(InflightBlock::new(total_slots)),
            Entry::Occupied(o) => {
                let ib = o.into_mut();
                if ib.total_slots() != total_slots {
                    error!(
                        "insert worker: slot count mismatch: key namespace={} expected={} got={}",
                        namespace,
                        ib.total_slots(),
                        total_slots
                    );
                    continue;
                }
                ib
            }
        };

        // Insert all slots for this hash
        let mut completed = false;
        for (slot_id, block) in slots {
            match inflight_block.insert_slot(slot_id, block, numa_node) {
                SlotInsertResult::Inserted {
                    completed: c,
                    footprint_added,
                } => {
                    inflight_bytes_added = inflight_bytes_added.saturating_add(footprint_added);
                    completed = c;
                    if completed {
                        break;
                    }
                }
                SlotInsertResult::Duplicate => {}
            }
        }

        if completed {
            let inflight_block = inflight.remove(&key).expect("just inserted");
            let total_footprint = inflight_block.footprint();
            inflight_bytes_removed = inflight_bytes_removed.saturating_add(total_footprint);
            let sealed = Arc::new(inflight_block.seal());

            // Brief lock: admit sealed block to cache
            if let Some(engine) = engine.upgrade() {
                let mut inner = engine.inner.lock();
                match inner.cache.insert(key.clone(), Arc::clone(&sealed)) {
                    CacheInsertOutcome::InsertedNew => {
                        record_cache_insert_new(total_footprint);
                    }
                    CacheInsertOutcome::AlreadyExists => {}
                    CacheInsertOutcome::Rejected => {
                        core_metrics().cache_block_admission_rejections.add(1, &[]);
                    }
                }
                drop(inner);

                // Seal notification (for SSD offload)
                if let Some(tx) = &engine.seal_notify_tx {
                    let _ = tx.send((key.clone(), Arc::downgrade(&sealed)));
                }
            }

            sealed_blocks.push((key, sealed));
        }
    }

    if inflight_bytes_added > 0 {
        record_inflight_bytes_added(inflight_bytes_added);
    }
    if inflight_bytes_removed > 0 {
        record_inflight_bytes_removed(inflight_bytes_removed);
    }

    // Send block hashes to metaserver worker (batched, fire-and-forget)
    if !sealed_blocks.is_empty()
        && let Some(engine) = engine.upgrade()
    {
        let metaserver_hashes: Vec<Vec<u8>> = sealed_blocks
            .iter()
            .map(|(key, _)| key.hash.clone())
            .collect();
        engine.send_metaserver_insert(namespace.to_owned(), metaserver_hashes);
    }

    // SSD offload (fire-and-forget internally)
    if !sealed_blocks.is_empty()
        && let Some(engine) = engine.upgrade()
    {
        engine.send_ssd_batch(&sealed_blocks);
    }
}

/// GC stale inflight blocks within the insert worker.
fn gc_inflight(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    max_age: std::time::Duration,
) -> usize {
    let before = inflight.len();

    inflight.retain(|key, block| {
        let age = block.age();
        if age > max_age {
            warn!(
                "GC: removing stale inflight block: namespace={} hash_len={} filled={} total={} age_secs={}",
                key.namespace,
                key.hash.len(),
                block.filled_count(),
                block.total_slots(),
                age.as_secs()
            );
            record_inflight_bytes_removed(block.footprint());
            false
        } else {
            true
        }
    });

    let cleaned = before - inflight.len();
    if cleaned > 0 {
        core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
        info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
    }
    cleaned
}

// ============================================================================
// Metaserver Worker (dedicated task, batches block hash inserts)
// ============================================================================

/// Background worker that receives block hash insert commands and batches them
/// into MetaServer gRPC calls, grouped by namespace.
async fn metaserver_worker_loop(
    mut rx: UnboundedReceiver<crate::MetaserverInsertCmd>,
    mut client: MetaServerClient<Channel>,
    node_url: String,
) {
    while let Some(cmd) = rx.recv().await {
        // Drain additional commands for batching
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        // Merge hashes by namespace
        let mut by_namespace: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for cmd in cmds {
            by_namespace
                .entry(cmd.namespace)
                .or_default()
                .extend(cmd.block_hashes);
        }

        for (namespace, block_hashes) in by_namespace {
            let count = block_hashes.len();
            let req = InsertBlockHashesRequest {
                namespace,
                block_hashes,
                node: node_url.clone(),
            };
            match client.insert_block_hashes(req).await {
                Ok(response) => {
                    debug!(
                        "MetaServer insert: sent {} hashes, inserted {}",
                        count,
                        response.into_inner().inserted_count
                    );
                }
                Err(err) => {
                    warn!("MetaServer insert failed: {}", err);
                }
            }
        }
    }

    info!("Metaserver worker shutting down");
}

#[cfg(test)]
impl StorageEngine {
    /// Insert a single block directly into the in-memory cache (test only).
    pub(crate) fn test_insert_cache(&self, key: BlockKey, block: Arc<SealedBlock>) {
        let mut inner = self.inner.lock();
        inner.cache.insert(key, block);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
        let (storage, _rx) =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let mut hashes: HashSet<Vec<u8>> = HashSet::new();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
        assert!(hashes.is_empty());
    }
}
