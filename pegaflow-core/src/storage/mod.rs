// ============================================================================
// StorageEngine: thin coordinator over three sub-components.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Sub-components:
// - ReadCache: sealed-block cache, pin/unpin semantics, PinToken RAII
// - PrefetchScheduler: backing-store prefetch, unified Mutex concurrency
// - WritePipeline: insert worker channel, metaserver, seal notifications
// ============================================================================

mod prefetch;
pub(crate) mod read_cache;
mod write_path;

use bytesize::ByteSize;
use log::{debug, info};
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use tonic::transport::Channel;

use crate::backing::{BackingStore, DEFAULT_MAX_PREFETCH_BLOCKS, SsdCacheConfig};
use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};

pub use read_cache::PinToken;
use read_cache::ReadCache;

use prefetch::PrefetchScheduler;
use write_path::{InsertDeps, WritePipeline};

// ============================================================================
// Constants
// ============================================================================

/// Number of LRU blocks to evict per iteration when reclaiming memory.
const RECLAIM_BATCH_SIZE: usize = 64;

// ============================================================================
// Public types
// ============================================================================

/// Notification sent when a block is sealed (for SSD offload, etc.)
pub type SealNotification = (BlockKey, Weak<SealedBlock>);

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity).
    pub hint_value_size_bytes: Option<usize>,
    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch).
    pub max_prefetch_blocks: usize,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
    /// Enable NUMA-aware memory allocation.
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

// ============================================================================
// StorageEngine (thin coordinator)
// ============================================================================

pub struct StorageEngine {
    /// Unified pinned memory allocator (handles both global and NUMA modes).
    allocator: Arc<PinnedAllocator>,

    /// Read path: sealed-block cache + pin/unpin.
    read_cache: Arc<ReadCache>,

    /// Backing-store prefetch scheduler.
    prefetch: PrefetchScheduler,

    /// Write pipeline: insert worker, seal notifications, metaserver.
    write_pipeline: Arc<WritePipeline>,

    /// Optional secondary backing store (SSD tier), shared with prefetch.
    backing: Option<Arc<dyn BackingStore>>,
}

impl StorageEngine {
    /// Create a new StorageEngine.
    ///
    /// Returns (engine, receiver) where receiver gets notified of sealed blocks.
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

        let ssd_enabled = ssd_cache_config.is_some();

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
                ssd_enabled,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                use_hugepages,
                ssd_enabled,
                unit_hint,
            ))
        };

        // Sub-components
        let read_cache = Arc::new(ReadCache::new(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
        ));

        let (seal_notify_tx, seal_notify_rx) = mpsc::unbounded_channel();
        let (write_pipeline, insert_rx) = WritePipeline::new(seal_notify_tx);
        let write_pipeline = Arc::new(write_pipeline);

        let is_numa = allocator.is_numa();
        let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
            let backing: Option<Arc<dyn BackingStore>> = ssd_cache_config.and_then(|ssd_cfg| {
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

            let prefetch = PrefetchScheduler::new(backing.clone(), max_prefetch_blocks);

            Self {
                allocator,
                read_cache: read_cache.clone(),
                prefetch,
                write_pipeline: write_pipeline.clone(),
                backing: backing.clone(),
            }
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let deps = Arc::new(InsertDeps {
                read_cache: engine.read_cache.clone(),
                write_pipeline: engine.write_pipeline.clone(),
                backing: engine.backing.clone(),
            });
            let weak_deps = Arc::downgrade(&deps);
            // Keep deps alive by leaking it into the thread. The worker holds
            // a Weak, so it won't prevent engine drop. The Arc is dropped when
            // the thread exits (channel closed).
            std::thread::Builder::new()
                .name("pegaflow-insert".into())
                .spawn(move || {
                    let _keep_alive = deps;
                    write_path::insert_worker_loop(insert_rx, weak_deps);
                })
                .expect("failed to spawn insert worker thread");
        }

        (engine, seal_notify_rx)
    }

    // ====================================================================
    // Delegation: MetaServer
    // ====================================================================

    pub(crate) fn set_metaserver_client(
        &self,
        client: MetaServerClient<Channel>,
        node_url: String,
    ) {
        self.write_pipeline.set_metaserver_client(client, node_url);
    }

    // ====================================================================
    // Delegation: Query flags
    // ====================================================================

    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.backing.is_some()
    }

    pub(crate) fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    // ====================================================================
    // Allocation
    // ====================================================================

    /// Allocate pinned memory, optionally from a specific NUMA node's pool.
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
            if let Some(alloc) = self.allocator.allocate(size, node) {
                return Some(alloc);
            }

            let (freed_blocks, _freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes);

            if freed_blocks == 0 || largest_free < requested_bytes {
                break;
            }
        }

        let (used, total) = self.allocator.usage();
        log::error!(
            "Pinned memory pool exhausted; cannot satisfy allocation: \
             requested={} used={} total={} numa={:?}",
            ByteSize(requested_bytes),
            ByteSize(used),
            ByteSize(total),
            numa_node
        );
        core_metrics().pool_alloc_failures.add(1, &[]);
        None
    }

    // ====================================================================
    // Delegation: Write path
    // ====================================================================

    pub(crate) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        self.write_pipeline.send_raw_insert(batch);
    }

    /// In-place filter for hashes NOT already sealed in cache.
    pub(crate) fn filter_hashes_not_in_cache_inplace(
        &self,
        namespace: &str,
        hashes: &mut HashSet<Vec<u8>>,
    ) {
        let namespace = namespace.to_string();
        hashes.retain(|hash| {
            let key = BlockKey::new(namespace.clone(), hash.clone());
            !self.read_cache.contains_key(&key)
        });
    }

    // ====================================================================
    // Delegation: Read path
    // ====================================================================

    /// Lookup multiple blocks for load operation. Returns PinTokens.
    ///
    /// Each PinToken auto-releases its pin on drop.
    pub(crate) fn cache_lookup_many(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<PinToken>, String> {
        self.read_cache
            .consume_pinned(instance_id, namespace, block_hashes)
    }

    /// Unpin blocks (cancellation path before consume).
    pub(crate) fn unpin_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> usize {
        self.read_cache
            .unpin_blocks(instance_id, namespace, block_hashes)
    }

    /// Pure memory-only prefix check. Returns `(hit, missing)`.
    pub(crate) fn check_prefix_memory_only(
        &self,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> (usize, usize) {
        self.read_cache.check_prefix_memory_only(namespace, hashes)
    }

    /// Check prefix blocks and schedule backing-store reads if needed.
    pub(crate) fn check_prefix_and_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        self.prefetch.check_and_prefetch(
            &self.read_cache,
            instance_id,
            req_id,
            namespace,
            hashes,
            num_workers,
        )
    }

    // ====================================================================
    // Eviction
    // ====================================================================

    fn reclaim_until_allocator_can_allocate(&self, required_bytes: u64) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (0, 0, self.allocator.largest_free_allocation());
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.allocator.largest_free_allocation();

        while largest_free < required_bytes {
            let used_before = self.allocator.usage().0;

            let evicted = self.read_cache.remove_lru_batch(RECLAIM_BATCH_SIZE);

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
                core_metrics().cache_resident_bytes.add(-(b as i64), &[]);
            }

            if still_referenced > 0 {
                core_metrics()
                    .cache_block_evictions_still_referenced
                    .add(still_referenced, &[]);
            }

            freed_bytes = freed_bytes.saturating_add(batch_bytes);
            freed_blocks += evicted.len();

            drop(evicted);
            let used_after = self.allocator.usage().0;
            let reclaimed = used_before.saturating_sub(used_after);
            if reclaimed > 0 {
                core_metrics()
                    .cache_eviction_reclaimed_bytes
                    .add(reclaimed, &[]);
            }

            largest_free = self.allocator.largest_free_allocation();
        }

        if freed_blocks > 0 {
            debug!(
                "Reclaimed cache blocks toward allocator request: \
                 freed_blocks={} freed_bytes={} largest_free={} required={}",
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

    // ====================================================================
    // GC
    // ====================================================================

    /// Remove stale inflight blocks older than `max_age`.
    pub(crate) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        self.write_pipeline.gc_stale_inflight(max_age).await
    }
}

#[cfg(test)]
impl StorageEngine {
    /// Insert a single block directly into the in-memory cache (test only).
    pub(crate) fn test_insert_cache(&self, key: BlockKey, block: Arc<SealedBlock>) {
        self.read_cache.test_insert(key, block);
    }

    /// Get the pin refcount for a (instance, block) pair (test only).
    pub(crate) fn test_pin_count(&self, instance_id: &str, key: &BlockKey) -> usize {
        self.read_cache.pin_count(instance_id, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> (Arc<StorageEngine>, UnboundedReceiver<SealNotification>) {
        StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[])
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
        let (storage, _rx) = make_engine();
        let mut hashes: HashSet<Vec<u8>> = HashSet::new();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
        assert!(hashes.is_empty());
    }

    #[tokio::test]
    async fn pin_consume_auto_releases() {
        let (storage, _rx) = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        // Insert into cache, then pin
        storage.test_insert_cache(key.clone(), block.clone());
        storage
            .read_cache
            .pin_blocks("inst1", 1, &[(key.clone(), block)]);

        assert_eq!(storage.test_pin_count("inst1", &key), 1);

        // consume_pinned creates PinToken
        let tokens = storage
            .cache_lookup_many("inst1", "ns", &[vec![1]])
            .unwrap();
        assert_eq!(tokens.len(), 1);

        // Pin still held by token
        assert_eq!(storage.test_pin_count("inst1", &key), 1);

        // Drop token → auto-release
        drop(tokens);
        assert_eq!(storage.test_pin_count("inst1", &key), 0);
    }

    #[tokio::test]
    async fn unpin_blocks_cancellation_path() {
        let (storage, _rx) = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key.clone(), block.clone());
        storage
            .read_cache
            .pin_blocks("inst1", 2, &[(key.clone(), block)]);

        assert_eq!(storage.test_pin_count("inst1", &key), 2);

        // Unpin once (simulating one worker cancellation)
        let unpinned = storage.unpin_blocks("inst1", "ns", &[vec![1]]);
        assert_eq!(unpinned, 1);
        assert_eq!(storage.test_pin_count("inst1", &key), 1);

        // Unpin again
        let unpinned = storage.unpin_blocks("inst1", "ns", &[vec![1]]);
        assert_eq!(unpinned, 1);
        assert_eq!(storage.test_pin_count("inst1", &key), 0);
    }

    #[tokio::test]
    async fn allocate_bounded_reclaim_terminates() {
        // With a tiny pool, allocation of a huge block should fail fast
        // (not loop forever) thanks to MAX_RECLAIM_ROUNDS.
        let (storage, _rx) =
            StorageEngine::new_with_config(4096, false, StorageConfig::default(), &[]);

        // Try to allocate more than the entire pool
        let result = storage.allocate(NonZeroU64::new(1 << 30).unwrap(), None);
        assert!(result.is_none(), "should fail, not loop forever");
    }

    #[tokio::test]
    async fn gc_stale_inflight_returns_zero_when_empty() {
        let (storage, _rx) = make_engine();
        let cleaned = storage
            .gc_stale_inflight(std::time::Duration::from_secs(60))
            .await;
        assert_eq!(cleaned, 0);
    }

    #[tokio::test]
    async fn multi_worker_pin_consume_lifecycle() {
        let (storage, _rx) = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key.clone(), block.clone());
        // Pin for 3 workers
        storage
            .read_cache
            .pin_blocks("inst1", 3, &[(key.clone(), block)]);
        assert_eq!(storage.test_pin_count("inst1", &key), 3);

        // Worker 0 consumes
        let tokens_0 = storage
            .cache_lookup_many("inst1", "ns", &[vec![1]])
            .unwrap();
        // Worker 1 consumes
        let tokens_1 = storage
            .cache_lookup_many("inst1", "ns", &[vec![1]])
            .unwrap();
        // Worker 2 consumes
        let tokens_2 = storage
            .cache_lookup_many("inst1", "ns", &[vec![1]])
            .unwrap();

        // All 3 tokens exist, refcount is 3
        assert_eq!(storage.test_pin_count("inst1", &key), 3);

        // Drop workers one at a time
        drop(tokens_0);
        assert_eq!(storage.test_pin_count("inst1", &key), 2);

        drop(tokens_1);
        assert_eq!(storage.test_pin_count("inst1", &key), 1);

        drop(tokens_2);
        assert_eq!(storage.test_pin_count("inst1", &key), 0);
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_removes_cached() {
        let (storage, _rx) = make_engine();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key1, block.clone());
        storage.test_insert_cache(key2, block);

        let mut hashes: HashSet<Vec<u8>> = [vec![1], vec![2], vec![3]].into_iter().collect();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);

        assert_eq!(hashes.len(), 1);
        assert!(hashes.contains(&vec![3]));
    }

    #[tokio::test]
    async fn consume_missing_pin_returns_error() {
        let (storage, _rx) = make_engine();
        // No pins exist — consume should fail
        let result = storage.cache_lookup_many("inst1", "ns", &[vec![1]]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn check_prefix_memory_only_basic() {
        let (storage, _rx) = make_engine();
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(BlockKey::new("ns".into(), vec![1]), block.clone());
        storage.test_insert_cache(BlockKey::new("ns".into(), vec![2]), block);

        // Full prefix hit
        let (hit, miss) = storage.check_prefix_memory_only("ns", &[vec![1], vec![2]]);
        assert_eq!(hit, 2);
        assert_eq!(miss, 0);

        // Prefix break at [3]
        let (hit, miss) = storage.check_prefix_memory_only("ns", &[vec![1], vec![2], vec![3]]);
        assert_eq!(hit, 2);
        assert_eq!(miss, 1);

        // First miss breaks entire prefix
        let (hit, miss) = storage.check_prefix_memory_only("ns", &[vec![3], vec![1]]);
        assert_eq!(hit, 0);
        assert_eq!(miss, 2);
    }
}
