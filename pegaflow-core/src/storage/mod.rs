// ============================================================================
// StorageEngine: thin coordinator over three sub-components.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Sub-components:
// - ReadCache: sealed-block cache, pin/unpin semantics, consume-on-load
// - PrefetchScheduler: backing-store prefetch, unified Mutex concurrency
// - WritePipeline: insert worker channel and seal notifications
// ============================================================================

mod prefetch;
pub(crate) mod read_cache;
mod write_path;

use bytesize::ByteSize;
use log::{debug, info};
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use crate::backing::{
    BackingStore, BackingStoreKind, BakingStoreConfig, DEFAULT_MAX_PREFETCH_BLOCKS, SsdCacheConfig,
};
use crate::block::BlockLookupResult;
use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};

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

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity).
    pub hint_value_size_bytes: Option<usize>,
    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch).
    pub max_prefetch_blocks: usize,
    /// Optional P2P baking store for inter-node ownership announcements.
    pub baking_store_config: Option<BakingStoreConfig>,
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
            baking_store_config: None,
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

    /// Write pipeline: insert worker channel.
    write_pipeline: Arc<WritePipeline>,

    /// Secondary backing stores ordered by priority.
    backing_stores: Vec<Arc<dyn BackingStore>>,
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
    ) -> Arc<Self> {
        let config = config.into();
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));
        let max_prefetch_blocks = config.max_prefetch_blocks;
        let baking_store_config = config.baking_store_config.clone();
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

        let (write_pipeline, insert_rx) = WritePipeline::new();
        let write_pipeline = Arc::new(write_pipeline);

        let is_numa = allocator.is_numa();
        let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
            let mut backing_stores = Vec::new();

            if let Some(baking_cfg) = baking_store_config.clone()
                && let Some(store) = crate::backing::new_p2p(baking_cfg)
            {
                backing_stores.push(store);
            }

            if let Some(ssd_cfg) = ssd_cache_config {
                let weak_engine = weak_engine.clone();
                if let Some(store) = crate::backing::new_ssd(
                    ssd_cfg,
                    move |size, numa_node| {
                        weak_engine
                            .upgrade()
                            .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
                    },
                    is_numa,
                ) {
                    backing_stores.push(store);
                }
            }

            let prefetch = PrefetchScheduler::new(backing_stores.clone(), max_prefetch_blocks);

            Self {
                allocator,
                read_cache: read_cache.clone(),
                prefetch,
                write_pipeline: write_pipeline.clone(),
                backing_stores,
            }
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let deps = Arc::new(InsertDeps {
                read_cache: engine.read_cache.clone(),
                backing_stores: engine.backing_stores.clone(),
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

        engine
    }

    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.backing_stores
            .iter()
            .any(|store| store.kind() == BackingStoreKind::Ssd)
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
        let hash_vec: Vec<Vec<u8>> = hashes.iter().cloned().collect();
        let keys: Vec<BlockKey> = hash_vec
            .iter()
            .map(|hash| BlockKey::new(namespace.clone(), hash.clone()))
            .collect();
        let present = self.read_cache.contains_keys(&keys);
        for (hash, is_present) in hash_vec.into_iter().zip(present) {
            if is_present {
                hashes.remove(&hash);
            }
        }
    }

    // ====================================================================
    // Delegation: Read path
    // ====================================================================

    /// Consume multiple pinned blocks for a load operation.
    ///
    /// Each returned [`Arc<SealedBlock>`] consumes one pin reservation.
    pub(crate) fn consume_pinned_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        self.read_cache
            .consume_pinned_blocks(instance_id, namespace, block_hashes)
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

    /// Non-prefix batch get: returns found sealed blocks and missing hashes.
    ///
    /// Used by the RDMA lease manager to look up arbitrary blocks.
    pub(crate) fn get_blocks_for_lease(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> BlockLookupResult {
        self.read_cache
            .get_blocks_for_lease(namespace, block_hashes)
    }

    /// Return `(base_ptr, size)` for every pinned memory region.
    ///
    /// Used for RDMA memory registration at server startup.
    pub(crate) fn pinned_pool_regions(&self) -> Vec<(u64, usize)> {
        self.allocator.memory_regions()
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
    /// Insert a block directly into the in-memory cache (test only).
    pub(crate) fn test_insert_cache(&self, key: BlockKey, block: Arc<SealedBlock>) {
        self.read_cache.batch_insert(vec![(key, block)]);
    }

    /// Get the pin refcount for a (instance, block) pair (test only).
    pub(crate) fn test_pin_count(&self, instance_id: &str, key: &BlockKey) -> usize {
        self.read_cache.pin_count(instance_id, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> Arc<StorageEngine> {
        StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[])
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
        let storage = make_engine();
        let mut hashes: HashSet<Vec<u8>> = HashSet::new();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
        assert!(hashes.is_empty());
    }

    #[tokio::test]
    async fn pin_consume_releases_immediately() {
        let storage = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        // Insert into cache, then pin
        storage.test_insert_cache(key.clone(), block.clone());
        storage
            .read_cache
            .pin_blocks("inst1", 1, &[(key.clone(), block)]);

        assert_eq!(storage.test_pin_count("inst1", &key), 1);

        // consume_pinned_blocks transfers Arc ownership and consumes the reservation
        let blocks = storage
            .consume_pinned_blocks("inst1", "ns", &[vec![1]])
            .unwrap();
        assert_eq!(blocks.len(), 1);

        // Reservation is consumed immediately on lookup
        assert_eq!(storage.test_pin_count("inst1", &key), 0);
    }

    #[tokio::test]
    async fn unpin_blocks_cancellation_path() {
        let storage = make_engine();
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
        let storage = StorageEngine::new_with_config(4096, false, StorageConfig::default(), &[]);

        // Try to allocate more than the entire pool
        let result = storage.allocate(NonZeroU64::new(1 << 30).unwrap(), None);
        assert!(result.is_none(), "should fail, not loop forever");
    }

    #[tokio::test]
    async fn gc_stale_inflight_returns_zero_when_empty() {
        let storage = make_engine();
        let cleaned = storage
            .gc_stale_inflight(std::time::Duration::from_secs(60))
            .await;
        assert_eq!(cleaned, 0);
    }

    #[tokio::test]
    async fn multi_worker_consume_decrements_one_reservation_per_call() {
        let storage = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key.clone(), block.clone());
        // Pin for 3 workers
        storage
            .read_cache
            .pin_blocks("inst1", 3, &[(key.clone(), block)]);
        assert_eq!(storage.test_pin_count("inst1", &key), 3);

        // Worker 0 consumes
        let blocks_0 = storage
            .consume_pinned_blocks("inst1", "ns", &[vec![1]])
            .unwrap();
        assert_eq!(blocks_0.len(), 1);
        assert_eq!(storage.test_pin_count("inst1", &key), 2);
        // Worker 1 consumes
        let blocks_1 = storage
            .consume_pinned_blocks("inst1", "ns", &[vec![1]])
            .unwrap();
        assert_eq!(blocks_1.len(), 1);
        assert_eq!(storage.test_pin_count("inst1", &key), 1);
        // Worker 2 consumes
        let blocks_2 = storage
            .consume_pinned_blocks("inst1", "ns", &[vec![1]])
            .unwrap();
        assert_eq!(blocks_2.len(), 1);
        assert_eq!(storage.test_pin_count("inst1", &key), 0);

        // The owned Arcs can outlive reservation accounting.
        assert_eq!(blocks_0.len() + blocks_1.len() + blocks_2.len(), 3);
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_removes_cached() {
        let storage = make_engine();
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
        let storage = make_engine();
        // No pins exist — consume should fail
        let result = storage.consume_pinned_blocks("inst1", "ns", &[vec![1]]);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn check_prefix_memory_only_basic() {
        let storage = make_engine();
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
