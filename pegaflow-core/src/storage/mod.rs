mod prefetch;
mod read_cache;
mod tier_attribution;
pub(crate) mod transfer_lock;
mod write_path;

use bytesize::ByteSize;
#[cfg(not(feature = "rdma"))]
use log::warn;
use log::{debug, info};
use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::{Arc, Weak};
use std::time::Duration;

use crate::backing::{AllocateFn, DEFAULT_MAX_PREFETCH_BLOCKS, SsdBackingStore, SsdCacheConfig};
#[cfg(feature = "rdma")]
use crate::backing::{RdmaFetchStore, RdmaTransport};
use crate::block::{BlockKey, PrefetchStatus, SealedBlock};
use crate::internode::MetaServerClient;
use crate::internode::metaserver_client::MetaServerClientConfig;
use crate::metrics::core_metrics;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};
use pegaflow_common::NumaNode;

use prefetch::PrefetchScheduler;
#[cfg(feature = "rdma")]
use prefetch::RdmaFetch;
use read_cache::ReadCache;
use write_path::{InsertDeps, WritePipeline};

const RECLAIM_BATCH_SIZE: usize = 64;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCacheCleanupStats {
    pub evicted_blocks: usize,
    pub evicted_bytes: u64,
    pub reclaimed_bytes: u64,
    pub still_referenced_blocks: u64,
}

#[derive(Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity).
    pub hint_value_size_bytes: Option<usize>,
    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch).
    pub max_prefetch_blocks: usize,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
    /// Optional RDMA NIC names for inter-node transfer (e.g. `["mlx5_0", "mlx5_1"]`).
    pub rdma_nic_names: Option<Vec<String>>,
    /// Enable NUMA-aware memory allocation.
    pub enable_numa_affinity: bool,
    /// Allocate each block separately instead of contiguous batch allocation.
    /// Reduces fragmentation when blocks are freed in different order.
    pub blockwise_alloc: bool,
    /// Transfer lock timeout for cross-node RDMA transfers.
    pub transfer_lock_timeout: Duration,
    /// MetaServer address for p2p block discovery + registration (None = disabled).
    pub metaserver_addr: Option<String>,
    /// This node's routable address (from --addr) used for MetaServer registration and as
    /// requester_id in transfer locks. Must be set when metaserver_addr is set.
    pub advertise_addr: Option<String>,
    /// MetaServer registration queue depth.
    pub metaserver_queue_depth: usize,
    /// Number of shards for the pinned memory pool (reduces allocator lock contention).
    pub pool_shards: usize,
    /// H2D/D2H transfer backend used by the GPU workers.
    pub transfer_mode: crate::TransferMode,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_lfu_admission: false,
            hint_value_size_bytes: None,
            max_prefetch_blocks: DEFAULT_MAX_PREFETCH_BLOCKS,
            ssd_cache_config: None,
            rdma_nic_names: None,
            enable_numa_affinity: true,
            blockwise_alloc: false,
            transfer_lock_timeout: Duration::from_secs(120),
            metaserver_addr: None,
            advertise_addr: None,
            metaserver_queue_depth: crate::internode::DEFAULT_METASERVER_QUEUE_DEPTH,
            pool_shards: 1,
            transfer_mode: crate::TransferMode::default(),
        }
    }
}

pub(crate) struct StorageEngine {
    allocator: Arc<PinnedAllocator>,
    read_cache: Arc<ReadCache>,
    prefetch: PrefetchScheduler,
    write_pipeline: Arc<WritePipeline>,
    ssd_store: Option<Arc<SsdBackingStore>>,
    #[cfg(feature = "rdma")]
    rdma_transport: Option<Arc<RdmaTransport>>,
    blockwise_alloc: bool,
    metaserver_client: Option<Arc<MetaServerClient>>,
    transfer_lock: Arc<transfer_lock::TransferLockManager>,
}

impl StorageEngine {
    #[cfg_attr(
        not(feature = "rdma"),
        expect(
            clippy::unnecessary_wraps,
            reason = "only RDMA transport init can fail; keep one signature across builds"
        )
    )]
    pub(crate) fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: StorageConfig,
        numa_nodes: &[NumaNode],
    ) -> Result<Arc<Self>, String> {
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));
        let max_prefetch_blocks = config.max_prefetch_blocks;
        let ssd_cache_config = config.ssd_cache_config;
        let rdma_nic_names = config.rdma_nic_names;
        let blockwise_alloc = config.blockwise_alloc;
        let transfer_lock_timeout = config.transfer_lock_timeout;

        // Create MetaServer client if configured
        let metaserver_client = config.metaserver_addr.as_ref().map(|addr| {
            let advertise = config
                .advertise_addr
                .clone()
                .unwrap_or_else(|| "127.0.0.1:50055".to_string());
            info!(
                "MetaServer client enabled: metaserver={}, advertise={}, queue_depth={}",
                addr, advertise, config.metaserver_queue_depth
            );
            let ms_config = MetaServerClientConfig::new(addr.clone(), advertise)
                .with_queue_depth(config.metaserver_queue_depth);
            Arc::new(MetaServerClient::new(ms_config))
        });

        if blockwise_alloc {
            info!("Blockwise allocation enabled for batch_save");
        }

        let ssd_enabled = ssd_cache_config.is_some();
        let pool_shards = config.pool_shards;

        // Create unified allocator based on NUMA configuration
        let allocator = if !numa_nodes.is_empty() {
            info!(
                "Creating NUMA-aware pinned pools for {} nodes",
                numa_nodes.len()
            );
            Arc::new(PinnedAllocator::new_numa(
                capacity_bytes,
                numa_nodes,
                pool_shards,
                use_hugepages,
                ssd_enabled,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                pool_shards,
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

        // RDMA transport must be created after the allocator so it can
        // register the pinned memory regions with the RDMA NICs.
        let rdma_nics = rdma_nic_names.as_deref().filter(|nics| !nics.is_empty());
        #[cfg(feature = "rdma")]
        let rdma_transport = if let Some(nics) = rdma_nics {
            let rdma = crate::backing::new_rdma(nics, &allocator)?;
            crate::metrics::register_rdma_gauges(&rdma);
            Some(rdma)
        } else {
            None
        };

        #[cfg(not(feature = "rdma"))]
        if rdma_nics.is_some() {
            warn!(
                "RDMA NICs were configured, but this binary was built without the `rdma` feature; ignoring RDMA config"
            );
        }

        let is_numa = allocator.is_numa();
        let engine = Arc::new_cyclic(move |weak_engine: &Weak<Self>| {
            // Build shared allocate_fn for backing stores.
            let alloc_weak = weak_engine.clone();
            let allocate_fn: AllocateFn = Arc::new(move |size, numa_node| {
                alloc_weak
                    .upgrade()
                    .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
            });

            let ssd_store = ssd_cache_config
                .map(|cfg| crate::backing::new_ssd(cfg, allocate_fn.clone(), is_numa));

            #[cfg(feature = "rdma")]
            let rdma_fetch = rdma_transport.as_ref().and_then(|rdma| {
                let ms = metaserver_client.as_ref()?;
                let advertise = config
                    .advertise_addr
                    .clone()
                    .unwrap_or_else(|| "127.0.0.1:50055".to_string());
                Some(RdmaFetch::new(Arc::new(RdmaFetchStore::new(
                    Arc::clone(ms),
                    Arc::clone(rdma),
                    allocate_fn.clone(),
                    advertise,
                ))))
            });
            #[cfg(not(feature = "rdma"))]
            let rdma_fetch = None;

            let prefetch =
                PrefetchScheduler::new(ssd_store.clone(), rdma_fetch, max_prefetch_blocks);

            let transfer_lock = Arc::new(transfer_lock::TransferLockManager::new(
                transfer_lock_timeout,
            ));

            Self {
                allocator,
                read_cache: read_cache.clone(),
                prefetch,
                write_pipeline: write_pipeline.clone(),
                ssd_store,
                #[cfg(feature = "rdma")]
                rdma_transport,
                blockwise_alloc,
                metaserver_client,
                transfer_lock,
            }
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let deps = Arc::new(InsertDeps {
                read_cache: engine.read_cache.clone(),
                ssd_store: engine.ssd_store.clone(),
                metaserver_client: engine.metaserver_client.clone(),
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

        Ok(engine)
    }

    pub(crate) fn is_ssd_enabled(&self) -> bool {
        self.ssd_store.is_some()
    }

    pub(crate) fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    /// Returns true if blockwise allocation is enabled.
    pub(crate) fn blockwise_alloc(&self) -> bool {
        self.blockwise_alloc
    }

    /// Allocate pinned memory, returning `None` if the pool is exhausted after eviction.
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
                self.reclaim_until_allocator_can_allocate(requested_bytes, node);

            if freed_blocks == 0 && largest_free < requested_bytes {
                // Final retry: absorb concurrent frees that may have raced with reclaim probing.
                if let Some(alloc) = self.allocator.allocate(size, node) {
                    return Some(alloc);
                }
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

    pub(crate) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        self.write_pipeline.send_raw_insert(batch);
    }

    /// Flush the write pipeline.
    ///
    /// Returns a receiver that resolves once all batches enqueued before this
    /// call have been processed by the insert worker.
    pub(crate) async fn flush_write_pipeline(&self) {
        if let Some(rx) = self.write_pipeline.flush() {
            let _ = rx.await;
        }
    }

    /// Flush the SSD writer: waits until all enqueued writes are committed.
    pub(crate) async fn flush_ssd(&self) {
        if let Some(ssd) = &self.ssd_store {
            ssd.flush().await;
        }
    }

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

    /// Evict all blocks from the resident in-memory read cache.
    ///
    /// This preserves backing-store copies. Blocks with
    /// outstanding references may remain allocated until those holders release
    /// the last `Arc`.
    pub(crate) fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        let used_before = self.allocator.usage().0;
        let evicted = self.read_cache.remove_all();
        if evicted.is_empty() {
            return MemoryCacheCleanupStats::default();
        }

        if let Some(client) = &self.metaserver_client {
            let entries: Vec<(String, Vec<u8>)> = evicted
                .iter()
                .map(|(key, _)| (key.namespace.clone(), key.hash.clone()))
                .collect();
            client.try_unregister(entries);
        }

        let mut evicted_bytes = 0u64;
        let mut still_referenced_blocks = 0u64;
        for (_key, block) in &evicted {
            let bytes = block.memory_footprint();
            evicted_bytes = evicted_bytes.saturating_add(bytes);
            if Arc::strong_count(block) > 1 {
                still_referenced_blocks += 1;
            }
            core_metrics()
                .cache_resident_bytes
                .add(-(bytes as i64), &[]);
        }

        let evicted_blocks = evicted.len();
        if still_referenced_blocks > 0 {
            core_metrics()
                .cache_block_evictions_still_referenced
                .add(still_referenced_blocks, &[]);
        }
        core_metrics()
            .cache_block_evictions
            .add(evicted_blocks as u64, &[]);

        drop(evicted);
        let reclaimed_bytes = used_before.saturating_sub(self.allocator.usage().0);
        if reclaimed_bytes > 0 {
            core_metrics()
                .cache_eviction_reclaimed_bytes
                .add(reclaimed_bytes, &[]);
        }

        info!(
            "Cleaned resident memory cache: evicted_blocks={} evicted_bytes={} reclaimed_bytes={} still_referenced_blocks={}",
            evicted_blocks,
            ByteSize(evicted_bytes),
            ByteSize(reclaimed_bytes),
            still_referenced_blocks
        );

        MemoryCacheCleanupStats {
            evicted_blocks,
            evicted_bytes,
            reclaimed_bytes,
            still_referenced_blocks,
        }
    }

    /// Check prefix blocks and schedule backing-store reads if needed.
    pub(crate) async fn check_prefix_and_prefetch(
        &self,
        req_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchStatus {
        self.prefetch
            .check_and_prefetch(&self.read_cache, req_id, namespace, hashes)
            .await
    }

    fn reclaim_until_allocator_can_allocate(
        &self,
        required_bytes: u64,
        target_node: NumaNode,
    ) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (
                0,
                0,
                self.allocator.largest_free_allocation_for_node(target_node),
            );
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.allocator.largest_free_allocation_for_node(target_node);

        while largest_free < required_bytes {
            let used_before = self.allocator.usage().0;

            let evicted = self.read_cache.remove_lru_batch(RECLAIM_BATCH_SIZE);

            if evicted.is_empty() {
                break;
            }

            // Notify MetaServer that evicted blocks are no longer available for RDMA fetch.
            if let Some(client) = &self.metaserver_client {
                let entries: Vec<(String, Vec<u8>)> = evicted
                    .iter()
                    .map(|(key, _)| (key.namespace.clone(), key.hash.clone()))
                    .collect();
                client.try_unregister(entries);
            }

            let mut batch_bytes = 0u64;
            let mut still_referenced = 0u64;
            for (_key, block) in &evicted {
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

            largest_free = self.allocator.largest_free_allocation_for_node(target_node);
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

    /// Remove stale inflight blocks and failed_remote entries.
    pub(crate) async fn gc_stale_inflight(
        &self,
        inflight_max_age: std::time::Duration,
        failed_remote_max_age: std::time::Duration,
    ) -> (usize, usize) {
        let cleaned = self
            .write_pipeline
            .gc_stale_inflight(inflight_max_age)
            .await;
        let (stale_prefetch, failed) = self
            .prefetch
            .gc_stale_entries(inflight_max_age, failed_remote_max_age);
        if stale_prefetch > 0 {
            log::debug!("gc: removed {stale_prefetch} stale prefetch entries");
        }
        if failed > 0 {
            log::debug!("gc: cleared {failed} stale failed_remote entries");
        }
        (cleaned, failed)
    }

    // ---- Cross-node transfer: serving side ----

    /// Look up specific blocks by key (non-prefix). For cross-node transfer.
    pub(crate) fn get_blocks_for_transfer(
        &self,
        keys: &[BlockKey],
    ) -> Vec<(BlockKey, Arc<SealedBlock>)> {
        self.read_cache.get_blocks(keys)
    }

    /// Lock blocks for a transfer session, returning the session ID.
    pub(crate) fn lock_blocks_for_transfer(
        &self,
        requester_id: &str,
        blocks: &[(BlockKey, Arc<SealedBlock>)],
    ) -> String {
        self.transfer_lock
            .lock_blocks(requester_id, blocks.to_vec())
    }

    /// Take a transfer session's blocks for a push. The returned Arcs keep
    /// the blocks alive while the RDMA WRITEs run.
    #[cfg(any(feature = "rdma", test))]
    pub(crate) fn take_transfer_session(
        &self,
        session_id: &str,
    ) -> Option<Vec<(BlockKey, Arc<SealedBlock>)>> {
        self.transfer_lock.take(session_id)
    }

    /// GC expired transfer lock sessions. Returns the number of sessions expired.
    pub(crate) fn gc_expired_transfer_locks(&self) -> usize {
        self.transfer_lock.gc_expired()
    }

    /// Return `(base_ptr, size)` for each contiguous pinned memory region.
    /// Used for RDMA memory registration.
    pub(crate) fn pinned_memory_regions(&self) -> Vec<(u64, usize)> {
        self.allocator
            .memory_regions()
            .into_iter()
            .map(|(ptr, len)| (ptr.as_ptr() as u64, len))
            .collect()
    }

    #[cfg(feature = "rdma")]
    pub(crate) fn rdma_transport(&self) -> Option<&Arc<RdmaTransport>> {
        self.rdma_transport.as_ref()
    }

    pub(crate) async fn shutdown_metaserver_client(&self) {
        if let Some(client) = &self.metaserver_client {
            client.shutdown().await;
        }
    }
}

#[cfg(test)]
impl StorageEngine {
    /// Insert a block directly into the in-memory cache (test only).
    pub(crate) fn test_insert_cache(&self, key: BlockKey, block: Arc<SealedBlock>) {
        self.read_cache.batch_insert(vec![(key, block)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> Arc<StorageEngine> {
        StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]).unwrap()
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
        let storage = make_engine();
        let mut hashes: HashSet<Vec<u8>> = HashSet::new();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
        assert!(hashes.is_empty());
    }

    #[tokio::test]
    async fn cleanup_memory_cache_evicts_all_resident_blocks() {
        let storage = make_engine();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);
        let block1 = Arc::new(SealedBlock::from_slots(Vec::new()));
        let block2 = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key1, Arc::clone(&block1));
        storage.test_insert_cache(key2, block2);

        let stats = storage.cleanup_memory_cache();
        assert_eq!(stats.evicted_blocks, 2);
        assert_eq!(stats.still_referenced_blocks, 1);
        assert_eq!(stats.reclaimed_bytes, 0);

        let stats = storage.cleanup_memory_cache();
        assert_eq!(stats, MemoryCacheCleanupStats::default());
    }

    #[tokio::test]
    async fn allocate_bounded_reclaim_terminates() {
        // With a tiny pool, allocation of a huge block should fail fast
        // (not loop forever) thanks to MAX_RECLAIM_ROUNDS.
        let storage =
            StorageEngine::new_with_config(4096, false, StorageConfig::default(), &[]).unwrap();

        // Try to allocate more than the entire pool
        let result = storage.allocate(NonZeroU64::new(1 << 30).unwrap(), None);
        assert!(result.is_none(), "should fail, not loop forever");
    }

    #[tokio::test]
    async fn gc_stale_inflight_returns_zero_when_empty() {
        let storage = make_engine();
        let (cleaned, failed) = storage
            .gc_stale_inflight(
                std::time::Duration::from_secs(60),
                std::time::Duration::from_secs(60),
            )
            .await;
        assert_eq!(cleaned, 0);
        assert_eq!(failed, 0);
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

    // ---- Cross-node transfer: serving side tests ----

    #[tokio::test]
    async fn get_blocks_for_transfer_returns_correct_blocks() {
        let storage = make_engine();
        let key1 = BlockKey::new("ns".into(), vec![1]);
        let key2 = BlockKey::new("ns".into(), vec![2]);
        let key3 = BlockKey::new("ns".into(), vec![3]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key1.clone(), block.clone());
        storage.test_insert_cache(key3.clone(), block.clone());

        // Request keys 1, 2, 3 — only 1 and 3 are present (non-prefix semantics)
        let result = storage.get_blocks_for_transfer(&[key1.clone(), key2, key3.clone()]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, key1);
        assert_eq!(result[1].0, key3);
    }

    #[tokio::test]
    async fn get_blocks_for_transfer_empty_keys() {
        let storage = make_engine();
        let result = storage.get_blocks_for_transfer(&[]);
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn pinned_memory_regions_returns_non_empty() {
        let storage = make_engine();
        let regions = storage.pinned_memory_regions();
        // With a 1 MiB pool, there should be at least one region
        assert!(
            !regions.is_empty(),
            "pinned_memory_regions should return at least one region"
        );
        // Each region should have a non-zero size
        for (_ptr, size) in &regions {
            assert!(*size > 0, "region size should be non-zero");
        }
    }

    #[tokio::test]
    async fn lock_and_take_transfer_session() {
        let storage = make_engine();
        let key = BlockKey::new("ns".into(), vec![1]);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));

        storage.test_insert_cache(key.clone(), block.clone());

        let session_id = storage.lock_blocks_for_transfer("node-a", &[(key, block)]);
        assert!(
            !session_id.is_empty(),
            "lock_blocks_for_transfer should return a UUID when enabled"
        );

        let taken = storage
            .take_transfer_session(&session_id)
            .expect("session should exist");
        assert_eq!(taken.len(), 1);
        assert!(storage.take_transfer_session(&session_id).is_none());
    }
}
