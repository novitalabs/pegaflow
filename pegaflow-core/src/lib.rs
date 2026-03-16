//! PegaFlow Core Engine
//!
//! A GPU-aware KV cache offloading engine with support for:
//! - Multi-tenant instance isolation
//! - Tensor parallelism (TP) across multiple GPUs
//! - Split-storage layout for efficient K/V batch transfers
//! - SSD caching tier
//! - Kubernetes service discovery for inter-node communication

#[macro_use]
mod trace;

pub mod allocator;
pub mod backing;
pub mod block;
mod cache;
pub mod gpu_worker;
pub mod instance;
pub mod lease;
pub mod logging;
pub mod metaserver;
mod metrics;
pub mod numa;
mod offload;
pub mod pinned_mem;
pub mod pinned_pool;
mod seal_offload;
mod storage;
pub mod sync_state;
mod transfer;

pub use backing::{
    BakingStoreConfig, DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
    DEFAULT_SSD_WRITE_INFLIGHT, DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};
pub use block::{
    BlockHash, BlockKey, BlockLookupResult, BlockStatus, LayerBlock, LayerSave, PrefetchStatus,
    SealedBlock,
};
pub use instance::{GpuContext, InstanceContext, KVCacheRegistration};
pub use lease::{LeaseConfig, LeaseError, LeaseGrant, LeaseManager};
pub use numa::NumaNode;
use numa::NumaTopology;
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use storage::StorageConfig;
pub use sync_state::{LoadState, LoadStateError};
pub use trace::{set_trace_sample_rate, should_sample};

// ============================================================================
// KV Cache Layout Notes
// ============================================================================
//
// PegaFlow currently prioritizes vLLM's layer-first (KV-first) tensor layout.
// This means all K segments are contiguous, followed by all V segments:
//
//   +---------------------------------------------------------------+
//   |  Layer0: KKKKKKKK.... | Layer0: VVVVVVVV.... | Layer1: K ...  |
//   +---------------------------------------------------------------+
//          ^ contiguous K blocks        ^ contiguous V blocks
//
// To support efficient batching during "load" (CPU -> GPU), we avoid
// interleaving K and V in a single contiguous block. Instead, we allocate
// all K segments in one contiguous CPU region, and all V segments in another.
// This Split-Storage approach allows merging K source pointers into a single
// cuMemcpy, significantly improving PCIe bandwidth utilization.
// ============================================================================

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use log::{debug, info, warn};
use uuid::Uuid;

use crate::backing::SSD_ALIGNMENT;
use crate::gpu_worker::{LayerLoadData, LoadBlock, LoadTask};
use crate::metrics::core_metrics;
use crate::storage::StorageEngine;

const DEFAULT_PINNED_POOL_BYTES: usize = 30 * 1024 * 1024 * 1024; // 30GB

/// Errors that can occur during engine operations.
#[derive(Debug)]
pub enum EngineError {
    /// Instance not found in the registry.
    InstanceMissing(String),
    /// GPU worker not found for the specified device.
    WorkerMissing(String, i32),
    /// Invalid argument provided.
    InvalidArgument(String),
    /// CUDA initialization or runtime error.
    CudaInit(String),
    /// Storage engine error.
    Storage(String),
    /// Internal lock poisoned.
    Poisoned(&'static str),
    /// Topology mismatch between registration and existing instance.
    TopologyMismatch(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::InstanceMissing(ctx) => write!(f, "instance {ctx} not found"),
            EngineError::WorkerMissing(ctx, device) => {
                write!(f, "device {device} not found in instance {ctx}")
            }
            EngineError::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            EngineError::CudaInit(msg) => write!(f, "failed to initialize CUDA: {msg}"),
            EngineError::Storage(msg) => write!(f, "storage error: {msg}"),
            EngineError::Poisoned(what) => write!(f, "internal lock poisoned: {what}"),
            EngineError::TopologyMismatch(msg) => write!(f, "topology mismatch: {msg}"),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<LoadStateError> for EngineError {
    fn from(err: LoadStateError) -> Self {
        EngineError::Storage(format!("LoadState: {err}"))
    }
}

/// Main engine for managing KV cache offloading.
///
/// `PegaEngine` is the top-level orchestrator that:
/// - Manages multiple inference instances
/// - Coordinates GPU worker pools for async transfers
/// - Interfaces with the storage engine for block caching
/// - Tracks GPU-NUMA topology for optimal memory locality
///
/// The engine is thread-safe and can be shared across async tasks.
pub struct PegaEngine {
    /// Active inference instances indexed by instance ID.
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
    /// Stable node identity for RDMA lease tracking.
    node_id: Uuid,
    /// Lease manager for RDMA P2P block transfer.
    lease_manager: Arc<LeaseManager>,
}

#[allow(clippy::new_without_default)]
impl PegaEngine {
    /// Create a new engine with default 30GB pinned memory pool.
    pub fn new() -> Self {
        Self::new_with_config(
            DEFAULT_PINNED_POOL_BYTES,
            false,
            storage::StorageConfig::default(),
        )
    }

    /// Create an engine with full custom configuration.
    ///
    /// If `storage_config.enable_numa_affinity` is true and the system has multiple
    /// NUMA nodes, per-node pinned memory pools are created for optimal bandwidth.
    pub fn new_with_config(
        pool_size: usize,
        use_hugepages: bool,
        storage_config: impl Into<storage::StorageConfig>,
    ) -> Self {
        // Detect GPU-NUMA topology
        let topology = Arc::new(NumaTopology::detect());
        topology.log_summary();

        // Resolve NUMA nodes based on config and topology
        let mut config = storage_config.into();
        let numa_nodes: Vec<numa::NumaNode> =
            if config.enable_numa_affinity && topology.is_multi_numa() {
                info!(
                    "Auto-enabling NUMA-aware memory allocation for {} nodes",
                    topology.num_nodes()
                );
                topology.numa_nodes().to_vec()
            } else {
                vec![]
            };

        let node_id = Uuid::new_v4();

        // Populate node_id into P2P config so the backing store can identify itself in lease RPCs.
        if let Some(ref mut baking_cfg) = config.baking_store_config {
            baking_cfg.node_id = node_id.to_string();
        }

        let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes);
        let lease_manager = Arc::new(LeaseManager::new(LeaseConfig::default()));
        info!("PegaEngine node_id={node_id}");

        PegaEngine {
            instances: RwLock::new(HashMap::new()),
            storage,
            topology,
            node_id,
            lease_manager,
        }
    }

    /// Get or create an instance with the specified topology.
    ///
    /// If an instance with the same ID exists but different topology,
    /// returns a `TopologyMismatch` error.
    fn get_or_create_instance(
        &self,
        instance_id: &str,
        namespace: &str,
        num_layers: usize,
        tp_size: usize,
        world_size: usize,
    ) -> Result<Arc<InstanceContext>, EngineError> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");

        if let Some(instance) = instances.get(instance_id) {
            // Already exists, verify topology
            instance
                .verify_topology(num_layers, tp_size, world_size)
                .map_err(|e| {
                    EngineError::TopologyMismatch(format!("instance {instance_id} {e}"))
                })?;
            return Ok(Arc::clone(instance));
        }

        // Create new instance
        let instance = InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            num_layers,
            tp_size,
            world_size,
        )
        .map_err(EngineError::InvalidArgument)?;

        let instance = Arc::new(instance);
        instances.insert(instance_id.to_string(), Arc::clone(&instance));
        Ok(instance)
    }

    /// Look up an instance by ID.
    fn get_instance(&self, instance_id: &str) -> Result<Arc<InstanceContext>, EngineError> {
        let instances = self.instances.read().expect("instances read lock poisoned");
        instances
            .get(instance_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))
    }

    /// Batch register multiple KV cache layers for a single GPU.
    ///
    /// This reduces gRPC round-trips from N to 1 and allows the engine to
    /// construct the GPU context with all layer registrations at once.
    #[allow(clippy::too_many_arguments)]
    pub fn register_context_layer_batch(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        tp_size: usize,
        world_size: usize,
        num_layers: usize,
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
    ) -> Result<(), EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(
                "device_id must be >= 0".to_string(),
            ));
        }

        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        let mut kv_caches = HashMap::with_capacity(batch_size);

        for i in 0..batch_size {
            let layer_name = &layer_names[i];
            let mut registration = KVCacheRegistration::new(
                data_ptrs[i],
                size_bytes_list[i],
                num_blocks_list[i],
                bytes_per_block_list[i],
                kv_stride_bytes_list[i],
                segments_list[i],
            )
            .map_err(|e| EngineError::InvalidArgument(format!("layer {layer_name}: {e}")))?;

            if ssd_enabled {
                registration = registration.with_ssd_padding(SSD_ALIGNMENT);
                if registration.padded_bytes_per_block != registration.bytes_per_block {
                    info!(
                        "SSD alignment padding: layer={layer_name}, bytes_per_block={} -> padded={}",
                        registration.bytes_per_block, registration.padded_bytes_per_block
                    );
                }
            }

            kv_caches.insert(layer_name.clone(), registration);
        }

        // Get or create instance
        let instance =
            self.get_or_create_instance(instance_id, namespace, num_layers, tp_size, world_size)?;

        // Get NUMA affinity for this GPU
        let numa_node = self.topology.numa_for_gpu(device_id);

        // Validate NUMA topology if NUMA-aware allocation is enabled
        if self.storage.is_numa_enabled() && numa_node.is_unknown() {
            return Err(EngineError::InvalidArgument(format!(
                "NUMA-aware allocation is enabled, but GPU {} NUMA affinity is unknown. \
                 Please ensure nvidia-smi is available and GPU NUMA topology is detectable, \
                 or disable NUMA-aware allocation.",
                device_id
            )));
        }

        // Register GPU with all layers
        instance.register_new_gpu(device_id, numa_node, kv_caches)?;

        info!(
            "Registered context batch: instance={instance_id}, namespace={namespace}, \
             device={device_id}, num_layers={num_layers}, tp_rank={tp_rank}/{tp_size}"
        );
        Ok(())
    }

    /// Unregister an instance and release all associated resources.
    pub fn unregister_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let removed = self
            .instances
            .write()
            .expect("instances write lock poisoned")
            .remove(instance_id);

        if removed.is_none() {
            return Err(EngineError::InstanceMissing(instance_id.to_string()));
        }
        info!("Unregistered instance: {}", instance_id);
        Ok(())
    }

    /// Unregister all instances, returning the IDs that were removed.
    pub fn unregister_all_instances(&self) -> Vec<String> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        let ids: Vec<String> = instances.keys().cloned().collect();
        instances.clear();
        if !ids.is_empty() {
            info!("Unregistered all instances: {:?}", ids);
        }
        ids
    }

    /// List all registered instance IDs.
    pub fn list_instance_ids(&self) -> Vec<String> {
        self.instances
            .read()
            .expect("instances read lock poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// Pure memory-only prefix query.
    ///
    /// Checks which prefix blocks are in the memory cache without triggering
    /// SSD prefetch or pinning blocks. Returns `(hit, missing)` counts.
    #[cfg_attr(feature = "tracing", fastrace::trace(name = "query.count_prefix_hit"))]
    pub fn count_prefix_hit_blocks(
        &self,
        instance_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<(usize, usize), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let metrics = core_metrics();

        let (hit, missing) = self
            .storage
            .check_prefix_memory_only(namespace, block_hashes);

        metrics.cache_block_hits.add(hit as u64, &[]);
        if missing > 0 {
            metrics.cache_block_misses.add(missing as u64, &[]);
        }

        Ok((hit, missing))
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Returns:
    /// - `Done { hit, missing: 0 }`: all blocks in memory cache
    /// - `Loading { hit, loading }`: some blocks being fetched from SSD
    /// - `Done { hit, missing }`: some blocks don't exist
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "query_prefetch.count_prefix_hit")
    )]
    pub fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<PrefetchStatus, EngineError> {
        if req_id.is_empty() {
            warn!("count_prefix_hit_blocks_with_prefetch: empty req_id, returning 0 hits");
            return Ok(PrefetchStatus::Done {
                hit: 0,
                missing: block_hashes.len(),
            });
        }

        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let world_size = instance.world_size();
        let metrics = core_metrics();

        let status = self.storage.check_prefix_and_prefetch(
            instance_id,
            req_id,
            namespace,
            block_hashes,
            world_size,
        );

        match &status {
            PrefetchStatus::Done { hit, missing } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
                if *missing > 0 {
                    metrics.cache_block_misses.add(*missing as u64, &[]);
                }
            }
            PrefetchStatus::Loading { hit, loading: _ } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Unpin blocks that were pinned during query.
    ///
    /// Used when load is cancelled or preempted before consumption.
    pub fn unpin_blocks(
        &self,
        instance_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<usize, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let unpinned = self
            .storage
            .unpin_blocks(instance_id, namespace, block_hashes);
        debug!(
            "unpin_blocks: instance_id={instance_id} blocks={} unpinned={unpinned}",
            block_hashes.len()
        );
        Ok(unpinned)
    }

    /// Batch load KV blocks for multiple layers asynchronously.
    ///
    /// Returns immediately after submitting the task to the GPU worker pool.
    /// The connector spin-waits on the `LoadState` until completion.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            load_state_shm,
            layer_names,
            block_ids,
            block_hashes,
        );

        if let Err(ref e) = result {
            log::error!("batch_load_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_load_kv_blocks_multi_layer_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();

        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        // Consume all pinned blocks reserved for this load.
        trace_scope!("load.cache_lookup", _s);
        let block_cache = self
            .storage
            .consume_pinned_blocks(instance_id, namespace, block_hashes)
            .map_err(EngineError::Storage)?;
        trace_drop!(_s);

        // Build load tasks for each layer
        trace_scope!("load.build_tasks");
        let mut layers = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_id = instance.get_layer_id(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} unknown for instance {instance_id}"
                ))
            })?;

            let registration = gpu.get_registration(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} not registered on device {device_id}"
                ))
            })?;

            let slot_id = instance.get_slot_index(layer_id, tp_rank)?;

            let blocks: Vec<LoadBlock> = block_ids
                .iter()
                .zip(block_cache.iter())
                .filter_map(|(block_id, block_entry)| {
                    let block_idx = usize::try_from(*block_id).ok()?;
                    let layer_block = block_entry.get_slot(slot_id)?.clone();
                    Some(LoadBlock {
                        block_idx,
                        layer_block,
                    })
                })
                .collect();

            if !blocks.is_empty() {
                layers.push(LayerLoadData {
                    layer_name: layer_name.to_string(),
                    registration,
                    blocks,
                });
            }
        }

        // Complete immediately if no blocks to load
        if layers.is_empty() {
            debug!("No blocks to load, completing immediately");
            LoadState::attach(load_state_shm)?.set_completed();
            return Ok(());
        }

        // Submit to worker pool (fire and forget)
        gpu.worker_pool().submit_load(LoadTask {
            layers,
            load_state_shm: load_state_shm.to_string(),
        })
    }

    /// Look up sealed blocks by hash for RDMA lease acquisition.
    ///
    /// Returns (found_blocks, missing_hashes). Does not stop at first miss.
    pub fn get_blocks_for_lease(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> BlockLookupResult {
        self.storage.get_blocks_for_lease(namespace, block_hashes)
    }

    /// Return `(base_ptr, size)` for every pinned memory region.
    ///
    /// Used for RDMA memory registration: each region must be registered
    /// with `TransferEngine::register_memory` so that one-sided
    /// RDMA reads can reach any sealed block in the pool.
    pub fn pinned_pool_regions(&self) -> Vec<(u64, usize)> {
        self.storage.pinned_pool_regions()
    }

    /// Stable node identity for RDMA lease tracking.
    pub fn node_id(&self) -> Uuid {
        self.node_id
    }

    /// Acquire a lease on sealed blocks for RDMA transfer.
    pub fn acquire_lease(
        &self,
        requester_node_id: &str,
        namespace: &str,
        found_blocks: Vec<(BlockKey, Arc<SealedBlock>)>,
        missing_hashes: Vec<Vec<u8>>,
        requested_duration_secs: u32,
    ) -> Result<LeaseGrant, LeaseError> {
        self.lease_manager.acquire(
            requester_node_id,
            namespace,
            found_blocks,
            missing_hashes,
            requested_duration_secs,
        )
    }

    /// Extend an existing lease.
    pub fn renew_lease(
        &self,
        lease_id: lease::LeaseId,
        requester_node_id: &str,
        extend_duration_secs: u32,
    ) -> Result<u64, LeaseError> {
        self.lease_manager
            .renew(lease_id, requester_node_id, extend_duration_secs)
    }

    /// Release a lease, dropping Arc references so blocks become evictable.
    pub fn release_lease(
        &self,
        lease_id: lease::LeaseId,
        requester_node_id: &str,
    ) -> Result<(), LeaseError> {
        self.lease_manager.release(lease_id, requester_node_id)
    }

    /// Sweep expired leases. Call periodically from a background task.
    pub fn sweep_expired_leases(&self) -> usize {
        self.lease_manager.sweep_expired()
    }

    /// Remove stale inflight blocks (background GC).
    ///
    /// Should be called periodically (e.g., every 5 minutes).
    pub async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        self.storage.gc_stale_inflight(max_age).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockKey, SealedBlock};

    /// Helper: create a PegaEngine with a small pool and register a test instance.
    fn setup_engine(instance_id: &str, namespace: &str) -> PegaEngine {
        let config = storage::StorageConfig {
            enable_lfu_admission: false,
            hint_value_size_bytes: None,
            max_prefetch_blocks: 100,
            baking_store_config: None,
            ssd_cache_config: None,
            enable_numa_affinity: false,
            transfer_engine: None,
        };
        let engine = PegaEngine::new_with_config(1 << 20, false, config);

        // Manually insert an InstanceContext (no GPU required)
        let instance = InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            2, // num_layers
            1, // tp_size
            1, // world_size
        )
        .unwrap();
        engine
            .instances
            .write()
            .unwrap()
            .insert(instance_id.to_string(), Arc::new(instance));

        engine
    }

    /// Helper: insert a block directly into the storage cache.
    fn insert_block(engine: &PegaEngine, namespace: &str, hash: Vec<u8>) {
        let key = BlockKey::new(namespace.to_string(), hash);
        let block = Arc::new(SealedBlock::from_slots(Vec::new()));
        engine.storage.test_insert_cache(key, block);
    }

    #[tokio::test]
    async fn all_blocks_cached_returns_done_full_hit() {
        let engine = setup_engine("inst1", "ns");
        insert_block(&engine, "ns", vec![1]);
        insert_block(&engine, "ns", vec![2]);
        insert_block(&engine, "ns", vec![3]);

        let hashes = vec![vec![1], vec![2], vec![3]];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 3);
                assert_eq!(missing, 0);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn prefix_hit_then_miss() {
        let engine = setup_engine("inst1", "ns");
        insert_block(&engine, "ns", vec![1]);
        insert_block(&engine, "ns", vec![2]);
        // vec![3] is NOT cached

        let hashes = vec![vec![1], vec![2], vec![3], vec![4]];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 2);
                assert_eq!(missing, 2); // [3] and [4]
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn all_blocks_missing() {
        let engine = setup_engine("inst1", "ns");

        let hashes = vec![vec![1], vec![2]];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 0);
                assert_eq!(missing, 2);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn empty_hashes_returns_done_zero() {
        let engine = setup_engine("inst1", "ns");

        let hashes: Vec<Vec<u8>> = vec![];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 0);
                assert_eq!(missing, 0);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn instance_not_found_returns_error() {
        let engine = setup_engine("inst1", "ns");

        let result =
            engine.count_prefix_hit_blocks_with_prefetch("nonexistent", "test-req", &[vec![1]]);

        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EngineError::InstanceMissing(id) if id == "nonexistent")
        );
    }

    #[tokio::test]
    async fn first_block_miss_makes_all_missing() {
        let engine = setup_engine("inst1", "ns");
        // Only [2] is cached, but [1] is not — prefix breaks at [1]
        insert_block(&engine, "ns", vec![2]);

        let hashes = vec![vec![1], vec![2], vec![3]];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 0);
                assert_eq!(missing, 3);
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn namespace_isolation() {
        let engine = setup_engine("inst1", "ns_a");
        // Block is cached under "ns_a", but query uses "ns_a" namespace
        insert_block(&engine, "ns_a", vec![1]);
        // Also insert a block under a different namespace
        insert_block(&engine, "ns_b", vec![2]);

        // Instance uses "ns_a", so only vec![1] should hit
        let hashes = vec![vec![1], vec![2]];
        let status = engine
            .count_prefix_hit_blocks_with_prefetch("inst1", "test-req", &hashes)
            .unwrap();

        match status {
            PrefetchStatus::Done { hit, missing } => {
                assert_eq!(hit, 1);
                assert_eq!(missing, 1); // vec![2] is under "ns_b", miss for "ns_a"
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    // ================================================================
    // count_prefix_hit_blocks (pure memory-only query)
    // ================================================================

    #[tokio::test]
    async fn query_all_blocks_cached() {
        let engine = setup_engine("inst1", "ns");
        insert_block(&engine, "ns", vec![1]);
        insert_block(&engine, "ns", vec![2]);
        insert_block(&engine, "ns", vec![3]);

        let (hit, missing) = engine
            .count_prefix_hit_blocks("inst1", &[vec![1], vec![2], vec![3]])
            .unwrap();

        assert_eq!(hit, 3);
        assert_eq!(missing, 0);
    }

    #[tokio::test]
    async fn query_prefix_hit_then_miss() {
        let engine = setup_engine("inst1", "ns");
        insert_block(&engine, "ns", vec![1]);
        insert_block(&engine, "ns", vec![2]);
        // vec![3] NOT cached

        let (hit, missing) = engine
            .count_prefix_hit_blocks("inst1", &[vec![1], vec![2], vec![3], vec![4]])
            .unwrap();

        assert_eq!(hit, 2);
        assert_eq!(missing, 2); // [3] and [4]
    }

    #[tokio::test]
    async fn query_all_blocks_missing() {
        let engine = setup_engine("inst1", "ns");

        let (hit, missing) = engine
            .count_prefix_hit_blocks("inst1", &[vec![1], vec![2]])
            .unwrap();

        assert_eq!(hit, 0);
        assert_eq!(missing, 2);
    }

    #[tokio::test]
    async fn query_empty_hashes() {
        let engine = setup_engine("inst1", "ns");

        let (hit, missing) = engine.count_prefix_hit_blocks("inst1", &[]).unwrap();

        assert_eq!(hit, 0);
        assert_eq!(missing, 0);
    }

    #[tokio::test]
    async fn query_instance_not_found() {
        let engine = setup_engine("inst1", "ns");

        let result = engine.count_prefix_hit_blocks("nonexistent", &[vec![1]]);

        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), EngineError::InstanceMissing(id) if id == "nonexistent")
        );
    }

    #[tokio::test]
    async fn query_first_block_miss_makes_all_missing() {
        let engine = setup_engine("inst1", "ns");
        // Only [2] is cached, but [1] is not — prefix breaks at [1]
        insert_block(&engine, "ns", vec![2]);

        let (hit, missing) = engine
            .count_prefix_hit_blocks("inst1", &[vec![1], vec![2], vec![3]])
            .unwrap();

        assert_eq!(hit, 0);
        assert_eq!(missing, 3);
    }

    #[tokio::test]
    async fn query_namespace_isolation() {
        let engine = setup_engine("inst1", "ns_a");
        insert_block(&engine, "ns_a", vec![1]);
        insert_block(&engine, "ns_b", vec![2]);

        let (hit, missing) = engine
            .count_prefix_hit_blocks("inst1", &[vec![1], vec![2]])
            .unwrap();

        assert_eq!(hit, 1);
        assert_eq!(missing, 1); // vec![2] is under "ns_b", miss for "ns_a"
    }

    /// Verify that `count_prefix_hit_blocks` is idempotent / side-effect-free:
    /// calling it multiple times yields the same result (no pinning or state mutation).
    #[tokio::test]
    async fn query_is_idempotent() {
        let engine = setup_engine("inst1", "ns");
        insert_block(&engine, "ns", vec![1]);
        insert_block(&engine, "ns", vec![2]);

        for _ in 0..3 {
            let (hit, missing) = engine
                .count_prefix_hit_blocks("inst1", &[vec![1], vec![2], vec![3]])
                .unwrap();
            assert_eq!(hit, 2);
            assert_eq!(missing, 1);
        }
    }
}
