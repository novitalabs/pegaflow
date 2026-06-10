//! PegaFlow Core Engine
//!
//! A GPU-aware KV cache offloading engine with support for:
//! - Multi-tenant instance isolation
//! - Tensor parallelism (TP) across multiple GPUs
//! - Split-storage layout for efficient K/V batch transfers
//! - SSD caching tier
//! - MetaServer-backed block discovery and RDMA fetch

#[macro_use]
mod trace;

mod allocator;
mod backing;
mod block;
mod cache;
mod gpu_worker;
mod instance;
mod internode;
mod lease;
pub use pegaflow_common::logging;
mod metrics;
mod offload;
mod pinned_mem;
mod pinned_pool;
mod seal_offload;
mod storage;
pub mod sync_state;
pub mod transfer;

pub use backing::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};
pub use block::{
    BlockHash, BlockKey, LayerBlock, LayerSave, PrefetchStatus, RawBlock, SealedBlock,
};
use instance::GpuRegistration;
pub use instance::{GpuContext, InstanceContext, KVCacheRegistration};
pub use internode::{DEFAULT_METASERVER_QUEUE_DEPTH, MetaServerClient, MetaServerClientConfig};
pub use lease::QueryLeaseId;
pub use pegaflow_common::NumaNode;
use pegaflow_common::NumaTopology;
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use storage::{MemoryCacheCleanupStats, StorageConfig};
pub use sync_state::{LoadState, LoadStateError};
pub use trace::{set_trace_sample_rate, should_sample};
pub use transfer::TransferMode;

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use log::{debug, info};

use crate::backing::SSD_ALIGNMENT;
use crate::gpu_worker::{LayerLoadData, LoadBlock, LoadCompletion, LoadTask};
use crate::lease::QueryLeaseManager;
use crate::metrics::core_metrics;
use crate::storage::StorageEngine;
use tokio::sync::oneshot;

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
    instances: Arc<RwLock<HashMap<String, Arc<InstanceContext>>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
    /// Query-ready blocks owned by opaque scheduler leases.
    query_leases: QueryLeaseManager,
    /// H2D/D2H transfer backend used by GPU worker pools.
    transfer_mode: TransferMode,
}

impl PegaEngine {
    /// Create an engine with full custom configuration.
    ///
    /// If `storage_config.enable_numa_affinity` is true and the system has multiple
    /// NUMA nodes, per-node pinned memory pools are created for optimal bandwidth.
    pub fn new_with_config(
        pool_size: usize,
        use_hugepages: bool,
        storage_config: storage::StorageConfig,
    ) -> Result<Self, EngineError> {
        let topology = Arc::new(NumaTopology::detect());
        topology.log_summary();

        let config = storage_config;
        let transfer_mode = config.transfer_mode;
        let numa_nodes: Vec<NumaNode> = if config.enable_numa_affinity && topology.is_multi_numa() {
            let gpu_numa_nodes = topology.gpu_numa_nodes();
            if gpu_numa_nodes.is_empty() {
                info!(
                    "Auto-enabling NUMA-aware memory allocation for {} CPU NUMA nodes; no GPU NUMA affinity detected",
                    topology.num_nodes()
                );
                topology.numa_nodes().to_vec()
            } else {
                info!(
                    "Auto-enabling NUMA-aware memory allocation for {} GPU-local NUMA nodes ({} CPU NUMA nodes detected)",
                    gpu_numa_nodes.len(),
                    topology.num_nodes()
                );
                gpu_numa_nodes
            }
        } else {
            vec![]
        };

        let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes)
            .map_err(EngineError::Storage)?;

        Ok(PegaEngine {
            instances: Arc::new(RwLock::new(HashMap::new())),
            storage,
            topology,
            query_leases: QueryLeaseManager::default(),
            transfer_mode,
        })
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
    ///
    /// Argument contract:
    /// - `device_id` must be non-negative; RPC callers validate this in service.rs.
    /// - `num_layers`, `tp_size`, and `world_size` must be non-zero.
    /// - `tp_rank` must be less than `tp_size`.
    /// - Layer metadata arrays must all have the same length as `layer_names`.
    /// - `layer_ids` must be connector-declared IDs in `0..num_layers`.
    /// - Registration sizes and pointers must describe a valid KV cache layout.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched registration RPC payload"
    )]
    pub fn register_context_layer_batch(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        tp_size: usize,
        world_size: usize,
        num_layers: usize,
        layer_names: &[String],
        layer_ids: &[usize],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
    ) -> Result<(), EngineError> {
        // Dense default: block stride == bytes_per_block (layer-first layout).
        self.register_context_layer_batch_strided(
            instance_id,
            namespace,
            device_id,
            tp_rank,
            pp_rank,
            tp_size,
            world_size,
            num_layers,
            layer_names,
            layer_ids,
            data_ptrs,
            size_bytes_list,
            num_blocks_list,
            bytes_per_block_list,
            kv_stride_bytes_list,
            segments_list,
            None,
        )
    }

    /// Like [`Self::register_context_layer_batch`] but with an explicit per-layer
    /// block stride (see [`KVCacheRegistration::with_block_stride`]). When
    /// `block_stride_bytes_list` is `Some` it must match `layer_names` in length;
    /// each entry overrides that layer's stride.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched registration RPC payload"
    )]
    pub fn register_context_layer_batch_strided(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        tp_size: usize,
        world_size: usize,
        num_layers: usize,
        layer_names: &[String],
        layer_ids: &[usize],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
        block_stride_bytes_list: Option<&[usize]>,
    ) -> Result<(), EngineError> {
        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        if layer_ids.len() != batch_size
            || data_ptrs.len() != batch_size
            || size_bytes_list.len() != batch_size
            || num_blocks_list.len() != batch_size
            || bytes_per_block_list.len() != batch_size
            || kv_stride_bytes_list.len() != batch_size
            || segments_list.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, layer_ids={}, data_ptrs={}, size_bytes={}, num_blocks={}, bytes_per_block={}, kv_stride_bytes={}, segments={}",
                layer_ids.len(),
                data_ptrs.len(),
                size_bytes_list.len(),
                num_blocks_list.len(),
                bytes_per_block_list.len(),
                kv_stride_bytes_list.len(),
                segments_list.len()
            )));
        }
        if let Some(strides) = block_stride_bytes_list
            && strides.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, block_stride_bytes={}",
                strides.len()
            )));
        }
        let mut kv_caches = HashMap::with_capacity(batch_size);
        let mut layer_ids_by_name = HashMap::with_capacity(batch_size);

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

            if let Some(strides) = block_stride_bytes_list {
                registration = registration.with_block_stride(strides[i]).map_err(|e| {
                    EngineError::InvalidArgument(format!("layer {layer_name}: {e}"))
                })?;
            }

            if ssd_enabled {
                registration = registration.with_ssd_padding(SSD_ALIGNMENT);
                if registration.padded_bytes_per_block != registration.bytes_per_block {
                    info!(
                        "SSD alignment padding: layer={layer_name}, bytes_per_block={} -> padded={}",
                        registration.bytes_per_block, registration.padded_bytes_per_block
                    );
                }
            }

            if kv_caches.insert(layer_name.clone(), registration).is_some() {
                return Err(EngineError::InvalidArgument(format!(
                    "duplicate layer name in registration batch: {layer_name}"
                )));
            }
            layer_ids_by_name.insert(layer_name.clone(), layer_ids[i]);
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
        instance.register_new_gpu(GpuRegistration {
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode: self.transfer_mode,
            kv_caches,
            layer_ids_by_name,
        })?;

        info!(
            "Registered context batch: instance={instance_id}, namespace={namespace}, \
             device={device_id}, num_layers={num_layers}, tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}"
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
        self.query_leases.release_instance(instance_id);
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
        drop(instances);
        for id in &ids {
            self.query_leases.release_instance(id);
        }
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

    /// Return the effective TP size registered for this instance.
    pub fn instance_tp_size(&self, instance_id: &str) -> Result<usize, EngineError> {
        Ok(self.get_instance(instance_id)?.tp_size())
    }

    /// Return the unique valid NUMA nodes used by a registered save group.
    pub fn registered_numa_nodes_for_save_group(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
    ) -> Result<Vec<NumaNode>, EngineError> {
        Ok(self
            .get_instance(instance_id)?
            .registered_numa_nodes_for_save_group(tp_rank, pp_rank))
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Argument contract:
    /// - `instance_id` must identify a registered instance.
    /// - `req_id` must be non-empty; RPC callers validate this in service.rs.
    /// - `block_hashes` may be empty.
    ///
    /// Returns:
    /// - `Ready { blocks, missing: 0 }`: all blocks in memory cache
    /// - `Loading`: some blocks being fetched from backing storage
    /// - `Ready { blocks, missing }`: terminal prefix result with a miss suffix
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "query_prefetch.count_prefix_hit")
    )]
    pub async fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        req_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<PrefetchStatus, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();

        let status = self
            .storage
            .check_prefix_and_prefetch(req_id, namespace, block_hashes)
            .await;

        match &status {
            PrefetchStatus::Ready { blocks, missing } => {
                let metrics = core_metrics();
                metrics.cache_block_hits.add(blocks.len() as u64, &[]);
                if *missing > 0 {
                    metrics.cache_block_misses.add(*missing as u64, &[]);
                }
            }
            PrefetchStatus::Loading => {}
        }

        Ok(status)
    }

    /// Create an opaque lease that owns query-ready blocks.
    pub fn create_query_lease(
        &self,
        instance_id: &str,
        blocks: Vec<Arc<SealedBlock>>,
    ) -> Result<QueryLeaseId, EngineError> {
        let instance = self.get_instance(instance_id)?;
        if blocks.is_empty() {
            return Err(EngineError::InvalidArgument(
                "query lease requires at least one block".to_string(),
            ));
        }
        Ok(self
            .query_leases
            .create(instance_id, blocks, instance.world_size()))
    }

    /// Release a query lease. Returns false when the lease is unknown or expired.
    pub fn release_query_lease(&self, lease: &QueryLeaseId) -> bool {
        self.query_leases.release(lease)
    }

    /// Evict all resident in-memory cache blocks while preserving backing-store data.
    pub fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        self.query_leases.sweep_expired();
        self.storage.cleanup_memory_cache()
    }

    /// Best-effort graceful unregister from MetaServer, if configured.
    pub async fn shutdown_metaserver_client(&self) {
        self.storage.shutdown_metaserver_client().await;
    }

    /// Batch load KV blocks for multiple layers asynchronously.
    ///
    /// Returns immediately after submitting the task to the GPU worker pool.
    /// The connector spin-waits on the `LoadState` until completion.
    #[allow(
        clippy::too_many_arguments,
        reason = "public API mirrors one batched load RPC payload"
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        load_state_shm: &str,
        layer_names: &[&str],
        loads: &[(QueryLeaseId, Vec<i32>)],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            layer_names,
            loads,
            LoadCompletion::Shm(load_state_shm.to_string()),
        );

        if let Err(ref e) = result {
            log::error!("batch_load_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
    }

    /// In-process variant of [`Self::batch_load_kv_blocks_multi_layer`]: instead
    /// of a caller-managed shared-memory `LoadState`, it returns a oneshot
    /// receiver that resolves when the GPU worker finishes the load (`Ok`) or it
    /// fails (`Err`). Poll it with `try_recv` to keep admission non-blocking, or
    /// await it. For in-process Rust embedders that register raw device pointers
    /// and have no second process to coordinate a `LoadState` with.
    ///
    /// On a pre-submit error the receiver is dropped (yields `RecvError`); the
    /// same error is returned synchronously here.
    pub fn batch_load_kv_blocks_multi_layer_inproc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_names: &[&str],
        loads: &[(QueryLeaseId, Vec<i32>)],
    ) -> Result<oneshot::Receiver<Result<(), EngineError>>, EngineError> {
        let (reply, rx) = oneshot::channel();
        self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            layer_names,
            loads,
            LoadCompletion::Channel(reply),
        )?;
        Ok(rx)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "internal helper keeps the public load API validation path explicit"
    )]
    fn batch_load_kv_blocks_multi_layer_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_names: &[&str],
        loads: &[(QueryLeaseId, Vec<i32>)],
        completion: LoadCompletion,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        instance.ensure_all_slots_registered()?;
        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        // Consume query leases reserved for this load.
        trace_scope!("load.cache_lookup", _s);
        let mut block_ids = Vec::new();
        let mut block_cache = Vec::new();
        for (lease, lease_block_ids) in loads {
            let blocks = self
                .query_leases
                .consume(instance_id, lease)
                .map_err(EngineError::Storage)?;
            if blocks.len() != lease_block_ids.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "query lease block count {} does not match destination block count {}",
                    blocks.len(),
                    lease_block_ids.len()
                )));
            }
            block_ids.extend_from_slice(lease_block_ids);
            block_cache.extend(blocks);
        }
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

            let mut blocks = Vec::with_capacity(block_ids.len());
            for (block_id, block_entry) in block_ids.iter().zip(block_cache.iter()) {
                let Ok(block_idx) = usize::try_from(*block_id) else {
                    continue;
                };
                let Some(block) = block_entry.get_slot(slot_id) else {
                    continue;
                };
                blocks.push(LoadBlock {
                    block_idx,
                    block: Arc::clone(block),
                });
            }

            if !blocks.is_empty() {
                layers.push(LayerLoadData {
                    layer_name: (*layer_name).to_string(),
                    registration,
                    blocks,
                });
            }
        }

        // Complete immediately if no blocks to load
        if layers.is_empty() {
            debug!("No blocks to load, completing immediately");
            completion.signal(Ok(()));
            return Ok(());
        }

        // Submit to worker pool (fire and forget)
        gpu.worker_pool()
            .submit_load(LoadTask { layers, completion })
    }

    /// Wait until all previously submitted save batches have been processed
    /// by the insert worker.
    ///
    /// This is a flush barrier: it guarantees that every `batch_save_kv_blocks_from_ipc`
    /// call that returned before this call will have its blocks inserted into the
    /// read cache (or inflight map) by the time this future resolves.
    pub async fn flush_saves(&self) {
        self.storage.flush_write_pipeline().await;
    }

    /// Flush write pipeline and SSD writer.
    ///
    /// Guarantees that all saves submitted before this call are both
    /// cache-visible and persisted to SSD (if SSD is enabled).
    pub async fn flush_all(&self) {
        self.storage.flush_write_pipeline().await;
        self.storage.flush_ssd().await;
    }

    /// Remove stale inflight blocks and failed_remote entries (background GC).
    ///
    /// Should be called periodically (e.g., every 30 seconds).
    pub async fn gc_stale_inflight(
        &self,
        inflight_max_age: std::time::Duration,
        failed_remote_max_age: std::time::Duration,
    ) -> (usize, usize) {
        self.storage
            .gc_stale_inflight(inflight_max_age, failed_remote_max_age)
            .await
    }

    // =========================================================================
    // Cross-node transfer: serving side
    // =========================================================================

    /// Lock the longest present, layout-homogeneous prefix of `block_hashes`
    /// for RDMA transfer. Returns the session ID, the number of blocks
    /// locked, and the per-slot layout template every locked block matches.
    ///
    /// Both sides derive all transfer addresses by replaying slot-major bump
    /// allocation over the template, so the prefix must be exact: a missing
    /// block or a block with a different layout truncates it.
    pub fn query_blocks_for_transfer(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        requester_id: &str,
    ) -> (
        String,
        usize,
        Vec<pegaflow_proto::proto::engine::TransferSlotInfo>,
    ) {
        let t0 = std::time::Instant::now();
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|h| BlockKey::new(namespace.to_string(), h.clone()))
            .collect();

        // `get_blocks_for_transfer` returns the present subset in request
        // order; cut it down to the contiguous prefix of the request.
        let found = self.storage.get_blocks_for_transfer(&keys);
        let lookup_elapsed = t0.elapsed();
        let mut found_iter = found.into_iter().peekable();
        let mut prefix: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();
        for key in &keys {
            match found_iter.peek() {
                Some((fk, _)) if fk == key => {
                    prefix.push(found_iter.next().expect("peeked entry exists"));
                }
                _ => break,
            }
        }

        let template = prefix
            .first()
            .map(|(_, block)| block_slot_template(block))
            .unwrap_or_default();
        // The scan dereferences every slot Arc of every block; fan large
        // prefixes out across threads. The first global mismatch is the
        // minimum of the chunk-local first mismatches.
        let homogeneous = if prefix.len() >= 256 {
            let threads = 4.min(prefix.len());
            let chunk = prefix.len().div_ceil(threads);
            let template = &template;
            let prefix = prefix.as_slice();
            std::thread::scope(|scope| {
                let workers: Vec<_> = (0..threads)
                    .map(|t| {
                        let lo = t * chunk;
                        let hi = ((t + 1) * chunk).min(prefix.len());
                        scope.spawn(move || {
                            prefix[lo..hi]
                                .iter()
                                .position(|(_, block)| !block_matches_template(block, template))
                                .map(|p| lo + p)
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .filter_map(|w| w.join().expect("homogeneity scan worker panicked"))
                    .min()
                    .unwrap_or(prefix.len())
            })
        } else {
            prefix
                .iter()
                .take_while(|(_, block)| block_matches_template(block, &template))
                .count()
        };
        prefix.truncate(homogeneous);

        let template_elapsed = t0.elapsed() - lookup_elapsed;
        let lock_start = std::time::Instant::now();
        let session_id = self.storage.lock_blocks_for_transfer(requester_id, &prefix);

        debug!(
            "query_blocks_for_transfer: namespace={namespace} requested={} locked_prefix={} session={session_id} lookup_ms={:.2} template_ms={:.2} lock_ms={:.2}",
            block_hashes.len(),
            prefix.len(),
            lookup_elapsed.as_secs_f64() * 1000.0,
            template_elapsed.as_secs_f64() * 1000.0,
            lock_start.elapsed().as_secs_f64() * 1000.0,
        );

        (session_id, prefix.len(), template)
    }

    /// GC expired transfer lock sessions.
    pub fn gc_expired_transfer_locks(&self) -> usize {
        self.storage.gc_expired_transfer_locks()
    }

    /// Return `(base_ptr, size)` for each contiguous pinned memory region.
    /// Used for RDMA memory registration.
    pub fn pinned_memory_regions(&self) -> Vec<(u64, usize)> {
        self.storage.pinned_memory_regions()
    }

    /// Returns true if RDMA transport is available.
    #[cfg(feature = "rdma")]
    pub fn has_rdma_transport(&self) -> bool {
        self.storage.rdma_transport().is_some()
    }

    /// Returns true if RDMA transport is available.
    #[cfg(not(feature = "rdma"))]
    pub fn has_rdma_transport(&self) -> bool {
        false
    }

    /// Serve a PushBlocks request: take the transfer session's locked blocks
    /// and RDMA-WRITE them into the requester's memory. Returns the number of
    /// bytes pushed once every WRITE has completed; the transfer lock is
    /// released when this function returns (success or failure).
    #[cfg(feature = "rdma")]
    pub async fn push_blocks_for_transfer(
        &self,
        request: &pegaflow_proto::proto::engine::PushBlocksRequest,
    ) -> Result<u64, PushBlocksError> {
        use crate::backing::{PushSegment, mr_desc_from_proto};

        let t0 = std::time::Instant::now();
        let rdma = self.storage.rdma_transport().ok_or_else(|| {
            PushBlocksError::Rejected("RDMA transport not configured".to_string())
        })?;

        // Holding the Arcs keeps the blocks alive while the WRITEs run.
        let session_blocks = self
            .storage
            .take_transfer_session(&request.transfer_session_id)
            .ok_or_else(|| {
                PushBlocksError::Rejected(format!(
                    "transfer session {} not found (expired or already pushed)",
                    request.transfer_session_id
                ))
            })?;

        // The replay below assumes the session is layout-homogeneous (the
        // query locked only such a prefix): every block has full per-slot
        // NUMA info identical to the first block's. Blocks rebuilt from SSD
        // or a previous RDMA fetch carry no per-slot NUMA info and must be
        // rejected, not panicked on.
        let slot_numas = session_blocks
            .first()
            .map(|(_, sealed)| sealed.slot_numas())
            .unwrap_or(&[]);
        for (key, sealed) in &session_blocks {
            if sealed.slot_numas().len() != sealed.slots().len()
                || sealed.slot_numas() != slot_numas
            {
                return Err(PushBlocksError::Rejected(format!(
                    "block {key:?} breaks the session's slot layout; it cannot be re-served"
                )));
            }
        }

        let mr_descs: Vec<_> = request
            .memory_regions
            .iter()
            .map(mr_desc_from_proto)
            .collect();

        // Bump-replay cursors, one per requester slab. The request carries
        // only slab bases; every destination address is derived here by
        // replaying the requester's allocation order with our own segment
        // sizes (the same sizes we returned in the query's slot_template).
        struct SlabCursor {
            mr_index: u32,
            next: u64,
            end: u64,
        }
        let mut slabs: std::collections::HashMap<u32, SlabCursor> =
            std::collections::HashMap::new();
        for slab in &request.slabs {
            if slab.mr_index as usize >= mr_descs.len() {
                return Err(PushBlocksError::Rejected(format!(
                    "slab mr_index {} out of bounds ({} regions in request)",
                    slab.mr_index,
                    mr_descs.len()
                )));
            }
            let end = slab.base_addr.checked_add(slab.capacity).ok_or_else(|| {
                PushBlocksError::Rejected(format!(
                    "slab 0x{:x}+{} overflows the address space",
                    slab.base_addr, slab.capacity
                ))
            })?;
            let cursor = SlabCursor {
                mr_index: slab.mr_index,
                next: slab.base_addr,
                end,
            };
            if slabs.insert(slab.numa_node, cursor).is_some() {
                return Err(PushBlocksError::Rejected(format!(
                    "duplicate slab for NUMA node {}",
                    slab.numa_node
                )));
            }
        }
        // Slot-major, K run before V run — the same order in which the
        // requester assigned destinations from its bump allocator. Each
        // layer's K (and V) segments of consecutive blocks are adjacent in
        // the pinned pool, so this order yields source- and
        // destination-contiguous runs that push_segments coalesces.
        //
        // Layout homogeneity (validated above) lets each (slot, K/V) run
        // resolve its slab cursor once instead of per segment.
        let session_elapsed = t0.elapsed();
        let build_start = std::time::Instant::now();
        let mut segments = Vec::new();
        for (slot_idx, slot_numa) in slot_numas.iter().enumerate() {
            let numa = slot_numa.0;
            for seg_idx in 0..2 {
                let slab = slabs.get_mut(&numa).ok_or_else(|| {
                    PushBlocksError::Rejected(format!("no requester slab for NUMA node {numa}"))
                })?;
                // Within one (slot, K/V) run the destination is a single bump
                // region, so source-contiguous segments of consecutive blocks
                // merge into one WRITE-sized segment. Layer allocations keep
                // consecutive blocks adjacent, so a run typically collapses
                // to one segment per (slot, K/V).
                let mut run = PushSegment {
                    src_addr: 0,
                    len: 0,
                    dst_mr_index: slab.mr_index,
                    dst_addr: 0,
                };
                for (_key, sealed) in &session_blocks {
                    let raw = &sealed.slots()[slot_idx];
                    let (Some(ptr), Some(size)) =
                        (raw.segment_ptr(seg_idx), raw.segment_size(seg_idx))
                    else {
                        continue;
                    };
                    let len = size as u64;
                    if len == 0 {
                        continue;
                    }
                    if len > slab.end - slab.next {
                        return Err(PushBlocksError::Rejected(format!(
                            "slab for NUMA node {numa} exhausted during replay: cursor=0x{:x} len={len} end=0x{:x}",
                            slab.next, slab.end
                        )));
                    }
                    let src = ptr.as_ptr() as u64;
                    if run.len > 0 && run.src_addr + run.len == src {
                        run.len += len;
                    } else {
                        if run.len > 0 {
                            segments.push(run);
                        }
                        run.src_addr = src;
                        run.dst_addr = slab.next;
                        run.len = len;
                    }
                    slab.next += len;
                }
                if run.len > 0 {
                    segments.push(run);
                }
            }
        }

        // The replayed cursors must land exactly on the requester's
        // allocation ends. Any divergence means the two sides disagree on
        // block count, sizes, or order — reject before submitting a WRITE.
        for (numa, slab) in &slabs {
            if slab.next != slab.end {
                return Err(PushBlocksError::Rejected(format!(
                    "bump replay mismatch on NUMA node {numa}: cursor ended at 0x{:x} but requester allocated up to 0x{:x}",
                    slab.next, slab.end
                )));
            }
        }

        let build_elapsed = build_start.elapsed();
        let rdma_start = std::time::Instant::now();
        let result = rdma.push_segments(&mr_descs, segments).await;
        info!(
            "PushBlocks timing: session_ms={:.2} build_ms={:.2} rdma_ms={:.2}",
            session_elapsed.as_secs_f64() * 1000.0,
            build_elapsed.as_secs_f64() * 1000.0,
            rdma_start.elapsed().as_secs_f64() * 1000.0,
        );
        result
    }

    /// Serve a PushBlocks request.
    #[cfg(not(feature = "rdma"))]
    pub async fn push_blocks_for_transfer(
        &self,
        _request: &pegaflow_proto::proto::engine::PushBlocksRequest,
    ) -> Result<u64, PushBlocksError> {
        Err(PushBlocksError::Rejected(
            "this binary was built without RDMA support".to_string(),
        ))
    }
}

/// Per-slot layout of a sealed block as the transfer protocol sees it.
/// The requester sizes its slabs from this template and both sides replay
/// slot-major bump allocation over it, so it is the protocol's source of
/// truth for segment sizes and NUMA placement.
fn block_slot_template(
    block: &SealedBlock,
) -> Vec<pegaflow_proto::proto::engine::TransferSlotInfo> {
    block
        .slots()
        .iter()
        .zip(block.slot_numas())
        .map(
            |(raw, numa)| pegaflow_proto::proto::engine::TransferSlotInfo {
                k_size: raw.segment_size(0).unwrap_or(0) as u64,
                v_size: raw.segment_size(1).unwrap_or(0) as u64,
                numa_node: numa.0,
            },
        )
        .collect()
}

/// Field-wise template match without materializing a per-block template Vec —
/// the query's homogeneity scan runs this over every candidate block.
fn block_matches_template(
    block: &SealedBlock,
    template: &[pegaflow_proto::proto::engine::TransferSlotInfo],
) -> bool {
    block.slots().len() == template.len()
        && block.slot_numas().len() == template.len()
        && block
            .slots()
            .iter()
            .zip(block.slot_numas())
            .zip(template)
            .all(|((raw, numa), t)| {
                raw.segment_size(0).unwrap_or(0) as u64 == t.k_size
                    && raw.segment_size(1).unwrap_or(0) as u64 == t.v_size
                    && numa.0 == t.numa_node
            })
}

/// Error from serving a PushBlocks request.
///
/// The split matters for the requester's memory safety: a `Rejected` push
/// never submitted an RDMA WRITE, so the requester may recycle its
/// destination buffers; after `Failed`, WRITEs may still be in flight and
/// the requester must leak them.
#[derive(Debug)]
pub enum PushBlocksError {
    /// Rejected before any RDMA WRITE was submitted.
    Rejected(String),
    /// Failed after WRITEs may have been submitted.
    Failed(String),
}

impl std::fmt::Display for PushBlocksError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(msg) => write!(f, "rejected: {msg}"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rdma")]
    #[test]
    fn rdma_initialization_failure_is_returned_to_caller() {
        let config = storage::StorageConfig {
            rdma_nic_names: Some(vec!["definitely-not-a-real-nic".to_string()]),
            ..storage::StorageConfig::default()
        };

        let err = match PegaEngine::new_with_config(1 << 20, false, config) {
            Ok(_) => panic!("engine startup must fail when RDMA NIC init fails"),
            Err(err) => err.to_string(),
        };

        assert!(err.contains("Failed to initialise RDMA transport"), "{err}");
        assert!(err.contains("definitely-not-a-real-nic"), "{err}");
    }

    #[cfg(not(feature = "rdma"))]
    #[test]
    fn rdma_config_is_ignored_without_feature() {
        let config = storage::StorageConfig {
            rdma_nic_names: Some(vec!["mlx5_0".to_string()]),
            ..storage::StorageConfig::default()
        };

        let engine = PegaEngine::new_with_config(1 << 20, false, config)
            .expect("no-RDMA build should ignore RDMA NIC config");

        assert!(!engine.has_rdma_transport());
    }

    #[cfg(not(feature = "rdma"))]
    #[tokio::test]
    async fn push_blocks_reports_missing_feature() {
        let engine = PegaEngine::new_with_config(1 << 20, false, storage::StorageConfig::default())
            .expect("engine should start without RDMA");

        let err = engine
            .push_blocks_for_transfer(&pegaflow_proto::proto::engine::PushBlocksRequest::default())
            .await
            .expect_err("no-RDMA build should reject push requests");

        assert!(
            matches!(err, PushBlocksError::Rejected(ref msg) if msg.contains("without RDMA")),
            "expected Rejected, got {err:?}"
        );
    }
}
