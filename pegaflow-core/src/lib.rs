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
mod layout;
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
pub use instance::{GpuContext, InstanceContext};
pub use internode::{
    DEFAULT_METASERVER_QUEUE_DEPTH, MetaServerClient, MetaServerClientConfig, P2pTransferService,
};
use layout::KVCacheLayout;
pub use lease::QueryLeaseId;
pub use pegaflow_common::NumaNode;
use pegaflow_common::NumaTopology;
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use storage::{DEFAULT_RDMA_QPS_PER_PEER, MemoryCacheCleanupStats, StorageConfig};
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
use crate::gpu_worker::{HostBlock, LayerTransferData, LoadCompletion, LoadTask, TransferBlock};
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
        tp_size: usize,
        world_size: usize,
        page_first: bool,
    ) -> Result<Arc<InstanceContext>, EngineError> {
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");

        if let Some(instance) = instances.get(instance_id) {
            // Already exists, verify topology
            instance
                .verify_topology(tp_size, world_size, page_first)
                .map_err(|e| {
                    EngineError::TopologyMismatch(format!("instance {instance_id} {e}"))
                })?;
            return Ok(Arc::clone(instance));
        }

        // Create new instance
        let instance = InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            tp_size,
            world_size,
            page_first,
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
    /// - `tp_size` and `world_size` must be non-zero.
    /// - `tp_rank` must be less than `tp_size`.
    /// - Layer metadata arrays must all have the same length as `layer_names`.
    /// - Registration sizes and pointers must describe a valid KV cache layout.
    ///
    /// The instance's layer-id space is sealed by the engine once `world_size`
    /// devices have registered; callers declare only the layers that actually
    /// exist on each device.
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
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
        transfer_mode: TransferMode,
        page_first: bool,
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
            layer_names,
            data_ptrs,
            size_bytes_list,
            num_blocks_list,
            bytes_per_block_list,
            kv_stride_bytes_list,
            segments_list,
            None,
            transfer_mode,
            page_first,
        )
    }

    /// Like [`Self::register_context_layer_batch`] but with an explicit per-layer
    /// block stride (see [`KVCacheRegistration::with_block_stride`]). When
    /// `block_stride_bytes_list` is `Some` it must match `layer_names` in length;
    /// each entry overrides that layer's stride.
    ///
    /// `transfer_mode` selects the GPU worker pools' H2D/D2H backend for this
    /// instance. The pools are spawned per (instance, GPU) at registration, so
    /// each instance can run a different backend.
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
        layer_names: &[String],
        data_ptrs: &[u64],
        size_bytes_list: &[usize],
        num_blocks_list: &[usize],
        bytes_per_block_list: &[usize],
        kv_stride_bytes_list: &[usize],
        segments_list: &[usize],
        block_stride_bytes_list: Option<&[usize]>,
        transfer_mode: TransferMode,
        page_first: bool,
    ) -> Result<(), EngineError> {
        // Build all registrations
        let ssd_enabled = self.storage.is_ssd_enabled();
        let batch_size = layer_names.len();
        if data_ptrs.len() != batch_size
            || size_bytes_list.len() != batch_size
            || num_blocks_list.len() != batch_size
            || bytes_per_block_list.len() != batch_size
            || kv_stride_bytes_list.len() != batch_size
            || segments_list.len() != batch_size
        {
            return Err(EngineError::InvalidArgument(format!(
                "registration metadata length mismatch: layer_names={batch_size}, data_ptrs={}, size_bytes={}, num_blocks={}, bytes_per_block={}, kv_stride_bytes={}, segments={}",
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

        for i in 0..batch_size {
            let layer_name = &layer_names[i];
            let mut layout = KVCacheLayout::new(
                data_ptrs[i],
                size_bytes_list[i],
                num_blocks_list[i],
                bytes_per_block_list[i],
                kv_stride_bytes_list[i],
                segments_list[i],
            )
            .map_err(|e| EngineError::InvalidArgument(format!("layer {layer_name}: {e}")))?;

            if let Some(strides) = block_stride_bytes_list {
                layout = layout.with_block_stride(strides[i]).map_err(|e| {
                    EngineError::InvalidArgument(format!("layer {layer_name}: {e}"))
                })?;
            }

            if ssd_enabled {
                layout = layout.with_ssd_padding(SSD_ALIGNMENT);
                if layout.padded_segment_bytes() != layout.segment_bytes() {
                    info!(
                        "SSD alignment padding: layer={layer_name}, bytes_per_block={} -> padded={}",
                        layout.segment_bytes(),
                        layout.padded_segment_bytes()
                    );
                }
            }

            if kv_caches.insert(layer_name.clone(), layout).is_some() {
                return Err(EngineError::InvalidArgument(format!(
                    "duplicate layer name in registration batch: {layer_name}"
                )));
            }
        }

        // Get or create instance
        let instance =
            self.get_or_create_instance(instance_id, namespace, tp_size, world_size, page_first)?;

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

        // Register GPU with all layers. The connector picks the backend per
        // model and sends it with the registration.
        instance.register_new_gpu(GpuRegistration {
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        })?;

        info!(
            "Registered context batch: instance={instance_id}, namespace={namespace}, \
             device={device_id}, layers={batch_size}, tp_rank={tp_rank}/{tp_size}, pp_rank={pp_rank}"
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
        wait_for_remote: bool,
    ) -> Result<PrefetchStatus, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();

        let status = self
            .storage
            .check_prefix_and_prefetch(req_id, namespace, block_hashes, wait_for_remote)
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
        loads: &[(QueryLeaseId, Vec<usize>)],
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
        loads: &[(QueryLeaseId, Vec<usize>)],
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
        loads: &[(QueryLeaseId, Vec<usize>)],
        completion: LoadCompletion,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
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
            // A stored block must carry exactly this instance's slot layout.
            // A mismatch means the namespace is shared by instances with
            // different layer sets (e.g. MTP enabled vs disabled) — loading
            // would silently leave layers uninitialized, so fail loudly and
            // let vLLM recompute.
            for block in &blocks {
                if block.slots().len() != topology.total_slots() {
                    return Err(EngineError::InvalidArgument(format!(
                        "stored block has {} slots but instance {instance_id} expects {}: \
                         namespace is shared by incompatible KV layouts",
                        block.slots().len(),
                        topology.total_slots()
                    )));
                }
            }
            block_ids.extend_from_slice(lease_block_ids);
            block_cache.extend(blocks);
        }
        trace_drop!(_s);

        // Build load tasks for each layer
        trace_scope!("load.build_tasks");
        let mut layers = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_id = topology.layer_id(layer_name)?;

            let layout = gpu.get_layout(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} not registered on device {device_id}"
                ))
            })?;

            let slot_id = topology.slot_index(layer_id, tp_rank)?;
            // Page-first: every layer reads from the one page slot (tp_rank) at
            // its sealed byte offset. Layer-first: offset 0 (the whole slot
            // RawBlock is the layer).
            let host_offset = topology
                .page_placement(layer_id)
                .map_or(0, |(offset, _)| offset);

            let mut blocks = Vec::with_capacity(block_ids.len());
            for (block_idx, block_entry) in block_ids.iter().copied().zip(block_cache.iter()) {
                if block_entry.get_slot(slot_id).is_none() {
                    return Err(EngineError::InvalidArgument(format!(
                        "stored block is missing slot {slot_id} for layer {layer_name}"
                    )));
                }
                blocks.push(TransferBlock {
                    block_idx,
                    block: HostBlock::Cached {
                        sealed: Arc::clone(block_entry),
                        slot_id,
                        offset: host_offset,
                    },
                });
            }

            if !blocks.is_empty() {
                layers.push(LayerTransferData {
                    layer_name: (*layer_name).to_string(),
                    layout,
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

    /// [`Self::flush_saves`] plus a MetaServer registration barrier: on return,
    /// every block saved before this call is cache-resident *and* its hash
    /// registration has been delivered to the MetaServer — or dropped after a
    /// failed attempt. Registration stays best-effort: this bounds *when*
    /// delivery is attempted, never *whether* it succeeds (a full queue drops
    /// hashes at enqueue time, a MetaServer outage drops the whole batch).
    ///
    /// The P/D handoff barrier: a prefill node calls this before signalling
    /// "KV ready", so a decode node's MetaServer query observes every
    /// registration that made it through; blocks whose registration was
    /// dropped are simply recomputed on the decode side. Ordering matters —
    /// the write pipeline enqueues the registrations as it seals blocks, so
    /// the pipeline must drain first. Without a MetaServer client this
    /// degrades to [`Self::flush_saves`].
    pub async fn flush_saves_and_registrations(&self) {
        self.storage.flush_write_pipeline().await;
        self.storage.flush_metaserver_registrations().await;
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

    /// Look up blocks and lock them for RDMA transfer. Returns metadata
    /// for each found block plus a session ID for later unlock.
    pub fn query_blocks_for_transfer(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        requester_id: &str,
    ) -> (String, Vec<(BlockKey, Arc<SealedBlock>)>) {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|h| BlockKey::new(namespace.to_string(), h.clone()))
            .collect();

        let found = self.storage.get_blocks_for_transfer(&keys);
        let session_id = self.storage.lock_blocks_for_transfer(requester_id, &found);

        debug!(
            "query_blocks_for_transfer: namespace={namespace} requested={} found={} session={session_id}",
            block_hashes.len(),
            found.len(),
        );

        (session_id, found)
    }

    pub fn transfer_lock_timeout(&self) -> std::time::Duration {
        self.storage.transfer_lock_timeout()
    }

    /// Release a transfer lock session. Returns the number of blocks released.
    pub fn release_transfer_lock(&self, session_id: &str) -> usize {
        self.storage.release_transfer_lock(session_id)
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

    /// Perform server-side RDMA handshake with connection reuse.
    ///
    /// If `client_handshake_bytes` is empty, the client believes it is already
    /// connected -- return our cached local metadata (or empty if not found).
    /// Otherwise, establish (or re-establish) a connection to the client.
    ///
    /// Returns `Err` if the handshake fails (bad client metadata, QP creation, etc.).
    #[cfg(feature = "rdma")]
    pub fn rdma_accept_handshake(
        &self,
        client_addr: &str,
        client_handshake_bytes: &[u8],
    ) -> Result<Vec<u8>, String> {
        let rdma = self
            .storage
            .rdma_transport()
            .ok_or_else(|| "RDMA transport not configured".to_string())?;

        if client_handshake_bytes.is_empty() {
            // Client thinks it's already connected -- return our cached meta if we have it
            return Ok(rdma
                .engine()
                .local_meta_for(client_addr)
                .map(|m| m.to_bytes())
                .unwrap_or_default());
        }

        let client_meta = pegaflow_transfer::HandshakeMetadata::from_bytes(client_handshake_bytes)
            .map_err(|e| format!("invalid client handshake metadata: {e}"))?;

        // Client sent handshake bytes → it has no connection. If we have a stale
        // one (e.g. client restarted), tear it down so get_or_prepare creates fresh QPs.
        rdma.engine().invalidate_connection(client_addr);

        let server_meta = match rdma
            .engine()
            .get_or_prepare(client_addr)
            .map_err(|e| format!("get_or_prepare failed: {e}"))?
        {
            pegaflow_transfer::ConnectionStatus::Prepared(m) => m,
            pegaflow_transfer::ConnectionStatus::Existing => {
                unreachable!("just invalidated connection for {client_addr}")
            }
            pegaflow_transfer::ConnectionStatus::Connecting => {
                return Err(format!("handshake to {client_addr} already in progress"));
            }
        };
        rdma.engine()
            .complete_handshake(client_addr, &server_meta, &client_meta)
            // Without the abort, the client stays in `connecting` forever and
            // every retry fails with "already in progress".
            .inspect_err(|_| rdma.engine().abort_handshake(client_addr, &server_meta))
            .map_err(|e| format!("complete_handshake failed: {e}"))?;
        info!("RDMA handshake accepted: client={client_addr}");
        Ok(server_meta.to_bytes())
    }

    /// Perform server-side RDMA handshake with connection reuse.
    #[cfg(not(feature = "rdma"))]
    pub fn rdma_accept_handshake(
        &self,
        _client_addr: &str,
        _client_handshake_bytes: &[u8],
    ) -> Result<Vec<u8>, String> {
        Err("this binary was built without RDMA support".to_string())
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
    #[test]
    fn rdma_handshake_reports_missing_feature() {
        let engine = PegaEngine::new_with_config(1 << 20, false, storage::StorageConfig::default())
            .expect("engine should start without RDMA");

        let err = engine
            .rdma_accept_handshake("127.0.0.1:50055", b"client-handshake")
            .expect_err("no-RDMA build should reject RDMA handshakes");

        assert_eq!(err, "this binary was built without RDMA support");
    }
}
