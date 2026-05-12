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

mod allocator;
mod backing;
mod block;
mod cache;
mod gpu_worker;
mod instance;
mod internode;
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
    BlockHash, BlockKey, BlockStatus, LayerBlock, LayerSave, RawBlock, ReserveLoadStatus,
    SealedBlock,
};
pub use instance::{GpuContext, InstanceContext, KVCacheRegistration};
pub use internode::{DEFAULT_METASERVER_QUEUE_DEPTH, MetaServerClient, MetaServerClientConfig};
pub use pegaflow_common::NumaNode;
use pegaflow_common::NumaTopology;
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use storage::{MemoryCacheCleanupStats, StorageConfig};
pub use sync_state::{LoadState, LoadStateError};
pub use trace::{set_trace_sample_rate, should_sample};

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, RwLock},
};

use log::{debug, info, warn};

use crate::backing::SSD_ALIGNMENT;
use crate::gpu_worker::{LayerLoadData, LoadBlock, LoadTask};
use crate::metrics::core_metrics;
use crate::storage::StorageEngine;

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
#[derive(Clone)]
pub struct PegaEngine {
    /// Active inference instances indexed by instance ID.
    instances: Arc<RwLock<HashMap<String, Arc<InstanceContext>>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
    /// GPU-NUMA topology for memory allocation decisions.
    topology: Arc<NumaTopology>,
}

struct ValidatedLoadPlan {
    total_blocks: usize,
    layers: Vec<ValidatedLoadLayer>,
}

struct ValidatedLoadLayer {
    name: String,
    registration: KVCacheRegistration,
    slot_id: usize,
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
    ) -> Self {
        let topology = Arc::new(NumaTopology::detect());
        topology.log_summary();

        let config = storage_config;
        let numa_nodes: Vec<NumaNode> = if config.enable_numa_affinity && topology.is_multi_numa() {
            info!(
                "Auto-enabling NUMA-aware memory allocation for {} nodes",
                topology.num_nodes()
            );
            topology.numa_nodes().to_vec()
        } else {
            vec![]
        };

        let storage = StorageEngine::new_with_config(pool_size, use_hugepages, config, &numa_nodes);

        PegaEngine {
            instances: Arc::new(RwLock::new(HashMap::new())),
            storage,
            topology,
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
        instance.register_new_gpu(device_id, tp_rank, pp_rank, numa_node, kv_caches)?;

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

    /// Reserve prefix hit blocks for a later load.
    ///
    /// Returns:
    /// - `Ready { hit, lease_id, .. }`: blocks are ready and covered by a load lease
    /// - `Loading { hit }`: backing prefetch is in progress; caller should retry
    #[cfg_attr(feature = "tracing", fastrace::trace(name = "reserve_load"))]
    pub async fn reserve_load(
        &self,
        instance_id: &str,
        request_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<ReserveLoadStatus, EngineError> {
        if request_id.is_empty() {
            warn!("reserve_load: empty request_id, returning 0 hits");
            return Ok(ReserveLoadStatus::Ready {
                hit: 0,
                missing: block_hashes.len(),
                lease_id: String::new(),
            });
        }

        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let world_size = instance.world_size();
        let metrics = core_metrics();

        let status = self
            .storage
            .reserve_load(instance_id, request_id, namespace, block_hashes, world_size)
            .await;

        match &status {
            ReserveLoadStatus::Ready { hit, missing, .. } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
                if *missing > 0 {
                    metrics.cache_block_misses.add(*missing as u64, &[]);
                }
            }
            ReserveLoadStatus::Loading { hit } => {
                metrics.cache_block_hits.add(*hit as u64, &[]);
            }
        }

        Ok(status)
    }

    /// Release a load lease that will not be consumed.
    pub fn release_load_lease(&self, load_lease_id: &str) -> bool {
        let released = self.storage.release_load_lease(load_lease_id);
        debug!("release_load_lease: lease={load_lease_id} released={released}");
        released
    }

    /// Evict all resident in-memory cache blocks while preserving backing-store data.
    pub fn cleanup_memory_cache(&self) -> MemoryCacheCleanupStats {
        self.storage.cleanup_memory_cache()
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
        leases: &[(&str, Vec<i32>)],
    ) -> Result<(), EngineError> {
        let load_state = LoadState::attach(load_state_shm)?;

        let result = self.batch_load_kv_blocks_multi_layer_inner(
            instance_id,
            tp_rank,
            device_id,
            load_state_shm,
            layer_names,
            leases,
        );

        if let Err(ref e) = result {
            log::error!("batch_load_kv_blocks_multi_layer pre-submit error: {e:?}");
            load_state.set_error();
        }

        result
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
        load_state_shm: &str,
        layer_names: &[&str],
        leases: &[(&str, Vec<i32>)],
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;

        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        let plan = self.validate_load_plan(
            &instance,
            &gpu,
            instance_id,
            tp_rank,
            device_id,
            layer_names,
            leases,
        )?;

        // Consume all load leases after validation succeeds.
        trace_scope!("load.cache_lookup", _s);
        let (block_ids, block_cache) =
            self.consume_load_leases(instance_id, leases, plan.total_blocks)?;
        trace_drop!(_s);

        // Build load tasks for each layer.
        trace_scope!("load.build_tasks");
        let layers = Self::build_layer_loads(plan.layers, &block_ids, &block_cache);

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

    #[allow(
        clippy::too_many_arguments,
        reason = "load validation needs the complete target context"
    )]
    fn validate_load_plan(
        &self,
        instance: &InstanceContext,
        gpu: &GpuContext,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_names: &[&str],
        leases: &[(&str, Vec<i32>)],
    ) -> Result<ValidatedLoadPlan, EngineError> {
        let total_blocks = self.validate_load_leases(instance_id, leases)?;
        let layers = Self::validate_load_layers(
            instance,
            gpu,
            instance_id,
            tp_rank,
            device_id,
            layer_names,
        )?;

        Ok(ValidatedLoadPlan {
            total_blocks,
            layers,
        })
    }

    fn validate_load_leases(
        &self,
        instance_id: &str,
        leases: &[(&str, Vec<i32>)],
    ) -> Result<usize, EngineError> {
        let mut total_blocks = 0usize;

        for (load_lease_id, block_ids) in leases {
            let lease_blocks = self
                .storage
                .load_lease_len(instance_id, load_lease_id)
                .map_err(EngineError::Storage)?;

            if lease_blocks != block_ids.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "lease {load_lease_id} contains {lease_blocks} blocks but request provided {} block_ids",
                    block_ids.len()
                )));
            }

            total_blocks += block_ids.len();
        }

        Ok(total_blocks)
    }

    fn validate_load_layers(
        instance: &InstanceContext,
        gpu: &GpuContext,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_names: &[&str],
    ) -> Result<Vec<ValidatedLoadLayer>, EngineError> {
        layer_names
            .iter()
            .map(|layer_name| {
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

                Ok(ValidatedLoadLayer {
                    name: (*layer_name).to_string(),
                    registration,
                    slot_id,
                })
            })
            .collect()
    }

    fn consume_load_leases(
        &self,
        instance_id: &str,
        leases: &[(&str, Vec<i32>)],
        total_blocks: usize,
    ) -> Result<(Vec<i32>, Vec<Arc<SealedBlock>>), EngineError> {
        let mut block_ids = Vec::with_capacity(total_blocks);
        let mut blocks = Vec::with_capacity(total_blocks);

        for (load_lease_id, lease_block_ids) in leases {
            let lease_blocks = self
                .storage
                .consume_load_lease(instance_id, load_lease_id)
                .map_err(EngineError::Storage)?;
            debug_assert_eq!(lease_blocks.len(), lease_block_ids.len());

            block_ids.extend_from_slice(lease_block_ids);
            blocks.extend(lease_blocks);
        }

        Ok((block_ids, blocks))
    }

    fn build_layer_loads(
        layers: Vec<ValidatedLoadLayer>,
        block_ids: &[i32],
        block_cache: &[Arc<SealedBlock>],
    ) -> Vec<LayerLoadData> {
        layers
            .into_iter()
            .filter_map(|layer| {
                let blocks: Vec<LoadBlock> = block_ids
                    .iter()
                    .zip(block_cache.iter())
                    .filter_map(|(block_id, block_entry)| {
                        let block_idx = usize::try_from(*block_id).ok()?;
                        let block = block_entry.get_slot(layer.slot_id)?.clone();
                        Some(LoadBlock { block_idx, block })
                    })
                    .collect();

                (!blocks.is_empty()).then_some(LayerLoadData {
                    layer_name: layer.name,
                    registration: layer.registration,
                    blocks,
                })
            })
            .collect()
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
    pub fn has_rdma_transport(&self) -> bool {
        self.storage.rdma_transport().is_some()
    }

    /// Perform server-side RDMA handshake with connection reuse.
    ///
    /// If `client_handshake_bytes` is empty, the client believes it is already
    /// connected -- return our cached local metadata (or empty if not found).
    /// Otherwise, establish (or re-establish) a connection to the client.
    ///
    /// Returns `Err` if the handshake fails (bad client metadata, QP creation, etc.).
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
            .map_err(|e| format!("complete_handshake failed: {e}"))?;
        info!("RDMA handshake accepted: client={client_addr}");
        Ok(server_meta.to_bytes())
    }
}
