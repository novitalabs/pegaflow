//! PegaFlow Core Engine
//!
//! A GPU-aware KV cache offloading engine with support for:
//! - Multi-tenant instance isolation
//! - Tensor parallelism (TP) across multiple GPUs
//! - Split-storage layout for efficient K/V batch transfers
//! - SSD caching tier

pub mod allocator;
pub mod block;
mod cache;
pub mod gpu_worker;
pub mod instance;
pub mod logging;
mod metrics;
pub mod numa;
pub mod pinned_mem;
pub mod pinned_pool;
mod seal_offload;
pub mod ssd_cache;
mod storage;
pub mod sync_state;
mod transfer;
mod uring;

pub use block::{
    BlockHash, BlockInsertError, BlockKey, BlockStatus, LayerBlock, PrefetchStatus, SealedBlock,
};
pub use instance::{GpuContext, InstanceContext, KVCacheRegistration};
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use ssd_cache::{
    DEFAULT_SSD_PREFETCH_INFLIGHT, DEFAULT_SSD_PREFETCH_QUEUE_DEPTH, DEFAULT_SSD_WRITE_INFLIGHT,
    DEFAULT_SSD_WRITE_QUEUE_DEPTH, SsdCacheConfig,
};
pub use storage::{SealNotification, StorageConfig};
pub use sync_state::{LoadState, LoadStateError};

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
    num::NonZeroU64,
    sync::{Arc, RwLock},
};

use log::{debug, info};

use crate::gpu_worker::{LayerLoadData, LoadBlock, LoadTask, SaveBlock};
use crate::metrics::core_metrics;
use crate::storage::{SSD_ALIGNMENT, StorageEngine};

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
///
/// The engine is thread-safe and can be shared across async tasks.
pub struct PegaEngine {
    /// Active inference instances indexed by instance ID.
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    /// Storage engine for pinned memory, block cache, and SSD tier.
    storage: Arc<StorageEngine>,
}

#[allow(clippy::new_without_default)]
impl PegaEngine {
    /// Create a new engine with default 30GB pinned memory pool.
    pub fn new() -> Self {
        let (engine, _rx) = Self::new_with_config(
            DEFAULT_PINNED_POOL_BYTES,
            false,
            storage::StorageConfig::default(),
        );
        engine
    }

    /// Create an engine with a custom pinned memory pool size.
    ///
    /// Set `use_hugepages` to true for 2MB huge pages (requires system configuration).
    pub fn new_with_pool_size(pool_size: usize, use_hugepages: bool) -> Self {
        let (engine, _rx) =
            Self::new_with_config(pool_size, use_hugepages, storage::StorageConfig::default());
        engine
    }

    /// Create an engine with full custom configuration.
    ///
    /// Returns the engine and a receiver for seal notifications (used for SSD offload).
    pub fn new_with_config(
        pool_size: usize,
        use_hugepages: bool,
        storage_config: impl Into<storage::StorageConfig>,
    ) -> (Self, tokio::sync::mpsc::UnboundedReceiver<SealNotification>) {
        let (storage, seal_notify_rx) =
            StorageEngine::new_with_config(pool_size, use_hugepages, storage_config);

        (
            PegaEngine {
                instances: RwLock::new(HashMap::new()),
                storage,
            },
            seal_notify_rx,
        )
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

    /// Register a KV cache layer with its memory layout.
    ///
    /// This validates the layout parameters, initializes the instance/GPU context
    /// if needed, and stores the registration for subsequent load/save operations.
    #[allow(clippy::too_many_arguments)]
    pub fn register_context_layer(
        &self,
        instance_id: &str,
        namespace: &str,
        device_id: i32,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
        tp_rank: usize,
        tp_size: usize,
        world_size: usize,
        num_layers: usize,
    ) -> Result<(), EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(
                "device_id must be >= 0".to_string(),
            ));
        }

        // Construct and validate registration
        let registration = KVCacheRegistration::new(
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        )
        .map_err(|e| EngineError::InvalidArgument(format!("layer {layer_name}: {e}")))?;

        // SSD alignment check
        if self.storage.is_ssd_enabled()
            && let Some(msg) = registration.check_ssd_alignment(SSD_ALIGNMENT)
        {
            return Err(EngineError::InvalidArgument(format!(
                "layer {layer_name}: {msg}"
            )));
        }

        // Get or create instance, then register new layer on GPU
        let instance =
            self.get_or_create_instance(instance_id, namespace, num_layers, tp_size, world_size)?;

        instance.register_new_gpu_layer(device_id, &layer_name, registration)?;

        info!(
            "Registered context: instance={instance_id}, namespace={namespace}, \
             device={device_id}, layer={layer_name}, num_blocks={num_blocks}, \
             bytes_per_block={bytes_per_block}, segments={segments}, tp_rank={tp_rank}/{tp_size}"
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

    /// Batch save KV blocks from multiple layers.
    ///
    /// More efficient than calling `save_kv_blocks_from_ipc` in a loop as it
    /// reduces Python-Rust boundary crossings.
    pub async fn batch_save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        saves: Vec<(String, Vec<i32>, Vec<Vec<u8>>)>,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace().to_string();

        for (layer_name, block_ids, block_hashes) in saves {
            self.save_kv_blocks_from_ipc_inner(
                instance_id,
                tp_rank,
                device_id,
                &namespace,
                &layer_name,
                block_ids,
                block_hashes,
            )
            .await?;
        }
        Ok(())
    }

    /// Save KV blocks for a single layer.
    pub async fn save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        layer_name: &str,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace().to_string();

        self.save_kv_blocks_from_ipc_inner(
            instance_id,
            tp_rank,
            device_id,
            &namespace,
            layer_name,
            block_ids,
            block_hashes,
        )
        .await
    }

    /// Internal implementation for saving KV blocks.
    #[allow(clippy::too_many_arguments)]
    async fn save_kv_blocks_from_ipc_inner(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        namespace: &str,
        layer_name: &str,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), EngineError> {
        if block_ids.len() != block_hashes.len() {
            return Err(EngineError::InvalidArgument(format!(
                "block_ids length {} does not match block_hashes {}",
                block_ids.len(),
                block_hashes.len()
            )));
        }

        let metrics = core_metrics();
        let timer = std::time::Instant::now();

        let instance = self.get_instance(instance_id)?;
        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        let layer_id = instance
            .get_layer_id(layer_name)
            .ok_or_else(|| EngineError::InvalidArgument(format!("layer {layer_name} unknown")))?;

        let registration = gpu.get_registration(layer_name).ok_or_else(|| {
            EngineError::InvalidArgument(format!("layer {layer_name} not registered on device"))
        })?;

        let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
        let total_slots = instance.total_slots();

        // Filter valid blocks
        let valid_blocks: Vec<(usize, Vec<u8>)> = block_ids
            .iter()
            .zip(block_hashes.iter())
            .filter(|(id, _)| **id >= 0)
            .filter_map(|(id, hash)| {
                let idx = *id as usize;
                if idx < registration.num_blocks {
                    Some((idx, hash.clone()))
                } else {
                    None
                }
            })
            .collect();

        if valid_blocks.is_empty() {
            return Ok(());
        }

        // Filter out blocks that already exist
        let valid_hashes: Vec<Vec<u8>> = valid_blocks.iter().map(|(_, h)| h.clone()).collect();
        let indices_to_save = self
            .storage
            .filter_blocks_to_save(namespace, &valid_hashes, slot_id);

        if indices_to_save.is_empty() {
            return Ok(());
        }

        let blocks_to_save: Vec<(usize, Vec<u8>)> = indices_to_save
            .into_iter()
            .map(|i| valid_blocks[i].clone())
            .collect();

        debug!(
            "Saving {} blocks for layer {layer_name} on instance {instance_id} rank {tp_rank}",
            blocks_to_save.len()
        );

        let block_size = registration.block_size_bytes;
        let num_blocks = blocks_to_save.len();

        // Handle split K/V layout
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            self.save_split_blocks(
                namespace,
                &registration,
                &blocks_to_save,
                block_size,
                slot_id,
                total_slots,
                gpu,
            )
            .await?;
        } else {
            // Contiguous layout
            self.save_contiguous_blocks(
                namespace,
                &registration,
                &blocks_to_save,
                block_size,
                slot_id,
                total_slots,
                gpu,
            )
            .await?;
        }

        let elapsed = timer.elapsed().as_secs_f64();
        let total_bytes = (block_size as u64)
            .checked_mul(num_blocks as u64)
            .unwrap_or(0);

        if num_blocks > 0 {
            metrics.save_bytes.add(total_bytes, &[]);
            metrics.save_duration_seconds.record(elapsed, &[]);
        }
        Ok(())
    }

    /// Save blocks with split K/V storage layout.
    #[allow(clippy::too_many_arguments)]
    async fn save_split_blocks(
        &self,
        namespace: &str,
        registration: &KVCacheRegistration,
        blocks_to_save: &[(usize, Vec<u8>)],
        block_size: usize,
        slot_id: usize,
        total_slots: usize,
        gpu: Arc<GpuContext>,
    ) -> Result<(), EngineError> {
        let segment_size = registration.bytes_per_block;
        let num_blocks = blocks_to_save.len();

        // Allocate separate regions for K and V
        let k_total_size = (segment_size as u64)
            .checked_mul(num_blocks as u64)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| {
                EngineError::Storage("allocation size overflow for K segments".to_string())
            })?;

        let mut k_allocation = self.storage.allocate(k_total_size).ok_or_else(|| {
            EngineError::Storage(
                "pinned pool exhausted while allocating K segment buffer".to_string(),
            )
        })?;

        let mut v_allocation = self.storage.allocate(k_total_size).ok_or_else(|| {
            EngineError::Storage(
                "pinned pool exhausted while allocating V segment buffer".to_string(),
            )
        })?;

        // Get base pointers
        let (k_base, v_base) = {
            let k_ptr = Arc::get_mut(&mut k_allocation)
                .expect("k_allocation must be uniquely owned")
                .as_mut_ptr();
            let v_ptr = Arc::get_mut(&mut v_allocation)
                .expect("v_allocation must be uniquely owned")
                .as_mut_ptr();
            (k_ptr as usize, v_ptr as usize)
        };

        // Build save blocks
        let save_blocks: Vec<SaveBlock> = blocks_to_save
            .iter()
            .enumerate()
            .map(|(i, (block_idx, _))| SaveBlock {
                block_idx: *block_idx,
                k_dst_ptr: (k_base + i * segment_size) as *mut u8,
                v_dst_ptr: Some((v_base + i * segment_size) as *mut u8),
            })
            .collect();

        // Execute GPU->CPU copy
        gpu.worker_pool()
            .save(registration.clone(), save_blocks)
            .await?;

        // Create sealed blocks and admit to cache
        let mut sealed_blocks = Vec::new();
        for (i, (_, block_hash)) in blocks_to_save.iter().enumerate() {
            let k_ptr = (k_base + i * segment_size) as *mut u8;
            let v_ptr = (v_base + i * segment_size) as *mut u8;

            let block = Arc::new(LayerBlock::new_split(
                k_ptr,
                v_ptr,
                block_size,
                Arc::clone(&k_allocation),
                Arc::clone(&v_allocation),
            ));

            if let Some(sealed) = self
                .storage
                .insert_slot(namespace, block_hash.clone(), slot_id, block, total_slots)
                .map_err(|e| EngineError::Storage(e.to_string()))?
            {
                sealed_blocks.push(sealed);
            }
        }

        self.storage.send_ssd_batch(&sealed_blocks);

        // Admit to cache (prefix-aware: stop on first rejection)
        for (key, block) in sealed_blocks {
            if !self.storage.cache_admit(key, block) {
                break;
            }
        }

        Ok(())
    }

    /// Save blocks with contiguous storage layout.
    #[allow(clippy::too_many_arguments)]
    async fn save_contiguous_blocks(
        &self,
        namespace: &str,
        registration: &KVCacheRegistration,
        blocks_to_save: &[(usize, Vec<u8>)],
        block_size: usize,
        slot_id: usize,
        total_slots: usize,
        gpu: Arc<GpuContext>,
    ) -> Result<(), EngineError> {
        let num_blocks = blocks_to_save.len();

        let total_size = (block_size as u64)
            .checked_mul(num_blocks as u64)
            .and_then(NonZeroU64::new)
            .ok_or_else(|| EngineError::Storage("allocation size overflow".to_string()))?;

        let mut allocation = self.storage.allocate(total_size).ok_or_else(|| {
            EngineError::Storage(
                "pinned pool exhausted while allocating contiguous block buffer".to_string(),
            )
        })?;

        let base_addr = Arc::get_mut(&mut allocation)
            .ok_or_else(|| EngineError::Storage("allocation shared unexpectedly".to_string()))?
            .as_mut_ptr() as usize;

        let save_blocks: Vec<SaveBlock> = blocks_to_save
            .iter()
            .enumerate()
            .map(|(i, (block_idx, _))| SaveBlock {
                block_idx: *block_idx,
                k_dst_ptr: (base_addr + i * block_size) as *mut u8,
                v_dst_ptr: None,
            })
            .collect();

        gpu.worker_pool()
            .save(registration.clone(), save_blocks)
            .await?;

        let mut sealed_blocks = Vec::new();
        for (i, (_, block_hash)) in blocks_to_save.iter().enumerate() {
            let cpu_ptr = (base_addr + i * block_size) as *mut u8;

            let block = Arc::new(LayerBlock::new_contiguous(
                cpu_ptr,
                block_size,
                Arc::clone(&allocation),
            ));

            if let Some(sealed) = self
                .storage
                .insert_slot(namespace, block_hash.clone(), slot_id, block, total_slots)
                .map_err(|e| EngineError::Storage(e.to_string()))?
            {
                sealed_blocks.push(sealed);
            }
        }

        self.storage.send_ssd_batch(&sealed_blocks);

        for (key, block) in sealed_blocks {
            if !self.storage.cache_admit(key, block) {
                break;
            }
        }

        Ok(())
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Returns:
    /// - `Done { hit, missing: 0 }`: all blocks in memory cache
    /// - `Loading { hit, loading }`: some blocks being fetched from SSD
    /// - `Done { hit, missing }`: some blocks don't exist
    pub fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<PrefetchStatus, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace();
        let world_size = instance.world_size();
        let metrics = core_metrics();

        let status = self.storage.check_prefix_and_prefetch(
            instance_id,
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
            PrefetchStatus::Loading { hit, .. } => {
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

        // Lookup all blocks (consumes pinned blocks)
        let block_cache = self
            .storage
            .cache_lookup_many(instance_id, namespace, block_hashes)
            .map_err(EngineError::Storage)?;

        // Build load tasks for each layer
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

    /// Remove stale inflight blocks (background GC).
    ///
    /// Should be called periodically (e.g., every 5 minutes).
    pub fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        self.storage.gc_stale_inflight(max_age)
    }
}
