//! Instance and GPU context management for PegaFlow.
//!
//! This module provides the hierarchical context structure for managing
//! multi-tenant inference instances and their associated GPU resources.

use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::CudaContext;
use log::info;

use crate::{EngineError, TransferMode, gpu_worker::GpuWorkerPool};
use pegaflow_common::NumaNode;

/// Layer metadata protected by a single mutex.
///
/// This struct consolidates layer name mappings and GPU contexts to avoid
/// potential deadlocks from acquiring multiple locks in inconsistent order.
struct LayerMetadata {
    /// Mapping from layer names to numeric IDs (0..num_layers).
    name_to_id: HashMap<String, usize>,

    /// Inverse mapping from IDs to layer names.
    names: Vec<Option<String>>,

    /// GPU contexts indexed by CUDA device ID.
    gpu_contexts: HashMap<i32, Arc<GpuContext>>,
}

/// Registration information for a KV cache layer.
///
/// This struct captures the memory layout of a layer's KV cache,
/// supporting both contiguous and split (K/V-separated) storage formats.
#[derive(Debug, Clone)]
pub struct KVCacheRegistration {
    /// GPU memory base pointer for this layer's KV cache.
    pub data_ptr: u64,

    /// Total size of the registered GPU memory region in bytes.
    pub size_bytes: usize,

    /// Number of blocks in this layer's cache.
    pub num_blocks: usize,

    /// Actual GPU-side segment size in bytes (one of K or V).
    /// Used for CUDA memcpy sizes and GPU offset calculations.
    pub bytes_per_block: usize,

    /// Actual GPU-side total block size (`bytes_per_block * segments`).
    pub block_size_bytes: usize,

    /// Stride in bytes between K and V segments for split storage layouts.
    /// Zero when using contiguous single-segment storage.
    pub kv_stride_bytes: usize,

    /// Number of segments per block (1 for contiguous, 2 for K/V split).
    pub segments: usize,

    /// CPU/SSD-side segment size, rounded up to SSD alignment.
    /// Controls pinned memory allocation stride and SSD iovec lengths.
    /// Equals `bytes_per_block` when no SSD padding is needed.
    pub padded_bytes_per_block: usize,

    /// CPU/SSD-side total block size (`padded_bytes_per_block * segments`).
    /// Becomes `RawBlock.total_size` → `SlotMeta.total_size()` for SSD I/O.
    pub padded_block_size_bytes: usize,
}

impl KVCacheRegistration {
    /// Construct and validate a new registration.
    ///
    /// # Errors
    /// Returns a simple error message string if validation fails.
    pub(crate) fn new(
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) -> Result<Self, String> {
        // Basic non-zero checks
        if data_ptr == 0 {
            return Err("data_ptr must not be null".into());
        }
        if size_bytes == 0 {
            return Err("size_bytes must be > 0".into());
        }
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            return Err("bytes_per_block, num_blocks, and segments must be non-zero".into());
        }
        if segments > 1 && kv_stride_bytes == 0 {
            return Err("kv_stride_bytes must be > 0 when segments > 1".into());
        }

        // Compute block_size_bytes (may overflow)
        let block_size_bytes = bytes_per_block
            .checked_mul(segments)
            .ok_or_else(|| "block_size_bytes overflow".to_string())?;

        // Validate memory layout doesn't overflow the buffer
        let max_block_offset = (num_blocks - 1)
            .checked_mul(bytes_per_block)
            .and_then(|o| o.checked_add((segments - 1).checked_mul(kv_stride_bytes)?))
            .ok_or_else(|| "memory layout overflow".to_string())?;

        let end = max_block_offset
            .checked_add(bytes_per_block)
            .ok_or_else(|| "memory layout end overflow".to_string())?;

        if end > size_bytes {
            return Err(format!(
                "registered memory too small: need {} bytes, got {}",
                end, size_bytes
            ));
        }

        Ok(Self {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            block_size_bytes,
            kv_stride_bytes,
            segments,
            padded_bytes_per_block: bytes_per_block,
            padded_block_size_bytes: block_size_bytes,
        })
    }

    /// Apply SSD alignment padding to segment sizes.
    ///
    /// Each segment (`bytes_per_block`) is rounded up to the next multiple of
    /// `alignment` so that every iovec in split writev is independently aligned.
    pub(crate) fn with_ssd_padding(mut self, alignment: usize) -> Self {
        let padded = self.bytes_per_block.next_multiple_of(alignment);
        self.padded_bytes_per_block = padded;
        self.padded_block_size_bytes = padded * self.segments;
        self
    }
}

/// Per-GPU execution context.
///
/// Each `GpuContext` manages:
/// - CUDA context lifetime for a specific device
/// - KV cache registrations for all layers on this GPU
/// - Asynchronous worker pool for load/save operations
/// - NUMA affinity for memory allocation optimization
pub struct GpuContext {
    /// CUDA device ID for diagnostics and duplicate registration checks.
    device_id: i32,

    /// Effective TP rank represented by this GPU context.
    tp_rank: usize,

    /// Pipeline-parallel rank represented by this GPU context.
    pp_rank: usize,

    /// Preferred NUMA node for this GPU (for memory allocation).
    preferred_numa: NumaNode,

    /// KV cache layout registrations by layer name.
    kv_caches: HashMap<String, KVCacheRegistration>,

    /// CUDA context handle (kept alive for the lifetime of this context).
    _cuda_ctx: Arc<CudaContext>,

    /// Worker thread pool for asynchronous GPU operations.
    worker_pool: GpuWorkerPool,
}

impl GpuContext {
    /// Create a new GPU context for the specified device.
    ///
    /// The `numa_node` should be obtained from `NumaTopology::numa_for_gpu()`.
    /// Worker threads will be pinned to the specified NUMA node for optimal
    /// memory locality during transfers.
    ///
    /// # Errors
    /// Returns `EngineError::CudaInit` if CUDA context creation or worker
    /// pool initialization fails.
    fn new(
        cuda_ctx: Arc<CudaContext>,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
        kv_caches: HashMap<String, KVCacheRegistration>,
    ) -> Result<Self, EngineError> {
        let worker_pool = GpuWorkerPool::spawn(device_id, numa_node, transfer_mode)?;

        Ok(Self {
            device_id,
            tp_rank,
            pp_rank,
            preferred_numa: numa_node,
            kv_caches,
            _cuda_ctx: cuda_ctx,
            worker_pool,
        })
    }

    /// Get the preferred NUMA node for this GPU.
    pub(crate) fn preferred_numa(&self) -> NumaNode {
        self.preferred_numa
    }

    /// CUDA device ID represented by this shard.
    pub(crate) fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Effective TP rank represented by this shard.
    pub(crate) fn tp_rank(&self) -> usize {
        self.tp_rank
    }

    /// Pipeline-parallel rank represented by this shard.
    pub(crate) fn pp_rank(&self) -> usize {
        self.pp_rank
    }

    /// Retrieve a layer's registration information.
    pub(crate) fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        self.kv_caches.get(layer_name).cloned()
    }

    /// Access the worker pool for submitting GPU operations.
    pub(crate) fn worker_pool(&self) -> &GpuWorkerPool {
        &self.worker_pool
    }
}

pub(crate) struct GpuRegistration {
    pub(crate) device_id: i32,
    pub(crate) tp_rank: usize,
    pub(crate) pp_rank: usize,
    pub(crate) numa_node: NumaNode,
    pub(crate) transfer_mode: TransferMode,
    pub(crate) kv_caches: HashMap<String, KVCacheRegistration>,
    pub(crate) layer_ids_by_name: HashMap<String, usize>,
}

/// Instance context for a model inference process.
///
/// An `InstanceContext` represents a single inference instance (e.g., one
/// `vllm serve` process) and manages all its GPU contexts and layer metadata.
/// It supports tensor parallelism via the `tp_size` parameter.
pub struct InstanceContext {
    /// Unique instance identifier.
    id: String,

    /// Namespace for model isolation (e.g., model name or tenant ID).
    namespace: String,

    /// Number of connector-declared logical layers for this instance.
    num_layers: usize,

    /// Tensor parallelism degree (number of GPUs per instance).
    tp_size: usize,

    /// Total world size (TP × PP × DP).
    world_size: usize,

    /// Layer metadata and GPU contexts protected by a single mutex.
    ///
    /// This consolidates previously separate mutexes to prevent deadlocks
    /// from inconsistent lock acquisition ordering.
    metadata: Mutex<LayerMetadata>,
}

impl InstanceContext {
    /// Create a new instance context.
    ///
    /// # Errors
    /// Returns an error string if topology parameters are invalid.
    pub(crate) fn new(
        id: String,
        namespace: String,
        num_layers: usize,
        tp_size: usize,
        world_size: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 || tp_size == 0 || world_size == 0 {
            return Err("num_layers, tp_size, and world_size must be > 0".into());
        }

        Ok(Self {
            id,
            namespace,
            num_layers,
            tp_size,
            world_size,
            metadata: Mutex::new(LayerMetadata {
                name_to_id: HashMap::new(),
                names: vec![None; num_layers],
                gpu_contexts: HashMap::new(),
            }),
        })
    }

    /// Register the connector-declared numeric ID for a layer name.
    #[cfg(test)]
    fn register_layer_id(&self, layer_name: &str, layer_id: usize) -> Result<(), EngineError> {
        if layer_id >= self.num_layers {
            return Err(EngineError::InvalidArgument(format!(
                "layer {layer_name} id {layer_id} out of range (num_layers {})",
                self.num_layers
            )));
        }

        let mut metadata = self.metadata.lock();

        if let Some(&id) = metadata.name_to_id.get(layer_name)
            && id != layer_id
        {
            return Err(EngineError::InvalidArgument(format!(
                "layer {layer_name} already registered with id {id}, got {layer_id}"
            )));
        }

        if let Some(existing_name) = &metadata.names[layer_id]
            && existing_name != layer_name
        {
            return Err(EngineError::InvalidArgument(format!(
                "layer id {layer_id} already registered for {existing_name}, got {layer_name}"
            )));
        }

        metadata.names[layer_id] = Some(layer_name.to_string());
        metadata.name_to_id.insert(layer_name.to_string(), layer_id);
        Ok(())
    }

    fn build_layer_id_assignments(
        &self,
        kv_caches: &HashMap<String, KVCacheRegistration>,
        layer_ids_by_name: &HashMap<String, usize>,
    ) -> Result<Vec<(String, usize)>, EngineError> {
        let mut assignments = Vec::with_capacity(kv_caches.len());
        let mut names_by_id = HashMap::with_capacity(kv_caches.len());

        for layer_name in kv_caches.keys() {
            let layer_id = *layer_ids_by_name.get(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("missing layer id for {layer_name}"))
            })?;

            if layer_id >= self.num_layers {
                return Err(EngineError::InvalidArgument(format!(
                    "layer {layer_name} id {layer_id} out of range (num_layers {})",
                    self.num_layers
                )));
            }

            if let Some(existing_name) = names_by_id.insert(layer_id, layer_name.as_str()) {
                return Err(EngineError::InvalidArgument(format!(
                    "layer id {layer_id} appears more than once in registration batch: {existing_name}, {layer_name}"
                )));
            }

            assignments.push((layer_name.clone(), layer_id));
        }

        Ok(assignments)
    }

    fn validate_layer_id_assignments(
        &self,
        metadata: &LayerMetadata,
        assignments: &[(String, usize)],
    ) -> Result<(), EngineError> {
        for (layer_name, layer_id) in assignments {
            if let Some(&id) = metadata.name_to_id.get(layer_name)
                && id != *layer_id
            {
                return Err(EngineError::InvalidArgument(format!(
                    "layer {layer_name} already registered with id {id}, got {layer_id}"
                )));
            }

            if let Some(existing_name) = &metadata.names[*layer_id]
                && existing_name != layer_name
            {
                return Err(EngineError::InvalidArgument(format!(
                    "layer id {layer_id} already registered for {existing_name}, got {layer_name}"
                )));
            }
        }

        Ok(())
    }

    fn commit_layer_id_assignments(metadata: &mut LayerMetadata, assignments: &[(String, usize)]) {
        for (layer_name, layer_id) in assignments {
            metadata.names[*layer_id] = Some(layer_name.clone());
            metadata.name_to_id.insert(layer_name.clone(), *layer_id);
        }
    }

    fn ensure_device_unregistered(
        metadata: &LayerMetadata,
        device_id: i32,
    ) -> Result<(), EngineError> {
        if metadata.gpu_contexts.contains_key(&device_id) {
            return Err(EngineError::InvalidArgument(format!(
                "GPU context for device {device_id} already exists"
            )));
        }
        Ok(())
    }

    /// Look up the numeric ID for a layer name.
    ///
    /// Returns `None` if the layer has not been registered.
    pub(crate) fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        let metadata = self.metadata.lock();
        metadata.name_to_id.get(layer_name).copied()
    }

    /// Calculate the total number of storage slots.
    ///
    /// Slots are organized as a flattened 2D array: `[layer][tp_rank]`.
    pub(crate) fn total_slots(&self) -> usize {
        self.num_layers * self.tp_size
    }

    /// Compute the slot index for a specific layer and TP rank.
    ///
    /// The slot index is used to locate blocks in the storage engine.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` if `layer_id` or `tp_rank`
    /// are out of bounds.
    pub(crate) fn get_slot_index(
        &self,
        layer_id: usize,
        tp_rank: usize,
    ) -> Result<usize, EngineError> {
        if layer_id >= self.num_layers {
            return Err(EngineError::InvalidArgument(format!(
                "layer_id {} out of range ({} layers)",
                layer_id, self.num_layers
            )));
        }
        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} out of range (tp_size {})",
                tp_rank, self.tp_size
            )));
        }
        Ok(layer_id * self.tp_size + tp_rank)
    }

    /// Build a GPU context for the specified device.
    ///
    /// This method lazily initializes CUDA contexts as devices are first accessed.
    /// The `numa_node` should be obtained from `NumaTopology::numa_for_gpu()`.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` for negative device IDs,
    /// or `EngineError::CudaInit` if CUDA context creation fails.
    fn build_gpu_context(
        &self,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
        numa_node: NumaNode,
        transfer_mode: TransferMode,
        kv_caches: HashMap<String, KVCacheRegistration>,
    ) -> Result<Arc<GpuContext>, EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {} must be >= 0",
                device_id
            )));
        }

        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;

        Ok(Arc::new(GpuContext::new(
            cuda_ctx,
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        )?))
    }

    /// Get an existing GPU context without creating one.
    pub(crate) fn get_gpu(&self, device_id: i32) -> Option<Arc<GpuContext>> {
        let metadata = self.metadata.lock();
        metadata.gpu_contexts.get(&device_id).cloned()
    }

    /// Get a GPU context and verify it belongs to the requested save group.
    pub(crate) fn get_gpu_for_save_group(
        &self,
        device_id: i32,
        tp_rank: usize,
        pp_rank: usize,
    ) -> Result<Arc<GpuContext>, EngineError> {
        let gpu = self
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(self.id.clone(), device_id))?;

        if gpu.tp_rank() != tp_rank || gpu.pp_rank() != pp_rank {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {device_id} is registered for tp_rank {}, pp_rank {}, but save requested tp_rank {tp_rank}, pp_rank {pp_rank}",
                gpu.tp_rank(),
                gpu.pp_rank()
            )));
        }

        Ok(gpu)
    }

    /// Register a new GPU with all its KV cache layers.
    ///
    /// # Errors
    /// - `EngineError::InvalidArgument` if GPU already registered
    /// - `EngineError::CudaInit` if GPU context creation fails
    pub(crate) fn register_new_gpu(
        &self,
        registration: GpuRegistration,
    ) -> Result<(), EngineError> {
        let GpuRegistration {
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
            layer_ids_by_name,
        } = registration;

        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} out of range (tp_size {})",
                tp_rank, self.tp_size
            )));
        }
        if kv_caches.len() != layer_ids_by_name.len() {
            return Err(EngineError::InvalidArgument(format!(
                "layer id count {} does not match layer cache count {}",
                layer_ids_by_name.len(),
                kv_caches.len()
            )));
        }

        let layer_id_assignments =
            self.build_layer_id_assignments(&kv_caches, &layer_ids_by_name)?;

        {
            let metadata = self.metadata.lock();
            Self::ensure_device_unregistered(&metadata, device_id)?;
            self.validate_layer_id_assignments(&metadata, &layer_id_assignments)?;
        }

        let ctx = self.build_gpu_context(
            device_id,
            tp_rank,
            pp_rank,
            numa_node,
            transfer_mode,
            kv_caches,
        )?;

        {
            let mut metadata = self.metadata.lock();
            Self::ensure_device_unregistered(&metadata, device_id)?;
            self.validate_layer_id_assignments(&metadata, &layer_id_assignments)?;
            Self::commit_layer_id_assignments(&mut metadata, &layer_id_assignments);
            metadata.gpu_contexts.insert(device_id, ctx);
        }

        info!("Initialized GPU context: device_id={device_id}, numa_node={numa_node}");
        Ok(())
    }

    /// Verify every declared logical layer has a registered slot for every TP rank.
    pub(crate) fn ensure_all_slots_registered(&self) -> Result<(), EngineError> {
        let metadata = self.metadata.lock();
        let total_slots = self.total_slots();
        let mut owners: Vec<Option<String>> = vec![None; total_slots];

        for gpu in metadata.gpu_contexts.values() {
            if gpu.tp_rank() >= self.tp_size {
                return Err(EngineError::InvalidArgument(format!(
                    "registered gpu device {} has tp_rank {} out of range (tp_size {})",
                    gpu.device_id(),
                    gpu.tp_rank(),
                    self.tp_size
                )));
            }

            for layer_name in gpu.kv_caches.keys() {
                let layer_id = metadata.name_to_id.get(layer_name).ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "layer {layer_name} registered on device {} without a layer id",
                        gpu.device_id()
                    ))
                })?;
                let slot_id = layer_id * self.tp_size + gpu.tp_rank();
                if slot_id >= total_slots {
                    return Err(EngineError::InvalidArgument(format!(
                        "slot_id {slot_id} out of range (total_slots {total_slots})"
                    )));
                }
                let owner = format!(
                    "device={} pp_rank={} tp_rank={} layer={}",
                    gpu.device_id(),
                    gpu.pp_rank(),
                    gpu.tp_rank(),
                    layer_name
                );
                if let Some(existing_owner) = owners[slot_id].replace(owner.clone()) {
                    return Err(EngineError::InvalidArgument(format!(
                        "slot {slot_id} registered twice: {existing_owner}; {owner}"
                    )));
                }
            }
        }

        let registered_slots = owners.iter().filter(|owner| owner.is_some()).count();
        if registered_slots != total_slots {
            let first_missing = owners
                .iter()
                .position(Option::is_none)
                .expect("missing slot must exist when counts differ");
            return Err(EngineError::InvalidArgument(format!(
                "instance {} has incomplete KV registration: registered_slots={} expected_slots={} first_missing_slot={}",
                self.id, registered_slots, total_slots, first_missing
            )));
        }

        Ok(())
    }

    /// Access the instance namespace.
    pub(crate) fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Access the effective TP size registered with the engine.
    pub(crate) fn tp_size(&self) -> usize {
        self.tp_size
    }

    /// Total worker count registered for this instance.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// Return unique valid NUMA nodes for registered shards in one save group.
    pub(crate) fn registered_numa_nodes_for_save_group(
        &self,
        tp_rank: usize,
        pp_rank: usize,
    ) -> Vec<NumaNode> {
        let metadata = self.metadata.lock();
        let mut nodes: Vec<NumaNode> = metadata
            .gpu_contexts
            .values()
            .filter(|gpu| gpu.tp_rank() == tp_rank && gpu.pp_rank() == pp_rank)
            .map(|gpu| gpu.preferred_numa())
            .filter(|node| node.is_valid())
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        nodes
    }

    /// Validate that a save placement hint targets a registered shard NUMA node.
    pub(crate) fn validate_save_numa_hint(
        &self,
        tp_rank: usize,
        pp_rank: usize,
        numa_node: NumaNode,
    ) -> Result<NumaNode, EngineError> {
        if !numa_node.is_valid() {
            return Err(EngineError::InvalidArgument(format!(
                "save NUMA hint must be a valid NUMA node, got {numa_node}"
            )));
        }

        let candidates = self.registered_numa_nodes_for_save_group(tp_rank, pp_rank);
        if candidates.contains(&numa_node) {
            return Ok(numa_node);
        }

        Err(EngineError::InvalidArgument(format!(
            "save NUMA hint {numa_node} is not registered for tp_rank {tp_rank}, pp_rank {pp_rank}; candidates={candidates:?}"
        )))
    }

    /// Verify that the topology matches expected values.
    ///
    /// Returns `Ok(())` if matches, or an error message describing the mismatch.
    pub(crate) fn verify_topology(
        &self,
        num_layers: usize,
        tp_size: usize,
        world_size: usize,
    ) -> Result<(), String> {
        if self.num_layers != num_layers || self.tp_size != tp_size || self.world_size != world_size
        {
            return Err(format!(
                "exists with layers={}, tp={}, world={}; \
                 requested layers={}, tp={}, world={}",
                self.num_layers, self.tp_size, self.world_size, num_layers, tp_size, world_size
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
