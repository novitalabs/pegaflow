//! Instance and GPU context management for PegaFlow.
//!
//! This module provides the hierarchical context structure for managing
//! multi-tenant inference instances and their associated GPU resources.

use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cudarc::driver::CudaContext;
use log::info;

use crate::{EngineError, gpu_worker::GpuWorkerPool};
use pegaflow_common::NumaNode;

/// Layer metadata protected by a single mutex.
///
/// This struct consolidates layer name mappings and GPU contexts to avoid
/// potential deadlocks from acquiring multiple locks in inconsistent order.
struct LayerMetadata {
    /// Mapping from layer names to numeric IDs (0..num_layers).
    name_to_id: HashMap<String, usize>,

    /// Inverse mapping from IDs to layer names.
    names: Vec<String>,

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
    /// CUDA device ID for this registered shard.
    device_id: i32,

    /// Effective TP rank represented by this GPU context.
    tp_rank: usize,

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
        numa_node: NumaNode,
        kv_caches: HashMap<String, KVCacheRegistration>,
    ) -> Result<Self, EngineError> {
        let worker_pool = GpuWorkerPool::spawn(device_id, numa_node)?;

        Ok(Self {
            device_id,
            tp_rank,
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

    /// Retrieve a layer's registration information.
    pub(crate) fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        self.kv_caches.get(layer_name).cloned()
    }

    /// Snapshot all layer registrations for topology-owned planning.
    pub(crate) fn layer_registrations(&self) -> Vec<(String, KVCacheRegistration)> {
        self.kv_caches
            .iter()
            .map(|(name, registration)| (name.clone(), registration.clone()))
            .collect()
    }

    /// Access the worker pool for submitting GPU operations.
    pub(crate) fn worker_pool(&self) -> &GpuWorkerPool {
        &self.worker_pool
    }
}

/// Instance context for a model inference process.
///
/// An `InstanceContext` represents a single inference instance (e.g., one
/// `vllm serve` process) and manages all its GPU contexts and layer metadata.
/// It supports tensor parallelism via the `tp_size` parameter.
pub struct InstanceContext {
    /// Unique instance identifier.
    _id: String,

    /// Namespace for model isolation (e.g., model name or tenant ID).
    namespace: String,

    /// Number of transformer layers in the model.
    /// Grows via `expand_num_layers` when PP ranks register additional layers.
    num_layers: AtomicUsize,

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

#[derive(Debug, Clone)]
pub(crate) struct InstanceLayerRegistration {
    pub layer_name: String,
    pub layer_id: usize,
    pub registration: KVCacheRegistration,
}

#[derive(Debug, Clone)]
pub(crate) struct InstanceShardTopology {
    pub shard_index: usize,
    pub device_id: i32,
    pub tp_rank: usize,
    pub preferred_numa: NumaNode,
    pub layers: Vec<InstanceLayerRegistration>,
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
            _id: id,
            namespace,
            num_layers: AtomicUsize::new(num_layers),
            tp_size,
            world_size,
            metadata: Mutex::new(LayerMetadata {
                name_to_id: HashMap::new(),
                names: Vec::new(),
                gpu_contexts: HashMap::new(),
            }),
        })
    }

    /// Get existing layer ID or allocate a new one.
    ///
    /// This is idempotent: calling multiple times with the same layer name
    /// returns the same ID. Used for MLA where multiple TP ranks register
    /// the same layer name on different devices.
    fn get_or_allocate_layer_id(&self, layer_name: &str) -> usize {
        let mut metadata = self.metadata.lock();

        if let Some(&id) = metadata.name_to_id.get(layer_name) {
            return id;
        }

        let id = metadata.names.len();
        metadata.names.push(layer_name.to_string());
        metadata.name_to_id.insert(layer_name.to_string(), id);
        // Grow num_layers to cover the newly allocated layer ID so that
        // total_slots() and get_slot_index() stay in bounds.
        self.num_layers.fetch_max(id + 1, Ordering::Relaxed);
        id
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
        self.num_layers.load(Ordering::Relaxed) * self.tp_size
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
        let num_layers = self.num_layers.load(Ordering::Relaxed);
        if layer_id >= num_layers {
            return Err(EngineError::InvalidArgument(format!(
                "layer_id {} out of range ({} layers)",
                layer_id, num_layers
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

    /// Get or create a GPU context for the specified device.
    ///
    /// This method lazily initializes CUDA contexts as devices are first accessed.
    /// The `numa_node` should be obtained from `NumaTopology::numa_for_gpu()`.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` for negative device IDs,
    /// or `EngineError::CudaInit` if CUDA context creation fails.
    fn create_gpu(
        &self,
        device_id: i32,
        tp_rank: usize,
        numa_node: NumaNode,
        kv_caches: HashMap<String, KVCacheRegistration>,
    ) -> Result<Arc<GpuContext>, EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {} must be >= 0",
                device_id
            )));
        }

        // Check if context already exists
        {
            let metadata = self.metadata.lock();
            if metadata.gpu_contexts.contains_key(&device_id) {
                return Err(EngineError::InvalidArgument(format!(
                    "GPU context for device {} already exists",
                    device_id
                )));
            }
        }

        // Create new context
        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;

        let ctx = Arc::new(GpuContext::new(
            cuda_ctx, device_id, tp_rank, numa_node, kv_caches,
        )?);

        // Insert and return
        let mut metadata = self.metadata.lock();

        // Double-check after acquiring lock
        if metadata.gpu_contexts.contains_key(&device_id) {
            return Err(EngineError::InvalidArgument(format!(
                "GPU context for device {} already exists",
                device_id
            )));
        }

        metadata.gpu_contexts.insert(device_id, Arc::clone(&ctx));

        info!(
            "Initialized GPU context: device_id={}, numa_node={}",
            device_id, numa_node
        );
        Ok(ctx)
    }

    /// Get an existing GPU context without creating one.
    pub(crate) fn get_gpu(&self, device_id: i32) -> Option<Arc<GpuContext>> {
        let metadata = self.metadata.lock();
        metadata.gpu_contexts.get(&device_id).cloned()
    }

    /// Register a new GPU with all its KV cache layers.
    ///
    /// # Errors
    /// - `EngineError::InvalidArgument` if GPU already registered
    /// - `EngineError::CudaInit` if GPU context creation fails
    pub(crate) fn register_new_gpu(
        &self,
        device_id: i32,
        tp_rank: usize,
        numa_node: NumaNode,
        kv_caches: HashMap<String, KVCacheRegistration>,
    ) -> Result<(), EngineError> {
        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} out of range (tp_size {})",
                tp_rank, self.tp_size
            )));
        }
        for layer_name in kv_caches.keys() {
            self.get_or_allocate_layer_id(layer_name);
        }
        self.create_gpu(device_id, tp_rank, numa_node, kv_caches)?;
        Ok(())
    }

    /// Snapshot registered device/tp/layer topology for instance-level planning.
    pub(crate) fn registered_shards(&self) -> Vec<InstanceShardTopology> {
        let metadata = self.metadata.lock();
        let mut shards: Vec<InstanceShardTopology> = metadata
            .gpu_contexts
            .values()
            .map(|gpu| {
                let mut layers: Vec<InstanceLayerRegistration> = gpu
                    .layer_registrations()
                    .into_iter()
                    .filter_map(|(layer_name, registration)| {
                        let layer_id = metadata.name_to_id.get(&layer_name).copied()?;
                        Some(InstanceLayerRegistration {
                            layer_name,
                            layer_id,
                            registration,
                        })
                    })
                    .collect();
                layers.sort_by_key(|layer| layer.layer_id);
                InstanceShardTopology {
                    shard_index: 0,
                    device_id: gpu.device_id(),
                    tp_rank: gpu.tp_rank(),
                    preferred_numa: gpu.preferred_numa(),
                    layers,
                }
            })
            .collect();
        shards.sort_by_key(|shard| (shard.tp_rank, shard.device_id));
        for (idx, shard) in shards.iter_mut().enumerate() {
            shard.shard_index = idx;
        }
        shards
    }

    /// Access the instance namespace.
    pub(crate) fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Access the world size.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// Verify that the topology matches expected values.
    ///
    /// Returns `Ok(())` if matches, or an error message describing the mismatch.
    pub(crate) fn verify_topology(
        &self,
        _num_layers: usize,
        tp_size: usize,
        world_size: usize,
    ) -> Result<(), String> {
        if self.tp_size != tp_size || self.world_size != world_size {
            return Err(format!(
                "exists with tp={}, world={}; \
                 requested tp={}, world={}",
                self.tp_size, self.world_size, tp_size, world_size
            ));
        }
        // num_layers grows automatically in get_or_allocate_layer_id
        // when PP ranks register new layers, so no check needed here.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_valid() {
        let reg = KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
        assert_eq!(reg.block_size_bytes, 1024);
    }

    #[test]
    fn registration_null_pointer_rejected() {
        assert!(KVCacheRegistration::new(0, 1024, 10, 64, 0, 1).is_err());
    }

    #[test]
    fn registration_memory_too_small() {
        let err = KVCacheRegistration::new(0x1000, 5120, 10, 1024, 0, 1).unwrap_err();
        assert!(err.contains("too small"));
    }

    #[test]
    fn padded_block_size() {
        // Unaligned: 8848 % 512 = 144, padded to 9216
        let reg = KVCacheRegistration::new(0x1000, 10_000_000, 100, 8848, 0, 1)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(reg.padded_bytes_per_block, 9216);
        assert_eq!(reg.padded_block_size_bytes, 9216);

        // Already aligned: no change
        let reg = KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(reg.padded_bytes_per_block, 1024);
        assert_eq!(reg.padded_block_size_bytes, 1024);

        // Split layout: padded per segment, total = padded * segments
        let reg = KVCacheRegistration::new(0x1000, 10_000_000, 100, 8848, 900_000, 2)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(reg.padded_bytes_per_block, 9216);
        assert_eq!(reg.padded_block_size_bytes, 9216 * 2);
    }

    /// Simulates the inference-side registration flow (single GPU).
    /// Covers: instance creation -> GPU context creation -> batch layer registration.
    #[test]
    fn inference_registration_flow() {
        // 1. Create instance (like get_or_create_instance)
        let instance = InstanceContext::new(
            "test-instance-1".to_string(),
            "model-ns".to_string(),
            64, // num_layers
            8,  // tp_size
            8,  // world_size
        )
        .expect("create instance");

        // Use UNKNOWN NUMA node for tests (no actual GPU/NUMA in CI)
        let numa = NumaNode::UNKNOWN;

        // 2. Build all layer registrations for device 0
        let mut kv_caches = HashMap::new();
        for layer_id in 0..4 {
            let layer_name = format!("layer_{}", layer_id);
            let reg = KVCacheRegistration::new(
                0x1000 + layer_id as u64 * 0x10000,
                1024 * 1024,
                100,
                1024,
                0,
                1,
            )
            .unwrap();
            kv_caches.insert(layer_name, reg);
        }

        // 3. Register all layers at once
        instance
            .register_new_gpu(0, 0, numa, kv_caches)
            .expect("register gpu with layers");

        // 4. Verify topology checking
        assert!(instance.verify_topology(64, 8, 8).is_ok());
        // num_layers can grow (PP ranks register incrementally), so smaller is ok
        assert!(instance.verify_topology(32, 8, 8).is_ok());
        // tp_size / world_size mismatch is still rejected
        assert!(instance.verify_topology(64, 4, 8).is_err());
        assert!(instance.verify_topology(64, 8, 4).is_err());

        // 5. Verify duplicate GPU registration fails
        let dup_caches = HashMap::from([(
            "layer_0".to_string(),
            KVCacheRegistration::new(0x2000, 1024 * 1024, 100, 1024, 0, 1).unwrap(),
        )]);
        let err = instance
            .register_new_gpu(0, 0, numa, dup_caches)
            .expect_err("duplicate GPU registration should fail");
        assert!(err.to_string().contains("already exists"));

        // 6. Verify we can get the registered layer back
        let gpu = instance.get_gpu(0).expect("get gpu context");
        let reg = gpu.get_registration("layer_0").expect("get registration");
        assert_eq!(reg.data_ptr, 0x1000);
        assert_eq!(reg.num_blocks, 100);
    }

    /// Tests MLA-style registration where the same layer name is used
    /// by multiple TP ranks. Verifies layer ID allocation is idempotent.
    #[test]
    fn mla_layer_id_allocation() {
        let instance = InstanceContext::new(
            "mla-instance".to_string(),
            "mla-ns".to_string(),
            10, // num_layers
            1,  // tp_size (MLA uses tp_size=1)
            8,  // world_size (8 TP ranks, but treated as 1 for storage)
        )
        .expect("create instance");

        // Simulate 8 TP ranks calling get_or_allocate_layer_id for the same layer
        // This is the MLA pattern where all ranks share the same KV data
        for _ in 0..8 {
            // All calls should return the same layer_id=0
            let layer_id = instance.get_or_allocate_layer_id("layer_0");
            assert_eq!(layer_id, 0, "all calls should get same layer_id");
        }

        // Verify only one layer was allocated
        assert_eq!(instance.get_layer_id("layer_0"), Some(0));

        // Allocate another layer
        let layer_id = instance.get_or_allocate_layer_id("layer_1");
        assert_eq!(layer_id, 1);

        // Verify both layers exist with correct IDs
        assert_eq!(instance.get_layer_id("layer_0"), Some(0));
        assert_eq!(instance.get_layer_id("layer_1"), Some(1));
    }
}
