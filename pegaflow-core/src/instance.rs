//! Instance and GPU context management for PegaFlow.
//!
//! This module provides the hierarchical context structure for managing
//! multi-tenant inference instances and their associated GPU resources.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use cudarc::driver::CudaContext;
use log::info;

use crate::{EngineError, gpu_worker::GpuWorkerPool};

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

/// Compute greatest common divisor using Euclidean algorithm.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
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

    /// Size of each block's segment in bytes (for one of K or V).
    pub bytes_per_block: usize,

    /// Total block size including all segments (`bytes_per_block * segments`).
    pub block_size_bytes: usize,

    /// Stride in bytes between K and V segments for split storage layouts.
    /// Zero when using contiguous single-segment storage.
    pub kv_stride_bytes: usize,

    /// Number of segments per block (1 for contiguous, 2 for K/V split).
    pub segments: usize,
}

impl KVCacheRegistration {
    /// Construct and validate a new registration.
    ///
    /// # Errors
    /// Returns a simple error message string if validation fails.
    pub fn new(
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
        })
    }

    /// Check SSD alignment requirement.
    ///
    /// Returns `None` if aligned, or `Some(error_message)` with fix hint.
    pub fn check_ssd_alignment(&self, alignment: usize) -> Option<String> {
        if self.bytes_per_block.is_multiple_of(alignment) {
            return None;
        }

        let factor = alignment / gcd(self.bytes_per_block, alignment);
        Some(format!(
            "SSD cache requires bytes_per_block to be aligned to {} bytes, but got {}. \
             Hint: multiply block_size by {} to achieve alignment.",
            alignment, self.bytes_per_block, factor
        ))
    }
}

/// Per-GPU execution context.
///
/// Each `GpuContext` manages:
/// - CUDA context lifetime for a specific device
/// - KV cache registrations for all layers on this GPU
/// - Asynchronous worker pool for load/save operations
pub struct GpuContext {
    /// KV cache layout registrations by layer name.
    kv_caches: Mutex<HashMap<String, KVCacheRegistration>>,

    /// CUDA context handle (kept alive for the lifetime of this context).
    _cuda_ctx: Arc<CudaContext>,

    /// Worker thread pool for asynchronous GPU operations.
    worker_pool: GpuWorkerPool,
}

impl GpuContext {
    /// Create a new GPU context for the specified device.
    ///
    /// # Errors
    /// Returns `EngineError::CudaInit` if CUDA context creation or worker
    /// pool initialization fails.
    pub fn new(cuda_ctx: Arc<CudaContext>, device_id: i32) -> Result<Self, EngineError> {
        let worker_pool = GpuWorkerPool::spawn(device_id)?;

        Ok(Self {
            kv_caches: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
            worker_pool,
        })
    }

    /// Register a new layer's KV cache layout.
    ///
    /// # Errors
    /// Returns an error if the layer is already registered on this GPU.
    pub fn register_new_layer(
        &self,
        layer_name: String,
        registration: KVCacheRegistration,
    ) -> Result<(), String> {
        let mut registrations = self.kv_caches.lock().expect("kv_caches lock poisoned");

        if registrations.contains_key(&layer_name) {
            return Err(format!(
                "layer {} already registered on this GPU",
                layer_name
            ));
        }

        registrations.insert(layer_name, registration);
        Ok(())
    }

    /// Retrieve a layer's registration information.
    pub fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        let registrations = self.kv_caches.lock().expect("kv_caches lock poisoned");
        registrations.get(layer_name).cloned()
    }

    /// Access the worker pool for submitting GPU operations.
    pub fn worker_pool(&self) -> &GpuWorkerPool {
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
    pub fn new(
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
            num_layers,
            tp_size,
            world_size,
            metadata: Mutex::new(LayerMetadata {
                name_to_id: HashMap::new(),
                names: Vec::new(),
                gpu_contexts: HashMap::new(),
            }),
        })
    }

    /// Allocate a new numeric ID for a layer name.
    ///
    /// # Returns
    /// - `Some(id)` if the layer was newly allocated
    /// - `None` if the layer already exists
    pub fn allocate_new_layer_id(&self, layer_name: &str) -> Option<usize> {
        let mut metadata = self.metadata.lock().expect("metadata lock poisoned");

        if metadata.name_to_id.contains_key(layer_name) {
            return None;
        }

        let id = metadata.names.len();
        metadata.names.push(layer_name.to_string());
        metadata.name_to_id.insert(layer_name.to_string(), id);
        Some(id)
    }

    /// Get existing layer ID or allocate a new one.
    ///
    /// This is idempotent: calling multiple times with the same layer name
    /// returns the same ID. Used for MLA where multiple TP ranks register
    /// the same layer name on different devices.
    pub fn get_or_allocate_layer_id(&self, layer_name: &str) -> usize {
        let mut metadata = self.metadata.lock().expect("metadata lock poisoned");

        if let Some(&id) = metadata.name_to_id.get(layer_name) {
            return id;
        }

        let id = metadata.names.len();
        metadata.names.push(layer_name.to_string());
        metadata.name_to_id.insert(layer_name.to_string(), id);
        id
    }

    /// Look up the numeric ID for a layer name.
    ///
    /// Returns `None` if the layer has not been registered.
    pub fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        let metadata = self.metadata.lock().expect("metadata lock poisoned");
        metadata.name_to_id.get(layer_name).copied()
    }

    /// Calculate the total number of storage slots.
    ///
    /// Slots are organized as a flattened 2D array: `[layer][tp_rank]`.
    pub fn total_slots(&self) -> usize {
        self.num_layers * self.tp_size
    }

    /// Compute the slot index for a specific layer and TP rank.
    ///
    /// The slot index is used to locate blocks in the storage engine.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` if `layer_id` or `tp_rank`
    /// are out of bounds.
    pub fn get_slot_index(&self, layer_id: usize, tp_rank: usize) -> Result<usize, EngineError> {
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

    /// Get or create a GPU context for the specified device.
    ///
    /// This method lazily initializes CUDA contexts as devices are first accessed.
    ///
    /// # Errors
    /// Returns `EngineError::InvalidArgument` for negative device IDs,
    /// or `EngineError::CudaInit` if CUDA context creation fails.
    pub fn ensure_gpu(&self, device_id: i32) -> Result<Arc<GpuContext>, EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {} must be >= 0",
                device_id
            )));
        }

        // Fast path: check if context exists
        {
            let metadata = self.metadata.lock().expect("metadata lock poisoned");
            if let Some(ctx) = metadata.gpu_contexts.get(&device_id) {
                return Ok(Arc::clone(ctx));
            }
        }

        // Slow path: create new context
        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;

        let ctx = Arc::new(GpuContext::new(cuda_ctx, device_id)?);

        // Insert and return
        let mut metadata = self.metadata.lock().expect("metadata lock poisoned");

        // Double-check after acquiring lock
        if let Some(existing) = metadata.gpu_contexts.get(&device_id) {
            return Ok(Arc::clone(existing));
        }

        metadata.gpu_contexts.insert(device_id, Arc::clone(&ctx));

        info!("Initialized GPU context: device_id={}", device_id);
        Ok(ctx)
    }

    /// Get an existing GPU context without creating one.
    pub fn get_gpu(&self, device_id: i32) -> Option<Arc<GpuContext>> {
        let metadata = self.metadata.lock().expect("metadata lock poisoned");
        metadata.gpu_contexts.get(&device_id).cloned()
    }

    /// Register a layer on a GPU.
    ///
    /// This method:
    /// 1. Ensures the GPU context exists (creates if needed)
    /// 2. Gets existing layer ID or allocates a new one (idempotent)
    /// 3. Registers the layer's KV cache on the GPU
    ///
    /// For MLA with TP > 1, multiple ranks may register the same layer name
    /// on different devices. The layer ID allocation is idempotent, but
    /// GPU registration will fail if the same layer is registered twice
    /// on the same device.
    ///
    /// # Errors
    /// - `EngineError::InvalidArgument` if layer already registered on this GPU
    /// - `EngineError::CudaInit` if GPU context creation fails
    pub fn register_new_gpu_layer(
        &self,
        device_id: i32,
        layer_name: &str,
        registration: KVCacheRegistration,
    ) -> Result<usize, EngineError> {
        // 1. Ensure GPU context
        let gpu = self.ensure_gpu(device_id)?;

        // 2. Get or allocate layer ID (idempotent for MLA multi-device registration)
        let layer_id = self.get_or_allocate_layer_id(layer_name);

        // 3. Register on GPU (fails if layer already registered on THIS device)
        gpu.register_new_layer(layer_name.to_string(), registration)
            .map_err(EngineError::InvalidArgument)?;

        Ok(layer_id)
    }

    /// Access the instance namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Access the number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Access the tensor parallelism size.
    pub fn tp_size(&self) -> usize {
        self.tp_size
    }

    /// Access the world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Verify that the topology matches expected values.
    ///
    /// Returns `Ok(())` if matches, or an error message describing the mismatch.
    pub fn verify_topology(
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
    fn ssd_alignment_check() {
        let reg = KVCacheRegistration::new(0x1000, 1024 * 1024, 100, 128, 0, 1).unwrap();
        assert!(reg.check_ssd_alignment(512).unwrap().contains("4"));
        assert!(reg.check_ssd_alignment(128).is_none());
    }

    /// Simulates the inference-side registration flow (single GPU).
    /// Covers: instance creation -> GPU context creation -> layer registration.
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

        // 2. Register multiple layers on device 0
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

            let id = instance
                .register_new_gpu_layer(0, &layer_name, reg)
                .expect("register layer");
            assert_eq!(id, layer_id);
        }

        // 3. Verify topology checking
        assert!(instance.verify_topology(64, 8, 8).is_ok());
        assert!(instance.verify_topology(32, 8, 8).is_err()); // wrong layers

        // 4. Verify duplicate layer registration on SAME device fails
        let dup_reg = KVCacheRegistration::new(0x2000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
        let err = instance
            .register_new_gpu_layer(0, "layer_0", dup_reg)
            .expect_err("duplicate on same device should fail");
        assert!(err.to_string().contains("already registered"));

        // 5. Verify we can get the registered layer back
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
