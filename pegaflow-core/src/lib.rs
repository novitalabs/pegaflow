pub mod allocator;
pub mod block;
mod cache;
pub mod gpu_worker;
mod metrics;
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
pub use pinned_pool::PinnedAllocation;
pub use seal_offload::SlotMeta;
pub use ssd_cache::SsdCacheConfig;
pub use storage::{SealNotification, StorageConfig};
pub use sync_state::{LoadState, LoadStateError};

// ============================================================================
// PegaEngine currently prioritizes vLLM's layer-first (KV-first) tensor layout.
// This means all K segments are contiguous, followed by all V segments, so the
// GPU memory picture looks like:
//
//   +---------------------------------------------------------------+
//   |  Layer0: KKKKKKKK.... | Layer0: VVVVVVVV.... | Layer1: K ...  |
//   +---------------------------------------------------------------+
//          ^ contiguous K blocks        ^ contiguous V blocks
//
// As long as vLLM keeps this layout we must respect its stride-based view and
// fall back to strided transfers; future refactors can add dedicated handling
// for other layouts without breaking this contract.
//
// To support efficient batching during "load" (CPU -> GPU), we now avoid
// storing K and V interleaved in a single contiguous block. Instead, we allocate
// all K segments for a saved batch in one contiguous CPU region, and all V segments
// in another. This Split-Storage approach ensures that when we load the batch back,
// the K source pointers are contiguous and can be merged into a single cuMemcpy,
// significantly improving PCIe bandwidth utilization compared to strided copies.
// ============================================================================

use cudarc::driver::CudaContext;
use std::{
    collections::HashMap,
    fmt,
    num::NonZeroU64,
    sync::{Arc, Mutex, RwLock},
};
use tracing::{debug, instrument};

use crate::gpu_worker::{GpuWorkerPool, LayerLoadData, LoadBlock, LoadTask, SaveBlock};
use crate::metrics::core_metrics;
use crate::storage::{StorageEngine, SSD_ALIGNMENT};

const DEFAULT_PINNED_POOL_BYTES: usize = 30 * 1024 * 1024 * 1024; // 30GB

/// Compute greatest common divisor using Euclidean algorithm
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[derive(Debug)]
pub enum EngineError {
    InstanceMissing(String),
    WorkerMissing(String, i32),
    InvalidArgument(String),
    CudaInit(String),
    Storage(String),
    Poisoned(&'static str),
    TopologyMismatch(String), // Context, Details
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

pub struct PegaEngine {
    /// Manages instances and their GPU contexts
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    /// Storage engine responsible for pinned allocations + block cache + SSD index
    storage: Arc<StorageEngine>,
}

#[derive(Debug, Clone)]
pub struct KVCacheRegistration {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    pub block_size_bytes: usize,
    /// Distance in bytes between K and V segments when KV-first layout is used.
    /// Zero when the layout stores a single segment per block.
    pub kv_stride_bytes: usize,
    /// Number of segments per block (1 for blocks-first, 2 for KV-first).
    pub segments: usize,
}

/// Context for a specific GPU (one CUDA device)
struct GpuContext {
    /// KV cache registrations for this GPU (Layer Name -> GPU Ptr Info)
    kv_caches: Mutex<HashMap<String, KVCacheRegistration>>,
    /// Hold CUDA context for the lifetime of the inference context
    _cuda_ctx: Arc<CudaContext>,
    /// Worker pool for async GPU operations (load/save)
    worker_pool: GpuWorkerPool,
}

impl GpuContext {
    fn new(cuda_ctx: Arc<CudaContext>, device_id: i32) -> Result<Self, EngineError> {
        let worker_pool = GpuWorkerPool::spawn(device_id)?;
        Ok(Self {
            kv_caches: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
            worker_pool,
        })
    }

    fn register_layer(&self, layer_name: String, registration: KVCacheRegistration) {
        let mut caches = self.kv_caches.lock().expect("kv_caches lock poisoned");
        caches.insert(layer_name, registration);
    }

    fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        let caches = self.kv_caches.lock().expect("kv_caches lock poisoned");
        caches.get(layer_name).cloned()
    }

    fn worker_pool(&self) -> &GpuWorkerPool {
        &self.worker_pool
    }
}

/// Global context for a Model Instance (across all device assignments)
struct InstanceContext {
    _id: String,
    /// Namespace for model isolation (fixed for the instance lifetime)
    namespace: String,
    num_layers: usize,
    tp_size: usize,
    /// Maps layer name to layer ID (0..num_layers)
    layer_name_to_id: Mutex<HashMap<String, usize>>,
    /// Inverse map to ensure stable ordering
    layer_names: Mutex<Vec<String>>,
    /// Active GPU contexts for this instance, keyed by CUDA device ID
    gpu_contexts: RwLock<HashMap<i32, Arc<GpuContext>>>,
}

impl InstanceContext {
    fn new(id: String, namespace: String, num_layers: usize, tp_size: usize) -> Self {
        Self {
            _id: id,
            namespace,
            num_layers,
            tp_size,
            layer_name_to_id: Mutex::new(HashMap::new()),
            layer_names: Mutex::new(Vec::new()),
            gpu_contexts: RwLock::new(HashMap::new()),
        }
    }

    fn get_or_create_layer_id(&self, layer_name: &str) -> usize {
        let mut map = self
            .layer_name_to_id
            .lock()
            .expect("layer_name_to_id lock poisoned");
        if let Some(&id) = map.get(layer_name) {
            return id;
        }

        let mut names = self.layer_names.lock().expect("layer_names lock poisoned");
        let id = names.len();
        names.push(layer_name.to_string());
        map.insert(layer_name.to_string(), id);
        id
    }

    fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        let map = self
            .layer_name_to_id
            .lock()
            .expect("layer_name_to_id lock poisoned");
        map.get(layer_name).copied()
    }

    fn total_slots(&self) -> usize {
        self.num_layers * self.tp_size
    }

    fn get_slot_index(&self, layer_id: usize, tp_rank: usize) -> Result<usize, EngineError> {
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

    fn ensure_gpu(&self, device_id: i32) -> Result<Arc<GpuContext>, EngineError> {
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(format!(
                "device_id {} must be >= 0",
                device_id
            )));
        }

        {
            let workers = self
                .gpu_contexts
                .read()
                .expect("gpu contexts read lock poisoned");
            if let Some(worker) = workers.get(&device_id) {
                return Ok(Arc::clone(worker));
            }
        }

        let mut workers = self
            .gpu_contexts
            .write()
            .expect("gpu contexts write lock poisoned");
        if let Some(worker) = workers.get(&device_id) {
            return Ok(Arc::clone(worker));
        }

        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let worker = Arc::new(GpuContext::new(cuda_ctx, device_id)?);
        workers.insert(device_id, Arc::clone(&worker));
        Ok(worker)
    }

    fn get_gpu(&self, device_id: i32) -> Option<Arc<GpuContext>> {
        let workers = self
            .gpu_contexts
            .read()
            .expect("gpu contexts read lock poisoned");
        workers.get(&device_id).cloned()
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PegaEngine {
    /// Create a new PegaEngine instance with default pool size and regular pages
    #[instrument(level = "info")]
    pub fn new() -> Self {
        let (engine, _rx) = Self::new_with_config(
            DEFAULT_PINNED_POOL_BYTES,
            false,
            storage::StorageConfig::default(),
        );
        engine
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    ///
    /// If `use_hugepages` is true, uses 2MB huge pages (requires system config).
    pub fn new_with_pool_size(pool_size: usize, use_hugepages: bool) -> Self {
        let (engine, _rx) =
            Self::new_with_config(pool_size, use_hugepages, storage::StorageConfig::default());
        engine
    }

    /// Create a new PegaEngine instance with custom pool size and storage config
    ///
    /// If `use_hugepages` is true, uses 2MB huge pages (requires system config).
    /// Returns (engine, seal_notify_rx) - pass receiver to `spawn_seal_offload_task` for SSD offload.
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

    fn get_or_create_instance(
        &self,
        instance_id: &str,
        namespace: &str,
        num_layers: usize,
        tp_size: usize,
    ) -> Result<Arc<InstanceContext>, EngineError> {
        // Fast path
        {
            let instances = self.instances.read().expect("instances read lock poisoned");
            if let Some(instance) = instances.get(instance_id) {
                // Verify topology matches
                if instance.num_layers != num_layers || instance.tp_size != tp_size {
                    return Err(EngineError::TopologyMismatch(format!(
                        "instance {instance_id} exists with layers={}, tp={}; requested layers={}, tp={}",
                        instance.num_layers, instance.tp_size, num_layers, tp_size
                    )));
                }
                return Ok(Arc::clone(instance));
            }
        }

        // Slow path
        let mut instances = self
            .instances
            .write()
            .expect("instances write lock poisoned");
        if let Some(instance) = instances.get(instance_id) {
            if instance.num_layers != num_layers || instance.tp_size != tp_size {
                return Err(EngineError::TopologyMismatch(format!(
                    "instance {instance_id} exists with layers={}, tp={}; requested layers={}, tp={}",
                    instance.num_layers, instance.tp_size, num_layers, tp_size
                )));
            }
            return Ok(Arc::clone(instance));
        }

        let instance = Arc::new(InstanceContext::new(
            instance_id.to_string(),
            namespace.to_string(),
            num_layers,
            tp_size,
        ));
        instances.insert(instance_id.to_string(), Arc::clone(&instance));
        Ok(instance)
    }

    fn get_instance(&self, instance_id: &str) -> Result<Arc<InstanceContext>, EngineError> {
        let instances = self.instances.read().expect("instances read lock poisoned");
        instances
            .get(instance_id)
            .cloned()
            .ok_or_else(|| EngineError::InstanceMissing(instance_id.to_string()))
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        err,
        fields(instance=%instance_id, rank=%tp_rank, device=%device_id, layer=%layer_name)
    )]
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
        num_layers: usize,
    ) -> Result<(), EngineError> {
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            return Err(EngineError::InvalidArgument(format!(
                "invalid KV cache layout for layer {layer_name}"
            )));
        }
        if segments > 1 && kv_stride_bytes == 0 {
            return Err(EngineError::InvalidArgument(format!(
                "kv_stride_bytes must be > 0 when segments > 1 for layer {layer_name}"
            )));
        }
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(
                "device_id must be >= 0".to_string(),
            ));
        }

        // Validate block sizing/strides up front to avoid runtime overflows
        let block_size_bytes = bytes_per_block.checked_mul(segments).ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "block size overflow: bytes_per_block={} segments={} for layer {layer_name}",
                bytes_per_block, segments
            ))
        })?;

        // Compute maximum offset + block size for the last segment of the last block
        let max_block_offset = num_blocks
            .checked_sub(1)
            .and_then(|idx| idx.checked_mul(bytes_per_block))
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "num_blocks {} too large for layer {layer_name}",
                    num_blocks
                ))
            })?;
        let max_segment_offset = segments
            .checked_sub(1)
            .and_then(|idx| idx.checked_mul(kv_stride_bytes))
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "segments/stride overflow for layer {layer_name} (segments={}, stride={})",
                    segments, kv_stride_bytes
                ))
            })?;

        let max_offset = max_block_offset
            .checked_add(max_segment_offset)
            .ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layout offset overflow for layer {layer_name}"
                ))
            })?;
        let end = max_offset.checked_add(bytes_per_block).ok_or_else(|| {
            EngineError::InvalidArgument(format!("layout end overflow for layer {layer_name}"))
        })?;
        if end > size_bytes {
            return Err(EngineError::InvalidArgument(format!(
                "registered memory too small for layer {layer_name}: need at least {} bytes, got {}",
                end,
                size_bytes
            )));
        }

        let instance = self.get_or_create_instance(instance_id, namespace, num_layers, tp_size)?;
        let worker = instance.ensure_gpu(device_id)?;

        // Register layer ID in global instance map
        instance.get_or_create_layer_id(&layer_name);

        if self.storage.is_ssd_enabled() && !bytes_per_block.is_multiple_of(SSD_ALIGNMENT) {
            let gcd = gcd(bytes_per_block, SSD_ALIGNMENT);
            let aligned_factor = SSD_ALIGNMENT / gcd;
            return Err(EngineError::InvalidArgument(format!(
                "SSD cache requires bytes_per_block to be aligned to {} bytes, but got {} for layer {}. \
                 Hint: bytes_per_block = stride * block_size. Current block_size likely needs to be multiplied by {} \
                 to achieve alignment. Consider using block_size that is a multiple of {}.",
                SSD_ALIGNMENT, bytes_per_block, layer_name, aligned_factor, aligned_factor
            )));
        }

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            block_size_bytes,
            kv_stride_bytes,
            segments,
        };

        worker.register_layer(layer_name, registration);
        Ok(())
    }

    /// Unregister instance
    #[instrument(level = "info", skip(self), err)]
    pub fn unregister_instance(&self, instance_id: &str) -> Result<(), EngineError> {
        let removed = self
            .instances
            .write()
            .expect("instances write lock poisoned")
            .remove(instance_id);

        if removed.is_none() {
            return Err(EngineError::InstanceMissing(instance_id.to_string()));
        }
        Ok(())
    }

    /// Batch save KV blocks from multiple layers.
    ///
    /// Each element in `saves` is a tuple of (layer_name, block_ids, block_hashes).
    /// This is more efficient than calling save_kv_blocks_from_ipc in a loop
    /// as it reduces Python-Rust boundary crossings.
    #[instrument(
        level = "debug",
        skip(self, saves),
        err,
        fields(instance=%instance_id, rank=%tp_rank, device=%device_id, layers=%saves.len())
    )]
    pub async fn batch_save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        device_id: i32,
        saves: Vec<(String, Vec<i32>, Vec<Vec<u8>>)>,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace.clone();

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

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        err,
        fields(instance=%instance_id, rank=%tp_rank, device=%device_id, layer=%layer_name, blocks=%block_ids.len())
    )]
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
        let namespace = &instance.namespace;

        self.save_kv_blocks_from_ipc_inner(
            instance_id,
            tp_rank,
            device_id,
            namespace,
            layer_name,
            block_ids,
            block_hashes,
        )
        .await
    }

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
        let worker = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        let layer_id = instance
            .get_layer_id(layer_name)
            .ok_or_else(|| EngineError::InvalidArgument(format!("layer {layer_name} unknown")))?;

        let registration = worker.get_registration(layer_name).ok_or_else(|| {
            EngineError::InvalidArgument(format!("layer {layer_name} not registered on device"))
        })?;

        let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
        let total_slots = instance.total_slots();

        // Collect blocks that need to be saved
        let mut blocks_to_save = Vec::with_capacity(block_ids.len());

        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            if *block_id < 0 {
                continue;
            }
            let block_idx = *block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(EngineError::InvalidArgument(format!(
                    "block {block_idx} out of range for layer {layer_name} ({} blocks registered)",
                    registration.num_blocks
                )));
            }

            // Check if this block_hash already has data for this slot
            // Skip if already sealed in cache or slot exists in inflight
            //
            // FIXME(race): There's a subtle race condition here. If cache contains the block
            // when checking layer 0 (skip), but gets evicted before checking layer 1 (write),
            // we create an inflight block with only slot 1 that can never complete.
            // Probability is near-zero with large pools, but for robustness we should add
            // a background GC thread to clean up inflight blocks older than ~1 hour.
            let needs_save = !self.storage.cache_contains(namespace, block_hash)
                && !self
                    .storage
                    .inflight_has_slot(namespace, block_hash, slot_id);

            if needs_save {
                blocks_to_save.push((block_idx, block_hash.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return Ok(());
        }

        debug!(
            "Saving {} blocks for layer {layer_name} on instance {instance_id} rank {tp_rank}",
            blocks_to_save.len()
        );

        let block_size = registration.block_size_bytes;
        let num_blocks = blocks_to_save.len();

        // For layer-first layout with KV stride, allocate separate regions for K and V
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;
            let k_total_size = (segment_size as u64)
                .checked_mul(num_blocks as u64)
                .and_then(NonZeroU64::new)
                .expect("allocation size overflow for K segments in layer {layer_name}");
            let v_total_size = k_total_size;

            // Allocate separate regions for K and V segments
            let mut k_allocation = self.storage.allocate(k_total_size).ok_or_else(|| {
                EngineError::Storage(
                    "pinned pool exhausted while allocating K segment buffer".to_string(),
                )
            })?;

            let mut v_allocation = self.storage.allocate(v_total_size).ok_or_else(|| {
                EngineError::Storage(
                    "pinned pool exhausted while allocating V segment buffer".to_string(),
                )
            })?;

            // Get mutable pointers before creating Arc clones
            // Store as usize to cross await point (pinned memory won't move)
            let (k_base_addr, v_base_addr) = {
                let k_ptr = Arc::get_mut(&mut k_allocation)
                    .expect("k_allocation must be uniquely owned here")
                    .as_mut_ptr();
                let v_ptr = Arc::get_mut(&mut v_allocation)
                    .expect("v_allocation must be uniquely owned here")
                    .as_mut_ptr();
                (k_ptr as usize, v_ptr as usize)
            };

            // Build SaveBlock list for worker
            let save_blocks: Vec<SaveBlock> = blocks_to_save
                .iter()
                .enumerate()
                .map(|(i, (block_idx, _))| SaveBlock {
                    block_idx: *block_idx,
                    k_dst_ptr: (k_base_addr + i * segment_size) as *mut u8,
                    v_dst_ptr: Some((v_base_addr + i * segment_size) as *mut u8),
                })
                .collect();

            // Execute GPU->CPU copy via worker pool
            worker
                .worker_pool()
                .save(registration.clone(), save_blocks)
                .await?;

            // Create LayerBlock objects after copying is done
            let mut sealed_blocks = Vec::new();
            for (i, (_, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let k_ptr = (k_base_addr + i * segment_size) as *mut u8;
                let v_ptr = (v_base_addr + i * segment_size) as *mut u8;

                let block = Arc::new(LayerBlock::new_split(
                    k_ptr,
                    v_ptr,
                    block_size,
                    Arc::clone(&k_allocation),
                    Arc::clone(&v_allocation),
                ));

                if let Some(sealed) = self
                    .storage
                    .insert_slot(namespace, block_hash, slot_id, block, total_slots)
                    .map_err(|e| EngineError::Storage(e.to_string()))?
                {
                    sealed_blocks.push(sealed);
                }
            }

            // Batch admit with prefix-aware early break:
            // If block[i] is rejected, block[i+1..] are useless for prefix matching
            for (key, block) in sealed_blocks {
                if !self.storage.cache_admit(key, block) {
                    break;
                }
            }
        } else {
            // Contiguous or single-segment layouts
            let block_size_u64 = u64::try_from(block_size).map_err(|_| {
                EngineError::Storage(format!(
                    "block size {} exceeds supported range for layer {layer_name}",
                    block_size
                ))
            })?;
            let total_size = block_size_u64
                .checked_mul(num_blocks as u64)
                .and_then(NonZeroU64::new)
                .ok_or_else(|| {
                    EngineError::Storage(format!(
                        "allocation size overflow while saving layer {layer_name}"
                    ))
                })?;

            let mut allocation = self.storage.allocate(total_size).ok_or_else(|| {
                EngineError::Storage(
                    "pinned pool exhausted while allocating contiguous block buffer".to_string(),
                )
            })?;

            // Store as usize to cross await point (pinned memory won't move)
            let base_addr = Arc::get_mut(&mut allocation)
                .ok_or_else(|| {
                    EngineError::Storage(format!(
                        "allocation shared unexpectedly while saving layer {layer_name}"
                    ))
                })?
                .as_mut_ptr() as usize;

            // Build SaveBlock list for worker
            let save_blocks: Vec<SaveBlock> = blocks_to_save
                .iter()
                .enumerate()
                .map(|(i, (block_idx, _))| SaveBlock {
                    block_idx: *block_idx,
                    k_dst_ptr: (base_addr + i * block_size) as *mut u8,
                    v_dst_ptr: None,
                })
                .collect();

            // Execute GPU->CPU copy via worker pool
            worker
                .worker_pool()
                .save(registration.clone(), save_blocks)
                .await?;

            // Create LayerBlock objects after copying is done
            let mut sealed_blocks = Vec::new();
            for (i, (_, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = (base_addr + i * block_size) as *mut u8;

                let block = Arc::new(LayerBlock::new_contiguous(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&allocation),
                ));

                if let Some(sealed) = self
                    .storage
                    .insert_slot(namespace, block_hash, slot_id, block, total_slots)
                    .map_err(|e| EngineError::Storage(e.to_string()))?
                {
                    sealed_blocks.push(sealed);
                }
            }

            // Batch admit with prefix-aware early break:
            // If block[i] is rejected, block[i+1..] are useless for prefix matching
            for (key, block) in sealed_blocks {
                if !self.storage.cache_admit(key, block) {
                    break;
                }
            }
        }

        let elapsed = timer.elapsed().as_secs_f64();
        let total_bytes = (block_size as u64)
            .checked_mul(num_blocks as u64)
            .unwrap_or(0);
        if num_blocks > 0 {
            metrics.save_bytes.add(total_bytes, &[]);
            metrics.save_duration_ms.record(elapsed * 1000.0, &[]);
        }
        Ok(())
    }

    /// Count prefix hit blocks with SSD prefetch support.
    ///
    /// Returns:
    /// - `Done { hit, missing: 0 }`: all requested blocks are in memory cache
    /// - `Loading { hit, loading }`: some blocks being fetched from SSD
    /// - `Done { hit, missing }`: some blocks don't exist anywhere
    ///
    /// When a cache miss is detected but SSD cache has the block, this method automatically
    /// starts a background prefetch from SSD. The caller should retry after a short delay.
    #[instrument(
        level = "debug",
        skip(self, block_hashes),
        err,
        fields(instance=%instance_id, requested = %block_hashes.len())
    )]
    pub fn count_prefix_hit_blocks_with_prefetch(
        &self,
        instance_id: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<PrefetchStatus, EngineError> {
        let instance = self.get_instance(instance_id)?;
        let namespace = &instance.namespace;
        let metrics = core_metrics();

        // Delegate to storage engine's unified check_prefix_and_prefetch
        let status = self
            .storage
            .check_prefix_and_prefetch(namespace, block_hashes);

        // Record metrics for terminal state
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

    /// Batch load KV blocks for multiple layers asynchronously.
    ///
    /// This submits a task to the GPU worker pool which performs all transfers.
    /// The function returns immediately after submitting the task.
    ///
    /// The connector creates a LoadState, passes the `load_state_shm` to this method,
    /// and then spin-waits on the state until it becomes non-zero (1=success, -1=error).
    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        err,
        fields(instance=%instance_id, rank=%tp_rank, device=%device_id, layers=%layer_names.len(), blocks=%block_ids.len())
    )]
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
        // Attach to LoadState early so we can set error if something fails
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

        // If we failed before submitting to worker, set error on LoadState
        if let Err(ref e) = result {
            tracing::error!(error = ?e, "batch_load_kv_blocks_multi_layer failed before worker");
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
        let namespace = &instance.namespace;

        let worker = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        // Lookup all block_hashes ONCE and cache the blocks
        let block_cache = self
            .storage
            .cache_lookup_many(namespace, block_hashes)
            .map_err(EngineError::Storage)?;

        // Build LoadTask with data for all layers
        let mut layers = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_id = instance.get_layer_id(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} unknown for instance {instance_id}"
                ))
            })?;

            let registration = worker.get_registration(layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "layer {layer_name} not registered on device {device_id}"
                ))
            })?;

            let slot_id = instance.get_slot_index(layer_id, tp_rank)?;

            // Collect valid blocks to load for this layer
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

        // If no layers have blocks to load, complete immediately
        if layers.is_empty() {
            debug!("No blocks to load, completing immediately");
            let load_state = LoadState::attach(load_state_shm)?;
            load_state.set_completed();
            return Ok(());
        }

        // Submit task to worker pool (fire and forget)
        let task = LoadTask {
            layers,
            load_state_shm: load_state_shm.to_string(),
        };

        worker.worker_pool().submit_load(task)
    }
}
