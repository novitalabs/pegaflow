pub mod allocator;
pub mod pinned_pool;
mod storage;
pub mod sync_state;
mod transfer;

pub use pinned_pool::PinnedAllocation;
pub use sync_state::LayerSyncState;

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

use cudarc::driver::{CudaContext, CudaEvent, CudaStream};
use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};
use tracing::{debug, info, instrument};

use crate::storage::{Block, StorageEngine};

const DEFAULT_PINNED_POOL_BYTES: usize = 30 * 1024 * 1024 * 1024; // 10GB

#[derive(Debug)]
pub enum EngineError {
    InstanceMissing(String),
    WorkerMissing(String, usize),
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
            EngineError::WorkerMissing(ctx, rank) => {
                write!(f, "worker rank {rank} not found in instance {ctx}")
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

pub struct PegaEngine {
    /// Manages instances and their workers
    instances: RwLock<HashMap<String, Arc<InstanceContext>>>,
    /// Storage engine responsible for pinned allocations + block cache
    storage: StorageEngine,
}

#[derive(Debug, Clone)]
pub struct KVCacheRegistration {
    pub data_ptr: u64,
    pub size_bytes: usize,
    pub num_blocks: usize,
    pub bytes_per_block: usize,
    /// Distance in bytes between K and V segments when KV-first layout is used.
    /// Zero when the layout stores a single segment per block.
    pub kv_stride_bytes: usize,
    /// Number of segments per block (1 for blocks-first, 2 for KV-first).
    pub segments: usize,
}

/// Context for a specific Worker process (one TP rank)
struct WorkerContext {
    /// KV cache registrations for this worker (Layer Name -> GPU Ptr Info)
    kv_caches: Mutex<HashMap<String, KVCacheRegistration>>,
    /// Single stream for all transfers to ensure sequential execution
    stream: Arc<CudaStream>,
    /// Track per-layer completion events for async loading
    layer_events: Mutex<HashMap<String, CudaEvent>>,
    /// Shared memory sync state for async layer loading (attached from connector)
    sync_state: Mutex<Option<Arc<LayerSyncState>>>,
    /// Hold CUDA context for the lifetime of the inference context
    _cuda_ctx: Arc<CudaContext>,
    device_id: i32,
}

impl WorkerContext {
    fn new(cuda_ctx: Arc<CudaContext>, device_id: i32) -> Self {
        let stream = cuda_ctx
            .new_stream()
            .expect("Failed to create stream for worker context");
        Self {
            kv_caches: Mutex::new(HashMap::new()),
            stream,
            layer_events: Mutex::new(HashMap::new()),
            sync_state: Mutex::new(None),
            _cuda_ctx: cuda_ctx,
            device_id,
        }
    }

    fn register_layer(&self, layer_name: String, registration: KVCacheRegistration) {
        let mut caches = self.kv_caches.lock().expect("kv_caches lock poisoned");
        caches.insert(layer_name, registration);
    }

    fn get_registration(&self, layer_name: &str) -> Option<KVCacheRegistration> {
        let caches = self.kv_caches.lock().expect("kv_caches lock poisoned");
        caches.get(layer_name).cloned()
    }

    fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    fn take_layer_event(&self, layer_name: &str) -> Option<CudaEvent> {
        let mut guard = self.layer_events.lock().expect("layer events map poisoned");
        guard.remove(layer_name)
    }

    fn set_sync_state(&self, sync_state: Arc<LayerSyncState>) {
        let mut guard = self.sync_state.lock().expect("sync_state lock poisoned");
        *guard = Some(sync_state);
    }

    fn get_sync_state(&self) -> Option<Arc<LayerSyncState>> {
        let guard = self.sync_state.lock().expect("sync_state lock poisoned");
        guard.clone()
    }
}

/// Global context for a Model Instance (across all TP ranks)
struct InstanceContext {
    id: String,
    num_layers: usize,
    tp_size: usize,
    /// Maps layer name to layer ID (0..num_layers)
    layer_name_to_id: Mutex<HashMap<String, usize>>,
    /// Inverse map to ensure stable ordering
    layer_names: Mutex<Vec<String>>,
    /// Active workers for this instance, keyed by TP rank
    workers: RwLock<HashMap<usize, Arc<WorkerContext>>>,
}

impl InstanceContext {
    fn new(id: String, num_layers: usize, tp_size: usize) -> Self {
        Self {
            id,
            num_layers,
            tp_size,
            layer_name_to_id: Mutex::new(HashMap::new()),
            layer_names: Mutex::new(Vec::new()),
            workers: RwLock::new(HashMap::new()),
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

    fn ensure_worker(
        &self,
        tp_rank: usize,
        device_id: i32,
    ) -> Result<Arc<WorkerContext>, EngineError> {
        if tp_rank >= self.tp_size {
            return Err(EngineError::InvalidArgument(format!(
                "tp_rank {} exceeds tp_size {}",
                tp_rank, self.tp_size
            )));
        }

        // Fast path: read lock
        {
            let workers = self.workers.read().expect("workers read lock poisoned");
            if let Some(worker) = workers.get(&tp_rank) {
                if worker.device_id != device_id {
                    return Err(EngineError::TopologyMismatch(format!(
                        "worker rank {tp_rank} already bound to device {}, requested {device_id}",
                        worker.device_id
                    )));
                }
                return Ok(Arc::clone(worker));
            }
        }

        // Slow path: write lock
        let mut workers = self.workers.write().expect("workers write lock poisoned");
        // Check again in case another thread inserted it
        if let Some(worker) = workers.get(&tp_rank) {
            if worker.device_id != device_id {
                return Err(EngineError::TopologyMismatch(format!(
                    "worker rank {tp_rank} already bound to device {}, requested {device_id}",
                    worker.device_id
                )));
            }
            return Ok(Arc::clone(worker));
        }

        // Create new worker
        let cuda_ctx = CudaContext::new(device_id as usize)
            .map_err(|e| EngineError::CudaInit(format!("{e:?}")))?;
        let worker = Arc::new(WorkerContext::new(cuda_ctx, device_id));
        workers.insert(tp_rank, Arc::clone(&worker));
        Ok(worker)
    }

    fn get_worker(&self, tp_rank: usize) -> Option<Arc<WorkerContext>> {
        let workers = self.workers.read().expect("workers read lock poisoned");
        workers.get(&tp_rank).cloned()
    }
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        Self::new_with_pool_size(DEFAULT_PINNED_POOL_BYTES)
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    pub fn new_with_pool_size(pool_size: usize) -> Self {
        let storage = StorageEngine::new(pool_size);
        PegaEngine {
            instances: RwLock::new(HashMap::new()),
            storage,
        }
    }

    fn get_or_create_instance(
        &self,
        instance_id: &str,
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
        fields(instance=%instance_id, rank=%tp_rank, layer=%layer_name)
    )]
    pub fn register_context_layer(
        &self,
        instance_id: &str,
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
        if device_id < 0 {
            return Err(EngineError::InvalidArgument(
                "device_id must be >= 0".to_string(),
            ));
        }

        let instance = self.get_or_create_instance(instance_id, num_layers, tp_size)?;
        let worker = instance.ensure_worker(tp_rank, device_id)?;

        // Register layer ID in global instance map
        instance.get_or_create_layer_id(&layer_name);

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        };

        worker.register_layer(layer_name, registration);
        Ok(())
    }

    /// Unregister instance
    #[instrument(level = "info", skip(self))]
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

    /// Attach a shared-memory sync state to a worker.
    ///
    /// The connector creates the sync state and passes the `shm_name` to the server.
    /// The server then attaches to the same shared memory region.
    #[instrument(level = "info", skip(self))]
    pub fn attach_sync_state(
        &self,
        instance_id: &str,
        tp_rank: usize,
        shm_name: &str,
        num_layers: usize,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let worker = instance
            .get_worker(tp_rank)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), tp_rank))?;

        let sync_state = LayerSyncState::attach(shm_name, num_layers)
            .map_err(|e| EngineError::Storage(format!("failed to attach sync state: {e:?}")))?;

        worker.set_sync_state(Arc::new(sync_state));
        info!(
            "Attached sync state for instance {} rank {} (shm={})",
            instance_id, tp_rank, shm_name
        );
        Ok(())
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(instance=%instance_id, rank=%tp_rank, layer=%layer_name, blocks=%block_ids.len())
    )]
    pub fn save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
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

        let instance = self.get_instance(instance_id)?;
        let worker = instance
            .get_worker(tp_rank)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), tp_rank))?;

        let layer_id = instance
            .get_layer_id(layer_name)
            .ok_or_else(|| EngineError::InvalidArgument(format!("layer {layer_name} unknown")))?;

        let registration = worker.get_registration(layer_name).ok_or_else(|| {
            EngineError::InvalidArgument(format!("layer {layer_name} not registered on worker"))
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
            let needs_save = !self.storage.slot_has_block(block_hash, slot_id);

            if needs_save {
                blocks_to_save.push((block_idx, block_hash.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return Ok(());
        }

        info!(
            "Saving {} blocks for layer {layer_name} on instance {instance_id} rank {tp_rank}",
            blocks_to_save.len()
        );

        let block_size = transfer::block_size(&registration).unwrap();
        let num_blocks = blocks_to_save.len();

        // For layer-first layout with KV stride, allocate separate regions for K and V
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;
            let k_total_size = segment_size * num_blocks;
            let v_total_size = segment_size * num_blocks;

            // Allocate separate regions for K and V segments
            let k_allocation = self.storage.allocate(k_total_size);
            let v_allocation = self.storage.allocate(v_total_size);
            let k_base_ptr = k_allocation.as_mut_ptr();
            let v_base_ptr = v_allocation.as_mut_ptr();

            // Calculate GPU offsets for batching
            let mut k_offsets_with_idx = Vec::with_capacity(num_blocks);
            let mut v_offsets_with_idx = Vec::with_capacity(num_blocks);

            for (i, (block_idx, _)) in blocks_to_save.iter().enumerate() {
                let k_offset = transfer::segment_offset(&registration, *block_idx, 0).unwrap();
                let v_offset = transfer::segment_offset(&registration, *block_idx, 1).unwrap();
                k_offsets_with_idx.push((k_offset, i));
                v_offsets_with_idx.push((v_offset, i));
            }

            // Sort by GPU offset to find contiguous ranges
            k_offsets_with_idx.sort_by_key(|&(offset, _)| offset);
            v_offsets_with_idx.sort_by_key(|&(offset, _)| offset);

            // Batch copy K segments
            transfer::batch_copy_segments(
                &k_offsets_with_idx,
                k_base_ptr,
                segment_size,
                &registration,
            )
            .unwrap();

            // Batch copy V segments
            transfer::batch_copy_segments(
                &v_offsets_with_idx,
                v_base_ptr,
                segment_size,
                &registration,
            )
            .unwrap();

            // Create Block objects after all copying is done
            for (i, (_, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let k_ptr = unsafe { k_base_ptr.add(i * segment_size) };
                let v_ptr = unsafe { v_base_ptr.add(i * segment_size) };

                // We now keep K and V data in separate allocations during their lifetime
                // This avoids the memory overwrite bug and keeps data contiguous for better batching next time
                let block = Arc::new(Block::new_split(
                    k_ptr,
                    v_ptr,
                    block_size,
                    Arc::clone(&k_allocation),
                    Arc::clone(&v_allocation),
                ));

                self.storage
                    .insert_block(block_hash, slot_id, block, total_slots);
            }
        } else {
            // Original logic for contiguous or single-segment layouts
            let total_size = block_size * num_blocks;
            let allocation = self.storage.allocate(total_size);
            let base_ptr = allocation.as_mut_ptr();

            // Copy blocks and create Block objects
            for (i, (block_idx, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = unsafe { base_ptr.add(i * block_size) };
                transfer::copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr).unwrap();

                let block = Arc::new(Block::new_contiguous(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&allocation),
                ));

                self.storage
                    .insert_block(block_hash, slot_id, block, total_slots);
            }
        }
        Ok(())
    }

    /// Count how many blocks from the prefix are available in CPU storage
    ///
    /// Returns the number of contiguous blocks available from the start.
    /// Stops counting at the first unavailable block.
    /// Uses the per-block completion status so schedulers only see fully saved blocks.
    ///
    /// Args:
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - usize: Number of contiguous blocks available from the prefix
    #[instrument(
        level = "debug",
        skip(self, block_hashes),
        fields(requested = %block_hashes.len()),
        ret
    )]
    pub fn count_prefix_hit_blocks(&self, block_hashes: &[Vec<u8>]) -> Result<usize, EngineError> {
        let mut hit_count = 0;

        for block_hash in block_hashes.iter() {
            // Storage engine handles "completion" atomically across all layers and TP ranks
            if !self.storage.block_is_complete(block_hash) {
                break;
            }
            hit_count += 1;
        }

        debug!(
            hit_count,
            total = block_hashes.len(),
            "Counted prefix hit blocks"
        );

        Ok(hit_count)
    }

    /// Batch load KV blocks for multiple layers with async completion notification.
    ///
    /// This method:
    /// 1. Looks up all block_hashes in storage ONCE
    /// 2. Submits async transfers for ALL layers
    /// 3. Spawns a background thread that waits for each layer's CUDA event
    ///    and marks the corresponding flag in shared memory sync_state
    ///
    /// The connector can then use `wait_layer()` on the sync_state to wait
    /// for each layer without ZMQ round-trips.
    ///
    /// Args:
    ///   - layer_names: List of layer names to load
    ///   - block_ids: GPU block IDs to load into (shared across all layers)
    ///   - block_hashes: Content hashes for each block (shared across all layers)
    ///
    /// Returns:
    ///   - Vec of (layer_name, bytes_transferred) for each successfully loaded layer
    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(instance=%instance_id, rank=%tp_rank, layers=%layer_names.len(), blocks=%block_ids.len())
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<(String, usize)>, EngineError> {
        let start_time = Instant::now();

        let instance = self.get_instance(instance_id)?;
        let worker = instance
            .get_worker(tp_rank)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), tp_rank))?;

        let stream = worker.stream();
        let sync_state = worker.get_sync_state();

        // Reset sync_state flags if available
        if let Some(ref ss) = sync_state {
            ss.reset();
        }

        // Step 1: Lookup all block_hashes ONCE and cache the blocks
        let shard_blocks_cache = self
            .storage
            .lookup_many(block_hashes)
            .map_err(EngineError::Storage)?;

        // Step 2: Submit async transfers for ALL layers, collect events
        let mut results = Vec::with_capacity(layer_names.len());
        let mut layer_events: Vec<(usize, CudaEvent)> = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_id = match instance.get_layer_id(layer_name) {
                Some(id) => id,
                None => {
                    info!("Layer {} unknown in instance, skipping", layer_name);
                    continue;
                }
            };

            let registration = match worker.get_registration(layer_name) {
                Some(reg) => reg,
                None => {
                    info!("Layer {} not registered on worker, skipping", layer_name);
                    continue;
                }
            };

            let slot_id = instance.get_slot_index(layer_id, tp_rank)?;

            // Collect valid blocks to load for this layer
            let mut blocks_to_load = Vec::with_capacity(block_ids.len());

            for (block_id, shard_blocks_arc) in block_ids.iter().zip(shard_blocks_cache.iter()) {
                let block_idx = *block_id as usize;

                let blocks = shard_blocks_arc.lock_blocks();
                if let Some(block) = blocks.get(slot_id).and_then(|opt| opt.as_ref()) {
                    blocks_to_load.push((block_idx, block.clone()));
                }
            }

            if blocks_to_load.is_empty() {
                // Still record event and mark done immediately if no blocks
                if let Ok(event) = stream.record_event(None) {
                    layer_events.push((layer_id, event));
                }
                continue;
            }

            // Perform async transfer
            let total_transfer;

            // Optimize for layer-first layout with KV stride
            if registration.segments == 2
                && registration.kv_stride_bytes > registration.bytes_per_block
            {
                let segment_size = registration.bytes_per_block;

                // Prepare K and V segments with their GPU destinations
                let mut k_transfers = Vec::with_capacity(blocks_to_load.len());
                let mut v_transfers = Vec::with_capacity(blocks_to_load.len());

                for (block_idx, block) in &blocks_to_load {
                    let k_gpu_offset = match transfer::segment_offset(&registration, *block_idx, 0)
                    {
                        Ok(offset) => offset,
                        Err(e) => {
                            info!("Failed to get K offset for layer {}: {}", layer_name, e);
                            continue;
                        }
                    };
                    let v_gpu_offset = match transfer::segment_offset(&registration, *block_idx, 1)
                    {
                        Ok(offset) => offset,
                        Err(e) => {
                            info!("Failed to get V offset for layer {}: {}", layer_name, e);
                            continue;
                        }
                    };

                    let k_cpu_ptr = block.k_ptr() as *const u8;
                    let v_cpu_ptr = if let Some(v_ptr) = block.v_ptr() {
                        v_ptr as *const u8
                    } else {
                        unsafe { k_cpu_ptr.add(segment_size) }
                    };

                    k_transfers.push((k_gpu_offset, k_cpu_ptr));
                    v_transfers.push((v_gpu_offset, v_cpu_ptr));
                }

                // Batch copy K segments (async)
                if let Err(e) = transfer::batch_copy_segments_to_gpu(
                    &k_transfers,
                    segment_size,
                    &registration,
                    &stream,
                ) {
                    info!("Failed to copy K segments for layer {}: {}", layer_name, e);
                    continue;
                }

                // Batch copy V segments (async)
                if let Err(e) = transfer::batch_copy_segments_to_gpu(
                    &v_transfers,
                    segment_size,
                    &registration,
                    &stream,
                ) {
                    info!("Failed to copy V segments for layer {}: {}", layer_name, e);
                    continue;
                }

                total_transfer = blocks_to_load.len() * segment_size * 2;
            } else {
                // Original logic for contiguous or single-segment layouts
                let mut transfer_size = 0;
                for (block_idx, block) in blocks_to_load {
                    match transfer::copy_block_cpu_to_gpu(
                        &registration,
                        block_idx,
                        block.k_ptr() as *const u8,
                        &stream,
                    ) {
                        Ok(_) => {
                            transfer_size += block.size();
                        }
                        Err(e) => {
                            info!(
                                "Failed to copy block {} for layer {}: {}",
                                block_idx, layer_name, e
                            );
                        }
                    }
                }
                total_transfer = transfer_size;
            }

            // Record event for this layer (after all transfers for this layer are queued)
            match stream.record_event(None) {
                Ok(event) => {
                    layer_events.push((layer_id, event));
                }
                Err(e) => {
                    info!(
                        "Failed to record CUDA event for layer {}: {:?}",
                        layer_name, e
                    );
                }
            }

            results.push((layer_name.to_string(), total_transfer));
        }

        // Step 3: Spawn background thread to wait for events and mark sync_state
        if let Some(sync_state) = sync_state {
            std::thread::spawn(move || {
                let start = std::time::Instant::now();

                for (layer_id, event) in layer_events {
                    // Wait for this layer's transfer to complete
                    if let Err(e) = event.synchronize() {
                        info!("Failed to sync event for layer {}: {:?}", layer_id, e);
                    }
                    // Mark layer as done in shared memory
                    sync_state.mark_layer_done(layer_id);
                }

                let elapsed = start.elapsed();
                info!("All layers synchronized in {:?} ", elapsed);
            });
        }

        let total_elapsed = (Instant::now() - start_time).as_secs_f64();
        info!(
            "batch_load_kv_blocks_multi_layer: submitted {} layers in {:.3}s",
            results.len(),
            total_elapsed
        );

        Ok(results)
    }

    /// Block until the most recent async transfer for a layer finishes.
    pub fn wait_for_layer_transfer(
        &self,
        instance_id: &str,
        tp_rank: usize,
        layer_name: &str,
    ) -> Result<(), EngineError> {
        let instance = self.get_instance(instance_id)?;
        let worker = instance
            .get_worker(tp_rank)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), tp_rank))?;

        let event = worker.take_layer_event(layer_name);

        if let Some(event) = event {
            event.synchronize().map_err(|e| {
                EngineError::Storage(format!("failed to sync layer {layer_name}: {e:?}"))
            })?;
        }
        Ok(())
    }
}
