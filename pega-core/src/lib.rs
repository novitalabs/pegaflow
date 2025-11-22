pub mod allocator;
pub mod pinned_pool;
mod transfer;

pub use pinned_pool::PinnedAllocation;

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
use moka::sync::Cache;
use std::{
    collections::HashMap,
    ptr::NonNull,
    sync::{Arc, Mutex, RwLock},
    time::Instant,
};
use tracing::{debug, info, instrument};

use crate::pinned_pool::PinnedMemoryPool;

const DEFAULT_PINNED_POOL_BYTES: usize = 20 * 1024 * 1024 * 1024; // 10GB
const CACHE_USAGE_RATIO: f64 = 0.90;

type BlockHash = Vec<u8>;

/// A vector of blocks for all layers, indexed by layer_id
/// Each entry is Option<Arc<Block>> to support partial layer storage
type LayerBlocks = Vec<Option<Arc<Block>>>;

/// Wrapper for LayerBlocks with fixed weight for cache eviction
struct LayerBlocksWithWeight {
    blocks: Mutex<LayerBlocks>,
}

impl LayerBlocksWithWeight {
    fn new(num_layers: usize) -> Self {
        Self {
            blocks: Mutex::new(vec![None; num_layers]),
        }
    }

    /// Fixed weight per block hash entry (assumes all layers populated)
    /// This avoids locking during cache weigher calls
    /// FIXME
    fn weight(&self) -> u32 {
        1 // Simple fixed weight, cache will evict based on entry count
    }
}

pub struct PegaEngine {
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    /// Wrapped in RwLock for thread-safe registration with concurrent reads
    kv_caches: RwLock<HashMap<String, KVCacheRegistration>>,
    /// Map layer names to layer IDs for efficient indexing
    /// Wrapped in RwLock for thread-safe registration with concurrent reads
    layer_name_to_id: RwLock<HashMap<String, usize>>,
    /// Ordered list of layer names (layer_id is the index into this vec)
    /// Wrapped in RwLock for thread-safe registration with concurrent reads
    layer_names: RwLock<Vec<String>>,
    /// Store saved KV blocks: block_hash -> Vec<Option<Arc<Block>>> (one per layer)
    /// Cache is already thread-safe, Mutex only protects the LayerBlocks Vec
    kv_storage: Cache<BlockHash, Arc<LayerBlocksWithWeight>>,
    /// Pinned memory pool for zero-copy GPU transfers
    pinned_pool: Arc<PinnedMemoryPool>,
    /// Single stream for all transfers to ensure sequential execution (Layer0 -> Layer1...)
    stream: Arc<CudaStream>,
    /// Track per-layer completion events for async loading
    layer_events: Mutex<HashMap<String, CudaEvent>>,
    _cuda_ctx: Arc<CudaContext>,
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

pub struct Block {
    /// Pointer to K segment (or combined data if contiguous)
    k_ptr: NonNull<u8>,
    /// Pointer to V segment (if stored separately)
    v_ptr: Option<NonNull<u8>>,
    size: usize,
    /// Shared RAII allocation handle for K memory (automatically freed when last reference drops)
    #[allow(dead_code)]
    k_allocation: Arc<PinnedAllocation>,
    /// Shared RAII allocation handle for V memory (if separate from K)
    #[allow(dead_code)]
    v_allocation: Option<Arc<PinnedAllocation>>,
}

impl Block {
    fn new_contiguous(ptr: *mut u8, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        Self {
            k_ptr: unsafe { NonNull::new_unchecked(ptr) },
            v_ptr: None,
            size,
            k_allocation: allocation,
            v_allocation: None,
        }
    }

    fn new_split(
        k_ptr: *mut u8,
        v_ptr: *mut u8,
        size: usize,
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
    ) -> Self {
        Self {
            k_ptr: unsafe { NonNull::new_unchecked(k_ptr) },
            v_ptr: Some(unsafe { NonNull::new_unchecked(v_ptr) }),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
        }
    }

    fn k_ptr(&self) -> *mut u8 {
        self.k_ptr.as_ptr()
    }

    fn v_ptr(&self) -> Option<*mut u8> {
        self.v_ptr.map(|ptr| ptr.as_ptr())
    }

    fn size(&self) -> usize {
        self.size
    }
}

// TODO: Add safety comments, it is a bit tricky
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        Self::new_with_pool_size(DEFAULT_PINNED_POOL_BYTES)
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    pub fn new_with_pool_size(pool_size: usize) -> Self {
        let pinned_pool = Arc::new(PinnedMemoryPool::new(pool_size));
        let cache_capacity = ((pool_size as f64) * CACHE_USAGE_RATIO).floor().max(1.0) as u64;
        // TODO: handle weigher more gracefully
        let kv_storage = Cache::builder()
            .max_capacity(cache_capacity)
            .weigher(|_, layer_blocks_with_weight: &Arc<LayerBlocksWithWeight>| {
                // Use fixed weight to avoid locking
                layer_blocks_with_weight.weight()
            })
            .build();

        // TODO: hard code device 0 for now
        let cuda_ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = cuda_ctx.new_stream().expect("Failed to create stream");

        PegaEngine {
            kv_caches: RwLock::new(HashMap::new()),
            layer_name_to_id: RwLock::new(HashMap::new()),
            layer_names: RwLock::new(Vec::new()),
            kv_storage,
            pinned_pool,
            stream: stream,
            layer_events: Mutex::new(HashMap::new()),
            _cuda_ctx: cuda_ctx,
        }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_kv_cache(
        &self,
        layer_name: String,
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        bytes_per_block: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) {
        if bytes_per_block == 0 || num_blocks == 0 || segments == 0 {
            panic!("Invalid KV cache layout for layer {}", layer_name);
        }

        let registration = KVCacheRegistration {
            data_ptr,
            size_bytes,
            num_blocks,
            bytes_per_block,
            kv_stride_bytes,
            segments,
        };

        // Acquire write locks for registration
        let mut layer_name_to_id = self.layer_name_to_id.write().expect("layer_name_to_id lock poisoned");
        let mut layer_names = self.layer_names.write().expect("layer_names lock poisoned");
        let mut kv_caches = self.kv_caches.write().expect("kv_caches lock poisoned");

        // Assign layer_id if this is a new layer
        if !layer_name_to_id.contains_key(&layer_name) {
            let layer_id = layer_names.len();
            layer_name_to_id.insert(layer_name.clone(), layer_id);
            layer_names.push(layer_name.clone());
        }

        kv_caches.insert(layer_name, registration);
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_all_kv_caches(&self) {
        // Acquire write locks for all registration metadata
        let mut kv_caches = self.kv_caches.write().expect("kv_caches lock poisoned");
        let mut layer_name_to_id = self.layer_name_to_id.write().expect("layer_name_to_id lock poisoned");
        let mut layer_names = self.layer_names.write().expect("layer_names lock poisoned");

        // Clear all registration metadata
        kv_caches.clear();
        layer_name_to_id.clear();
        layer_names.clear();
    }

    /// Allocate pinned memory from the pool. Returns RAII guard. Panics when the allocation cannot be satisfied.
    fn allocate_pinned(&self, size: usize) -> PinnedAllocation {
        self.pinned_pool.allocate(size)
    }

    /// Get the layer_id for a given layer_name
    fn get_layer_id(&self, layer_name: &str) -> Option<usize> {
        let layer_name_to_id = self.layer_name_to_id.read().expect("layer_name_to_id lock poisoned");
        layer_name_to_id.get(layer_name).copied()
    }

    /// Get the total number of layers
    fn num_layers(&self) -> usize {
        let layer_names = self.layer_names.read().expect("layer_names lock poisoned");
        layer_names.len()
    }

    /// Create a new LayerBlocksWithWeight
    fn new_layer_blocks(&self) -> Arc<LayerBlocksWithWeight> {
        Arc::new(LayerBlocksWithWeight::new(self.num_layers()))
    }

    fn record_layer_event(&self, layer_name: &str, event: CudaEvent) {
        let mut guard = self.layer_events.lock().expect("layer events map poisoned");

        guard.insert(layer_name.to_string(), event);
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn save_kv_blocks_from_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) {
        assert_eq!(
            block_ids.len(),
            block_hashes.len(),
            "block_ids and block_hashes must have equal length"
        );

        let layer_id = self
            .get_layer_id(&layer_name)
            .unwrap_or_else(|| panic!("Layer {} not registered", layer_name));

        // Acquire read lock for kv_caches and clone the registration
        let registration = {
            let kv_caches = self.kv_caches.read().expect("kv_caches lock poisoned");
            kv_caches
                .get(&layer_name)
                .cloned()
                .unwrap_or_else(|| panic!("Layer {} not registered", layer_name))
        };

        // Collect blocks that need to be saved
        let mut blocks_to_save = Vec::with_capacity(block_ids.len());

        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            if *block_id < 0 {
                continue;
            }
            let block_idx = *block_id as usize;
            assert!(
                block_idx < registration.num_blocks,
                "Block {} out of range for layer {} ({} blocks registered)",
                block_idx,
                layer_name,
                registration.num_blocks
            );

            // Check if this block_hash already has data for this layer
            let needs_save = if let Some(layer_blocks) = self.kv_storage.get(block_hash) {
                let blocks = layer_blocks.blocks.lock().unwrap();
                blocks.get(layer_id).and_then(|opt| opt.as_ref()).is_none()
            } else {
                true
            };

            if needs_save {
                blocks_to_save.push((block_idx, block_hash.clone()));
            }
        }

        if blocks_to_save.is_empty() {
            return;
        }

        let block_size = transfer::block_size(&registration).unwrap();
        let num_blocks = blocks_to_save.len();

        // For layer-first layout with KV stride, allocate separate regions for K and V
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;
            let k_total_size = segment_size * num_blocks;
            let v_total_size = segment_size * num_blocks;

            // Allocate separate regions for K and V segments
            let k_allocation = self.allocate_pinned(k_total_size);
            let v_allocation = self.allocate_pinned(v_total_size);
            let k_base_ptr = k_allocation.as_mut_ptr();
            let v_base_ptr = v_allocation.as_mut_ptr();
            let k_shared_allocation = Arc::new(k_allocation);
            let v_shared_allocation = Arc::new(v_allocation);

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
                    Arc::clone(&k_shared_allocation),
                    Arc::clone(&v_shared_allocation),
                ));

                // Insert or update the LayerBlocks for this block_hash
                let layer_blocks = self
                    .kv_storage
                    .entry(block_hash)
                    .or_insert_with(|| self.new_layer_blocks())
                    .value()
                    .clone();

                let mut blocks = layer_blocks.blocks.lock().unwrap();
                blocks[layer_id] = Some(block);
            }
        } else {
            // Original logic for contiguous or single-segment layouts
            let total_size = block_size * num_blocks;
            let allocation = self.allocate_pinned(total_size);
            let base_ptr = allocation.as_mut_ptr();
            let shared_allocation = Arc::new(allocation);

            // Copy blocks and create Block objects
            for (i, (block_idx, block_hash)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = unsafe { base_ptr.add(i * block_size) };
                transfer::copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr).unwrap();

                let block = Arc::new(Block::new_contiguous(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&shared_allocation),
                ));

                // Insert or update the LayerBlocks for this block_hash
                let layer_blocks = self
                    .kv_storage
                    .entry(block_hash)
                    .or_insert_with(|| self.new_layer_blocks())
                    .value()
                    .clone();

                let mut blocks = layer_blocks.blocks.lock().unwrap();
                blocks[layer_id] = Some(block);
            }
        }
    }

    /// Count how many blocks from the prefix are available in CPU storage
    ///
    /// Returns the number of contiguous blocks available from the start.
    /// Stops counting at the first unavailable block.
    /// Uses the first registered layer (layer_id = 0) for availability check.
    ///
    /// Args:
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - usize: Number of contiguous blocks available from the prefix
    #[instrument(
        level = "info",
        skip(self, block_hashes),
        fields(requested = %block_hashes.len()),
        ret
    )]
    pub fn count_prefix_hit_blocks(&self, block_hashes: &[Vec<u8>]) -> usize {
        // Use first layer for availability check (layer_id = 0)
        // If no layers registered, all blocks are unavailable
        if self.num_layers() == 0 {
            return 0;
        }

        let layer_id = 0;
        let mut hit_count = 0;

        for block_hash in block_hashes.iter() {
            let available = if let Some(layer_blocks) = self.kv_storage.get(block_hash) {
                let blocks = layer_blocks.blocks.lock().unwrap();
                blocks.get(layer_id).and_then(|opt| opt.as_ref()).is_some()
            } else {
                false
            };

            if !available {
                break;
            }
            hit_count += 1;
        }

        debug!(
            hit_count,
            total = block_hashes.len(),
            "Counted prefix hit blocks"
        );

        hit_count
    }

    /// Batch load KV blocks for multiple layers with shared block mapping
    ///
    /// This method optimizes loading the same blocks across multiple layers by:
    /// 1. Looking up all block_hashes in storage ONCE
    /// 2. For each layer, extracting blocks from the cached LayerBlocks
    /// 3. Performing transfers for each layer
    ///
    /// This reduces hash table lookups from O(layers × blocks) to O(blocks)
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
        fields(layers = %layer_names.len(), blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn batch_load_kv_blocks_multi_layer(
        &self,
        layer_names: &[&str],
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<(String, usize)>, String> {
        let start_time = Instant::now();

        // Step 1: Lookup all block_hashes ONCE and cache the LayerBlocks
        let mut layer_blocks_cache = Vec::with_capacity(block_hashes.len());
        for block_hash in block_hashes {
            let layer_blocks = self
                .kv_storage
                .get(block_hash)
                .ok_or_else(|| format!("Missing KV block hash"))?;
            layer_blocks_cache.push(layer_blocks);
        }

        let lookup_time = Instant::now();
        info!(
            "batch_load_kv_blocks_multi_layer: lookup time: {} us (for {} blocks)",
            (lookup_time - start_time).as_micros(),
            block_hashes.len()
        );

        // Step 2: For each layer, extract blocks and perform transfer
        let mut results = Vec::with_capacity(layer_names.len());

        for layer_name in layer_names {
            let layer_start = Instant::now();

            let layer_id = match self.get_layer_id(layer_name) {
                Some(id) => id,
                None => {
                    info!("Layer {} not registered, skipping", layer_name);
                    continue;
                }
            };

            // Acquire read lock for kv_caches and clone the registration
            let registration = {
                let kv_caches = self.kv_caches.read().expect("kv_caches lock poisoned");
                match kv_caches.get(*layer_name).cloned() {
                    Some(reg) => reg,
                    None => {
                        info!("Layer {} not registered, skipping", layer_name);
                        continue;
                    }
                }
            };

            // Collect valid blocks to load for this layer
            let mut blocks_to_load = Vec::with_capacity(block_ids.len());

            for (block_id, layer_blocks_arc) in block_ids.iter().zip(layer_blocks_cache.iter()) {
                let block_idx = *block_id as usize;

                let blocks = layer_blocks_arc.blocks.lock().unwrap();
                if let Some(block) = blocks.get(layer_id).and_then(|opt| opt.as_ref()) {
                    blocks_to_load.push((block_idx, block.clone()));
                }
            }

            if blocks_to_load.is_empty() {
                info!("No blocks to load for layer {}", layer_name);
                continue;
            }

            // Perform transfer using existing logic
            let mut total_transfer = 0;
            let stream = self.stream.clone();

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
                        // If it was stored contiguously (e.g. old format), V follows K
                        unsafe { k_cpu_ptr.add(segment_size) }
                    };

                    k_transfers.push((k_gpu_offset, k_cpu_ptr));
                    v_transfers.push((v_gpu_offset, v_cpu_ptr));
                }

                // Sort by GPU offset for batching
                k_transfers.sort_by_key(|&(offset, _)| offset);
                v_transfers.sort_by_key(|&(offset, _)| offset);

                // Batch copy K segments
                if let Err(e) = transfer::batch_copy_segments_to_gpu(
                    &k_transfers,
                    segment_size,
                    &registration,
                    &stream,
                ) {
                    info!("Failed to copy K segments for layer {}: {}", layer_name, e);
                    continue;
                }

                // Batch copy V segments
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
                for (block_idx, block) in blocks_to_load {
                    match transfer::copy_block_cpu_to_gpu(
                        &registration,
                        block_idx,
                        block.k_ptr() as *const u8,
                        &stream,
                    ) {
                        Ok(_) => {
                            total_transfer += block.size();
                        }
                        Err(e) => {
                            info!(
                                "Failed to copy block {} for layer {}: {}",
                                block_idx, layer_name, e
                            );
                        }
                    }
                }
            }

            // Record event for this layer
            match stream.record_event(None) {
                Ok(event) => {
                    self.record_layer_event(layer_name, event);
                }
                Err(e) => {
                    info!(
                        "Failed to record CUDA event for layer {}: {:?}",
                        layer_name, e
                    );
                }
            }

            let layer_elapsed = (Instant::now() - layer_start).as_secs_f64();
            let bandwidth = if layer_elapsed > 0.0 {
                total_transfer as f64 / layer_elapsed
            } else {
                0.0
            };
            debug!(
                layer = layer_name,
                total_transfer,
                elapsed_us = (Instant::now() - layer_start).as_micros(),
                bandwidth_gbps = bandwidth / 1e9,
                "Completed layer transfer"
            );

            results.push((layer_name.to_string(), total_transfer));
        }

        let total_elapsed = (Instant::now() - start_time).as_secs_f64();
        info!(
            "batch_load_kv_blocks_multi_layer: completed {} layers in {:.3}s",
            results.len(),
            total_elapsed
        );

        Ok(results)
    }

    /// Block until the most recent async transfer for a layer finishes.
    pub fn wait_for_layer_transfer(&self, layer_name: &str) -> Result<(), String> {
        let event = {
            let mut guard = self.layer_events.lock().expect("layer events map poisoned");
            guard.remove(layer_name)
        };

        if let Some(event) = event {
            event
                .synchronize()
                .map_err(|e| format!("Failed to sync layer {layer_name}: {e:?}"))?;
        }
        Ok(())
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: PegaEngine can be safely sent between threads
// - PinnedMemoryPool owns the CUDA allocation
// - CUDA context is thread-safe (Arc<CudaContext>)
unsafe impl Send for PegaEngine {}
unsafe impl Sync for PegaEngine {}
