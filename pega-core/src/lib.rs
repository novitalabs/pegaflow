pub mod allocator;
pub mod pinned_pool;

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
// ============================================================================

use std::{collections::HashMap, sync::Arc, time::Instant};

use allocator::Allocation;
use moka::sync::Cache;
use tracing::{debug, info, instrument};

use crate::pinned_pool::PinnedMemoryPool;

const DEFAULT_PINNED_POOL_BYTES: usize = 10 * 1024 * 1024 * 1024; // 10GB
const CACHE_USAGE_RATIO: f64 = 0.90;

type BlockKey = (String, Vec<u8>);

struct PendingBlock {
    block_idx: usize,
    key: BlockKey,
    slot: usize,
}

pub struct PegaEngine {
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    kv_caches: HashMap<String, KVCacheRegistration>,
    /// Store saved KV blocks (layer_name, block_hash) -> block data
    kv_storage: Cache<BlockKey, Arc<Block>>,
    /// Pinned memory pool for zero-copy GPU transfers
    pinned_pool: Arc<PinnedMemoryPool>,
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
    /// Pointer to pinned memory (not owned, managed by PegaEngine's pool)
    ptr: *mut u8,
    size: usize,
    /// Allocation handle for freeing memory (shared across multiple blocks)
    allocation: Arc<Allocation>,
    pool: Arc<PinnedMemoryPool>,
}

impl Block {
    fn new(
        ptr: *mut u8,
        size: usize,
        allocation: Arc<Allocation>,
        pool: Arc<PinnedMemoryPool>,
    ) -> Self {
        Self {
            ptr,
            size,
            allocation,
            pool,
        }
    }

    fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    fn size(&self) -> usize {
        self.size
    }

    fn weight(&self) -> u32 {
        u32::try_from(self.size)
            .expect("KV block larger than u32::MAX bytes is not supported for caching")
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        // Only free when this is the last reference to the allocation
        if Arc::strong_count(&self.allocation) == 1 {
            self.pool.free(*self.allocation);
        }
    }
}

// Safety: Block wraps a read-only pinned allocation; freeing is handled through its Drop.
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
        let kv_storage = Cache::builder()
            .max_capacity(cache_capacity)
            .weigher(|_, block: &Arc<Block>| block.weight())
            .build();

        PegaEngine {
            kv_caches: HashMap::new(),
            kv_storage,
            pinned_pool,
        }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "debug",
        skip(self),
        fields(layer = %layer_name, size_bytes, num_blocks, bytes_per_block)
    )]
    pub fn register_kv_cache(
        &mut self,
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

        self.kv_caches.insert(layer_name, registration);
    }

    /// Unregister all KV cache handles
    #[instrument(level = "info", skip(self))]
    pub fn unregister_all_kv_caches(&mut self) {
        self.kv_caches.clear();
    }

    /// Get the number of registered KV caches
    #[instrument(level = "debug", skip(self), ret)]
    pub fn num_registered_kv_caches(&self) -> usize {
        self.kv_caches.len()
    }

    /// Allocate pinned memory from the pool. Panics when the allocation cannot be satisfied.
    fn allocate_pinned(&self, size: usize) -> (Allocation, *mut u8) {
        self.pinned_pool.allocate(size)
    }

    /// Get pinned memory usage statistics
    pub fn get_pinned_memory_usage(&self) -> (usize, usize) {
        self.pinned_pool.usage()
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
    )]
    pub fn save_kv_blocks_from_ipc(
        &mut self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) {
        assert_eq!(
            block_ids.len(),
            block_hashes.len(),
            "block_ids and block_hashes must have equal length"
        );

        let registration = self
            .kv_caches
            .get(&layer_name)
            .unwrap_or_else(|| panic!("Layer {} not registered", layer_name));

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

            let key = (layer_name.clone(), block_hash.clone());
            if !self.kv_storage.contains_key(&key) {
                blocks_to_save.push((block_idx, key));
            }
        }

        if blocks_to_save.is_empty() {
            return;
        }

        let block_size = self.block_size(&registration).unwrap();
        let num_blocks = blocks_to_save.len();

        // For layer-first layout with KV stride, allocate separate regions for K and V
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;
            let k_total_size = segment_size * num_blocks;
            let v_total_size = segment_size * num_blocks;

            // Allocate separate regions for K and V segments
            let (k_allocation, k_base_ptr) = self.allocate_pinned(k_total_size);
            let (_v_allocation, v_base_ptr) = self.allocate_pinned(v_total_size);
            let k_shared_allocation = Arc::new(k_allocation);
            let _v_shared_allocation = Arc::new(_v_allocation);

            // Calculate GPU offsets for batching
            let mut k_offsets_with_idx = Vec::with_capacity(num_blocks);
            let mut v_offsets_with_idx = Vec::with_capacity(num_blocks);

            for (i, (block_idx, _)) in blocks_to_save.iter().enumerate() {
                let k_offset = self.segment_offset(&registration, *block_idx, 0).unwrap();
                let v_offset = self.segment_offset(&registration, *block_idx, 1).unwrap();
                k_offsets_with_idx.push((k_offset, i));
                v_offsets_with_idx.push((v_offset, i));
            }

            // Sort by GPU offset to find contiguous ranges
            k_offsets_with_idx.sort_by_key(|&(offset, _)| offset);
            v_offsets_with_idx.sort_by_key(|&(offset, _)| offset);

            // Batch copy K segments
            self.batch_copy_segments(&k_offsets_with_idx, k_base_ptr, segment_size, &registration)
                .unwrap();

            // Batch copy V segments
            self.batch_copy_segments(&v_offsets_with_idx, v_base_ptr, segment_size, &registration)
                .unwrap();

            // Create Block objects after all copying is done
            for (i, (_, key)) in blocks_to_save.into_iter().enumerate() {
                let k_ptr = unsafe { k_base_ptr.add(i * segment_size) };
                let v_ptr = unsafe { v_base_ptr.add(i * segment_size) };

                // Create a single block that spans both K and V data
                // We'll store them contiguously: K data followed by V data
                let combined_ptr = k_ptr; // Start with K data
                unsafe {
                    // Copy V data right after K data
                    std::ptr::copy_nonoverlapping(v_ptr, k_ptr.add(segment_size), segment_size);
                }

                let block = Arc::new(Block::new(
                    combined_ptr,
                    block_size,
                    Arc::clone(&k_shared_allocation), // Use K allocation for the combined block
                    Arc::clone(&self.pinned_pool),
                ));
                self.kv_storage.insert(key, block);
            }
        } else {
            // Original logic for contiguous or single-segment layouts
            let total_size = block_size * num_blocks;
            let (allocation, base_ptr) = self.allocate_pinned(total_size);
            let shared_allocation = Arc::new(allocation);

            // Copy blocks and create Block objects
            for (i, (block_idx, key)) in blocks_to_save.into_iter().enumerate() {
                let cpu_ptr = unsafe { base_ptr.add(i * block_size) };
                self.copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr)
                    .unwrap();

                let block = Arc::new(Block::new(
                    cpu_ptr,
                    block_size,
                    Arc::clone(&shared_allocation),
                    Arc::clone(&self.pinned_pool),
                ));
                self.kv_storage.insert(key, block);
            }
        }
    }

    /// Copy data from GPU to CPU
    #[instrument(level = "debug", skip(self, cpu_buffer), fields(offset, size), err)]
    fn copy_gpu_to_cpu(
        &self,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &mut [u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        let src_ptr = gpu_base_ptr + offset as u64;
        let dst_ptr = cpu_buffer.as_mut_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyDtoH_v2(dst_ptr as *mut std::ffi::c_void, src_ptr, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH failed: {:?}", result));
            }
        }

        Ok(())
    }

    /// Get storage statistics
    /// Returns (num_blocks, total_bytes)
    #[instrument(level = "info", skip(self), ret)]
    pub fn get_storage_stats(&self) -> (usize, usize) {
        let num_blocks = usize::try_from(self.kv_storage.entry_count()).unwrap_or(usize::MAX);
        let total_bytes = usize::try_from(self.kv_storage.weighted_size()).unwrap_or(usize::MAX);
        (num_blocks, total_bytes)
    }

    /// Remove a KV block and free its memory
    #[instrument(level = "info", skip(self, block_hash), fields(layer = %layer_name))]
    pub fn remove_kv_block(&mut self, layer_name: String, block_hash: Vec<u8>) -> bool {
        let key = (layer_name, block_hash);
        if self.kv_storage.contains_key(&key) {
            self.kv_storage.invalidate(&key);
            self.kv_storage.run_pending_tasks();
            true
        } else {
            false
        }
    }

    /// Clear all stored KV blocks and free their memory
    #[instrument(level = "info", skip(self))]
    pub fn clear_all_kv_blocks(&mut self) {
        self.kv_storage.invalidate_all();
        self.kv_storage.run_pending_tasks();
    }

    /// Check which KV blocks are available in CPU storage
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_hashes: List of block hashes to check
    ///
    /// Returns:
    ///   - Vec<bool>: For each hash, true if available in storage
    #[instrument(
        level = "debug",
        skip(self, block_hashes),
        fields(layer = %layer_name, requested = %block_hashes.len()),
        ret
    )]
    pub fn check_kv_blocks_availability(
        &self,
        layer_name: String,
        block_hashes: Vec<Vec<u8>>,
    ) -> Vec<bool> {
        let mut availability = Vec::with_capacity(block_hashes.len());

        for (idx, block_hash) in block_hashes.iter().enumerate() {
            let key = (layer_name.clone(), block_hash.clone());
            let available = self.kv_storage.contains_key(&key);
            availability.push(available);

            let hash_preview: Vec<u8> = block_hash.iter().copied().take(8).collect();
            debug!(
                block_index = idx,
                available,
                hash_prefix = ?hash_preview,
                "Checked KV block availability"
            );
        }

        let num_available = availability.iter().filter(|&&x| x).count();
        debug!(
            num_available,
            total = block_hashes.len(),
            "Completed KV block availability check"
        );

        availability
    }

    /// Load KV blocks from CPU memory to GPU via IPC handle
    ///
    /// Args:
    ///   - layer_name: Name of the layer
    ///   - block_ids: GPU block IDs to load into
    ///   - block_hashes: Content hashes for each block
    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn load_kv_blocks_to_ipc(
        &self,
        layer_name: &str,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<usize, String> {
        let start_time = Instant::now();
        if block_ids.len() != block_hashes.len() {
            return Err("block_ids and block_hashes must have equal length".into());
        }

        let Some(registration) = self.kv_caches.get(layer_name) else {
            return Err(format!("Layer {} not registered", layer_name));
        };

        // Collect valid blocks to load
        let mut blocks_to_load = Vec::new();
        for (block_id, block_hash) in block_ids.iter().zip(block_hashes.iter()) {
            if *block_id < 0 {
                continue;
            }

            let block_idx = *block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(format!(
                    "Block {} out of range for layer {} ({} blocks registered)",
                    block_idx, layer_name, registration.num_blocks
                ));
            }

            let key = (layer_name.to_string(), block_hash.clone());
            let Some(block) = self.kv_storage.get(&key) else {
                return Err(format!("Missing KV block for layer {}", layer_name));
            };

            let expected_size = registration
                .bytes_per_block
                .checked_mul(registration.segments)
                .ok_or_else(|| "Stored block size overflow".to_string())?;
            if block.size() != expected_size {
                return Err(format!(
                    "Stored block size mismatch for layer {}: {} vs {}",
                    layer_name,
                    block.size(),
                    expected_size
                ));
            }

            blocks_to_load.push((block_idx, block));
        }

        let mut total_transfer = 0;

        // Optimize for layer-first layout with KV stride
        if registration.segments == 2 && registration.kv_stride_bytes > registration.bytes_per_block
        {
            let segment_size = registration.bytes_per_block;

            // Prepare K and V segments with their GPU destinations
            let mut k_transfers = Vec::with_capacity(blocks_to_load.len());
            let mut v_transfers = Vec::with_capacity(blocks_to_load.len());

            for (block_idx, block) in &blocks_to_load {
                let k_gpu_offset = self.segment_offset(&registration, *block_idx, 0)?;
                let v_gpu_offset = self.segment_offset(&registration, *block_idx, 1)?;

                // K segment is at the beginning of stored block, V segment follows
                let k_cpu_ptr = block.ptr() as *const u8;
                let v_cpu_ptr = unsafe { k_cpu_ptr.add(segment_size) };

                k_transfers.push((k_gpu_offset, k_cpu_ptr));
                v_transfers.push((v_gpu_offset, v_cpu_ptr));
            }

            // Sort by GPU offset for batching
            k_transfers.sort_by_key(|&(offset, _)| offset);
            v_transfers.sort_by_key(|&(offset, _)| offset);

            // Batch copy K segments
            self.batch_copy_segments_to_gpu(&k_transfers, segment_size, &registration)?;

            // Batch copy V segments
            self.batch_copy_segments_to_gpu(&v_transfers, segment_size, &registration)?;

            total_transfer = blocks_to_load.len() * segment_size * 2;
        } else {
            // Original logic for contiguous or single-segment layouts
            for (block_idx, block) in blocks_to_load {
                self.copy_block_cpu_to_gpu(&registration, block_idx, block.ptr() as *const u8)?;
                total_transfer += block.size();
            }
        }

        let end_time = Instant::now();
        // print cost
        let elapsed = (end_time - start_time).as_secs_f64();
        let bandwidth = if elapsed > 0.0 {
            total_transfer as f64 / elapsed
        } else {
            0.0
        };
        info!(
            total_transfer,
            elapsed_us = (end_time - start_time).as_micros(),
            bandwidth_gbps = bandwidth / 1e9,
            "Completed load_kv_blocks_to_ipc"
        );
        Ok(total_transfer)
    }

    /// Calculate the byte offset for a given block/segment combination.
    fn segment_offset(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        segment_idx: usize,
    ) -> Result<usize, String> {
        if segment_idx >= registration.segments {
            return Err("Segment index out of range".to_string());
        }

        let base = block_idx
            .checked_mul(registration.bytes_per_block)
            .ok_or_else(|| "Block offset overflow".to_string())?;

        let segment_offset = segment_idx
            .checked_mul(registration.kv_stride_bytes)
            .ok_or_else(|| "Segment offset overflow".to_string())?;

        let offset = base
            .checked_add(segment_offset)
            .ok_or_else(|| "Combined offset overflow".to_string())?;

        if offset + registration.bytes_per_block > registration.size_bytes {
            return Err(format!(
                "Block {} segment {} exceeds registered memory (offset {}, size {}, limit {})",
                block_idx,
                segment_idx,
                offset,
                registration.bytes_per_block,
                registration.size_bytes
            ));
        }

        Ok(offset)
    }

    /// Copy data from CPU to GPU
    #[instrument(level = "debug", skip(self, cpu_buffer), fields(offset, size), err)]
    fn copy_cpu_to_gpu(
        &self,
        gpu_base_ptr: u64,
        offset: usize,
        cpu_buffer: &[u8],
        size: usize,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        if cpu_buffer.len() < size {
            return Err(format!(
                "CPU buffer too small: {} bytes, need {} bytes",
                cpu_buffer.len(),
                size
            ));
        }

        let dst_ptr = gpu_base_ptr + offset as u64;
        let src_ptr = cpu_buffer.as_ptr();

        unsafe {
            // Use synchronous copy for simplicity
            let result = sys::cuMemcpyHtoD_v2(dst_ptr, src_ptr as *const std::ffi::c_void, size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD failed: {:?}", result));
            }
        }

        Ok(())
    }

    /// Batch copy segments by finding and merging contiguous ranges
    fn batch_copy_segments(
        &self,
        offsets_with_idx: &[(usize, usize)],
        base_ptr: *mut u8,
        segment_size: usize,
        registration: &KVCacheRegistration,
    ) -> Result<(), String> {
        let total_segments = offsets_with_idx.len();
        let mut batch_count = 0;
        let mut i = 0;

        while i < offsets_with_idx.len() {
            let (start_offset, start_idx) = offsets_with_idx[i];
            let mut _end_idx = start_idx;
            let mut count = 1;

            // Find contiguous range
            for j in i + 1..offsets_with_idx.len() {
                let (offset, idx) = offsets_with_idx[j];
                if offset == start_offset + count * segment_size {
                    _end_idx = idx;
                    count += 1;
                } else {
                    break;
                }
            }

            // Perform batch copy for this contiguous range
            let total_size = count * segment_size;
            let cpu_ptr = unsafe { base_ptr.add(start_idx * segment_size) };
            let buffer = unsafe { std::slice::from_raw_parts_mut(cpu_ptr, total_size) };
            self.copy_gpu_to_cpu(registration.data_ptr, start_offset, buffer, total_size)?;

            batch_count += 1;
            i += count;
        }

        debug!(
            "GPU->CPU batch copy: {} segments -> {} batches ({}x reduction)",
            total_segments,
            batch_count,
            total_segments as f32 / batch_count as f32
        );

        Ok(())
    }

    /// Batch copy segments from CPU to GPU by finding and merging contiguous ranges
    fn batch_copy_segments_to_gpu(
        &self,
        transfers: &[(usize, *const u8)],
        segment_size: usize,
        registration: &KVCacheRegistration,
    ) -> Result<(), String> {
        let total_segments = transfers.len();
        let mut batch_count = 0;
        let mut i = 0;

        while i < transfers.len() {
            let (start_gpu_offset, start_cpu_ptr) = transfers[i];
            let mut count = 1;

            // Find contiguous range in GPU memory
            for j in i + 1..transfers.len() {
                let (gpu_offset, _) = transfers[j];
                if gpu_offset == start_gpu_offset + count * segment_size {
                    count += 1;
                } else {
                    break;
                }
            }

            if count > 1 {
                // Check if CPU pointers are also contiguous
                let cpu_contiguous = (1..count).all(|k| {
                    let (_, cpu_ptr) = transfers[i + k];
                    let expected_ptr = unsafe { start_cpu_ptr.add(k * segment_size) };
                    cpu_ptr == expected_ptr
                });

                if cpu_contiguous {
                    // Batch copy contiguous range
                    let total_size = count * segment_size;
                    let buffer = unsafe { std::slice::from_raw_parts(start_cpu_ptr, total_size) };
                    self.copy_cpu_to_gpu(
                        registration.data_ptr,
                        start_gpu_offset,
                        buffer,
                        total_size,
                    )?;
                    batch_count += 1;
                    i += count;
                    continue;
                }
            }

            // Copy individual segment
            let buffer = unsafe { std::slice::from_raw_parts(start_cpu_ptr, segment_size) };
            self.copy_cpu_to_gpu(
                registration.data_ptr,
                start_gpu_offset,
                buffer,
                segment_size,
            )?;
            batch_count += 1;
            i += 1;
        }

        debug!(
            "CPU->GPU batch copy: {} segments -> {} batches ({}x reduction)",
            total_segments,
            batch_count,
            total_segments as f32 / batch_count as f32
        );

        Ok(())
    }

    fn block_size(&self, registration: &KVCacheRegistration) -> Result<usize, String> {
        registration
            .bytes_per_block
            .checked_mul(registration.segments)
            .ok_or_else(|| "Block size overflow".to_string())
    }

    fn is_contiguous_layout(registration: &KVCacheRegistration) -> bool {
        registration.segments <= 1 || registration.kv_stride_bytes == registration.bytes_per_block
    }

    fn copy_block_gpu_to_cpu(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        dst_ptr: *mut u8,
    ) -> Result<(), String> {
        if Self::is_contiguous_layout(registration) {
            let block_size = self.block_size(registration)?;
            let offset = self.segment_offset(registration, block_idx, 0)?;
            let buffer = unsafe { std::slice::from_raw_parts_mut(dst_ptr, block_size) };
            self.copy_gpu_to_cpu(registration.data_ptr, offset, buffer, block_size)
        } else {
            self.copy_gpu_to_cpu_strided(registration, block_idx, dst_ptr)
        }
    }

    fn copy_block_cpu_to_gpu(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        src_ptr: *const u8,
    ) -> Result<(), String> {
        if Self::is_contiguous_layout(registration) {
            let block_size = self.block_size(registration)?;
            let offset = self.segment_offset(registration, block_idx, 0)?;
            let buffer = unsafe { std::slice::from_raw_parts(src_ptr, block_size) };
            self.copy_cpu_to_gpu(registration.data_ptr, offset, buffer, block_size)
        } else {
            self.copy_cpu_to_gpu_strided(registration, block_idx, src_ptr)
        }
    }

    fn copy_gpu_to_cpu_strided(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        dst_ptr: *mut u8,
    ) -> Result<(), String> {
        // Copy each segment separately using regular cuMemcpy
        for segment_idx in 0..registration.segments {
            let src_offset = self.segment_offset(registration, block_idx, segment_idx)?;
            let dst_offset = segment_idx * registration.bytes_per_block;
            let dst_segment_ptr = unsafe { dst_ptr.add(dst_offset) };
            let buffer = unsafe {
                std::slice::from_raw_parts_mut(dst_segment_ptr, registration.bytes_per_block)
            };

            self.copy_gpu_to_cpu(
                registration.data_ptr,
                src_offset,
                buffer,
                registration.bytes_per_block,
            )?;
        }
        Ok(())
    }

    fn copy_cpu_to_gpu_strided(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        src_ptr: *const u8,
    ) -> Result<(), String> {
        // Copy each segment separately using regular cuMemcpy
        for segment_idx in 0..registration.segments {
            let dst_offset = self.segment_offset(registration, block_idx, segment_idx)?;
            let src_offset = segment_idx * registration.bytes_per_block;
            let src_segment_ptr = unsafe { src_ptr.add(src_offset) };
            let buffer = unsafe {
                std::slice::from_raw_parts(src_segment_ptr, registration.bytes_per_block)
            };

            self.copy_cpu_to_gpu(
                registration.data_ptr,
                dst_offset,
                buffer,
                registration.bytes_per_block,
            )?;
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_cudarc_basic() {
        // Get a stream for GPU 0
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // copy a rust slice to the device
        let _inp = stream.clone_htod(&[1.0f32; 100]).unwrap();

        // or allocate directly
        let _out = stream.alloc_zeros::<f32>(100).unwrap();
    }

    #[test]
    fn test_gpu_to_cpu_copy() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Allocate and initialize data on GPU
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let gpu_data = stream.clone_htod(&test_data).unwrap();

        // 3. Copy from GPU to CPU
        let cpu_data: Vec<f32> = stream.clone_dtoh(&gpu_data).unwrap();

        // 4. Verify the data
        assert_eq!(cpu_data, test_data);
        println!("GPU->CPU copy test passed! Data: {:?}", cpu_data);
    }

    #[test]
    fn test_gpu_to_cpu_copy_bf16() {
        // 1. Create context and stream
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        // 2. Simulate KV cache block: [2, 16, 12, 64] * bf16 (2 bytes)
        let block_size = 2 * 16 * 12 * 64;
        let test_data: Vec<u8> = (0..block_size).map(|i| (i % 256) as u8).collect();

        // 3. Copy to GPU
        let gpu_block = stream.clone_htod(&test_data).unwrap();

        // 4. Copy back to CPU
        let cpu_block: Vec<u8> = stream.clone_dtoh(&gpu_block).unwrap();

        // 5. Verify
        assert_eq!(cpu_block.len(), block_size);
        assert_eq!(cpu_block, test_data);
        println!(
            "GPU->CPU BF16 block copy test passed! Block size: {} bytes",
            block_size
        );
    }
}
