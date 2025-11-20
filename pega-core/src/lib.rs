pub mod allocator;

use std::{collections::HashMap, sync::Mutex, time::Instant};

use allocator::{Allocation, ScaledOffsetAllocator};
use tracing::{debug, info, instrument};

const DEFAULT_PINNED_POOL_BYTES: usize = 10 * 1024 * 1024 * 1024; // 10GB

pub struct PegaEngine {
    /// Store registered KV cache pointers (new IPC wrapper): layer_name -> registration
    kv_caches: HashMap<String, KVCacheRegistration>,
    /// Store saved KV blocks: (layer_name, block_hash) -> block data
    kv_storage: HashMap<(String, Vec<u8>), Block>,
    /// Pinned memory pool for zero-copy GPU transfers
    pinned_pool_ptr: *mut u8,
    /// Allocator for managing pinned memory
    allocator: Mutex<ScaledOffsetAllocator>,
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

#[derive(Clone)]
pub struct Block {
    /// Pointer to pinned memory (not owned, managed by PegaEngine's pool)
    pub ptr: *mut u8,
    pub size: usize,
    /// Allocation handle for freeing memory
    allocation: Allocation,
}

impl PegaEngine {
    /// Create a new PegaEngine instance
    #[instrument(level = "info")]
    pub fn new() -> Self {
        Self::new_with_pool_size(DEFAULT_PINNED_POOL_BYTES)
    }

    /// Create a new PegaEngine instance with a custom pinned memory pool size
    pub fn new_with_pool_size(pool_size: usize) -> Self {
        use cudarc::driver::sys;

        if pool_size == 0 {
            panic!("Pinned memory pool size must be greater than zero");
        }

        let mut pool_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        unsafe {
            let result = sys::cuMemAllocHost_v2(&mut pool_ptr, pool_size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                panic!("Failed to allocate pinned memory pool: {:?}", result);
            }
        }

        // Create allocator for managing the pinned memory pool
        let allocator = ScaledOffsetAllocator::new(pool_size as u64)
            .expect("Failed to create memory allocator");

        info!(
            "Allocated pinned memory pool: {} GB ({} bytes)",
            pool_size as f64 / 1e9,
            pool_size
        );

        PegaEngine {
            kv_caches: HashMap::new(),
            kv_storage: HashMap::new(),
            pinned_pool_ptr: pool_ptr as *mut u8,
            allocator: Mutex::new(allocator),
        }
    }

    /// Register a KV cache region with its layout info
    #[instrument(
        level = "info",
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
        if size == 0 {
            panic!("Cannot allocate zero bytes from the pinned pool");
        }

        let mut allocator = self.allocator.lock().unwrap();

        let allocation = match allocator.allocate(size as u64) {
            Ok(Some(allocation)) => allocation,
            Ok(None) => {
                let report = allocator.storage_report();
                panic!(
                    "Pinned memory pool exhausted! Requested: {:.2} MB, Free: {:.2} MB, Largest: {:.2} MB",
                    size as f64 / 1e6,
                    report.total_free_bytes as f64 / 1e6,
                    report.largest_free_allocation_bytes as f64 / 1e6
                );
            }
            Err(err) => panic!("Pinned memory allocation error: {}", err),
        };

        let ptr = unsafe { self.pinned_pool_ptr.add(allocation.offset_bytes as usize) };
        (allocation, ptr)
    }

    /// Free pinned memory allocation
    fn free_pinned(&self, allocation: Allocation) {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(allocation);
    }

    /// Get pinned memory usage statistics
    pub fn get_pinned_memory_usage(&self) -> (usize, usize) {
        let allocator = self.allocator.lock().unwrap();
        let report = allocator.storage_report();
        let total = allocator.total_bytes() as usize;
        let used = total - report.total_free_bytes as usize;
        (used, total)
    }

    #[instrument(
        level = "debug",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn save_kv_blocks_from_ipc(
        &mut self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        if block_ids.len() != block_hashes.len() {
            return Err("block_ids and block_hashes must have equal length".into());
        }

        let Some(registration) = self.kv_caches.get(&layer_name) else {
            return Err(format!("Layer {} not registered", layer_name));
        };

        for (block_id, block_hash) in block_ids.into_iter().zip(block_hashes.into_iter()) {
            if block_id < 0 {
                continue;
            }
            let block_idx = block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(format!(
                    "Block {} out of range for layer {} ({} blocks registered)",
                    block_idx, layer_name, registration.num_blocks
                ));
            }

            // Skip if already stored
            if self
                .kv_storage
                .contains_key(&(layer_name.clone(), block_hash.clone()))
            {
                continue;
            }

            let block_size = self.block_size(&registration)?;
            let (allocation, cpu_ptr) = self.allocate_pinned(block_size);

            self.copy_block_gpu_to_cpu(&registration, block_idx, cpu_ptr)?;

            info!("insert key {}-{:?} to kv_storage", layer_name, block_hash);

            self.kv_storage.insert(
                (layer_name.clone(), block_hash),
                Block {
                    ptr: cpu_ptr,
                    size: block_size,
                    allocation,
                },
            );
        }

        Ok(())
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
        let num_blocks = self.kv_storage.len();
        let total_bytes: usize = self.kv_storage.values().map(|block| block.size).sum();
        (num_blocks, total_bytes)
    }

    /// Remove a KV block and free its memory
    #[instrument(level = "info", skip(self, block_hash), fields(layer = %layer_name))]
    pub fn remove_kv_block(&mut self, layer_name: String, block_hash: Vec<u8>) -> bool {
        let key = (layer_name, block_hash);
        if let Some(block) = self.kv_storage.remove(&key) {
            self.free_pinned(block.allocation);
            true
        } else {
            false
        }
    }

    /// Clear all stored KV blocks and free their memory
    #[instrument(level = "info", skip(self))]
    pub fn clear_all_kv_blocks(&mut self) {
        let allocations: Vec<Allocation> = self
            .kv_storage
            .drain()
            .map(|(_, block)| block.allocation)
            .collect();

        for allocation in allocations {
            self.free_pinned(allocation);
        }
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
        level = "info",
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
        level = "info",
        skip(self, block_ids, block_hashes),
        fields(layer = %layer_name, blocks = %block_ids.len(), hashes = %block_hashes.len()),
        err
    )]
    pub fn load_kv_blocks_to_ipc(
        &self,
        layer_name: String,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
    ) -> Result<(), String> {
        let start_time = Instant::now();
        if block_ids.len() != block_hashes.len() {
            return Err("block_ids and block_hashes must have equal length".into());
        }

        let Some(registration) = self.kv_caches.get(&layer_name) else {
            return Err(format!("Layer {} not registered", layer_name));
        };

        let mut total_transfer = 0;
        for (block_id, block_hash) in block_ids.into_iter().zip(block_hashes.into_iter()) {
            if block_id < 0 {
                continue;
            }

            let block_idx = block_id as usize;
            if block_idx >= registration.num_blocks {
                return Err(format!(
                    "Block {} out of range for layer {} ({} blocks registered)",
                    block_idx, layer_name, registration.num_blocks
                ));
            }

            let key = (layer_name.clone(), block_hash.clone());
            let Some(block) = self.kv_storage.get(&key) else {
                return Err(format!("Missing KV block for layer {}", layer_name));
            };

            let expected_size = registration
                .bytes_per_block
                .checked_mul(registration.segments)
                .ok_or_else(|| "Stored block size overflow".to_string())?;
            if block.size != expected_size {
                return Err(format!(
                    "Stored block size mismatch for layer {}: {} vs {}",
                    layer_name, block.size, expected_size
                ));
            }

            // Copy each segment from pinned memory to GPU
            self.copy_block_cpu_to_gpu(&registration, block_idx, block.ptr)?;
            total_transfer += block.size;
        }

        let end_time = Instant::now();
        // print cost
        info!(
            "load_kv_blocks_to_ipc: total_transfer = {} bytes, time = {} us",
            total_transfer,
            (end_time - start_time).as_micros()
        );
        Ok(())
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
        use cudarc::driver::sys;

        if registration.kv_stride_bytes == 0 {
            return Err("Invalid KV stride for strided copy".into());
        }

        let offset = self.segment_offset(registration, block_idx, 0)?;
        let device_ptr = registration.data_ptr + offset as u64;
        let request = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
            srcHost: std::ptr::null(),
            srcDevice: device_ptr as sys::CUdeviceptr,
            srcArray: std::ptr::null_mut(),
            srcPitch: registration.kv_stride_bytes,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_HOST,
            dstHost: dst_ptr as *mut std::ffi::c_void,
            dstDevice: 0,
            dstArray: std::ptr::null_mut(),
            dstPitch: registration.bytes_per_block,
            WidthInBytes: registration.bytes_per_block,
            Height: registration.segments,
        };

        unsafe {
            let result = sys::cuMemcpy2D_v2(&request);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpy2D (DtoH) failed: {:?}", result));
            }
        }

        Ok(())
    }

    fn copy_cpu_to_gpu_strided(
        &self,
        registration: &KVCacheRegistration,
        block_idx: usize,
        src_ptr: *const u8,
    ) -> Result<(), String> {
        use cudarc::driver::sys;

        if registration.kv_stride_bytes == 0 {
            return Err("Invalid KV stride for strided copy".into());
        }

        let offset = self.segment_offset(registration, block_idx, 0)?;
        let device_ptr = registration.data_ptr + offset as u64;
        let request = sys::CUDA_MEMCPY2D_st {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_HOST,
            srcHost: src_ptr as *const std::ffi::c_void,
            srcDevice: 0,
            srcArray: std::ptr::null_mut(),
            srcPitch: registration.bytes_per_block,
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: sys::CUmemorytype_enum::CU_MEMORYTYPE_DEVICE,
            dstHost: std::ptr::null_mut(),
            dstDevice: device_ptr as sys::CUdeviceptr,
            dstArray: std::ptr::null_mut(),
            dstPitch: registration.kv_stride_bytes,
            WidthInBytes: registration.bytes_per_block,
            Height: registration.segments,
        };

        unsafe {
            let result = sys::cuMemcpy2D_v2(&request);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(format!("cuMemcpy2D (HtoD) failed: {:?}", result));
            }
        }

        Ok(())
    }
}

impl Default for PegaEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PegaEngine {
    fn drop(&mut self) {
        use cudarc::driver::sys;

        if !self.pinned_pool_ptr.is_null() {
            unsafe {
                let result = sys::cuMemFreeHost(self.pinned_pool_ptr as *mut std::ffi::c_void);
                if result != sys::cudaError_enum::CUDA_SUCCESS {
                    eprintln!("Warning: Failed to free pinned memory pool: {:?}", result);
                }
            }
            info!("Freed pinned memory pool");
        }
    }
}

// Safety: PegaEngine can be safely sent between threads
// - pinned_pool_ptr is managed exclusively by this struct
// - AtomicUsize provides thread-safe access to the offset
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
