use std::{
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use tracing::info;

use crate::allocator::{Allocation, ScaledOffsetAllocator};

/// RAII guard for a pinned memory allocation.
/// Automatically frees the allocation when dropped.
pub struct PinnedAllocation {
    allocation: Allocation,
    ptr: NonNull<u8>,
    pool: Arc<PinnedMemoryPool>,
}

impl PinnedAllocation {
    /// Get a const pointer to the allocated memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the allocated memory
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes
    pub fn size(&self) -> usize {
        self.allocation.size_bytes as usize
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Automatically free the allocation when the guard is dropped
        self.pool.free_internal(self.allocation);
    }
}

/// Manages a CUDA pinned memory pool and a byte-addressable allocator.
pub struct PinnedMemoryPool {
    base_ptr: NonNull<u8>,
    allocator: Mutex<ScaledOffsetAllocator>,
}

impl PinnedMemoryPool {
    /// Allocate a new pinned memory pool of `pool_size` bytes.
    pub fn new(pool_size: usize) -> Self {
        use cudarc::driver::sys;

        if pool_size == 0 {
            panic!("Pinned memory pool size must be greater than zero");
        }

        let mut pool_ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        let base_ptr = unsafe {
            let result = sys::cuMemAllocHost_v2(&mut pool_ptr, pool_size);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                panic!("Failed to allocate pinned memory pool: {:?}", result);
            }
            NonNull::new(pool_ptr as *mut u8).expect("cuMemAllocHost returned null pointer")
        };

        let allocator = ScaledOffsetAllocator::new(pool_size as u64)
            .expect("Failed to create memory allocator");

        info!(
            "Allocated pinned memory pool: {} GB ({} bytes)",
            pool_size as f64 / 1e9,
            pool_size
        );

        Self {
            base_ptr,
            allocator: Mutex::new(allocator),
        }
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    pub fn allocate(self: &Arc<Self>, size: usize) -> Option<PinnedAllocation> {
        if size == 0 {
            panic!("Cannot allocate zero bytes from the pinned pool");
        }

        let mut allocator = self.allocator.lock().unwrap();

        let allocation = match allocator.allocate(size as u64) {
            Ok(Some(allocation)) => allocation,
            Ok(None) => {
                return None; // Pool exhausted, caller can retry after eviction
            }
            Err(err) => panic!("Pinned memory allocation error: {}", err),
        };

        let ptr = unsafe { self.base_ptr.as_ptr().add(allocation.offset_bytes as usize) };
        let ptr = NonNull::new(ptr).expect("PinnedMemoryPool returned null pointer");

        Some(PinnedAllocation {
            allocation,
            ptr,
            pool: Arc::clone(self),
        })
    }

    /// Internal method to free a pinned memory allocation.
    /// This is called automatically by PinnedAllocation's Drop implementation.
    /// Users should not call this directly - use PinnedAllocation RAII instead.
    pub(crate) fn free_internal(&self, allocation: Allocation) {
        let mut allocator = self.allocator.lock().unwrap();
        allocator.free(allocation);
    }

    /// Get (used_bytes, total_bytes) for the pool.
    pub fn usage(&self) -> (usize, usize) {
        let allocator = self.allocator.lock().unwrap();
        let report = allocator.storage_report();
        let total = allocator.total_bytes() as usize;
        let used = total - report.total_free_bytes as usize;
        (used, total)
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        use cudarc::driver::sys;

        unsafe {
            let result = sys::cuMemFreeHost(self.base_ptr.as_ptr() as *mut std::ffi::c_void);
            if result != sys::cudaError_enum::CUDA_SUCCESS {
                eprintln!("Warning: Failed to free pinned memory pool: {:?}", result);
            }
        }
        info!("Freed pinned memory pool");
    }
}

// TODO: fix this
unsafe impl Send for PinnedMemoryPool {}
unsafe impl Sync for PinnedMemoryPool {}
