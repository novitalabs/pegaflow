use std::{mem::MaybeUninit, ptr::NonNull};

/// A simple memory pool that manages fixed-size chunks of memory.
///
/// This pool pre-allocates a contiguous buffer and manages allocation/deallocation
/// of fixed-size chunks within that buffer. It uses unsafe operations for performance
/// and does not perform bounds checking on freed pointers.
pub(crate) struct MemoryPool {
    chunk_size: usize,
    buffer: Vec<MaybeUninit<u8>>,
    free_list: Vec<NonNull<u8>>,
}

impl MemoryPool {
    /// Creates a new memory pool with the specified chunk size and number of chunks.
    ///
    /// All chunks are initially available for allocation.
    pub(crate) fn new(chunk_size: usize, num_chunks: usize) -> Self {
        let mut buffer = Vec::with_capacity(chunk_size * num_chunks);
        unsafe { buffer.set_len(chunk_size * num_chunks) };
        let ptr = unsafe { NonNull::new_unchecked(buffer.as_mut_ptr() as *mut u8) };
        let free_list = (0..num_chunks)
            .map(|i| unsafe { ptr.byte_add(i * chunk_size) })
            .collect();
        Self {
            chunk_size,
            buffer,
            free_list,
        }
    }

    /// Returns the size of each chunk in bytes.
    pub(crate) fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns a pointer to the start of the underlying buffer.
    pub(crate) fn buffer_ptr(&self) -> NonNull<u8> {
        unsafe { NonNull::new_unchecked(self.buffer.as_ptr() as *mut u8) }
    }

    /// Returns the length of the underlying buffer in bytes.
    pub(crate) fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Allocates a uninitialized memory chunk from the pool.
    /// Returns `None` if no chunks are available.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the returned pointer is not used after being freed.
    pub(crate) unsafe fn alloc(&mut self) -> Option<NonNull<u8>> {
        self.free_list.pop()
    }

    /// Frees a previously allocated chunk back to the pool.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was previously returned by `alloc()` from this pool
    /// - `ptr` is not used after being freed
    /// - `ptr` is not freed multiple times
    pub(crate) unsafe fn free(&mut self, ptr: NonNull<u8>) {
        self.free_list.push(ptr);
    }
}
