//! Low-level pinned memory allocation for CUDA.
//!
//! This module provides two allocation strategies:
//!
//! 1. **Regular pages** (`PinnedMemory::allocate`): Uses `mmap(MAP_POPULATE)` + `cudaHostRegister`.
//!    Works on any system but slower (~20s for 30GB).
//!
//! 2. **Huge pages** (`PinnedMemory::allocate_hugepages`): Uses `mmap(MAP_HUGETLB)` + `cudaHostRegister`.
//!    Much faster (~6s for 30GB) but requires pre-configured huge pages:
//!    ```bash
//!    # Reserve huge pages (size from /proc/meminfo, typically 2MB)
//!    sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'
//!    ```
//!
//! # Safety
//!
//! The memory returned is:
//! - Page-aligned (4KB or huge page size depending on allocation method)
//! - Pinned and registered with CUDA for DMA transfers
//! - Valid for the lifetime of the `PinnedMemory` struct
//! - Automatically unmapped and unregistered on drop

use std::io;
use std::ptr::NonNull;
use std::sync::OnceLock;

use cudarc::runtime::sys as rt;

const PAGE_SIZE: usize = 4096;

/// Cached huge page size from /proc/meminfo
static HUGE_PAGE_SIZE: OnceLock<Option<usize>> = OnceLock::new();

/// Read the system's default huge page size from /proc/meminfo.
/// Returns None if reading or parsing fails.
fn get_huge_page_size() -> Option<usize> {
    *HUGE_PAGE_SIZE.get_or_init(read_hugepage_size_from_proc)
}

/// Parse Hugepagesize from /proc/meminfo (in kB, convert to bytes)
fn read_hugepage_size_from_proc() -> Option<usize> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in content.lines() {
        // Format: "Hugepagesize:       2048 kB"
        if line.starts_with("Hugepagesize:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // parts = ["Hugepagesize:", "2048", "kB"]
            if parts.len() == 3 && parts[2] == "kB" {
                let kb: usize = parts[1].parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

/// Get the system's default huge page size in bytes.
///
/// Reads from `/proc/meminfo` on first call, then caches the result.
/// Returns `None` if the system doesn't support huge pages or `/proc/meminfo` is unavailable.
pub fn huge_page_size() -> Option<usize> {
    get_huge_page_size()
}

/// Error type for pinned memory allocation.
#[derive(Debug)]
pub enum PinnedMemError {
    /// mmap failed
    MmapFailed(io::Error),
    /// cudaHostRegister failed
    CudaRegisterFailed(rt::cudaError),
    /// Size must be greater than zero
    ZeroSize,
    /// Failed to determine huge page size from /proc/meminfo
    HugePageSizeUnavailable,
}

impl std::fmt::Display for PinnedMemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MmapFailed(e) => write!(f, "mmap failed: {}", e),
            Self::CudaRegisterFailed(e) => write!(f, "cudaHostRegister failed: {:?}", e),
            Self::ZeroSize => write!(f, "size must be greater than zero"),
            Self::HugePageSizeUnavailable => write!(
                f,
                "cannot determine huge page size: Hugepagesize not found in /proc/meminfo"
            ),
        }
    }
}

impl std::error::Error for PinnedMemError {}

/// Allocation strategy for pinned memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Regular 4KB pages with MAP_POPULATE
    Regular,
    /// Huge pages (size from /proc/meminfo, requires system configuration)
    HugePages,
}

/// RAII wrapper for pinned memory allocated via mmap + cudaHostRegister.
///
/// Memory is automatically unmapped and unregistered when dropped.
pub struct PinnedMemory {
    ptr: NonNull<u8>,
    size: usize,
    strategy: AllocStrategy,
}

// SAFETY: PinnedMemory owns a pinned memory region that is:
// - Fixed in physical memory (pinned by CUDA)
// - Safe to access from any thread
// The pointer is valid for the lifetime of this struct.
unsafe impl Send for PinnedMemory {}
unsafe impl Sync for PinnedMemory {}

impl PinnedMemory {
    /// Allocate pinned memory using regular pages.
    ///
    /// Uses `mmap(MAP_POPULATE)` to pre-fault pages, then registers with CUDA.
    /// This works on any system but is slower than huge pages.
    ///
    /// # Errors
    ///
    /// Returns error if mmap or cudaHostRegister fails.
    pub fn allocate(size: usize) -> Result<Self, PinnedMemError> {
        Self::allocate_internal(size, AllocStrategy::Regular)
    }

    /// Allocate pinned memory using huge pages.
    ///
    /// Uses `mmap(MAP_HUGETLB)` for fast allocation, then registers with CUDA.
    /// Much faster than regular pages but requires system configuration:
    ///
    /// ```bash
    /// # Reserve huge pages (size depends on system, typically 2MB)
    /// sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'  # for 30GB with 2MB pages
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `MmapFailed` if huge pages are not configured or insufficient.
    pub fn allocate_hugepages(size: usize) -> Result<Self, PinnedMemError> {
        Self::allocate_internal(size, AllocStrategy::HugePages)
    }

    fn allocate_internal(size: usize, strategy: AllocStrategy) -> Result<Self, PinnedMemError> {
        if size == 0 {
            return Err(PinnedMemError::ZeroSize);
        }

        let (aligned_size, mmap_flags) = match strategy {
            AllocStrategy::Regular => {
                let aligned = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE;
                (aligned, flags)
            }
            AllocStrategy::HugePages => {
                let huge_page_size =
                    get_huge_page_size().ok_or(PinnedMemError::HugePageSizeUnavailable)?;
                let aligned = (size + huge_page_size - 1) & !(huge_page_size - 1);
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB;
                (aligned, flags)
            }
        };

        // SAFETY: mmap with MAP_ANONYMOUS creates a new anonymous mapping.
        // The flags are valid and size is non-zero and aligned.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                mmap_flags,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(PinnedMemError::MmapFailed(std::io::Error::last_os_error()));
        }

        // SAFETY: ptr is valid from mmap, and we're registering it with CUDA.
        // cudaHostRegister pins the memory and makes it accessible for DMA.
        let result =
            unsafe { rt::cudaHostRegister(ptr, aligned_size, rt::cudaHostRegisterDefault) };

        if result != rt::cudaError::cudaSuccess {
            // Clean up mmap on failure
            // SAFETY: ptr was returned by mmap with the same size
            unsafe { libc::munmap(ptr, aligned_size) };
            return Err(PinnedMemError::CudaRegisterFailed(result));
        }

        // SAFETY: mmap succeeded and returned non-null pointer
        let ptr = NonNull::new(ptr as *mut u8).expect("mmap returned null but not MAP_FAILED");

        Ok(Self {
            ptr,
            size: aligned_size,
            strategy,
        })
    }

    /// Get a raw pointer to the allocated memory.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the allocated memory.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes.
    ///
    /// This is the aligned size, which may be larger than the requested size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the allocation strategy used.
    #[inline]
    pub fn strategy(&self) -> AllocStrategy {
        self.strategy
    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        // SAFETY: ptr was registered with cudaHostRegister in allocate_internal
        unsafe {
            let result = rt::cudaHostUnregister(self.ptr.as_ptr() as *mut libc::c_void);
            if result != rt::cudaError::cudaSuccess {
                eprintln!("Warning: cudaHostUnregister failed: {:?}", result);
            }
        }

        // SAFETY: ptr was allocated by mmap with the same size
        unsafe {
            if libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size) == -1 {
                let err = std::io::Error::last_os_error();
                eprintln!("Warning: munmap failed: {}", err);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_regular() {
        // Skip if no CUDA context available
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }

        let mem = PinnedMemory::allocate(4096).unwrap();
        assert!(mem.size() >= 4096);
        assert_eq!(mem.strategy(), AllocStrategy::Regular);
    }

    #[test]
    fn test_zero_size_fails() {
        let result = PinnedMemory::allocate(0);
        assert!(matches!(result, Err(PinnedMemError::ZeroSize)));
    }

    #[test]
    fn test_read_hugepage_size() {
        // Hugepagesize is always present in /proc/meminfo on Linux
        let size = read_hugepage_size_from_proc();
        assert!(size.is_some(), "Hugepagesize should exist in /proc/meminfo");

        let size = size.unwrap();
        // Common sizes: 2MB (default), 1GB
        assert!(
            size >= 2 * 1024 * 1024,
            "Hugepage size should be at least 2MB"
        );
        assert!(
            size.is_power_of_two(),
            "Hugepage size should be power of two"
        );
    }
}
