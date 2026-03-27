//! Low-level pinned memory allocation for CUDA.
//!
//! This module provides two allocation strategies:
//!
//! 1. **Write-combined** (`PinnedMemory::allocate_write_combined`): Uses `cudaHostAlloc` with write-combined flag.
//!    Optimized for CPU-to-GPU transfers with better write performance.
//!
//! 2. **Huge pages** (`PinnedMemory::allocate_hugepages`): Uses `mmap(MAP_HUGETLB)` + `cudaHostRegister`.
//!    Much faster allocation for large buffers but requires pre-configured huge pages:
//!    ```bash
//!    # Reserve huge pages (size from /proc/meminfo, typically 2MB)
//!    sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'
//!    ```
//!
//! # Safety
//!
//! The memory returned is:
//! - Pinned and registered with CUDA for DMA transfers
//! - Valid for the lifetime of the `PinnedMemory` struct
//! - Automatically freed/unmapped and unregistered on drop

use std::io;
use std::ptr::NonNull;
use std::sync::OnceLock;

use cudarc::runtime::sys as rt;

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

/// Error type for pinned memory allocation.
#[derive(Debug)]
pub enum PinnedMemError {
    /// mmap failed
    MmapFailed(io::Error),
    /// cudaHostAlloc failed
    CudaAllocFailed(rt::cudaError),
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
            Self::CudaAllocFailed(e) => write!(f, "cudaHostAlloc failed: {:?}", e),
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
    /// Regular pinned memory via cudaHostAlloc (flags=0).
    /// Safe for both CPU reads and writes; use when SSD offload is enabled.
    Regular,
    /// Write-combined via cudaHostAlloc.
    /// Fast for CPU→GPU (write-only) workloads, but CPU reads are extremely slow.
    WriteCombined,
    /// Huge pages (size from /proc/meminfo, requires system configuration)
    HugePages,
}

/// RAII wrapper for CUDA pinned memory.
///
/// Memory is automatically freed/unmapped and unregistered when dropped.
pub struct PinnedMemory {
    ptr: NonNull<u8>,
    size: usize,
    strategy: AllocStrategy,
}

impl std::fmt::Debug for PinnedMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedMemory")
            .field("ptr", &format!("{:p}", self.ptr.as_ptr()))
            .field("size", &self.size)
            .field("strategy", &self.strategy)
            .finish()
    }
}

// SAFETY: PinnedMemory owns a pinned memory region that is:
// - Fixed in physical memory (pinned by CUDA)
// - Safe to access from any thread
// The pointer is valid for the lifetime of this struct.
// Sync is safe because PinnedMemory exposes only an immutable pointer via
// as_ptr() and has no interior mutability.
unsafe impl Send for PinnedMemory {}
unsafe impl Sync for PinnedMemory {}

impl PinnedMemory {
    /// Allocate regular pinned memory (flags=0).
    ///
    /// Safe for both CPU reads and writes. Use when SSD offload is enabled,
    /// since write-combined memory has extremely slow CPU reads.
    pub(crate) fn allocate_regular(size: usize) -> Result<Self, PinnedMemError> {
        Self::allocate_internal(size, AllocStrategy::Regular)
    }

    /// Allocate pinned memory using write-combined mode.
    ///
    /// Uses `cudaHostAlloc` with `cudaHostAllocWriteCombined` flag.
    /// Fast for CPU→GPU transfers but CPU reads are extremely slow.
    /// Do NOT use when SSD offload is enabled.
    #[allow(dead_code)]
    pub(crate) fn allocate_write_combined(size: usize) -> Result<Self, PinnedMemError> {
        Self::allocate_internal(size, AllocStrategy::WriteCombined)
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
    pub(crate) fn allocate_hugepages(size: usize) -> Result<Self, PinnedMemError> {
        Self::allocate_internal(size, AllocStrategy::HugePages)
    }

    fn allocate_internal(size: usize, strategy: AllocStrategy) -> Result<Self, PinnedMemError> {
        if size == 0 {
            return Err(PinnedMemError::ZeroSize);
        }

        let (ptr, aligned_size) = match strategy {
            AllocStrategy::Regular => {
                let mut ptr: *mut libc::c_void = std::ptr::null_mut();
                // SAFETY: ptr is a valid stack pointer; size is validated non-zero above.
                let result = unsafe { rt::cudaHostAlloc(&mut ptr, size, 0) };
                if result != rt::cudaError::cudaSuccess {
                    return Err(PinnedMemError::CudaAllocFailed(result));
                }
                (ptr, size)
            }
            AllocStrategy::WriteCombined => {
                let mut ptr: *mut libc::c_void = std::ptr::null_mut();
                // SAFETY: ptr is a valid stack pointer; size is validated non-zero above.
                let result =
                    unsafe { rt::cudaHostAlloc(&mut ptr, size, rt::cudaHostAllocWriteCombined) };
                if result != rt::cudaError::cudaSuccess {
                    return Err(PinnedMemError::CudaAllocFailed(result));
                }
                (ptr, size)
            }
            AllocStrategy::HugePages => {
                let huge_page_size =
                    get_huge_page_size().ok_or(PinnedMemError::HugePageSizeUnavailable)?;
                let aligned = (size + huge_page_size - 1) & !(huge_page_size - 1);
                let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB;

                // SAFETY: All parameters are valid: null hint, aligned non-zero size,
                // valid protection flags, MAP_ANONYMOUS with fd=-1. Return is checked
                // against MAP_FAILED before use.
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        aligned,
                        libc::PROT_READ | libc::PROT_WRITE,
                        flags,
                        -1,
                        0,
                    )
                };

                if ptr == libc::MAP_FAILED {
                    return Err(PinnedMemError::MmapFailed(std::io::Error::last_os_error()));
                }

                // Register with CUDA for DMA
                // SAFETY: ptr is a valid mmap'd region of `aligned` bytes (checked above).
                let result =
                    unsafe { rt::cudaHostRegister(ptr, aligned, rt::cudaHostRegisterDefault) };

                if result != rt::cudaError::cudaSuccess {
                    unsafe { libc::munmap(ptr, aligned) };
                    return Err(PinnedMemError::CudaRegisterFailed(result));
                }

                (ptr, aligned)
            }
        };

        // SAFETY: allocation succeeded and returned non-null pointer
        let ptr = NonNull::new(ptr as *mut u8).expect("allocation returned null");

        Ok(Self {
            ptr,
            size: aligned_size,
            strategy,
        })
    }

    /// Get a raw pointer to the allocated memory.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes.
    ///
    /// This is the aligned size, which may be larger than the requested size.
    #[inline]
    pub(crate) fn size(&self) -> usize {
        self.size
    }
}

impl Drop for PinnedMemory {
    fn drop(&mut self) {
        match self.strategy {
            AllocStrategy::Regular | AllocStrategy::WriteCombined => {
                // SAFETY: ptr was allocated with cudaHostAlloc
                let result = unsafe { rt::cudaFreeHost(self.ptr.as_ptr() as *mut libc::c_void) };
                if result != rt::cudaError::cudaSuccess {
                    eprintln!("Warning: cudaFreeHost failed: {:?}", result);
                }
            }
            AllocStrategy::HugePages => {
                // SAFETY: ptr was registered with cudaHostRegister
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_write_combined() {
        // Skip if no CUDA context available
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }

        let mem = PinnedMemory::allocate_write_combined(4096).unwrap();
        assert!(mem.size() >= 4096);
    }

    #[test]
    fn test_zero_size_fails() {
        let result = PinnedMemory::allocate_write_combined(0);
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
