//! Low-level pinned memory allocation for CUDA.
//!
//! Three strategies, all returning DMA-pinned memory:
//!
//! 1. **Regular** (`allocate_regular`): lazy `mmap` + parallel page pre-touch +
//!    `cudaHostRegister`. Each touch thread is pinned to the target NUMA node
//!    so first-touch places every page on that node's local memory.
//!
//! 2. **HugePages** (`allocate_hugepages`): `mmap(MAP_HUGETLB)` + parallel
//!    pre-touch + `cudaHostRegister`. Same NUMA-aware touch. Requires reserved
//!    hugepages:
//!    ```bash
//!    sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'  # 30GB at 2MB pages
//!    ```
//!
//! 3. **WriteCombined** (`allocate_write_combined`): `cudaHostAlloc` with WC
//!    flag. CPU reads are extremely slow — don't use when SSD offload is on.
//!    NUMA placement follows the calling thread's affinity (caller is expected
//!    to wrap with `run_on_numa`).
//!
//! See `examples/pinned_alloc_parallel.rs` for the benchmarks motivating the
//! parallel pre-touch path.
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
use pegaflow_common::{NumaNode, pin_thread_to_numa_node};

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
pub(crate) enum PinnedMemError {
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
pub(crate) enum AllocStrategy {
    /// `mmap` + parallel pre-touch + `cudaHostRegister`.
    /// Safe for both CPU reads and writes; use when SSD offload is enabled.
    Regular,
    /// `cudaHostAlloc` with write-combined flag.
    /// Fast for CPU→GPU (write-only) workloads, but CPU reads are extremely slow.
    WriteCombined,
    /// `mmap(MAP_HUGETLB)` + parallel pre-touch + `cudaHostRegister`.
    /// Requires reserved hugepages.
    HugePages,
}

/// RAII wrapper for CUDA pinned memory.
///
/// Memory is automatically freed/unmapped and unregistered when dropped.
pub(crate) struct PinnedMemory {
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
    /// Allocate regular pinned memory via `mmap` + parallel pre-touch + register.
    ///
    /// All pages are first-touched by threads pinned to `node`, so the entire
    /// region lands NUMA-local to that node. Pass `NumaNode::UNKNOWN` to skip
    /// pinning and rely on the calling thread's existing affinity.
    pub(crate) fn allocate_regular(size: usize, node: NumaNode) -> Result<Self, PinnedMemError> {
        Self::allocate_mmap_register(size, AllocStrategy::Regular, node)
    }

    /// Allocate write-combined pinned memory via `cudaHostAlloc`.
    ///
    /// Fast for CPU→GPU transfers but CPU reads are extremely slow.
    /// Do NOT use when SSD offload is enabled. NUMA placement follows the
    /// calling thread's affinity at allocation time.
    pub(crate) fn allocate_write_combined(size: usize) -> Result<Self, PinnedMemError> {
        if size == 0 {
            return Err(PinnedMemError::ZeroSize);
        }
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: ptr is a valid stack pointer; size is validated non-zero above.
        let result = unsafe { rt::cudaHostAlloc(&mut ptr, size, rt::cudaHostAllocWriteCombined) };
        if result != rt::cudaError::cudaSuccess {
            return Err(PinnedMemError::CudaAllocFailed(result));
        }
        let ptr = NonNull::new(ptr as *mut u8).expect("cudaHostAlloc returned null");
        Ok(Self {
            ptr,
            size,
            strategy: AllocStrategy::WriteCombined,
        })
    }

    /// Allocate hugepage-backed pinned memory.
    ///
    /// Uses `mmap(MAP_HUGETLB)` + parallel pre-touch + `cudaHostRegister`.
    /// Touch threads are pinned to `node` for NUMA-local first-touch.
    ///
    /// Requires reserved hugepages:
    /// ```bash
    /// sudo sh -c 'echo 15360 > /proc/sys/vm/nr_hugepages'  # 30GB at 2MB pages
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `MmapFailed` if huge pages are not configured or insufficient.
    pub(crate) fn allocate_hugepages(size: usize, node: NumaNode) -> Result<Self, PinnedMemError> {
        Self::allocate_mmap_register(size, AllocStrategy::HugePages, node)
    }

    fn allocate_mmap_register(
        size: usize,
        strategy: AllocStrategy,
        node: NumaNode,
    ) -> Result<Self, PinnedMemError> {
        if size == 0 {
            return Err(PinnedMemError::ZeroSize);
        }

        let (flags, aligned_size) = match strategy {
            AllocStrategy::HugePages => {
                let huge_page_size =
                    get_huge_page_size().ok_or(PinnedMemError::HugePageSizeUnavailable)?;
                let aligned = (size + huge_page_size - 1) & !(huge_page_size - 1);
                (
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                    aligned,
                )
            }
            AllocStrategy::Regular => (libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, size),
            AllocStrategy::WriteCombined => {
                unreachable!("WriteCombined does not use the mmap path")
            }
        };

        // SAFETY: null hint + anonymous mapping with valid prot/flags; the
        // result is checked against MAP_FAILED before any use.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                aligned_size,
                libc::PROT_READ | libc::PROT_WRITE,
                flags,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(PinnedMemError::MmapFailed(io::Error::last_os_error()));
        }

        parallel_pre_touch(ptr.cast::<u8>(), aligned_size, node);

        // SAFETY: ptr is a valid mapping of aligned_size bytes (checked above).
        let result =
            unsafe { rt::cudaHostRegister(ptr, aligned_size, rt::cudaHostRegisterDefault) };
        if result != rt::cudaError::cudaSuccess {
            // SAFETY: ptr was successfully mmap'd above.
            unsafe { libc::munmap(ptr, aligned_size) };
            return Err(PinnedMemError::CudaRegisterFailed(result));
        }

        let ptr = NonNull::new(ptr as *mut u8).expect("mmap returned null");
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
            AllocStrategy::WriteCombined => {
                // SAFETY: ptr was allocated with cudaHostAlloc.
                let result = unsafe { rt::cudaFreeHost(self.ptr.as_ptr() as *mut libc::c_void) };
                if result != rt::cudaError::cudaSuccess {
                    eprintln!("Warning: cudaFreeHost failed: {:?}", result);
                }
            }
            AllocStrategy::Regular | AllocStrategy::HugePages => {
                // SAFETY: ptr was registered with cudaHostRegister.
                let unreg =
                    unsafe { rt::cudaHostUnregister(self.ptr.as_ptr() as *mut libc::c_void) };
                if unreg != rt::cudaError::cudaSuccess {
                    eprintln!("Warning: cudaHostUnregister failed: {:?}", unreg);
                }
                // SAFETY: ptr was allocated by mmap with the same size.
                let unmap =
                    unsafe { libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, self.size) };
                if unmap == -1 {
                    let err = io::Error::last_os_error();
                    eprintln!("Warning: munmap failed: {}", err);
                }
            }
        }
    }
}

/// Fault in every page of `[ptr, ptr+size)` across worker threads pinned to
/// `node`. Each thread owns a disjoint chunk and writes one byte per page
/// (plus a tail byte) to force materialization. First-touch on a pinned
/// thread places each page on `node`'s local memory.
///
/// If `node.is_unknown()`, threads run with the calling thread's existing
/// affinity (typically set by `run_on_numa` at the call site).
fn parallel_pre_touch(ptr: *mut u8, size: usize, node: NumaNode) {
    let page = page_size();
    let threads = touch_threads();
    let chunk = size.div_ceil(threads);
    // *mut u8 is not Send; smuggle as usize and reconstruct inside the scope.
    // Lifetimes are bounded by thread::scope.
    let base = ptr as usize;

    std::thread::scope(|s| {
        for i in 0..threads {
            let off = i * chunk;
            if off >= size {
                break;
            }
            let len = chunk.min(size - off);
            s.spawn(move || {
                if node.is_valid()
                    && let Err(e) = pin_thread_to_numa_node(node)
                {
                    log::warn!("pre-touch NUMA pin to {} failed: {}", node, e);
                }
                let p = (base + off) as *mut u8;
                let mut off = 0usize;
                while off < len {
                    // SAFETY: chunk is a disjoint sub-range of the caller's mapping.
                    unsafe { p.add(off).write_volatile(0u8) };
                    off += page;
                }
                // SAFETY: len > 0 is enforced by the chunk bookkeeping above.
                unsafe { p.add(len - 1).write_volatile(0u8) };
            });
        }
    });
}

fn touch_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(8, |n| n.get())
        .max(1)
}

fn page_size() -> usize {
    // SAFETY: sysconf with a valid name; non-positive result is fallback-handled.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 { v as usize } else { 4096 }
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
    fn test_allocate_regular() {
        if cudarc::driver::CudaContext::new(0).is_err() {
            return;
        }

        let mem = PinnedMemory::allocate_regular(4096, NumaNode::UNKNOWN).unwrap();
        assert!(mem.size() >= 4096);
    }

    #[test]
    fn test_zero_size_fails() {
        assert!(matches!(
            PinnedMemory::allocate_write_combined(0),
            Err(PinnedMemError::ZeroSize)
        ));
        assert!(matches!(
            PinnedMemory::allocate_regular(0, NumaNode::UNKNOWN),
            Err(PinnedMemError::ZeroSize)
        ));
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
