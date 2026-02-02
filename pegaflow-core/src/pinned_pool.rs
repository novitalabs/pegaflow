use parking_lot::Mutex;
use std::{collections::HashMap, num::NonZeroU64, ptr::NonNull, sync::Arc};

use bytesize::ByteSize;
use log::{error, info, warn};

use crate::allocator::{Allocation, ScaledOffsetAllocator};
use crate::metrics::core_metrics;
use crate::numa::{NumaNode, run_on_numa};
use crate::pinned_mem::PinnedMemory;

/// RAII guard for a pinned memory allocation.
/// Automatically frees the allocation when dropped.
pub struct PinnedAllocation {
    allocation: Allocation,
    ptr: NonNull<u8>,
    pool: Arc<PinnedMemoryPool>,
}

// SAFETY: PinnedAllocation points to CUDA pinned memory which is fixed in physical
// memory and safe to access from any thread. The NonNull<u8> is just a pointer to
// this pinned memory region.
unsafe impl Send for PinnedAllocation {}
unsafe impl Sync for PinnedAllocation {}

impl PinnedAllocation {
    /// Get a const pointer to the allocated memory
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the allocated memory
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the size of the allocation in bytes
    pub fn size(&self) -> NonZeroU64 {
        self.allocation.size_bytes
    }
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        // Automatically free the allocation when the guard is dropped
        self.pool.free_internal(&self.allocation);
    }
}

/// Manages a CUDA pinned memory pool and a byte-addressable allocator.
#[derive(Debug)]
pub(crate) struct PinnedMemoryPool {
    /// Backing pinned memory (handles mmap + cudaHostRegister)
    backing: PinnedMemory,
    allocator: Mutex<ScaledOffsetAllocator>,
}

impl PinnedMemoryPool {
    /// Upper bound for simultaneous allocations in the pinned pool.
    const MAX_ALLOCS: u32 = 4_000_000;
    /// Alignment for unit size (512 bytes for Direct I/O compatibility)
    const UNIT_ALIGNMENT: u64 = 512;

    /// Calculate unit size that fits pool_size into u32 range
    fn compute_unit_size(pool_size: u64, hint: Option<NonZeroU64>) -> u64 {
        let max_units = u32::MAX as u64;
        let min_unit_for_capacity = pool_size.div_ceil(max_units);
        let base = hint.map(|h| h.get()).unwrap_or(Self::UNIT_ALIGNMENT);
        let unit = base.max(min_unit_for_capacity);
        unit.div_ceil(Self::UNIT_ALIGNMENT) * Self::UNIT_ALIGNMENT
    }

    /// Allocate a new pinned memory pool of `pool_size` bytes.
    ///
    /// If `use_hugepages` is true, uses huge pages (requires system config).
    /// If `unit_size_hint` is provided, the allocator rounds allocations up to this size.
    pub fn new(pool_size: usize, use_hugepages: bool, unit_size_hint: Option<NonZeroU64>) -> Self {
        if pool_size == 0 {
            panic!("Pinned memory pool size must be greater than zero");
        }

        let backing = if use_hugepages {
            info!("Allocating pinned memory pool with huge pages");
            PinnedMemory::allocate_hugepages(pool_size)
                .expect("Failed to allocate pinned memory pool with huge pages")
        } else {
            info!("Allocating pinned memory pool with regular pages");
            PinnedMemory::allocate(pool_size).expect("Failed to allocate pinned memory pool")
        };

        let actual_size = backing.size() as u64;
        let unit_size = Self::compute_unit_size(actual_size, unit_size_hint);

        info!(
            "Pinned pool: size={}, unit_size={}, max_units={}",
            ByteSize(actual_size),
            ByteSize(unit_size),
            actual_size.div_ceil(unit_size)
        );

        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(actual_size) {
            metrics.pool_capacity_bytes.add(capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric update: capacity_bytes={}",
                actual_size
            );
        }

        let allocator = ScaledOffsetAllocator::new_with_unit_size_and_max_allocs(
            actual_size,
            unit_size,
            Self::MAX_ALLOCS,
        )
        .unwrap_or_else(|err| {
            panic!(
                "Failed to create memory allocator (size={}, unit={}): {}",
                ByteSize(actual_size),
                ByteSize(unit_size),
                err
            )
        });

        Self {
            backing,
            allocator: Mutex::new(allocator),
        }
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    pub fn allocate(self: &Arc<Self>, size: NonZeroU64) -> Option<PinnedAllocation> {
        // Allocation is done under lock, metrics and pointer computation outside lock
        let allocation = {
            let mut allocator = self.allocator.lock();
            match allocator.allocate(size.get()) {
                Ok(Some(allocation)) => allocation,
                Ok(None) => return None, // Pool exhausted, caller can retry after eviction
                Err(err) => {
                    error!(
                        "Pinned memory allocation error: {} (requested {}): requested_bytes={}",
                        err,
                        ByteSize(size.get()),
                        size.get()
                    );
                    return None;
                }
            }
        };

        let offset: usize = allocation
            .offset_bytes
            .try_into()
            .expect("allocation offset exceeds usize");
        let ptr = unsafe { self.backing.as_ptr().add(offset) };
        let ptr = NonNull::new(ptr as *mut u8).expect("PinnedMemoryPool returned null pointer");

        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            core_metrics().pool_used_bytes.add(size_i64, &[]);
        }

        Some(PinnedAllocation {
            allocation,
            ptr,
            pool: Arc::clone(self),
        })
    }

    /// Internal method to free a pinned memory allocation.
    /// This is called automatically by PinnedAllocation's Drop implementation.
    /// Users should not call this directly - use PinnedAllocation RAII instead.
    pub(crate) fn free_internal(&self, allocation: &Allocation) {
        // Free under lock, metrics update outside lock
        {
            let mut allocator = self.allocator.lock();
            allocator.free(allocation);
        }

        let size_bytes = allocation.size_bytes.get();
        if let Ok(size_i64) = i64::try_from(size_bytes) {
            core_metrics().pool_used_bytes.add(-size_i64, &[]);
        }
    }

    /// Get (used_bytes, total_bytes) for the pool.
    pub fn usage(&self) -> (u64, u64) {
        let allocator = self.allocator.lock();
        let report = allocator.storage_report();
        let total = allocator.total_bytes();
        let used = total - report.total_free_bytes;
        (used, total)
    }

    /// Largest contiguous free region currently available, in bytes.
    pub fn largest_free_allocation(&self) -> u64 {
        let allocator = self.allocator.lock();
        allocator.storage_report().largest_free_allocation_bytes
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        let metrics = core_metrics();
        let capacity_bytes = self.backing.size();
        if let Ok(capacity_i64) = i64::try_from(capacity_bytes) {
            metrics.pool_capacity_bytes.add(-capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric cleanup: capacity_bytes={}",
                capacity_bytes
            );
        }
    }
}

// PinnedMemory handles cleanup in its Drop impl, no manual Drop needed here.

// SAFETY: The pool owns a PinnedMemory backing that remains valid for the lifetime
// of the pool. All mutations of the allocator state are guarded by the internal
// `Mutex`. CUDA pinned host memory can be accessed from any host thread.
unsafe impl Send for PinnedMemoryPool {}
unsafe impl Sync for PinnedMemoryPool {}

// ============================================================================
// NUMA-Aware Pool Management
// ============================================================================

/// Manages multiple pinned memory pools, one per NUMA node.
///
/// Each pool is allocated on its respective NUMA node using first-touch policy
/// (memory is allocated from a thread pinned to that NUMA node).
///
/// This enables NUMA-local memory allocation for GPU workers, which is critical
/// for achieving optimal D2H/H2D transfer bandwidth on multi-socket systems.
///
/// # Note
/// This implementation does NOT provide fallback for unknown NUMA nodes.
/// If a GPU's NUMA affinity cannot be determined, the system should either:
/// - Disable NUMA-aware allocation (use global pool)
/// - Fail early during registration with a clear error
#[derive(Debug)]
pub(crate) struct NumaAwarePinnedPools {
    /// Per-NUMA pools indexed by NUMA node ID
    pools: HashMap<u32, Arc<PinnedMemoryPool>>,
}

impl NumaAwarePinnedPools {
    /// Create NUMA-aware pools, evenly distributing capacity across nodes.
    ///
    /// # Arguments
    /// * `total_capacity` - Total memory to allocate across all NUMA nodes
    /// * `numa_nodes` - List of NUMA nodes to create pools for
    /// * `use_hugepages` - Whether to use huge pages for allocation
    /// * `unit_size_hint` - Optional hint for allocator unit size
    ///
    /// # Behavior
    /// - Each NUMA node gets `total_capacity / num_nodes` bytes
    /// - Pools are allocated on threads pinned to their respective NUMA nodes
    ///
    /// # Panics
    /// Panics if any NUMA pool allocation fails. This is a fail-fast behavior
    /// to prevent silent partial failures that would cause mysterious allocation
    /// errors later when GPUs try to use the missing pool.
    pub fn new(
        total_capacity: usize,
        numa_nodes: &[NumaNode],
        use_hugepages: bool,
        unit_size_hint: Option<NonZeroU64>,
    ) -> Self {
        let num_nodes = numa_nodes.len();
        if num_nodes == 0 {
            warn!("No NUMA nodes provided, creating empty NumaAwarePinnedPools");
            return Self {
                pools: HashMap::new(),
            };
        }

        let per_node_capacity = total_capacity / num_nodes;
        info!(
            "Creating NUMA-aware pools: total={}, nodes={}, per_node={}",
            ByteSize(total_capacity as u64),
            num_nodes,
            ByteSize(per_node_capacity as u64)
        );

        let mut pools = HashMap::new();

        for node in numa_nodes {
            if node.is_unknown() {
                continue;
            }

            let node_id = node.0;
            let hint = unit_size_hint;

            // Allocate pool on a thread pinned to this NUMA node
            let result = run_on_numa(*node, move || {
                PinnedMemoryPool::new(per_node_capacity, use_hugepages, hint)
            });

            match result {
                Ok(pool) => {
                    info!(
                        "Created pinned pool on NUMA{}: capacity={}",
                        node_id,
                        ByteSize(per_node_capacity as u64)
                    );
                    pools.insert(node_id, Arc::new(pool));
                }
                Err(e) => {
                    panic!(
                        "Failed to create pool on NUMA{}: {}. \
                         This is a fatal error during initialization. \
                         Please check system resources or disable NUMA affinity.",
                        node_id, e
                    );
                }
            }
        }

        Self { pools }
    }

    /// Allocate memory from the pool for a specific NUMA node.
    ///
    ///
    /// Returns `None` if:
    /// - `numa_node` is `UNKNOWN` (should be caught at registration time)
    /// - The NUMA node has no pool
    /// - The pool is exhausted
    pub fn allocate(&self, numa_node: NumaNode, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
        if numa_node.is_unknown() {
            error!("UNEXPECTED: allocate called with UNKNOWN NUMA node");
            return None;
        }

        self.pools.get(&numa_node.0)?.allocate(size).map(Arc::new)
    }

    /// Get aggregate usage across all pools: (used_bytes, total_bytes)
    pub(crate) fn total_usage(&self) -> (u64, u64) {
        let mut used = 0u64;
        let mut total = 0u64;

        for pool in self.pools.values() {
            let (u, t) = pool.usage();
            used += u;
            total += t;
        }

        (used, total)
    }
}

// ============================================================================
// Unified Allocator Interface
// ============================================================================

/// Unified pinned memory allocator that hides NUMA details from callers.
///
/// This enum encapsulates both global and NUMA-aware allocation strategies,
/// providing a single interface for the rest of the system.
#[derive(Debug)]
pub(crate) enum PinnedAllocator {
    /// Global pool (NUMA disabled or single-node systems)
    Global(Arc<PinnedMemoryPool>),
    /// NUMA-aware pools (multi-socket systems)
    Numa(NumaAwarePinnedPools),
}

impl PinnedAllocator {
    /// Create a new global allocator.
    pub(crate) fn new_global(
        capacity: usize,
        use_hugepages: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Self {
        let pool = Arc::new(PinnedMemoryPool::new(capacity, use_hugepages, unit_hint));
        Self::Global(pool)
    }

    /// Create a new NUMA-aware allocator.
    ///
    /// If `numa_nodes` is empty, falls back to a global allocator.
    pub(crate) fn new_numa(
        capacity: usize,
        numa_nodes: &[NumaNode],
        use_hugepages: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Self {
        if numa_nodes.is_empty() {
            warn!(
                "NUMA allocator requested but no nodes provided, falling back to global allocator"
            );
            return Self::new_global(capacity, use_hugepages, unit_hint);
        }
        Self::Numa(NumaAwarePinnedPools::new(
            capacity,
            numa_nodes,
            use_hugepages,
            unit_hint,
        ))
    }

    /// Allocate pinned memory.
    ///
    /// For global allocators, `numa_node` is ignored.
    /// For NUMA allocators, allocates from the specified node's pool.
    pub(crate) fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: NumaNode,
    ) -> Option<Arc<PinnedAllocation>> {
        match self {
            Self::Global(pool) => pool.allocate(size).map(Arc::new),
            Self::Numa(pools) => pools.allocate(numa_node, size),
        }
    }

    /// Get aggregate usage: (used_bytes, total_bytes)
    pub(crate) fn usage(&self) -> (u64, u64) {
        match self {
            Self::Global(pool) => pool.usage(),
            Self::Numa(pools) => pools.total_usage(),
        }
    }

    /// Get the largest contiguous free region available.
    ///
    /// For NUMA allocators, returns the sum of all pools' free space
    /// (conservative estimate for fragmentation).
    pub(crate) fn largest_free_allocation(&self) -> u64 {
        match self {
            Self::Global(pool) => pool.largest_free_allocation(),
            Self::Numa(pools) => {
                // For monitoring purposes, return the minimum largest free region
                // across all NUMA nodes. This reflects the most constrained node
                // and is accurate for checking if a contiguous allocation can succeed.
                pools
                    .pools
                    .values()
                    .map(|p| p.largest_free_allocation())
                    .min()
                    .unwrap_or(0)
            }
        }
    }

    /// Check if this is a NUMA allocator.
    pub(crate) fn is_numa(&self) -> bool {
        matches!(self, Self::Numa(_))
    }
}

// SAFETY: PinnedAllocator owns Arc<PinnedMemoryPool> or NumaAwarePinnedPools,
// both of which are Send + Sync.
unsafe impl Send for PinnedAllocator {}
unsafe impl Sync for PinnedAllocator {}
