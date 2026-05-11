use parking_lot::Mutex;
use std::{
    collections::HashMap,
    num::NonZeroU64,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bytesize::ByteSize;
use log::{error, info, warn};

use crate::allocator::{Allocation, ScaledOffsetAllocator};
use crate::metrics::core_metrics;
use crate::pinned_mem::PinnedMemory;
use pegaflow_common::{NumaNode, run_on_numa};

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
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable pointer to the allocated memory
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get the underlying NonNull pointer.
    pub(crate) fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
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
    allocatable_bytes: u64,
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
    /// If `ssd_enabled` is true, uses regular pinned memory instead of write-combined,
    /// because SSD reads back data into CPU memory and write-combined has extremely slow CPU reads.
    /// If `unit_size_hint` is provided, the allocator rounds allocations up to this size.
    fn new(
        pool_size: usize,
        use_hugepages: bool,
        ssd_enabled: bool,
    ) -> Result<Self, crate::pinned_mem::PinnedMemError> {
        let backing = if use_hugepages {
            PinnedMemory::allocate_hugepages(pool_size)
        } else if ssd_enabled {
            PinnedMemory::allocate_regular(pool_size)
        } else {
            PinnedMemory::allocate_write_combined(pool_size)
        }?; unit_size_hint: Option<NonZeroU64>,
    ) -> Self {
        assert!(
            pool_size != 0,
            "Pinned memory pool size must be greater than zero"
        );

        let backing = if use_hugepages {
            info!("Allocating pinned memory pool with huge pages");
            PinnedMemory::allocate_hugepages(pool_size)
                .expect("Failed to allocate pinned memory pool with huge pages")
        } else if ssd_enabled {
            info!("Allocating pinned memory pool with regular pages (SSD enabled)");
            PinnedMemory::allocate_regular(pool_size)
                .expect("Failed to allocate regular pinned memory pool")
        } else {
            info!("Allocating pinned memory pool with write-combined pages");
            PinnedMemory::allocate_write_combined(pool_size)
                .expect("Failed to allocate pinned memory pool")
        };

        let actual_size = backing.size() as u64;
        let unit_size = Self::compute_unit_size(actual_size, unit_size_hint);
        let max_units = actual_size / unit_size;
        let allocatable_bytes = max_units * unit_size;

        info!(
            "Pinned pool: backing_size={}, unit_size={}, max_units={}, allocatable_size={}",
            ByteSize(actual_size),
            ByteSize(unit_size),
            max_units,
            ByteSize(allocatable_bytes)
        );

        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(allocatable_bytes) {
            metrics.pool_capacity_bytes.add(capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric update: allocatable_bytes={}",
                allocatable_bytes
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
            allocatable_bytes,
        }
    }

    /// Allocate pinned memory from the pool. Returns None when the allocation cannot be satisfied.
    /// Returns a RAII guard that automatically frees the allocation when dropped.
    fn allocate(self: &Arc<Self>, size: NonZeroU64) -> Option<PinnedAllocation> {
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

        let size_bytes = allocation.size_bytes.get();
        let backing_size = self.backing.size() as u64;
        let allocation_end = allocation.offset_bytes.checked_add(size_bytes);
        assert!(
            matches!(allocation_end, Some(end) if end <= backing_size),
            "pinned allocator returned allocation outside backing memory: offset={} size={} backing_size={}",
            allocation.offset_bytes,
            size_bytes,
            backing_size
        );

        let offset = allocation.offset_bytes as usize;
        let ptr = unsafe { self.backing.as_ptr().add(offset) };
        let ptr = NonNull::new(ptr as *mut u8).expect("PinnedMemoryPool returned null pointer");

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
    fn free_internal(&self, allocation: &Allocation) {
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
    fn usage(&self) -> (u64, u64) {
        let allocator = self.allocator.lock();
        let report = allocator.storage_report();
        let total = allocator.total_bytes();
        let used = total - report.total_free_bytes;
        (used, total)
    }

    /// Largest contiguous free region currently available, in bytes.
    fn largest_free_allocation(&self) -> u64 {
        let allocator = self.allocator.lock();
        allocator.storage_report().largest_free_allocation_bytes
    }

    /// Return the base pointer and length of the backing pinned memory.
    fn memory_region(&self) -> (NonNull<u8>, usize) {
        let ptr = NonNull::new(self.backing.as_ptr() as *mut u8)
            .expect("PinnedMemoryPool backing pointer is null");
        (ptr, self.backing.size())
    }
}

impl Drop for PinnedMemoryPool {
    fn drop(&mut self) {
        let metrics = core_metrics();
        if let Ok(capacity_i64) = i64::try_from(self.allocatable_bytes) {
            metrics.pool_capacity_bytes.add(-capacity_i64, &[]);
        } else {
            error!(
                "Pinned pool capacity exceeds i64::MAX; skipping capacity metric cleanup: allocatable_bytes={}",
                self.allocatable_bytes
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
// Sharded Pool Wrapper
// ============================================================================

/// Wrapper that distributes allocations across multiple `PinnedMemoryPool` shards
/// using round-robin. Each shard has its own backing memory and allocator, reducing
/// lock contention under concurrent save workloads.
///
/// `PinnedAllocation` returned by `allocate` holds an `Arc` to the specific shard
/// it was allocated from, so `Drop` automatically frees back to the correct shard.
#[derive(Debug)]
pub(crate) struct ShardedPinnedPool {
    shards: Vec<Arc<PinnedMemoryPool>>,
    cursor: AtomicUsize,
}

impl ShardedPinnedPool {
    /// Create a sharded pool by splitting `total_capacity` evenly across `num_shards` pools.
    ///
    /// When `num_shards <= 1`, a single pool is created (equivalent to the old behavior).
    fn new(
        total_capacity: usize,
        num_shards: usize,
        use_hugepages: bool,
        ssd_enabled: bool,
        unit_size_hint: Option<NonZeroU64>,
    ) -> Self {
        let num_shards = num_shards.max(1);
        let per_shard = total_capacity / num_shards;

        if num_shards > 1 {
            info!(
                "Creating sharded pinned pool: total={}, shards={}, per_shard={}",
                ByteSize(total_capacity as u64),
                num_shards,
                ByteSize(per_shard as u64)
            );
        }

        let shards: Vec<Arc<PinnedMemoryPool>> = (0..num_shards)
            .map(|_| {
                Arc::new(PinnedMemoryPool::new(
                    per_shard,
                    use_hugepages,
                    ssd_enabled,
                    unit_size_hint,
                ))
            })
            .collect();

        Self {
            shards,
            cursor: AtomicUsize::new(0),
        }
    }

    /// Allocate from shards using round-robin with fallback to all shards.
    fn allocate(&self, size: NonZeroU64) -> Option<PinnedAllocation> {
        let n = self.shards.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % n;

        for i in 0..n {
            let idx = (start + i) % n;
            if let Some(alloc) = self.shards[idx].allocate(size) {
                return Some(alloc);
            }
        }
        None
    }

    /// Aggregate usage across all shards.
    fn usage(&self) -> (u64, u64) {
        let mut used = 0u64;
        let mut total = 0u64;
        for shard in &self.shards {
            let (u, t) = shard.usage();
            used += u;
            total += t;
        }
        (used, total)
    }

    /// Largest contiguous free region across all shards.
    fn largest_free_allocation(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.largest_free_allocation())
            .max()
            .unwrap_or(0)
    }

    /// Return all backing memory regions (one per shard).
    fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        self.shards.iter().map(|s| s.memory_region()).collect()
    }
}

// SAFETY: ShardedPinnedPool owns Arc<PinnedMemoryPool> which are Send + Sync.
// The AtomicUsize cursor is inherently thread-safe.
unsafe impl Send for ShardedPinnedPool {}
unsafe impl Sync for ShardedPinnedPool {}

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
    /// Per-NUMA sharded pools indexed by NUMA node ID
    pools: HashMap<u32, ShardedPinnedPool>,
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
    fn new(
        total_capacity: usize,
        numa_nodes: &[NumaNode],
        num_shards: usize,
        use_hugepages: bool,
        ssd_enabled: bool,
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
            "Creating NUMA-aware pools: total={}, nodes={}, per_node={}, shards_per_node={}",
            ByteSize(total_capacity as u64),
            num_nodes,
            ByteSize(per_node_capacity as u64),
            num_shards
        );

        let mut pools = HashMap::new();

        for node in numa_nodes {
            if node.is_unknown() {
                continue;
            }

            let node_id = node.0;
            let hint = unit_size_hint;

            // Allocate sharded pool on a thread pinned to this NUMA node
            let result = run_on_numa(*node, move || {
                ShardedPinnedPool::new(
                    per_node_capacity,
                    num_shards,
                    use_hugepages,
                    ssd_enabled,
                    hint,
                )
            });

            match result {
                Ok(pool) => {
                    info!(
                        "Created pinned pool on NUMA{}: capacity={}",
                        node_id,
                        ByteSize(per_node_capacity as u64)
                    );
                    pools.insert(node_id, pool);
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
    fn allocate(&self, numa_node: NumaNode, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
        if numa_node.is_unknown() {
            error!("UNEXPECTED: allocate called with UNKNOWN NUMA node");
            return None;
        }

        self.pools.get(&numa_node.0)?.allocate(size).map(Arc::new)
    }

    /// Largest contiguous free region for a specific NUMA node.
    fn largest_free_allocation_for_node(&self, numa_node: NumaNode) -> u64 {
        if numa_node.is_unknown() {
            return 0;
        }
        self.pools
            .get(&numa_node.0)
            .map(|pool| pool.largest_free_allocation())
            .unwrap_or(0)
    }

    /// Return all backing memory regions across all NUMA nodes and shards.
    fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        self.pools
            .values()
            .flat_map(|p| p.memory_regions())
            .collect()
    }

    /// Get aggregate usage across all pools: (used_bytes, total_bytes)
    fn total_usage(&self) -> (u64, u64) {
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
    Global(ShardedPinnedPool),
    /// NUMA-aware pools (multi-socket systems)
    Numa(NumaAwarePinnedPools),
}

impl PinnedAllocator {
    /// Create a new global allocator.
    pub(crate) fn new_global(
        capacity: usize,
        num_shards: usize,
        use_hugepages: bool,
        ssd_enabled: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Self {
        Self::Global(ShardedPinnedPool::new(
            capacity,
            num_shards,
            use_hugepages,
            ssd_enabled,
            unit_hint,
        ))
    }

    /// Create a new NUMA-aware allocator.
    ///
    /// If `numa_nodes` is empty, falls back to a global allocator.
    pub(crate) fn new_numa(
        capacity: usize,
        numa_nodes: &[NumaNode],
        num_shards: usize,
        use_hugepages: bool,
        ssd_enabled: bool,
        unit_hint: Option<NonZeroU64>,
    ) -> Self {
        if numa_nodes.is_empty() {
            warn!(
                "NUMA allocator requested but no nodes provided, falling back to global allocator"
            );
            return Self::new_global(capacity, num_shards, use_hugepages, ssd_enabled, unit_hint);
        }
        Self::Numa(NumaAwarePinnedPools::new(
            capacity,
            numa_nodes,
            num_shards,
            use_hugepages,
            ssd_enabled,
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

    /// Largest contiguous free region for the requested allocation target.
    ///
    /// For global allocators, `numa_node` is ignored.
    /// For NUMA allocators, this checks only the target node's pool.
    pub(crate) fn largest_free_allocation_for_node(&self, numa_node: NumaNode) -> u64 {
        match self {
            Self::Global(pool) => pool.largest_free_allocation(),
            Self::Numa(pools) => pools.largest_free_allocation_for_node(numa_node),
        }
    }

    /// Check if this is a NUMA allocator.
    pub(crate) fn is_numa(&self) -> bool {
        matches!(self, Self::Numa(_))
    }

    /// Return all backing memory regions as `(ptr, len)` pairs.
    pub(crate) fn memory_regions(&self) -> Vec<(NonNull<u8>, usize)> {
        match self {
            Self::Global(pool) => pool.memory_regions(),
            Self::Numa(pools) => pools.memory_regions(),
        }
    }
}

// SAFETY: PinnedAllocator owns ShardedPinnedPool or NumaAwarePinnedPools,
// both of which are Send + Sync.
unsafe impl Send for PinnedAllocator {}
unsafe impl Sync for PinnedAllocator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_largest_free_for_node_is_not_global_min() {
        let mut pools = HashMap::new();
        pools.insert(
            0,
            ShardedPinnedPool::new(4096, 1, false, false, NonZeroU64::new(512)),
        );
        pools.insert(
            1,
            ShardedPinnedPool::new(4096, 1, false, false, NonZeroU64::new(512)),
        );
        let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

        let pinned = match &allocator {
            PinnedAllocator::Numa(pools) => pools.pools.get(&0).unwrap(),
            _ => unreachable!(),
        };
        let _held = pinned.allocate(NonZeroU64::new(3584).unwrap()).unwrap();

        let global_min = match &allocator {
            PinnedAllocator::Numa(pools) => pools
                .pools
                .values()
                .map(|p| p.largest_free_allocation())
                .min()
                .unwrap_or(0),
            _ => unreachable!(),
        };
        let node1_largest = allocator.largest_free_allocation_for_node(NumaNode(1));

        assert_eq!(global_min, 512);
        assert_eq!(node1_largest, 4096);
    }

    #[test]
    fn numa_largest_free_for_unknown_node_is_zero() {
        let mut pools = HashMap::new();
        pools.insert(
            0,
            ShardedPinnedPool::new(4096, 1, false, false, NonZeroU64::new(512)),
        );
        let allocator = PinnedAllocator::Numa(NumaAwarePinnedPools { pools });

        assert_eq!(
            allocator.largest_free_allocation_for_node(NumaNode::UNKNOWN),
            0
        );
    }
}
