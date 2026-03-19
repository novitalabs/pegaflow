// ============================================================================
// Block types for StorageEngine
// ============================================================================

use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use crate::pinned_pool::PinnedAllocation;
use pegaflow_common::NumaNode;

// ============================================================================
// BlockKey
// ============================================================================

/// Key for identifying blocks in storage, including namespace for model isolation.
///
/// NOTE: Using String for namespace is simple but adds ~20-50 bytes overhead per key.
/// Future optimization: intern namespaces to u32 IDs (saves memory, faster comparison).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockKey {
    /// Namespace for model isolation (e.g., model name, or empty string for shared storage)
    pub namespace: String,
    /// Block content hash
    pub hash: Vec<u8>,
}

impl BlockKey {
    pub fn new(namespace: String, hash: Vec<u8>) -> Self {
        Self { namespace, hash }
    }

    /// Estimate the memory size of this BlockKey in bytes
    /// Used for cache size-aware eviction policies
    pub fn estimated_size(&self) -> u64 {
        // Size = namespace string capacity + hash vec capacity + struct overhead (48 bytes)
        // Using capacity() instead of len() to account for actual heap-allocated memory
        (self.namespace.capacity() + self.hash.capacity() + 48) as u64
    }
}

pub type BlockHash = Vec<u8>;

/// Per-layer save input: layer name + block IDs + content hashes.
pub struct LayerSave {
    pub layer_name: String,
    pub block_ids: Vec<i32>,
    pub block_hashes: Vec<Vec<u8>>,
}

// ============================================================================
// Block Status and Prefetch Status
// ============================================================================

/// Status of a block in the storage hierarchy
#[derive(Debug, Clone)]
pub enum BlockStatus {
    /// Block is in memory cache, ready to use
    Cached,
    /// Block is being written (inflight)
    Inflight,
    /// Block is being prefetched from SSD
    Prefetching,
    /// Block exists in SSD, can trigger prefetch
    InSsd,
    /// Block not found anywhere
    Miss,
}

/// Result of checking prefix hits with prefetch support
#[derive(Debug, Clone)]
pub enum PrefetchStatus {
    /// Blocks are being prefetched - caller should retry
    Loading { hit: usize, loading: usize },
    /// Terminal state: hit/missing counts final (missing=0 means full hit)
    Done { hit: usize, missing: usize },
}

// ============================================================================
// Segment + RawBlock (storage-level, layout-agnostic)
// ============================================================================

/// A single contiguous memory segment in pinned memory.
pub(crate) struct Segment {
    ptr: NonNull<u8>,
    size: usize,
    /// Held for RAII drop semantics -- memory freed when last Arc drops.
    _allocation: Arc<PinnedAllocation>,
}

impl Segment {
    pub(crate) fn new(ptr: NonNull<u8>, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        Self {
            ptr,
            size,
            _allocation: allocation,
        }
    }
}

// Safety: Segment holds NonNull<u8> pointing to pinned memory whose lifetime
// is managed by Arc<PinnedAllocation>. PinnedAllocation itself is Send+Sync.
unsafe impl Send for Segment {}
unsafe impl Sync for Segment {}

/// Storage-level block: an ordered list of opaque memory segments.
/// No awareness of K/V layout -- that's the caller's concern.
///
/// RawBlock automatically derives Send+Sync from Segment.
pub struct RawBlock {
    segments: Box<[Segment]>,
    /// Total size across all segments (for footprint tracking).
    total_size: usize,
}

impl RawBlock {
    /// Create from an arbitrary list of segments.
    /// Caller decides the segment layout; RawBlock is layout-agnostic.
    pub(crate) fn new(segments: Vec<Segment>) -> Self {
        debug_assert!(!segments.is_empty(), "RawBlock requires at least 1 segment");
        let total_size = segments.iter().map(|s| s.size).sum();
        Self {
            segments: segments.into_boxed_slice(),
            total_size,
        }
    }

    /// Number of segments.
    pub(crate) fn num_segments(&self) -> usize {
        self.segments.len()
    }

    /// Get segment pointer by index.
    pub(crate) fn segment_ptr(&self, index: usize) -> Option<NonNull<u8>> {
        self.segments.get(index).map(|s| s.ptr)
    }

    /// Get segment size by index.
    pub(crate) fn segment_size(&self, index: usize) -> Option<usize> {
        self.segments.get(index).map(|s| s.size)
    }

    /// Iterator over (NonNull<u8>, size) pairs -- used for SSD I/O.
    pub(crate) fn segment_iovecs(&self) -> impl Iterator<Item = (NonNull<u8>, usize)> + '_ {
        self.segments.iter().map(|s| (s.ptr, s.size))
    }

    /// Total memory footprint.
    pub(crate) fn memory_footprint(&self) -> u64 {
        self.total_size as u64
    }
}

// ============================================================================
// LayerBlock (thin KV wrapper, service/GPU layer only)
// ============================================================================

/// Layer-aware view: interprets RawBlock segments as K/V cache data.
/// Lives in the service/GPU layer, NOT in storage.
///
/// Invariant: inner RawBlock has >= 1 segment.
pub struct LayerBlock {
    raw: Arc<RawBlock>,
}

impl LayerBlock {
    /// Wrap a RawBlock as a KV layer block.
    /// Panics if the block has zero segments (violated invariant from construction).
    pub fn new(raw: Arc<RawBlock>) -> Self {
        debug_assert!(
            raw.num_segments() > 0,
            "LayerBlock requires at least 1 segment"
        );
        Self { raw }
    }

    /// K segment pointer (always segment 0).
    pub fn k_ptr(&self) -> *const u8 {
        self.raw.segment_ptr(0).unwrap().as_ptr()
    }

    /// K segment size.
    pub fn k_size(&self) -> usize {
        self.raw.segment_size(0).unwrap()
    }

    /// V segment pointer (segment 1 if split, None if contiguous).
    pub fn v_ptr(&self) -> Option<*const u8> {
        self.raw.segment_ptr(1).map(|p| p.as_ptr() as *const u8)
    }

    /// V segment size (None if contiguous).
    pub fn v_size(&self) -> Option<usize> {
        self.raw.segment_size(1)
    }
}

// ============================================================================
// Sealed Block (read path, immutable)
// ============================================================================

/// Immutable block after all slots are filled. Exposed to callers.
pub struct SealedBlock {
    slots: Box<[Arc<RawBlock>]>,
    footprint: u64,
    /// Per-slot NUMA affinity, carried from InflightBlock for SSD write path.
    /// Empty when reconstructed from SSD prefetch (NUMA info lives in SlotMeta).
    slot_numas: Vec<NumaNode>,
}

impl SealedBlock {
    pub(crate) fn get_slot(&self, slot_id: usize) -> Option<&Arc<RawBlock>> {
        self.slots.get(slot_id)
    }

    pub(crate) fn memory_footprint(&self) -> u64 {
        self.footprint
    }

    /// Get all slots (for serialization)
    pub(crate) fn slots(&self) -> &[Arc<RawBlock>] {
        &self.slots
    }

    /// Per-slot NUMA affinity for SSD write path.
    pub(crate) fn slot_numas(&self) -> &[NumaNode] {
        &self.slot_numas
    }

    /// Create from a vec of slots (for deserialization / prefetch rebuild)
    pub(crate) fn from_slots(slots: Vec<Arc<RawBlock>>) -> Self {
        let footprint = slots.iter().map(|s| s.memory_footprint()).sum();
        Self {
            slots: slots.into_boxed_slice(),
            footprint,
            slot_numas: Vec::new(),
        }
    }

    /// Create from slots with pre-computed footprint (internal use)
    fn from_slots_with_footprint(
        slots: Box<[Arc<RawBlock>]>,
        footprint: u64,
        slot_numas: Vec<NumaNode>,
    ) -> Self {
        Self {
            slots,
            footprint,
            slot_numas,
        }
    }
}

// ============================================================================
// SlotInsertResult
// ============================================================================

/// Result of inserting a slot into an inflight block.
pub(crate) enum SlotInsertResult {
    /// Slot was newly inserted.
    Inserted {
        completed: bool,
        footprint_added: u64,
    },
    /// Slot already existed (no-op).
    Duplicate,
}

// ============================================================================
// Inflight Block (write path, mutable)
// ============================================================================

/// Block that is still being written. Internal to StorageEngine.
pub(crate) struct InflightBlock {
    slots: Vec<Option<Arc<RawBlock>>>,
    remaining: usize,
    total_slots: usize,
    footprint: u64,
    created_at: Instant,
    /// Per-slot NUMA node affinity, tracked during insertion for deterministic
    /// per-block NUMA assignment when the block is sealed.
    slot_numas: Vec<NumaNode>,
}

impl InflightBlock {
    pub(crate) fn new(total_slots: usize) -> Self {
        Self {
            slots: vec![None; total_slots],
            remaining: total_slots,
            total_slots,
            footprint: 0,
            created_at: Instant::now(),
            slot_numas: vec![NumaNode::UNKNOWN; total_slots],
        }
    }

    /// Returns the age of this inflight block.
    pub(crate) fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the number of filled slots.
    pub(crate) fn filled_count(&self) -> usize {
        self.total_slots - self.remaining
    }

    /// Returns the total number of slots.
    pub(crate) fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Returns the current memory footprint of all inserted slots.
    pub(crate) fn footprint(&self) -> u64 {
        self.footprint
    }

    /// Insert a slot idempotently. Duplicate inserts are no-ops.
    pub(crate) fn insert_slot(
        &mut self,
        slot_id: usize,
        block: Arc<RawBlock>,
        numa_node: NumaNode,
    ) -> SlotInsertResult {
        debug_assert!(
            slot_id < self.total_slots,
            "slot_id {} must be < total_slots {}",
            slot_id,
            self.total_slots
        );

        if self.slots[slot_id].is_some() {
            return SlotInsertResult::Duplicate;
        }

        let footprint_added = block.memory_footprint();
        self.footprint += footprint_added;
        self.slots[slot_id] = Some(block);
        self.slot_numas[slot_id] = numa_node;
        self.remaining = self
            .remaining
            .checked_sub(1)
            .expect("remaining should not underflow");

        SlotInsertResult::Inserted {
            completed: self.remaining == 0,
            footprint_added,
        }
    }

    /// Seal the block, converting to immutable SealedBlock.
    /// Panics if not all slots are filled.
    pub(crate) fn seal(self) -> SealedBlock {
        let slots: Vec<Arc<RawBlock>> = self
            .slots
            .into_iter()
            .map(|opt| opt.expect("all slots must be filled before sealing"))
            .collect();
        SealedBlock::from_slots_with_footprint(
            slots.into_boxed_slice(),
            self.footprint,
            self.slot_numas,
        )
    }
}
