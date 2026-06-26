// ============================================================================
// Block types for StorageEngine
// ============================================================================

use std::fmt;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

use crate::pinned_pool::{MappedPinnedPtr, PinnedAllocation};
use pegaflow_common::NumaNode;

// ============================================================================
// BlockKey
// ============================================================================

pub use pegaflow_common::BlockKey;

pub type BlockHash = Vec<u8>;

/// Per-layer save input: layer name + block IDs + content hashes.
pub struct LayerSave {
    pub layer_name: String,
    pub block_ids: Vec<usize>,
    pub block_hashes: Vec<Vec<u8>>,
}

// ============================================================================
// Prefetch Status
// ============================================================================

/// Result of checking prefix hits with SSD prefetch support
#[derive(Clone)]
pub enum PrefetchStatus {
    /// Blocks are being prefetched - caller should retry
    Loading,
    /// Terminal state: all ready prefix blocks are owned by the caller.
    Ready {
        blocks: Vec<Arc<SealedBlock>>,
        missing: usize,
    },
}

impl fmt::Debug for PrefetchStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loading => f.write_str("Loading"),
            Self::Ready { blocks, missing } => f
                .debug_struct("Ready")
                .field("blocks", &blocks.len())
                .field("missing", missing)
                .finish(),
        }
    }
}

// ============================================================================
// Segment + RawBlock (storage-level, layout-agnostic)
// ============================================================================

/// A single contiguous memory segment in pinned memory.
pub(crate) struct Segment {
    ptr: MappedPinnedPtr,
    size: usize,
    /// Held for RAII drop semantics -- memory freed when last Arc drops.
    _allocation: Arc<PinnedAllocation>,
}

impl Segment {
    pub(crate) fn new(ptr: NonNull<u8>, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        let ptr = allocation.mapped_ptr_for_host_range(ptr, size);
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
    segments: BlockSegments,
    /// Total size across all segments (for footprint tracking).
    total_size: usize,
}

enum BlockSegments {
    One(Segment),
    Two([Segment; 2]),
    Many(Box<[Segment]>),
}

impl BlockSegments {
    fn from_vec(segments: Vec<Segment>) -> Self {
        match segments.len() {
            1 => {
                let mut segments = segments.into_iter();
                Self::One(segments.next().expect("length checked"))
            }
            2 => {
                let mut segments = segments.into_iter();
                Self::Two([
                    segments.next().expect("length checked"),
                    segments.next().expect("length checked"),
                ])
            }
            _ => Self::Many(segments.into_boxed_slice()),
        }
    }

    fn as_slice(&self) -> &[Segment] {
        match self {
            Self::One(segment) => std::slice::from_ref(segment),
            Self::Two(segments) => segments,
            Self::Many(segments) => segments,
        }
    }
}

impl RawBlock {
    /// Create from an arbitrary list of segments.
    /// Caller decides the segment layout; RawBlock is layout-agnostic.
    pub(crate) fn new(segments: Vec<Segment>) -> Self {
        debug_assert!(!segments.is_empty(), "RawBlock requires at least 1 segment");
        let total_size = segments.iter().map(|s| s.size).sum();
        Self {
            segments: BlockSegments::from_vec(segments),
            total_size,
        }
    }

    pub(crate) fn single_segment(segment: Segment) -> Self {
        let total_size = segment.size;
        Self {
            segments: BlockSegments::One(segment),
            total_size,
        }
    }

    pub(crate) fn two_segments(k_segment: Segment, v_segment: Segment) -> Self {
        let total_size = k_segment.size + v_segment.size;
        Self {
            segments: BlockSegments::Two([k_segment, v_segment]),
            total_size,
        }
    }

    /// Number of segments.
    pub(crate) fn num_segments(&self) -> usize {
        self.segments.as_slice().len()
    }

    /// Get segment pointer by index.
    pub(crate) fn segment_ptr(&self, index: usize) -> Option<NonNull<u8>> {
        self.segments.as_slice().get(index).map(|s| s.ptr.host())
    }

    /// Get mapped host/device pointer pair by index.
    pub(crate) fn segment_mapped_ptr(&self, index: usize) -> Option<MappedPinnedPtr> {
        self.segments.as_slice().get(index).map(|s| s.ptr)
    }

    /// Get segment size by index.
    pub(crate) fn segment_size(&self, index: usize) -> Option<usize> {
        self.segments.as_slice().get(index).map(|s| s.size)
    }

    /// Iterator over (NonNull<u8>, size) pairs -- used for SSD I/O.
    pub(crate) fn segment_iovecs(&self) -> impl Iterator<Item = (NonNull<u8>, usize)> + '_ {
        self.segments
            .as_slice()
            .iter()
            .map(|s| (s.ptr.host(), s.size))
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
pub struct LayerBlock<'a> {
    raw: &'a RawBlock,
}

impl<'a> LayerBlock<'a> {
    /// Wrap a RawBlock as a KV layer block.
    /// Panics if the block has zero segments (violated invariant from construction).
    pub fn new(raw: &'a RawBlock) -> Self {
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

    /// Total size across all segments (K + V if split).
    pub fn size(&self) -> usize {
        self.raw.memory_footprint() as usize
    }
}

// ============================================================================
// Sealed Block (read path, immutable)
// ============================================================================

/// Immutable block after all slots are filled. Exposed to callers.
pub struct SealedBlock {
    slots: Box<[RawBlock]>,
    footprint: u64,
    /// Per-slot NUMA affinity; always covers every slot (`len == slots.len()`).
    /// Used by the SSD write path and advertised on cross-node transfer.
    slot_numas: Vec<NumaNode>,
}

impl SealedBlock {
    pub(crate) fn get_slot(&self, slot_id: usize) -> Option<&RawBlock> {
        self.slots.get(slot_id)
    }

    pub(crate) fn memory_footprint(&self) -> u64 {
        self.footprint
    }

    /// Get all slots (for serialization / cross-node transfer)
    pub fn slots(&self) -> &[RawBlock] {
        &self.slots
    }

    /// Per-slot NUMA affinity (used by SSD write path and cross-node transfer).
    pub fn slot_numas(&self) -> &[NumaNode] {
        &self.slot_numas
    }

    /// Create from per-slot `(block, NUMA)` pairs (deserialization / prefetch rebuild).
    /// Pairing keeps each slot and its advertised NUMA affinity in lockstep, so the
    /// cross-node serve path can never truncate slots against a short `slot_numas`.
    pub(crate) fn from_slots(slots: Vec<(RawBlock, NumaNode)>) -> Self {
        let mut blocks = Vec::with_capacity(slots.len());
        let mut slot_numas = Vec::with_capacity(slots.len());
        let mut footprint = 0u64;
        for (block, numa) in slots {
            footprint = footprint.saturating_add(block.memory_footprint());
            blocks.push(block);
            slot_numas.push(numa);
        }
        Self {
            slots: blocks.into_boxed_slice(),
            footprint,
            slot_numas,
        }
    }

    /// Create from slots with pre-computed footprint (internal use)
    fn from_slots_with_footprint(
        slots: Box<[RawBlock]>,
        footprint: u64,
        slot_numas: Vec<NumaNode>,
    ) -> Self {
        Self {
            slots,
            footprint,
            slot_numas,
        }
    }

    /// Create from a fully populated, slot-id ordered insert batch.
    pub(crate) fn from_ordered_slot_inserts(
        slots: Vec<(usize, RawBlock)>,
        total_slots: usize,
        numa_node: NumaNode,
    ) -> Result<Self, Vec<(usize, RawBlock)>> {
        if slots.len() != total_slots
            || slots
                .iter()
                .enumerate()
                .any(|(expected, (slot_id, _))| *slot_id != expected)
        {
            return Err(slots);
        }

        let mut footprint = 0u64;
        let mut blocks = Vec::with_capacity(total_slots);
        for (_, block) in slots {
            footprint = footprint.saturating_add(block.memory_footprint());
            blocks.push(block);
        }

        Ok(Self::from_slots_with_footprint(
            blocks.into_boxed_slice(),
            footprint,
            vec![numa_node; total_slots],
        ))
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
    slots: Vec<Option<RawBlock>>,
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
            slots: (0..total_slots).map(|_| None).collect(),
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
        block: RawBlock,
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
        let slots: Vec<RawBlock> = self
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
