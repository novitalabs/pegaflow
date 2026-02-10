// ============================================================================
// Block types for StorageEngine
// ============================================================================

use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use crate::pinned_pool::PinnedAllocation;

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
}

pub type BlockHash = Vec<u8>;

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
// LayerBlock (pinned memory holder)
// ============================================================================

/// CPU block data stored in pinned memory for a single layer/TP slot.
pub struct LayerBlock {
    /// Pointer to K segment (or combined data if contiguous)
    k_ptr: std::ptr::NonNull<u8>,
    /// Pointer to V segment (if stored separately)
    v_ptr: Option<std::ptr::NonNull<u8>>,
    size: usize,
    /// Shared RAII allocation handle for K memory (automatically freed when last reference drops)
    #[allow(dead_code)]
    k_allocation: Arc<PinnedAllocation>,
    /// Shared RAII allocation handle for V memory (if separate from K)
    #[allow(dead_code)]
    v_allocation: Option<Arc<PinnedAllocation>>,
}

impl LayerBlock {
    pub fn new_contiguous(ptr: *mut u8, size: usize, allocation: Arc<PinnedAllocation>) -> Self {
        let k_ptr =
            std::ptr::NonNull::new(ptr).expect("contiguous block K pointer must be non-null");
        Self {
            k_ptr,
            v_ptr: None,
            size,
            k_allocation: allocation,
            v_allocation: None,
        }
    }

    pub fn new_split(
        k_ptr: *mut u8,
        v_ptr: *mut u8,
        size: usize,
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
    ) -> Self {
        let k_ptr = std::ptr::NonNull::new(k_ptr).expect("split block K pointer must be non-null");
        let v_ptr = std::ptr::NonNull::new(v_ptr).expect("split block V pointer must be non-null");
        Self {
            k_ptr,
            v_ptr: Some(v_ptr),
            size,
            k_allocation,
            v_allocation: Some(v_allocation),
        }
    }

    pub fn k_ptr(&self) -> *const u8 {
        self.k_ptr.as_ptr()
    }

    pub fn v_ptr(&self) -> Option<*const u8> {
        self.v_ptr.map(|ptr| ptr.as_ptr() as *const u8)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Total pinned memory occupied by this layer block.
    pub fn memory_footprint(&self) -> u64 {
        self.size as u64
    }
}

// Safety: pinned memory ownership is tracked by Arc counters on the allocations.
unsafe impl Send for LayerBlock {}
unsafe impl Sync for LayerBlock {}

// ============================================================================
// Sealed Block (read path, immutable)
// ============================================================================

/// Immutable block after all slots are filled. Exposed to callers.
pub struct SealedBlock {
    slots: Box<[Arc<LayerBlock>]>,
    footprint: u64,
}

impl SealedBlock {
    pub fn get_slot(&self, slot_id: usize) -> Option<&Arc<LayerBlock>> {
        self.slots.get(slot_id)
    }

    pub fn memory_footprint(&self) -> u64 {
        self.footprint
    }

    /// Get all slots (for serialization)
    pub fn slots(&self) -> &[Arc<LayerBlock>] {
        &self.slots
    }

    /// Create from a vec of slots (for deserialization)
    pub fn from_slots(slots: Vec<Arc<LayerBlock>>) -> Self {
        let footprint = slots.iter().map(|s| s.memory_footprint()).sum();
        Self {
            slots: slots.into_boxed_slice(),
            footprint,
        }
    }

    /// Create from slots with pre-computed footprint (internal use)
    pub(crate) fn from_slots_with_footprint(slots: Box<[Arc<LayerBlock>]>, footprint: u64) -> Self {
        Self { slots, footprint }
    }
}

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug)]
pub enum BlockInsertError {
    SlotOutOfBounds { slot_id: usize, total_slots: usize },
    SlotCountMismatch { expected: usize, got: usize },
}

impl fmt::Display for BlockInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockInsertError::SlotOutOfBounds {
                slot_id,
                total_slots,
            } => {
                write!(
                    f,
                    "slot_id {} out of bounds ({} slots)",
                    slot_id, total_slots
                )
            }
            BlockInsertError::SlotCountMismatch { expected, got } => {
                write!(f, "slot count mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

impl std::error::Error for BlockInsertError {}

// ============================================================================
// Inflight Block (write path, mutable)
// ============================================================================

/// Block that is still being written. Internal to StorageEngine.
pub(crate) struct InflightBlock {
    slots: Vec<Option<Arc<LayerBlock>>>,
    remaining: usize,
    total_slots: usize,
    footprint: u64,
    created_at: Instant,
}

impl InflightBlock {
    pub fn new(total_slots: usize) -> Self {
        Self {
            slots: vec![None; total_slots],
            remaining: total_slots,
            total_slots,
            footprint: 0,
            created_at: Instant::now(),
        }
    }

    /// Returns the age of this inflight block.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }

    /// Returns the number of filled slots.
    pub fn filled_count(&self) -> usize {
        self.total_slots - self.remaining
    }

    /// Returns the total number of slots.
    pub fn total_slots(&self) -> usize {
        self.total_slots
    }

    /// Returns the current memory footprint of all inserted slots.
    pub fn footprint(&self) -> u64 {
        self.footprint
    }

    pub fn slot_exists(&self, slot_id: usize) -> bool {
        self.slots
            .get(slot_id)
            .and_then(|opt| opt.as_ref())
            .is_some()
    }

    /// Fast path for save hot path.
    ///
    /// Preconditions are expected to be validated by caller:
    /// - `slot_id < self.total_slots`
    /// - `total_slots` consistency was checked where the inflight block is looked up
    ///
    /// Returns:
    /// - `None` if the slot is already filled (no-op)
    /// - `Some(completed)` if this call inserted a slot
    pub fn try_insert_slot_fast(&mut self, slot_id: usize, block: Arc<LayerBlock>) -> Option<bool> {
        debug_assert!(
            slot_id < self.total_slots,
            "slot_id {} must be < total_slots {}",
            slot_id,
            self.total_slots
        );

        if self.slots[slot_id].is_some() {
            return None;
        }

        self.footprint += block.memory_footprint();
        self.slots[slot_id] = Some(block);
        self.remaining = self
            .remaining
            .checked_sub(1)
            .expect("remaining should not underflow");
        Some(self.remaining == 0)
    }

    /// Seal the block, converting to immutable SealedBlock.
    /// Panics if not all slots are filled.
    pub fn seal(self) -> SealedBlock {
        let slots: Vec<Arc<LayerBlock>> = self
            .slots
            .into_iter()
            .map(|opt| opt.expect("all slots must be filled before sealing"))
            .collect();
        SealedBlock::from_slots_with_footprint(slots.into_boxed_slice(), self.footprint)
    }
}
