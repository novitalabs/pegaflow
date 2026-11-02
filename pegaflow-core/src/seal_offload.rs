// ============================================================================
// Seal Offload Types
//
// Shared types for sealed block metadata, used by SSD cache.
// ============================================================================

use std::ptr::NonNull;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::backing::SSD_ALIGNMENT;
use crate::block::{RawBlock, Segment};
use crate::pinned_pool::PinnedAllocation;
use pegaflow_common::NumaNode;

/// Per-slot metadata (one slot = one layer's KV cache).
///
/// Layout-agnostic: uses per-segment sizes instead of `is_split` boolean.
#[derive(Debug, Clone)]
pub struct SlotMeta {
    /// Per-segment sizes. SmallVec inlines up to 2 elements on the stack
    /// (covers K-only MLA and K+V layouts; spills to heap for 3+ segments).
    pub segment_sizes: SmallVec<[u64; 2]>,
    /// Pre-computed total size across all segments (cached for consistency with RawBlock).
    pub total_size: u64,
    /// NUMA node affinity for this slot's GPU.
    pub numa_node: NumaNode,
}

impl SlotMeta {
    /// Create a new SlotMeta, caching total_size from segment_sizes.
    pub fn new(segment_sizes: SmallVec<[u64; 2]>, numa_node: NumaNode) -> Self {
        let total_size = segment_sizes.iter().sum();
        Self {
            segment_sizes,
            total_size,
            numa_node,
        }
    }

    /// Total size across all segments.
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Number of segments.
    pub fn num_segments(&self) -> usize {
        self.segment_sizes.len()
    }

    /// Bytes the slot takes on SSD and in a prefetch buffer: every segment
    /// rounded up to [`SSD_ALIGNMENT`].
    pub fn io_size(&self) -> u64 {
        self.segment_sizes.iter().map(|&s| io_len(s)).sum()
    }
}

/// `len` rounded up to the O_DIRECT alignment.
///
/// KV layouts pad segments themselves, so this is the identity for them.
/// Object segments keep their logical size; their allocation is padded, so
/// the rounded tail stays inside it.
pub(crate) fn io_len(len: u64) -> u64 {
    len.next_multiple_of(SSD_ALIGNMENT as u64)
}

/// Reconstruct a `RawBlock` from a pinned allocation at a given offset.
///
/// # Safety
/// Caller must ensure `allocation.as_ptr() + offset + meta.io_size()` is within bounds.
pub(crate) unsafe fn reconstruct_raw_block(
    meta: &SlotMeta,
    allocation: Arc<PinnedAllocation>,
    base_offset: usize,
) -> RawBlock {
    let base_ptr = allocation.as_ptr() as *mut u8;
    debug_assert!(
        !base_ptr.is_null(),
        "reconstruct_raw_block: allocation pointer must be non-null"
    );
    debug_assert!(
        base_offset.checked_add(meta.io_size() as usize).is_some(),
        "reconstruct_raw_block: offset + io_size overflows usize"
    );
    let mut offset = base_offset;
    let mut segments = Vec::with_capacity(meta.segment_sizes.len());
    for &seg_size in &meta.segment_sizes {
        // Safety: caller guarantees base_ptr + offset + seg_size is within allocation.
        // base_ptr is non-null (from PinnedAllocation::NonNull), so add() preserves non-null.
        let ptr = unsafe {
            NonNull::new(base_ptr.add(offset))
                .expect("segment pointer within allocation must be non-null")
        };
        segments.push(Segment::new(
            ptr,
            seg_size as usize,
            Arc::clone(&allocation),
        ));
        offset += io_len(seg_size) as usize;
    }
    RawBlock::new(segments)
}

/// Build iovecs for writing a slot's RawBlock to SSD, each segment's length
/// rounded by [`io_len`].
#[inline]
pub(crate) fn write_iovecs(slot: &RawBlock) -> Vec<(*const u8, usize)> {
    slot.segment_iovecs()
        .map(|(ptr, size)| (ptr.as_ptr() as *const u8, io_len(size as u64) as usize))
        .collect()
}

/// Build iovecs for reading a slot from SSD into a buffer.
/// `base` is the buffer base pointer, `offset` is the slot's offset within the buffer.
///
/// # Safety
/// Caller must ensure `base + offset + io_size` is within a valid allocation.
#[inline]
pub(crate) unsafe fn read_iovecs(
    meta: &SlotMeta,
    base: *mut u8,
    offset: usize,
) -> Vec<(*mut u8, usize)> {
    debug_assert!(
        !base.is_null(),
        "read_iovecs: base pointer must be non-null"
    );
    let mut current_offset = offset;
    let mut result = Vec::with_capacity(meta.segment_sizes.len());
    for &seg_size in &meta.segment_sizes {
        let size = io_len(seg_size) as usize;
        // Safety: caller ensures base + current_offset + size is within valid allocation.
        let ptr = unsafe { base.add(current_offset) };
        result.push((ptr, size));
        current_offset += size;
    }
    debug_assert_eq!(
        current_offset - offset,
        meta.io_size() as usize,
        "read_iovecs: accumulated offset mismatch"
    );
    result
}
