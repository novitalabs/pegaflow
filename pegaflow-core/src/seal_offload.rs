// ============================================================================
// Seal Offload Types
//
// Shared types for sealed block metadata, used by SSD cache.
// ============================================================================

use crate::numa::NumaNode;

/// Per-slot metadata (one slot = one layer's KV cache)
#[derive(Debug, Clone)]
pub struct SlotMeta {
    /// K and V stored separately (split) or together (contiguous)
    pub is_split: bool,
    /// Total size in bytes (K + V combined)
    pub size: u64,
    /// NUMA node affinity for this slot's GPU
    pub numa_node: NumaNode,
}

use std::sync::Arc;

use crate::block::LayerBlock;
use crate::pinned_pool::PinnedAllocation;

impl SlotMeta {
    /// Reconstruct a `LayerBlock` from a pinned allocation at a given offset.
    ///
    /// # Safety
    /// Caller must ensure `allocation.as_ptr() + offset + self.size` is within bounds.
    pub(crate) unsafe fn make_layer_block(
        &self,
        allocation: Arc<PinnedAllocation>,
        offset: usize,
    ) -> Arc<LayerBlock> {
        let base_ptr = allocation.as_ptr() as *mut u8;
        let slot_size = self.size as usize;

        if self.is_split {
            let half = slot_size / 2;
            let k_ptr = unsafe { base_ptr.add(offset) };
            let v_ptr = unsafe { base_ptr.add(offset + half) };
            Arc::new(LayerBlock::new_split(
                k_ptr,
                v_ptr,
                slot_size,
                Arc::clone(&allocation),
                allocation,
            ))
        } else {
            let ptr = unsafe { base_ptr.add(offset) };
            Arc::new(LayerBlock::new_contiguous(ptr, slot_size, allocation))
        }
    }

    /// Build iovecs for writing a slot to SSD.
    /// Split layout: [K, V], Contiguous layout: [KV]
    #[inline]
    pub(crate) fn write_iovecs(&self, slot: &LayerBlock) -> Vec<(*const u8, usize)> {
        if self.is_split {
            let half = self.size as usize / 2;
            vec![(slot.k_ptr(), half), (slot.v_ptr().unwrap(), half)]
        } else {
            vec![(slot.k_ptr(), self.size as usize)]
        }
    }

    /// Build iovecs for reading a slot from SSD into a buffer.
    /// `base` is the buffer base pointer, `offset` is the slot's offset within the buffer.
    ///
    /// # Safety
    /// Caller must ensure `base + offset + size` is within a valid allocation.
    #[inline]
    pub(crate) unsafe fn read_iovecs(&self, base: *mut u8, offset: usize) -> Vec<(*mut u8, usize)> {
        let size = self.size as usize;
        if self.is_split {
            let half = size / 2;
            // SAFETY: Caller ensures base + offset + size is within valid allocation
            unsafe { vec![(base.add(offset), half), (base.add(offset + half), half)] }
        } else {
            // SAFETY: Caller ensures base + offset + size is within valid allocation
            unsafe { vec![(base.add(offset), size)] }
        }
    }
}
