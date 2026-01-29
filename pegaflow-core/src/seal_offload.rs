// ============================================================================
// Seal Offload Types
//
// Shared types for sealed block metadata, used by SSD cache.
// ============================================================================

use serde::{Deserialize, Serialize};

/// Per-slot metadata (one slot = one layer's KV cache)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotMeta {
    /// K and V stored separately (split) or together (contiguous)
    pub is_split: bool,
    /// Total size in bytes (K + V combined)
    pub size: u64,
}

use crate::block::LayerBlock;

impl SlotMeta {
    /// Build iovecs for writing a slot to SSD.
    /// Split layout: [K, V], Contiguous layout: [KV]
    #[inline]
    pub fn write_iovecs(&self, slot: &LayerBlock) -> Vec<(*const u8, usize)> {
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
    pub unsafe fn read_iovecs(&self, base: *mut u8, offset: usize) -> Vec<(*mut u8, usize)> {
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
