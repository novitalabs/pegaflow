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
