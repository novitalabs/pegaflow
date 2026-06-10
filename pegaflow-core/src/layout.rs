//! Per-layer KV cache layout.
//!
//! Every layout PegaFlow supports (dense layer-first, fused-buffer strided,
//! MLA single-segment, K/V split) is a parameterization of one affine formula:
//! `addr = data_ptr + block_idx * block_stride + segment_idx * kv_stride`.
//!
//! All validation happens at construction. After that,
//! [`KVCacheLayout::block_copies`] is the only layout question the GPU copy
//! paths ask, and it cannot produce an out-of-bounds range for a valid block
//! index. Padded sizes govern pinned-memory strides and SSD iovecs; GPU copies
//! always use actual (unpadded) sizes.

/// How a block's segments are addressed on the GPU.
#[derive(Debug, Clone, Copy)]
enum SegmentLayout {
    /// The whole block is one contiguous range: a single segment, or K/V
    /// adjacent with no gap (`kv_stride == segment_bytes`).
    Contiguous,
    /// K and V live in separate regions, `kv_stride_bytes` apart: two copies
    /// per block.
    Split { kv_stride_bytes: usize },
}

/// One contiguous device address range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockCopy {
    pub addr: u64,
    pub bytes: usize,
}

/// Device address ranges making up one block on the GPU.
pub(crate) enum BlockCopies {
    /// One contiguous copy.
    Contiguous(BlockCopy),
    /// Two copies: K and V segments in separate regions.
    Split { k: BlockCopy, v: BlockCopy },
}

/// Layout of one layer's KV cache: GPU addressing plus the host-side block
/// shape derived from it.
#[derive(Debug, Clone)]
pub(crate) struct KVCacheLayout {
    /// GPU memory base pointer for this layer's KV cache.
    data_ptr: u64,
    /// Total size of the registered GPU memory region in bytes.
    size_bytes: usize,
    /// Number of blocks in this layer's cache.
    num_blocks: usize,
    /// Byte step between consecutive blocks (per segment region for split
    /// layouts). Defaults to `segment_bytes` (dense); overridden via
    /// [`KVCacheLayout::with_block_stride`] for fused buffers.
    block_stride_bytes: usize,
    /// GPU-side segment size in bytes (one of K or V).
    segment_bytes: usize,
    /// Segments per block (1 contiguous/MLA, 2 for K/V).
    segments: usize,
    /// CPU/SSD-side segment stride, `segment_bytes` rounded up to SSD
    /// alignment. Equals `segment_bytes` when SSD is disabled.
    padded_segment_bytes: usize,
    seg: SegmentLayout,
}

impl KVCacheLayout {
    /// Construct and validate a layout. Rejects null/oversized regions and
    /// overlapping segment configurations (crash early instead of copying
    /// garbage later).
    ///
    /// `segment_bytes` is the size of ONE segment — for K/V split layouts
    /// that is K or V alone, not the whole block. (The registration RPC calls
    /// this field `bytes_per_block` for historical reasons; passing a whole
    /// split block here would mostly pass validation with every address
    /// wrong.)
    pub(crate) fn new(
        data_ptr: u64,
        size_bytes: usize,
        num_blocks: usize,
        segment_bytes: usize,
        kv_stride_bytes: usize,
        segments: usize,
    ) -> Result<Self, String> {
        if data_ptr == 0 {
            return Err("data_ptr must not be null".into());
        }
        if size_bytes == 0 {
            return Err("size_bytes must be > 0".into());
        }
        if segment_bytes == 0 || num_blocks == 0 || segments == 0 {
            return Err("segment_bytes, num_blocks, and segments must be non-zero".into());
        }
        data_ptr
            .checked_add(size_bytes as u64)
            .ok_or_else(|| "data_ptr + size_bytes overflows the address space".to_string())?;
        let block_bytes = segment_bytes
            .checked_mul(segments)
            .ok_or_else(|| "block size overflow".to_string())?;

        let seg = if segments == 1 {
            SegmentLayout::Contiguous
        } else if kv_stride_bytes == 0 {
            return Err("kv_stride_bytes must be > 0 when segments > 1".into());
        } else if kv_stride_bytes < segment_bytes {
            return Err(format!(
                "kv_stride_bytes {kv_stride_bytes} < segment_bytes {segment_bytes}: segments would overlap"
            ));
        } else if kv_stride_bytes == segment_bytes {
            SegmentLayout::Contiguous
        } else if segments == 2 {
            SegmentLayout::Split { kv_stride_bytes }
        } else {
            return Err(format!(
                "split layout (kv_stride_bytes {kv_stride_bytes} > segment_bytes {segment_bytes}) supports exactly 2 segments, got {segments}"
            ));
        };

        let layout = Self {
            data_ptr,
            size_bytes,
            num_blocks,
            block_stride_bytes: match seg {
                SegmentLayout::Contiguous => block_bytes,
                SegmentLayout::Split { .. } => segment_bytes,
            },
            segment_bytes,
            segments,
            padded_segment_bytes: segment_bytes,
            seg,
        };
        layout.check_bounds()?;
        Ok(layout)
    }

    /// Override the per-block stride for a non-contiguous per-layer view —
    /// e.g. one layer's blocks inside a fused buffer holding all layers per
    /// block, where consecutive blocks sit `stride` apart but each copy spans
    /// only the layer's own bytes.
    ///
    /// # Errors
    /// Stride smaller than a block's extent (blocks would overlap), or the
    /// strided layout exceeds the registered region.
    pub(crate) fn with_block_stride(mut self, stride: usize) -> Result<Self, String> {
        let min_stride = match self.seg {
            SegmentLayout::Contiguous => self.block_bytes(),
            SegmentLayout::Split { .. } => self.segment_bytes,
        };
        if stride < min_stride {
            return Err(format!(
                "block_stride {stride} must be >= {min_stride}: blocks would overlap"
            ));
        }
        self.block_stride_bytes = stride;
        self.check_bounds()?;
        Ok(self)
    }

    /// Apply SSD alignment padding to the host-side segment stride so every
    /// iovec in a split writev is independently aligned.
    pub(crate) fn with_ssd_padding(mut self, alignment: usize) -> Self {
        self.padded_segment_bytes = self.segment_bytes.next_multiple_of(alignment);
        self
    }

    /// Validate that all addressed bytes fit in `size_bytes` and no two block
    /// ranges alias the same device memory.
    fn check_bounds(&self) -> Result<(), String> {
        let end = match self.seg {
            SegmentLayout::Contiguous => {
                let block_bytes = self.block_bytes();
                if self.block_stride_bytes < block_bytes {
                    return Err(format!(
                        "block_stride {} must be >= block_bytes {block_bytes}: blocks would overlap",
                        self.block_stride_bytes
                    ));
                }
                (self.num_blocks - 1)
                    .checked_mul(self.block_stride_bytes)
                    .and_then(|o| o.checked_add(block_bytes))
                    .ok_or_else(|| "memory layout overflow".to_string())?
            }
            SegmentLayout::Split { kv_stride_bytes } => {
                if self.block_stride_bytes < self.segment_bytes {
                    return Err(format!(
                        "block_stride {} must be >= segment_bytes {}: blocks would overlap",
                        self.block_stride_bytes, self.segment_bytes
                    ));
                }
                self.check_split_segments_disjoint(kv_stride_bytes)?;
                let last_segment_end = (self.num_blocks - 1)
                    .checked_mul(self.block_stride_bytes)
                    .and_then(|o| o.checked_add(self.segment_bytes))
                    .ok_or_else(|| "memory layout overflow".to_string())?;
                kv_stride_bytes
                    .checked_add(last_segment_end)
                    .ok_or_else(|| "memory layout overflow".to_string())?
            }
        };
        if end > self.size_bytes {
            return Err(format!(
                "registered memory too small: need {end} bytes, got {}",
                self.size_bytes
            ));
        }
        Ok(())
    }

    fn check_split_segments_disjoint(&self, kv_stride_bytes: usize) -> Result<(), String> {
        let max_block_distance = self.num_blocks - 1;
        let nearest = kv_stride_bytes / self.block_stride_bytes;

        for distance in [nearest, nearest.saturating_add(1)] {
            if distance == 0 || distance > max_block_distance {
                continue;
            }
            let block_distance_bytes = distance
                .checked_mul(self.block_stride_bytes)
                .ok_or_else(|| "memory layout overflow".to_string())?;
            let gap = kv_stride_bytes.abs_diff(block_distance_bytes);
            if gap < self.segment_bytes {
                return Err(format!(
                    "kv_stride_bytes {kv_stride_bytes} overlaps blocks {distance} apart"
                ));
            }
        }

        Ok(())
    }

    /// Device address ranges for `block_idx`.
    ///
    /// Construction already proved the last block fits, so any valid index is
    /// in bounds and the arithmetic below cannot overflow.
    pub(crate) fn block_copies(&self, block_idx: usize) -> Result<BlockCopies, String> {
        if block_idx >= self.num_blocks {
            return Err(format!(
                "block {block_idx} out of range ({} blocks)",
                self.num_blocks
            ));
        }
        let base = self.data_ptr + (block_idx * self.block_stride_bytes) as u64;
        Ok(match self.seg {
            SegmentLayout::Contiguous => BlockCopies::Contiguous(BlockCopy {
                addr: base,
                bytes: self.block_bytes(),
            }),
            SegmentLayout::Split { kv_stride_bytes } => BlockCopies::Split {
                k: BlockCopy {
                    addr: base,
                    bytes: self.segment_bytes,
                },
                v: BlockCopy {
                    addr: base + kv_stride_bytes as u64,
                    bytes: self.segment_bytes,
                },
            },
        })
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// True when K and V need separate pinned segment pools.
    pub(crate) fn is_split(&self) -> bool {
        matches!(self.seg, SegmentLayout::Split { .. })
    }

    /// Actual (unpadded) GPU-side segment size.
    pub(crate) fn segment_bytes(&self) -> usize {
        self.segment_bytes
    }

    /// Actual (unpadded) total block size on the GPU.
    fn block_bytes(&self) -> usize {
        self.segment_bytes * self.segments
    }

    /// Host-side per-segment stride (SSD-aligned).
    pub(crate) fn padded_segment_bytes(&self) -> usize {
        self.padded_segment_bytes
    }

    /// Host-side total block size: pinned allocation footprint and
    /// `RawBlock.total_size` → `SlotMeta.total_size()` for SSD I/O.
    pub(crate) fn padded_block_bytes(&self) -> usize {
        self.padded_segment_bytes * self.segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contiguous_addr(layout: &KVCacheLayout, block_idx: usize) -> u64 {
        match layout.block_copies(block_idx).unwrap() {
            BlockCopies::Contiguous(c) => c.addr,
            BlockCopies::Split { .. } => panic!("expected contiguous copies"),
        }
    }

    #[test]
    fn dense_layout_addresses() {
        let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
        assert!(!layout.is_split());
        assert_eq!(layout.padded_block_bytes(), 1024);
        assert_eq!(contiguous_addr(&layout, 0), 0x1000);
        assert_eq!(contiguous_addr(&layout, 5), 0x1000 + 5 * 1024);
        assert!(layout.block_copies(100).is_err());
    }

    #[test]
    fn split_layout_addresses() {
        // 100 K blocks contiguous, then 100 V blocks: kv_stride = region size.
        let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 100 * 1024, 2).unwrap();
        assert!(layout.is_split());
        match layout.block_copies(3).unwrap() {
            BlockCopies::Split { k, v } => {
                assert_eq!(k.addr, 0x1000 + 3 * 1024);
                assert_eq!(v.addr, 0x1000 + (100 + 3) * 1024);
                assert_eq!(k.bytes, 1024);
                assert_eq!(v.bytes, 1024);
            }
            BlockCopies::Contiguous(_) => panic!("expected split copies"),
        }
    }

    #[test]
    fn adjacent_kv_collapses_to_contiguous() {
        // kv_stride == segment_bytes: one copy of both segments.
        let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 1024, 2).unwrap();
        assert!(!layout.is_split());
        match layout.block_copies(0).unwrap() {
            BlockCopies::Contiguous(c) => assert_eq!(c.bytes, 2048),
            BlockCopies::Split { .. } => panic!("expected contiguous copies"),
        }
        assert_eq!(contiguous_addr(&layout, 1), 0x1000 + 2048);
    }

    #[test]
    fn overlapping_kv_stride_rejected() {
        let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 512, 2).unwrap_err();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn split_kv_regions_must_not_overlap() {
        let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 2048, 2).unwrap_err();
        assert!(err.contains("overlaps blocks 2 apart"), "{err}");
    }

    #[test]
    fn sparse_split_layout_can_have_interleaved_disjoint_segments() {
        let layout = KVCacheLayout::new(0x1000, 1_600, 2, 100, 500, 2)
            .unwrap()
            .with_block_stride(1_000)
            .unwrap();

        match layout.block_copies(0).unwrap() {
            BlockCopies::Split { k, v } => {
                assert_eq!(k.addr, 0x1000);
                assert_eq!(v.addr, 0x1000 + 500);
            }
            BlockCopies::Contiguous(_) => panic!("expected split copies"),
        }
        match layout.block_copies(1).unwrap() {
            BlockCopies::Split { k, v } => {
                assert_eq!(k.addr, 0x1000 + 1_000);
                assert_eq!(v.addr, 0x1000 + 1_500);
            }
            BlockCopies::Contiguous(_) => panic!("expected split copies"),
        }
    }

    #[test]
    fn adjacent_kv_default_stride_keeps_blocks_disjoint() {
        let layout = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 1024, 2).unwrap();

        let block0 = match layout.block_copies(0).unwrap() {
            BlockCopies::Contiguous(c) => c,
            BlockCopies::Split { .. } => panic!("expected contiguous copies"),
        };
        let block1 = match layout.block_copies(1).unwrap() {
            BlockCopies::Contiguous(c) => c,
            BlockCopies::Split { .. } => panic!("expected contiguous copies"),
        };

        assert_eq!(block0.addr + block0.bytes as u64, block1.addr);
    }

    #[test]
    fn block_stride_decouples_step_from_copy_size() {
        let segment_bytes = 4096 * 2;
        let page_stride_bytes = 8192 * 2;
        let num_blocks = 8;
        let size_bytes = num_blocks * page_stride_bytes;

        let layout = KVCacheLayout::new(0x10000, size_bytes, num_blocks, segment_bytes, 0, 1)
            .unwrap()
            .with_block_stride(page_stride_bytes)
            .unwrap();

        assert_eq!(
            contiguous_addr(&layout, 3),
            0x10000 + 3 * page_stride_bytes as u64
        );
        assert_eq!(
            contiguous_addr(&layout, 7),
            0x10000 + 7 * page_stride_bytes as u64
        );
    }

    #[test]
    fn block_stride_defaults_to_dense_and_validates() {
        let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1).unwrap();
        assert_eq!(contiguous_addr(&layout, 5), 0x1000 + 5 * 1024);

        // Stride smaller than the block: overlap.
        assert!(
            KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
                .unwrap()
                .with_block_stride(512)
                .is_err()
        );

        // Strided layout exceeds the registered region.
        assert!(
            KVCacheLayout::new(0x1000, 8 * 1024, 8, 1024, 0, 1)
                .unwrap()
                .with_block_stride(4096)
                .is_err()
        );

        // Split K/V regions that are disjoint with the default dense stride can
        // overlap again if the explicit per-block stride grows too far.
        let err = KVCacheLayout::new(0x1000, 200 * 1024, 100, 1024, 100 * 1024, 2)
            .unwrap()
            .with_block_stride(2048)
            .unwrap_err();
        assert!(err.contains("overlaps blocks 50 apart"), "{err}");
    }

    #[test]
    fn null_pointer_rejected() {
        assert!(KVCacheLayout::new(0, 1024, 10, 64, 0, 1).is_err());
    }

    #[test]
    fn memory_too_small_rejected() {
        let err = KVCacheLayout::new(0x1000, 5120, 10, 1024, 0, 1).unwrap_err();
        assert!(err.contains("too small"), "{err}");
    }

    #[test]
    fn ssd_padding() {
        // Unaligned: 8848 % 512 = 144, padded to 9216.
        let layout = KVCacheLayout::new(0x1000, 10_000_000, 100, 8848, 0, 1)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(layout.padded_segment_bytes(), 9216);
        assert_eq!(layout.padded_block_bytes(), 9216);

        // Already aligned: no change.
        let layout = KVCacheLayout::new(0x1000, 1024 * 1024, 100, 1024, 0, 1)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(layout.padded_segment_bytes(), 1024);
        assert_eq!(layout.padded_block_bytes(), 1024);

        // Split layout: padded per segment, total = padded * segments.
        let layout = KVCacheLayout::new(0x1000, 10_000_000, 100, 8848, 900_000, 2)
            .unwrap()
            .with_ssd_padding(512);
        assert_eq!(layout.padded_segment_bytes(), 9216);
        assert_eq!(layout.padded_block_bytes(), 9216 * 2);
    }
}
