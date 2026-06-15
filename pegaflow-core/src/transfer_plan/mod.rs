//! Build a compact [`TransferPlan`] from in-memory sealed blocks.

mod encode;
#[cfg(feature = "rdma")]
mod materialize;
mod merge;
#[cfg(feature = "rdma")]
mod rebuild;
mod schema;
mod view;

pub(crate) use encode::encode_transfer_plan_bytes;
#[cfg(feature = "rdma")]
pub(crate) use materialize::materialize_transfer_plan;
#[cfg(feature = "rdma")]
pub(crate) use rebuild::rebuild_sealed_blocks;

#[cfg(all(test, feature = "rdma"))]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use pegaflow_common::NumaNode;

    use crate::BlockKey;
    use crate::block::{LayerBlock, RawBlock, SealedBlock, Segment};
    use crate::pinned_pool::PinnedAllocator;

    use super::encode::encode_transfer_plan;
    use super::{encode_transfer_plan_bytes, materialize_transfer_plan, rebuild_sealed_blocks};
    use crate::backing::AllocateFn;

    fn make_stride_blocks(
        count: usize,
        stride: usize,
        seg_bytes: usize,
    ) -> (
        Vec<(BlockKey, Arc<SealedBlock>)>,
        Arc<crate::pinned_pool::PinnedAllocation>,
    ) {
        let allocator = Arc::new(PinnedAllocator::new_global(
            64 * 1024 * 1024,
            1,
            false,
            false,
            None,
        ));
        let total = stride
            .checked_mul(count)
            .and_then(|n| NonZeroU64::new(n as u64))
            .expect("allocation size");
        let slab = allocator
            .allocate(total, NumaNode::UNKNOWN)
            .expect("slab allocation");

        let mut found = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * stride;
            let host = slab.mapped_ptr().add(offset).host();
            for b in 0..seg_bytes {
                unsafe {
                    *host.as_ptr().add(b) = i as u8;
                }
            }
            let segment = Segment::new(host, seg_bytes, Arc::clone(&slab));
            let key = BlockKey::new("bench-ns".to_string(), vec![i as u8]);
            found.push((
                key,
                Arc::new(SealedBlock::from_slots(vec![RawBlock::single_segment(
                    segment,
                )])),
            ));
        }
        (found, slab)
    }

    fn test_allocate_fn() -> AllocateFn {
        let allocator = Arc::new(PinnedAllocator::new_global(
            64 * 1024 * 1024,
            1,
            false,
            false,
            None,
        ));
        Arc::new(move |size, _numa| allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN))
    }

    fn simulate_transfer(
        plan: &pegaflow_transfer_wire::TransferPlan,
        found_blocks: &[(BlockKey, Arc<SealedBlock>)],
        materialized: &super::materialize::RebuildContext,
    ) {
        use super::materialize::chunk_ptr_at;

        for placement in &plan.placements {
            let seg = &plan.slot_schemas[placement.slot_idx as usize].segments
                [placement.segment_idx as usize];
            let seg_len = seg.bytes;
            let stride = seg.block_stride;

            for i in 0..placement.block_count {
                let block_idx = (placement.block_start + i) as usize;
                let raw = &found_blocks[block_idx].1.slots()[placement.slot_idx as usize];
                let layer = LayerBlock::new(raw);
                let src = if placement.segment_idx == 0 {
                    layer.k_ptr()
                } else {
                    layer.v_ptr().expect("split segment")
                };
                let offset_in_chunk = placement.offset_in_chunk + (i as usize) * stride;
                let (dst_ptr, _) =
                    chunk_ptr_at(materialized, plan, placement.chunk_idx, offset_in_chunk)
                        .expect("chunk ptr");
                let dst = dst_ptr.as_ptr();
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, seg_len);
                }
            }
        }
    }

    #[test]
    fn encode_materialize_rebuild_round_trip() {
        let (found, _slab) = make_stride_blocks(8, 128, 64);
        let plan = encode_transfer_plan(&found).expect("encode");
        assert!(!plan.placements.is_empty());

        let materialized =
            materialize_transfer_plan(&plan, &test_allocate_fn()).expect("materialize");
        assert_eq!(materialized.rebuild.slabs.len(), 1);
        assert_eq!(materialized.rdma_descs.len(), plan.remote_chunks.len());
        simulate_transfer(&plan, &found, &materialized.rebuild);
        let rebuilt =
            rebuild_sealed_blocks(&plan, &materialized.rebuild, "bench-ns").expect("rebuild");
        assert_eq!(rebuilt.len(), found.len());

        for (orig, (key, sealed)) in found.iter().zip(rebuilt.iter()) {
            assert_eq!(orig.0.hash, key.hash);
            let orig_raw = &orig.1.slots()[0];
            let rebuilt_raw = &sealed.slots()[0];
            let orig_layer = LayerBlock::new(orig_raw);
            let rebuilt_layer = LayerBlock::new(rebuilt_raw);
            assert_eq!(orig_layer.k_size(), rebuilt_layer.k_size());
            unsafe {
                assert_eq!(
                    std::slice::from_raw_parts(orig_layer.k_ptr(), orig_layer.k_size()),
                    std::slice::from_raw_parts(rebuilt_layer.k_ptr(), rebuilt_layer.k_size()),
                );
            }
        }
    }

    #[test]
    fn batch_encode_wire_size_scales_with_blocks_not_slots() {
        let block_count = 512;
        let (found, _) = make_stride_blocks(block_count, 256, 128);
        let plan = encode_transfer_plan(&found).expect("encode");
        assert_eq!(plan.placements.len(), 1);
        assert_eq!(plan.placements[0].block_count, block_count as u32);

        let wire = encode_transfer_plan_bytes(&found).expect("postcard encode");
        assert!(wire.len() < 64 * 1024, "wire bytes={}", wire.len());
        assert!(wire.len() < block_count * 64, "wire bytes={}", wire.len());
    }

    #[test]
    fn materialize_allocates_one_slab_per_numa() {
        let (found, _) = make_stride_blocks(4, 128, 64);
        let plan = encode_transfer_plan(&found).expect("encode");
        let materialized =
            materialize_transfer_plan(&plan, &test_allocate_fn()).expect("materialize");
        assert_eq!(materialized.rebuild.slabs.len(), 1);
        assert!(
            materialized.rebuild.slabs[0].length >= plan.total_remote_bytes() as usize,
            "slab should cover all chunk bytes on NUMA"
        );
    }
}
