//! Benchmark-only fixtures for the cross-node `rebuild_sealed_blocks` hot path.
//!
//! Builds a vLLM-shaped transfer plan (`slots = layers * tp`, split K/V) through
//! the real encode -> materialize pipeline, then exposes a single `rebuild()`
//! call for criterion to time. Gated behind the `bench` feature so it never
//! ships in production builds.

use std::num::NonZeroU64;
use std::sync::Arc;

use pegaflow_common::NumaNode;
use pegaflow_transfer_wire::TransferPlan;

use crate::BlockKey;
use crate::backing::AllocateFn;
use crate::block::{RawBlock, SealedBlock, Segment};
use crate::pinned_pool::PinnedAllocator;

use super::materialize::RebuildContext;
use super::{
    encode::encode_transfer_plan, encode_transfer_plan_bytes, materialize_transfer_plan,
    rebuild_sealed_blocks,
};

const NAMESPACE: &str = "rebuild-bench";
const SEGMENTS_PER_SLOT: usize = 2; // split K/V

/// A prepared transfer plan plus its materialized local slab, ready to rebuild
/// repeatedly. Construction runs the real encode + materialize pipeline; only
/// [`Self::rebuild`] should be timed.
pub struct RebuildFixture {
    plan: TransferPlan,
    rebuild: RebuildContext,
    cells: usize,
}

impl RebuildFixture {
    /// Build a fixture of `blocks` sealed blocks, each with `slots` slots (one
    /// per layer*rank) of split K/V segments of `seg_bytes` each.
    ///
    /// Rebuild cost is driven by cell count (`blocks * slots * 2`), not segment
    /// byte size, so `seg_bytes` can stay tiny to keep the slab small.
    pub fn new(blocks: usize, slots: usize, seg_bytes: usize) -> Self {
        assert!(blocks > 0 && slots > 0 && seg_bytes > 0);
        let found = build_source_blocks(blocks, slots, seg_bytes);
        let plan = encode_transfer_plan(&found).expect("encode transfer plan");
        // Source blocks only feed encode; drop them so the bench measures
        // rebuild against the freshly materialized slab alone.
        drop(found);

        let allocate_fn = bench_allocate_fn(plan.total_remote_bytes());
        let materialized =
            materialize_transfer_plan(&plan, &allocate_fn).expect("materialize transfer plan");
        let cells = blocks * slots * SEGMENTS_PER_SLOT;
        Self {
            plan,
            rebuild: materialized.rebuild,
            cells,
        }
    }

    /// Run the production rebuild once; returns the rebuilt block count.
    pub fn rebuild(&self) -> usize {
        rebuild_sealed_blocks(&self.plan, &self.rebuild, NAMESPACE)
            .expect("rebuild sealed blocks")
            .len()
    }

    /// `(placements, remote_chunks, cells)` for reporting.
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.plan.placements.len(),
            self.plan.remote_chunks.len(),
            self.cells,
        )
    }
}

/// Source blocks plus their encoded wire bytes, for benching the query path:
/// `encode` is the serving-side hot path behind `QueryBlocksForTransfer`,
/// `decode` is the requesting side parsing + validating the response.
pub struct QueryFixture {
    found: Vec<(BlockKey, Arc<SealedBlock>)>,
    bytes: Vec<u8>,
}

impl QueryFixture {
    pub fn new(blocks: usize, slots: usize, seg_bytes: usize) -> Self {
        assert!(blocks > 0 && slots > 0 && seg_bytes > 0);
        let found = build_source_blocks(blocks, slots, seg_bytes);
        let bytes = encode_transfer_plan_bytes(&found).expect("encode transfer plan bytes");
        Self { found, bytes }
    }

    /// Server-side: build the `TransferPlan` from sealed blocks and postcard-encode.
    pub fn encode(&self) -> usize {
        encode_transfer_plan_bytes(&self.found)
            .expect("encode transfer plan bytes")
            .len()
    }

    /// Client-side: decode + validate the wire bytes back into a `TransferPlan`.
    pub fn decode(&self) -> usize {
        TransferPlan::decode_from_slice(&self.bytes)
            .expect("decode transfer plan")
            .block_hashes
            .len()
    }

    pub fn wire_len(&self) -> usize {
        self.bytes.len()
    }
}

fn bench_allocate_fn(total_bytes: u64) -> AllocateFn {
    let capacity = (total_bytes as usize)
        .next_power_of_two()
        .max(64 * 1024 * 1024);
    let allocator = Arc::new(PinnedAllocator::new_global(capacity, 1, false, false, None));
    Arc::new(move |size, _numa| allocator.allocate(NonZeroU64::new(size)?, NumaNode::UNKNOWN))
}

fn build_source_blocks(
    blocks: usize,
    slots: usize,
    seg_bytes: usize,
) -> Vec<(BlockKey, Arc<SealedBlock>)> {
    // One slab; cell (slot s, segment g, block i) lives at byte offset
    //   ((s * 2 + g) * blocks + i) * seg_bytes.
    // For a fixed (slot, segment) the blocks are contiguous with stride ==
    // seg_bytes, so encode coalesces each (slot, segment) into one run: the
    // plan ends up with `slots * 2` placements of `blocks` each, mirroring the
    // real vLLM-shaped layout (layers*tp slots, split K/V).
    let total = (blocks * slots * SEGMENTS_PER_SLOT * seg_bytes) as u64;
    let capacity = (total as usize).next_power_of_two().max(64 * 1024 * 1024);
    let allocator = Arc::new(PinnedAllocator::new_global(capacity, 1, false, false, None));
    let slab = allocator
        .allocate(
            NonZeroU64::new(total).expect("non-zero slab"),
            NumaNode::UNKNOWN,
        )
        .expect("source slab");
    let base = slab.mapped_ptr();

    let mut found = Vec::with_capacity(blocks);
    for i in 0..blocks {
        let mut slot_inserts = Vec::with_capacity(slots);
        for s in 0..slots {
            let k_off = ((s * SEGMENTS_PER_SLOT) * blocks + i) * seg_bytes;
            let v_off = ((s * SEGMENTS_PER_SLOT + 1) * blocks + i) * seg_bytes;
            let k_seg = Segment::new(base.add(k_off).host(), seg_bytes, Arc::clone(&slab));
            let v_seg = Segment::new(base.add(v_off).host(), seg_bytes, Arc::clone(&slab));
            slot_inserts.push((s, RawBlock::two_segments(k_seg, v_seg)));
        }
        let mut hash = Vec::with_capacity(4);
        hash.extend_from_slice(&(i as u32).to_le_bytes());
        let key = BlockKey::new(NAMESPACE.to_string(), hash);
        let sealed = SealedBlock::from_ordered_slot_inserts(slot_inserts, slots, NumaNode(0))
            .unwrap_or_else(|_| panic!("ordered slot inserts for block {i}"));
        found.push((key, Arc::new(sealed)));
    }
    found
}
