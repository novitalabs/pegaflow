use std::ptr::NonNull;
use std::sync::Arc;

use crate::block::Segment;
use crate::pd_receive::{PdReceiveLayerLayout, PdReceiveLoadPlan, PdReceiveSlabDesc};
use crate::{EngineError, InstanceContext, PinnedAllocation, RawBlock, SealedBlock};

#[derive(Debug, Clone)]
pub struct PrepareLoadRequest {
    pub instance_id: String,
    pub request_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub num_prompt_tokens: u64,
    pub num_computed_tokens: u64,
    pub virtual_block_size: u64,
    pub decode_request_id: Option<String>,
    pub decode_expected_writes: usize,
}

#[derive(Debug, Clone)]
pub struct PreparedLoadItem {
    pub plan_id: u64,
    pub block_ids: Vec<i32>,
}

#[derive(Debug, Clone)]
pub enum PrepareLoadOutcome {
    NoPlan,
    Plan { plan_id: u64, num_tokens: u64 },
}

#[derive(Clone)]
struct LayerShardSourceBlocks {
    slot_id: usize,
    blocks: Vec<Arc<RawBlock>>,
}

#[derive(Clone)]
pub(crate) struct PreparedLoadPlan {
    /// Source blocks grouped by layer/tensor-parallel slot.
    source_blocks: Vec<LayerShardSourceBlocks>,
}

impl PreparedLoadPlan {
    pub(crate) fn from_cache_blocks(
        instance: &InstanceContext,
        block_entries: &[Arc<SealedBlock>],
    ) -> Result<Self, EngineError> {
        let mut source_blocks = Vec::new();

        for shard in instance.registered_shards() {
            for layer in shard.layers {
                let slot_id = instance.get_slot_index(layer.layer_id, shard.tp_rank)?;
                let mut blocks = Vec::with_capacity(block_entries.len());
                for (block_idx, sealed) in block_entries.iter().enumerate() {
                    let block = sealed.get_slot(slot_id).cloned().ok_or_else(|| {
                        EngineError::InvalidArgument(format!(
                            "prepared cache block {block_idx} missing slot {slot_id}"
                        ))
                    })?;
                    blocks.push(block);
                }
                Self::insert_slot(&mut source_blocks, slot_id, blocks)?;
            }
        }

        Ok(Self { source_blocks })
    }

    pub(crate) fn from_staged_load_plans(
        load_plans: &[PdReceiveLoadPlan],
        num_blocks: usize,
    ) -> Result<Self, EngineError> {
        let mut source_blocks = Vec::new();

        for load_plan in load_plans {
            for receive_layer in &load_plan.layers {
                let layout = &receive_layer.layout;
                if layout.num_blocks < num_blocks {
                    return Err(EngineError::InvalidArgument(format!(
                        "P/D receive layer {} has {} staged blocks, prepare needs {}",
                        layout.layer_name, layout.num_blocks, num_blocks
                    )));
                }
                let mut blocks = Vec::with_capacity(num_blocks);
                for staged_block_idx in 0..num_blocks {
                    blocks.push(pd_receive_raw_block(
                        &receive_layer.slab,
                        layout,
                        &receive_layer.allocation,
                        staged_block_idx,
                    )?);
                }
                Self::insert_slot(&mut source_blocks, layout.slot_id, blocks)?;
            }
        }

        Ok(Self { source_blocks })
    }

    pub(crate) fn block_count(&self) -> usize {
        self.source_blocks
            .first()
            .map_or(0, |source| source.blocks.len())
    }

    pub(crate) fn blocks_for_slot(&self, slot_id: usize) -> Option<&[Arc<RawBlock>]> {
        self.source_blocks
            .iter()
            .find(|source| source.slot_id == slot_id)
            .map(|source| source.blocks.as_slice())
    }

    fn insert_slot(
        source_blocks: &mut Vec<LayerShardSourceBlocks>,
        slot_id: usize,
        blocks: Vec<Arc<RawBlock>>,
    ) -> Result<(), EngineError> {
        if source_blocks.iter().any(|source| source.slot_id == slot_id) {
            return Err(EngineError::InvalidArgument(format!(
                "prepared load plan has duplicate slot {slot_id}"
            )));
        }
        source_blocks.push(LayerShardSourceBlocks { slot_id, blocks });
        Ok(())
    }
}

fn pd_receive_segment_ptr(
    slab: &PdReceiveSlabDesc,
    layout: &PdReceiveLayerLayout,
    staged_block_idx: usize,
    segment_idx: usize,
) -> Result<NonNull<u8>, EngineError> {
    let block_offset = layout
        .block_stride
        .checked_mul(staged_block_idx as u64)
        .ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "P/D receive block offset overflow for layer {}",
                layout.layer_name
            ))
        })?;
    let segment_offset = layout
        .padded_segment_stride
        .checked_mul(segment_idx as u64)
        .ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "P/D receive segment offset overflow for layer {}",
                layout.layer_name
            ))
        })?;
    let offset = layout
        .layer_offset
        .checked_add(block_offset)
        .and_then(|value| value.checked_add(segment_offset))
        .ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "P/D receive offset overflow for layer {}",
                layout.layer_name
            ))
        })?;
    let end = offset.checked_add(layout.segment_size).ok_or_else(|| {
        EngineError::InvalidArgument(format!(
            "P/D receive segment end overflow for layer {}",
            layout.layer_name
        ))
    })?;
    if end > slab.size {
        return Err(EngineError::InvalidArgument(format!(
            "P/D receive segment out of slab bounds for layer {}: end={} slab_size={}",
            layout.layer_name, end, slab.size
        )));
    }
    let addr = slab.base_ptr.checked_add(offset).ok_or_else(|| {
        EngineError::InvalidArgument(format!(
            "P/D receive pointer overflow for layer {}",
            layout.layer_name
        ))
    })?;
    let addr = usize::try_from(addr).map_err(|_| {
        EngineError::InvalidArgument(format!(
            "P/D receive pointer does not fit usize for layer {}",
            layout.layer_name
        ))
    })?;
    NonNull::new(addr as *mut u8).ok_or_else(|| {
        EngineError::InvalidArgument(format!(
            "P/D receive pointer is null for layer {}",
            layout.layer_name
        ))
    })
}

fn pd_receive_raw_block(
    slab: &PdReceiveSlabDesc,
    layout: &PdReceiveLayerLayout,
    allocation: &Arc<PinnedAllocation>,
    staged_block_idx: usize,
) -> Result<Arc<RawBlock>, EngineError> {
    let segment_size = usize::try_from(layout.segment_size).map_err(|_| {
        EngineError::InvalidArgument(format!(
            "P/D receive segment size does not fit usize for layer {}: {}",
            layout.layer_name, layout.segment_size
        ))
    })?;
    let mut segments = Vec::with_capacity(layout.segment_count);
    for segment_idx in 0..layout.segment_count {
        segments.push(Segment::new(
            pd_receive_segment_ptr(slab, layout, staged_block_idx, segment_idx)?,
            segment_size,
            Arc::clone(allocation),
        ));
    }
    Ok(Arc::new(RawBlock::new(segments)))
}
