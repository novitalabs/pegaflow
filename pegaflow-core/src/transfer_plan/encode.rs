use std::sync::Arc;

use pegaflow_transfer_wire::TransferPlan;

use crate::BlockKey;
use crate::block::SealedBlock;

use super::merge::build_transfer_geometry;
use super::schema::{TransferPlanError, derive_slot_schemas};
use super::view::block_views_from_found;

pub(crate) fn encode_transfer_plan(
    found_blocks: &[(BlockKey, Arc<SealedBlock>)],
) -> Result<TransferPlan, TransferPlanError> {
    let views = block_views_from_found(found_blocks)?;
    if views.is_empty() {
        return Ok(TransferPlan {
            slot_schemas: vec![],
            remote_chunks: vec![],
            placements: vec![],
            block_hashes: vec![],
        });
    }

    let slot_schemas = derive_slot_schemas(&views)?;
    let (remote_chunks, placements) = build_transfer_geometry(&views, &slot_schemas);
    let block_hashes = views.into_iter().map(|v| v.hash).collect();

    let plan = TransferPlan {
        slot_schemas,
        remote_chunks,
        placements,
        block_hashes,
    };
    plan.validate()
        .map_err(|e| TransferPlanError::new(e.to_string()))?;
    Ok(plan)
}

pub(crate) fn encode_transfer_plan_bytes(
    found_blocks: &[(BlockKey, Arc<SealedBlock>)],
) -> Result<Vec<u8>, String> {
    let plan = encode_transfer_plan(found_blocks).map_err(|e| e.to_string())?;
    plan.encode_to_vec().map_err(|e| e.to_string())
}
