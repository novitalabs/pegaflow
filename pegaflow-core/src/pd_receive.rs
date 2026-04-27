use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use uuid::Uuid;

use crate::pinned_pool::PinnedAllocation;
use crate::{EngineError, NumaNode, PegaEngine};

const DEFAULT_PD_RECEIVE_TTL: Duration = Duration::from_secs(30);
type PdReceiveRequestKey = (String, String);

#[derive(Debug, Clone)]
pub struct PdReceivePrepareRequest {
    pub instance_id: String,
    pub request_id: String,
    pub block_hashes: Vec<Vec<u8>>,
    pub num_blocks: usize,
    pub expected_imm_count: usize,
    pub expire_after: Option<Duration>,
}

#[derive(Debug, Clone)]
pub struct PdReceivePrepareResponse {
    pub handle: String,
    pub imm_data: u32,
    pub expires_at_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PdReceiveSlabDesc {
    pub base_ptr: u64,
    pub size: u64,
    pub numa_node: NumaNode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdReceiveRankDesc {
    pub receive_rank: usize,
    pub device_id: i32,
    pub tp_rank: usize,
    pub slab_index: usize,
    pub numa_node: NumaNode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdReceiveLayerLayout {
    pub layer_name: String,
    pub receive_rank: usize,
    pub slab_index: usize,
    pub layer_offset: u64,
    pub block_stride: u64,
    pub segment_count: usize,
    pub segment_size: u64,
    pub padded_segment_stride: u64,
    pub num_blocks: usize,
    pub slot_id: usize,
}

#[derive(Debug, Clone)]
pub struct PdReceiveDescriptor {
    pub handle: String,
    pub imm_data: u32,
    pub expires_at_ms: u64,
    pub data_ready: bool,
    pub ranks: Vec<PdReceiveRankDesc>,
    pub slabs: Vec<PdReceiveSlabDesc>,
    pub layers: Vec<PdReceiveLayerLayout>,
    pub block_hashes: Vec<Vec<u8>>,
}

pub(crate) struct PdReceiveLoadPlan {
    pub layers: Vec<PdReceiveLoadLayer>,
}

pub(crate) struct PdReceiveLoadLayer {
    pub layout: PdReceiveLayerLayout,
    pub slab: PdReceiveSlabDesc,
    pub allocation: Arc<PinnedAllocation>,
}

pub(crate) enum PdReceiveCheckoutStatus {
    Pending,
    Ready(Vec<PdReceiveLoadPlan>),
    Expired,
}

#[derive(Debug, Clone)]
pub enum PdReceiveDescriptorLookup {
    Pending,
    Ready(PdReceiveDescriptor),
    Failed,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PdReceiveLeaseState {
    Prepared,
    Writing,
    Ready,
}

#[derive(Debug, Clone)]
pub(crate) struct PdReceiveLayerPlan {
    pub layer_name: String,
    pub block_stride: usize,
    pub segment_count: usize,
    pub segment_size: usize,
    pub padded_segment_stride: usize,
    pub slot_id: usize,
}

struct PdReceiveSlab {
    desc: PdReceiveSlabDesc,
    allocation: Option<Arc<PinnedAllocation>>,
}

#[allow(dead_code)]
struct PdReceiveLease {
    instance_id: String,
    request_id: String,
    handle: String,
    imm_data: u32,
    state: PdReceiveLeaseState,
    expected_imm_count: usize,
    observed_imm_count: usize,
    created_at: Instant,
    expires_at: Instant,
    expires_at_ms: u64,
    ranks: Vec<PdReceiveRankDesc>,
    slabs: Vec<PdReceiveSlab>,
    layers: Vec<PdReceiveLayerLayout>,
    block_hashes: Vec<Vec<u8>>,
}

impl PdReceiveLease {
    fn prepare_response(&self) -> PdReceivePrepareResponse {
        PdReceivePrepareResponse {
            handle: self.handle.clone(),
            imm_data: self.imm_data,
            expires_at_ms: self.expires_at_ms,
        }
    }

    fn data_ready(&self) -> bool {
        self.state == PdReceiveLeaseState::Ready
    }

    fn descriptor(&self, receive_rank: Option<usize>) -> Option<PdReceiveDescriptor> {
        if let Some(receive_rank) = receive_rank {
            let rank = self
                .ranks
                .iter()
                .find(|rank| rank.receive_rank == receive_rank)?;
            let slab = self.slabs.get(rank.slab_index)?;
            slab.allocation.as_ref()?;
            let slab = slab.desc.clone();
            let mut rank = rank.clone();
            rank.slab_index = 0;
            let layers = self
                .layers
                .iter()
                .filter(|layer| layer.receive_rank == receive_rank)
                .map(|layer| {
                    let mut layer = layer.clone();
                    layer.slab_index = 0;
                    layer
                })
                .collect();
            return Some(PdReceiveDescriptor {
                handle: self.handle.clone(),
                imm_data: self.imm_data,
                expires_at_ms: self.expires_at_ms,
                data_ready: self.data_ready(),
                ranks: vec![rank],
                slabs: vec![slab],
                layers,
                block_hashes: self.block_hashes.clone(),
            });
        }

        Some(PdReceiveDescriptor {
            handle: self.handle.clone(),
            imm_data: self.imm_data,
            expires_at_ms: self.expires_at_ms,
            data_ready: self.data_ready(),
            ranks: self.ranks.clone(),
            slabs: self.slabs.iter().map(|slab| slab.desc.clone()).collect(),
            layers: self.layers.clone(),
            block_hashes: self.block_hashes.clone(),
        })
    }
}

#[derive(Default)]
struct PdReceiveState {
    by_request: HashMap<PdReceiveRequestKey, String>,
    by_handle: HashMap<String, PdReceiveLease>,
    by_imm: HashMap<u32, String>,
}

pub(crate) struct PdReceiveManager {
    state: Mutex<PdReceiveState>,
    next_imm: AtomicU32,
}

pub(crate) struct PdReceiveLeaseInput {
    pub instance_id: String,
    pub request_id: String,
    pub expected_imm_count: usize,
    pub expires_at: Instant,
    pub expires_at_ms: u64,
    pub ranks: Vec<PdReceiveRankDesc>,
    pub slabs: Vec<(PdReceiveSlabDesc, Arc<PinnedAllocation>)>,
    pub layers: Vec<PdReceiveLayerLayout>,
    pub block_hashes: Vec<Vec<u8>>,
}

impl Default for PdReceiveManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PdReceiveManager {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(PdReceiveState::default()),
            next_imm: AtomicU32::new(1),
        }
    }

    pub(crate) fn existing_prepare_response(
        &self,
        instance_id: &str,
        request_id: &str,
    ) -> Option<PdReceivePrepareResponse> {
        let now = Instant::now();
        let mut state = self.state.lock();
        Self::gc_expired_locked(&mut state, now);
        let handle = state
            .by_request
            .get(&request_key(instance_id, request_id))?
            .clone();
        state
            .by_handle
            .get(&handle)
            .map(PdReceiveLease::prepare_response)
    }

    pub(crate) fn insert_prepared(&self, input: PdReceiveLeaseInput) -> PdReceivePrepareResponse {
        let now = Instant::now();
        let mut state = self.state.lock();
        Self::gc_expired_locked(&mut state, now);

        let request_key = request_key(&input.instance_id, &input.request_id);
        if let Some(handle) = state.by_request.get(&request_key)
            && let Some(lease) = state.by_handle.get(handle)
        {
            return lease.prepare_response();
        }

        let handle = Uuid::new_v4().to_string();
        let imm_data = self.allocate_imm_data(&state);
        let lease = PdReceiveLease {
            instance_id: input.instance_id,
            request_id: input.request_id,
            handle: handle.clone(),
            imm_data,
            state: PdReceiveLeaseState::Prepared,
            expected_imm_count: input.expected_imm_count.max(1),
            observed_imm_count: 0,
            created_at: now,
            expires_at: input.expires_at,
            expires_at_ms: input.expires_at_ms,
            ranks: input.ranks,
            slabs: input
                .slabs
                .into_iter()
                .map(|(desc, allocation)| PdReceiveSlab {
                    desc,
                    allocation: Some(allocation),
                })
                .collect(),
            layers: input.layers,
            block_hashes: input.block_hashes,
        };
        let response = lease.prepare_response();

        state.by_request.insert(request_key, handle.clone());
        state.by_imm.insert(imm_data, handle.clone());
        state.by_handle.insert(handle, lease);
        response
    }

    pub(crate) fn get_descriptor(
        &self,
        dst_instance_id: &str,
        request_id: &str,
        receive_rank: Option<usize>,
        handle: Option<&str>,
    ) -> PdReceiveDescriptorLookup {
        let now = Instant::now();
        let mut state = self.state.lock();
        Self::gc_expired_locked(&mut state, now);

        let lease_handle = if let Some(handle) = handle.filter(|h| !h.is_empty()) {
            handle.to_string()
        } else {
            match state
                .by_request
                .get(&request_key(dst_instance_id, request_id))
            {
                Some(handle) => handle.clone(),
                None => return PdReceiveDescriptorLookup::Pending,
            }
        };

        let Some(lease) = state.by_handle.get(&lease_handle) else {
            return PdReceiveDescriptorLookup::Pending;
        };
        if lease.instance_id != dst_instance_id || lease.request_id != request_id {
            return PdReceiveDescriptorLookup::Pending;
        }
        let expired = lease.expires_at <= now;
        if expired {
            Self::remove_locked(&mut state, &lease_handle);
            return PdReceiveDescriptorLookup::Expired;
        }
        let lease = state
            .by_handle
            .get(&lease_handle)
            .expect("lease checked above");

        match lease.state {
            PdReceiveLeaseState::Prepared
            | PdReceiveLeaseState::Writing
            | PdReceiveLeaseState::Ready => match lease.descriptor(receive_rank) {
                Some(descriptor) => PdReceiveDescriptorLookup::Ready(descriptor),
                None => PdReceiveDescriptorLookup::Failed,
            },
        }
    }

    pub(crate) fn try_checkout_ready(
        &self,
        dst_instance_id: &str,
        request_id: &str,
        handle: &str,
    ) -> Result<PdReceiveCheckoutStatus, EngineError> {
        let now = Instant::now();
        let mut state = self.state.lock();

        let lease_handle = if !handle.is_empty() {
            handle.to_string()
        } else {
            state
                .by_request
                .get(&request_key(dst_instance_id, request_id))
                .cloned()
                .ok_or_else(|| {
                    EngineError::InvalidArgument(format!(
                        "P/D receive lease not found for checkout: instance={dst_instance_id} request={request_id}"
                    ))
                })?
        };

        let Some(lease) = state.by_handle.get(&lease_handle) else {
            return Err(EngineError::InvalidArgument(format!(
                "P/D receive lease handle not found for checkout: {lease_handle}"
            )));
        };
        if lease.instance_id != dst_instance_id || lease.request_id != request_id {
            return Err(EngineError::InvalidArgument(format!(
                "P/D receive lease identity mismatch for checkout: handle={lease_handle}"
            )));
        }
        if lease.expires_at <= now {
            Self::remove_locked(&mut state, &lease_handle);
            return Ok(PdReceiveCheckoutStatus::Expired);
        }
        match lease.state {
            PdReceiveLeaseState::Prepared | PdReceiveLeaseState::Writing => {
                return Ok(PdReceiveCheckoutStatus::Pending);
            }
            PdReceiveLeaseState::Ready => {}
        }

        let load_plans = Self::load_plans_from_lease(lease, dst_instance_id, request_id)?;
        Self::remove_locked(&mut state, &lease_handle);
        Ok(PdReceiveCheckoutStatus::Ready(load_plans))
    }

    fn load_plans_from_lease(
        lease: &PdReceiveLease,
        dst_instance_id: &str,
        request_id: &str,
    ) -> Result<Vec<PdReceiveLoadPlan>, EngineError> {
        let mut load_plans = Vec::with_capacity(lease.ranks.len());
        for rank in &lease.ranks {
            let slab = lease.slabs.get(rank.slab_index).ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "P/D receive rank {} references missing slab {}",
                    rank.receive_rank, rank.slab_index
                ))
            })?;
            let allocation = slab.allocation.as_ref().ok_or_else(|| {
                EngineError::InvalidArgument(format!(
                    "P/D receive rank {} staging slab has already been checked out: instance={dst_instance_id} request={request_id}",
                    rank.receive_rank
                ))
            })?;
            let layers: Vec<PdReceiveLoadLayer> = lease
                .layers
                .iter()
                .filter(|layer| layer.receive_rank == rank.receive_rank)
                .map(|layout| PdReceiveLoadLayer {
                    layout: layout.clone(),
                    slab: slab.desc.clone(),
                    allocation: Arc::clone(allocation),
                })
                .collect();
            if layers.is_empty() {
                return Err(EngineError::InvalidArgument(format!(
                    "P/D receive rank {} has no layer layouts",
                    rank.receive_rank
                )));
            }
            load_plans.push(PdReceiveLoadPlan { layers });
        }
        if load_plans.is_empty() {
            return Err(EngineError::InvalidArgument(format!(
                "P/D receive lease has no ranks: instance={dst_instance_id} request={request_id}"
            )));
        }
        Ok(load_plans)
    }

    #[allow(dead_code)]
    pub(crate) fn observe_imm(&self, imm_data: u32) -> bool {
        let now = Instant::now();
        let mut state = self.state.lock();
        Self::gc_expired_locked(&mut state, now);
        let Some(handle) = state.by_imm.get(&imm_data).cloned() else {
            return false;
        };
        let Some(lease) = state.by_handle.get_mut(&handle) else {
            return false;
        };
        if lease.state == PdReceiveLeaseState::Prepared {
            lease.state = PdReceiveLeaseState::Writing;
        }
        lease.observed_imm_count = lease.observed_imm_count.saturating_add(1);
        if lease.observed_imm_count >= lease.expected_imm_count {
            lease.state = PdReceiveLeaseState::Ready;
        }
        true
    }

    fn allocate_imm_data(&self, state: &PdReceiveState) -> u32 {
        loop {
            let imm = self.next_imm.fetch_add(1, Ordering::Relaxed);
            if imm != 0 && !state.by_imm.contains_key(&imm) {
                return imm;
            }
        }
    }

    fn gc_expired_locked(state: &mut PdReceiveState, now: Instant) {
        let expired: Vec<String> = state
            .by_handle
            .iter()
            .filter_map(|(handle, lease)| {
                if lease.expires_at <= now {
                    Some(handle.clone())
                } else {
                    None
                }
            })
            .collect();
        for handle in expired {
            Self::remove_locked(state, &handle);
        }
    }

    fn remove_locked(state: &mut PdReceiveState, handle: &str) -> Option<PdReceiveLease> {
        let lease = state.by_handle.remove(handle)?;
        state
            .by_request
            .remove(&request_key(&lease.instance_id, &lease.request_id));
        state.by_imm.remove(&lease.imm_data);
        Some(lease)
    }
}

impl PegaEngine {
    pub fn prepare_staging_receive(
        &self,
        request: PdReceivePrepareRequest,
    ) -> Result<PdReceivePrepareResponse, EngineError> {
        validate_prepare_request(&request)?;
        if let Some(response) = self
            .pd_receive
            .existing_prepare_response(&request.instance_id, &request.request_id)
        {
            return Ok(response);
        }

        let num_blocks = resolved_num_blocks(&request)?;
        let instance = self.get_instance(&request.instance_id)?;
        let receive_ranks = instance.registered_shards();
        if receive_ranks.is_empty() {
            return Err(EngineError::InvalidArgument(format!(
                "instance {} has no registered receive ranks",
                request.instance_id
            )));
        }

        let mut seen_receive_ranks = HashSet::with_capacity(receive_ranks.len());
        let mut ranks = Vec::with_capacity(receive_ranks.len());
        let mut slabs = Vec::with_capacity(receive_ranks.len());
        let mut layers = Vec::new();

        for shard in receive_ranks {
            let receive_rank = shard.tp_rank;
            if !seen_receive_ranks.insert(receive_rank) {
                return Err(EngineError::InvalidArgument(format!(
                    "instance {} registered duplicate tp_rank {} for P/D receive",
                    request.instance_id, receive_rank
                )));
            }
            if shard.layers.is_empty() {
                return Err(EngineError::InvalidArgument(format!(
                    "instance {} receive_rank {} has no registered layers",
                    request.instance_id, receive_rank
                )));
            }

            let slab_index = slabs.len();
            let mut layer_plans = Vec::with_capacity(shard.layers.len());
            for layer in shard.layers {
                if num_blocks > layer.registration.num_blocks {
                    return Err(EngineError::InvalidArgument(format!(
                        "requested num_blocks {} exceeds registered capacity {} for layer {} receive_rank {}",
                        num_blocks, layer.registration.num_blocks, layer.layer_name, receive_rank
                    )));
                }
                let slot_id = instance.get_slot_index(layer.layer_id, shard.tp_rank)?;
                layer_plans.push(PdReceiveLayerPlan {
                    layer_name: layer.layer_name,
                    block_stride: layer.registration.padded_block_size_bytes,
                    segment_count: layer.registration.segments,
                    segment_size: layer.registration.bytes_per_block,
                    padded_segment_stride: layer.registration.padded_bytes_per_block,
                    slot_id,
                });
            }

            let (rank_layers, slab_size) =
                build_layer_layouts(receive_rank, slab_index, &layer_plans, num_blocks)?;
            let alloc_size = NonZeroU64::new(slab_size).ok_or_else(|| {
                EngineError::InvalidArgument("P/D receive allocation size is zero".to_string())
            })?;
            let numa_node = shard.preferred_numa;
            let allocation = self
                .storage
                .allocate(alloc_size, Some(numa_node))
                .ok_or_else(|| {
                    EngineError::Storage(format!(
                        "pinned pool exhausted while allocating P/D receive slab: bytes={slab_size} numa={numa_node}"
                    ))
                })?;
            let base_ptr = allocation.as_non_null().as_ptr() as u64;

            ranks.push(PdReceiveRankDesc {
                receive_rank,
                device_id: shard.device_id,
                tp_rank: shard.tp_rank,
                slab_index,
                numa_node,
            });
            slabs.push((
                PdReceiveSlabDesc {
                    base_ptr,
                    size: slab_size,
                    numa_node,
                },
                allocation,
            ));
            layers.extend(rank_layers);
        }

        let expire_after = request.expire_after.unwrap_or(DEFAULT_PD_RECEIVE_TTL);
        let now = Instant::now();
        let expires_at = now.checked_add(expire_after).ok_or_else(|| {
            EngineError::InvalidArgument("expire_after overflows Instant".to_string())
        })?;
        let expires_at_ms = unix_ms_after(expire_after);
        let default_imm_count = ranks.len().saturating_mul(self.pd_receive_imm_fanout());
        let expected_imm_count = if request.expected_imm_count == 0 {
            default_imm_count
        } else {
            request.expected_imm_count
        };

        Ok(self.pd_receive.insert_prepared(PdReceiveLeaseInput {
            instance_id: request.instance_id,
            request_id: request.request_id,
            expected_imm_count,
            expires_at,
            expires_at_ms,
            ranks,
            slabs,
            layers,
            block_hashes: request.block_hashes,
        }))
    }

    pub fn get_pd_receive_descriptor(
        &self,
        dst_instance_id: &str,
        request_id: &str,
        receive_rank: Option<usize>,
        handle: Option<&str>,
    ) -> PdReceiveDescriptorLookup {
        self.pd_receive
            .get_descriptor(dst_instance_id, request_id, receive_rank, handle)
    }

    fn pd_receive_imm_fanout(&self) -> usize {
        self.storage
            .rdma_transport()
            .map(|rdma| rdma.nic_count())
            .unwrap_or(1)
            .max(1)
    }
}

fn request_key(instance_id: &str, request_id: &str) -> PdReceiveRequestKey {
    (instance_id.to_string(), request_id.to_string())
}

fn validate_prepare_request(request: &PdReceivePrepareRequest) -> Result<(), EngineError> {
    if request.instance_id.is_empty() {
        return Err(EngineError::InvalidArgument(
            "instance_id must not be empty".to_string(),
        ));
    }
    if request.request_id.is_empty() {
        return Err(EngineError::InvalidArgument(
            "request_id must not be empty".to_string(),
        ));
    }
    let _ = resolved_num_blocks(request)?;
    Ok(())
}

fn resolved_num_blocks(request: &PdReceivePrepareRequest) -> Result<usize, EngineError> {
    let hash_blocks = request.block_hashes.len();
    if hash_blocks > 0 {
        if request.num_blocks > 0 && hash_blocks > request.num_blocks {
            return Err(EngineError::InvalidArgument(format!(
                "num_blocks {} is smaller than block_hashes length {}",
                request.num_blocks, hash_blocks
            )));
        }
        return Ok(request.num_blocks.max(hash_blocks));
    }
    if request.num_blocks == 0 {
        return Err(EngineError::InvalidArgument(
            "num_blocks must be > 0 when block_hashes is empty".to_string(),
        ));
    }
    Ok(request.num_blocks)
}

pub(crate) fn build_layer_layouts(
    receive_rank: usize,
    slab_index: usize,
    layer_plans: &[PdReceiveLayerPlan],
    num_blocks: usize,
) -> Result<(Vec<PdReceiveLayerLayout>, u64), EngineError> {
    let mut offset = 0u64;
    let mut layouts = Vec::with_capacity(layer_plans.len());
    for plan in layer_plans {
        let block_stride = u64::try_from(plan.block_stride).map_err(|_| {
            EngineError::InvalidArgument(format!(
                "block_stride for layer {} does not fit into u64",
                plan.layer_name
            ))
        })?;
        let layer_bytes = block_stride.checked_mul(num_blocks as u64).ok_or_else(|| {
            EngineError::InvalidArgument(format!(
                "P/D receive layer size overflow for layer {}",
                plan.layer_name
            ))
        })?;
        layouts.push(PdReceiveLayerLayout {
            layer_name: plan.layer_name.clone(),
            receive_rank,
            slab_index,
            layer_offset: offset,
            block_stride,
            segment_count: plan.segment_count,
            segment_size: plan.segment_size as u64,
            padded_segment_stride: plan.padded_segment_stride as u64,
            num_blocks,
            slot_id: plan.slot_id,
        });
        offset = offset.checked_add(layer_bytes).ok_or_else(|| {
            EngineError::InvalidArgument("P/D receive slab size overflow".to_string())
        })?;
    }
    Ok((layouts, offset))
}

fn unix_ms_after(duration: Duration) -> u64 {
    let deadline = SystemTime::now()
        .checked_add(duration)
        .unwrap_or(UNIX_EPOCH);
    deadline
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_layouts_use_one_slab_with_per_layer_offsets() {
        let plans = vec![
            PdReceiveLayerPlan {
                layer_name: "layer_0".to_string(),
                block_stride: 128,
                segment_count: 1,
                segment_size: 128,
                padded_segment_stride: 128,
                slot_id: 0,
            },
            PdReceiveLayerPlan {
                layer_name: "layer_1".to_string(),
                block_stride: 256,
                segment_count: 2,
                segment_size: 100,
                padded_segment_stride: 128,
                slot_id: 4,
            },
        ];

        let (layouts, slab_size) = build_layer_layouts(3, 2, &plans, 3).unwrap();

        assert_eq!(slab_size, 128 * 3 + 256 * 3);
        assert_eq!(layouts.len(), 2);
        assert_eq!(layouts[0].receive_rank, 3);
        assert_eq!(layouts[0].slab_index, 2);
        assert_eq!(layouts[0].layer_offset, 0);
        assert_eq!(layouts[0].block_stride, 128);
        assert_eq!(layouts[1].slab_index, 2);
        assert_eq!(layouts[1].layer_offset, 128 * 3);
        assert_eq!(layouts[1].segment_count, 2);
        assert_eq!(layouts[1].padded_segment_stride, 128);
    }

    #[test]
    fn resolved_num_blocks_allows_partial_hash_metadata() {
        let request = PdReceivePrepareRequest {
            instance_id: "i".to_string(),
            request_id: "r".to_string(),
            block_hashes: vec![vec![1], vec![2]],
            num_blocks: 3,
            expected_imm_count: 1,
            expire_after: None,
        };

        assert_eq!(resolved_num_blocks(&request).unwrap(), 3);
    }

    #[test]
    fn resolved_num_blocks_rejects_hash_count_above_num_blocks() {
        let request = PdReceivePrepareRequest {
            instance_id: "i".to_string(),
            request_id: "r".to_string(),
            block_hashes: vec![vec![1], vec![2], vec![3]],
            num_blocks: 2,
            expected_imm_count: 1,
            expire_after: None,
        };

        let err = resolved_num_blocks(&request).unwrap_err();
        assert!(err.to_string().contains("smaller than block_hashes"));
    }
}
