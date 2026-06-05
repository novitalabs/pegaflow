// ============================================================================
// Offload: GPU→CPU save path for KV cache blocks.
//
// Phases 0-3 (validation, hash filter, pinned allocation, GPU copy) run on the
// RPC path. Phase 4 (build RawBlocks, group by hash, insert into inflight)
// is deferred to the storage insert worker via `RawSaveBatch`.
// ============================================================================

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::Arc;

use log::{debug, info};

use crate::block::{BlockKey, LayerSave, RawBlock, Segment};
use crate::pinned_pool::MappedPinnedPtr;

/// Grouped insert entries: each hash maps to its per-slot `RawBlock`s.
pub(crate) type InsertEntries = Vec<(BlockKey, Vec<(usize, Arc<RawBlock>)>)>;
use crate::gpu_worker::{SaveBlock, SaveLayerData};
use crate::instance::KVCacheRegistration;
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use crate::{EngineError, PegaEngine};
use pegaflow_common::NumaNode;

// ============================================================================
// Types sent to the insert worker (deferred Phase 4)
// ============================================================================

/// How a layer's blocks are laid out in pinned memory after GPU copy.
///
/// Sizes here are *padded* (SSD-aligned). GPU copies use actual (unpadded)
/// sizes from `KVCacheRegistration`; the padding tail is unused.
pub(crate) enum LayerAlloc {
    Split {
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
        k_base: MappedPinnedPtr,
        v_base: MappedPinnedPtr,
        /// Per-segment stride in pinned memory (padded for SSD alignment).
        padded_segment_size: usize,
    },
    Contiguous {
        allocation: Arc<PinnedAllocation>,
        base: MappedPinnedPtr,
    },
}

impl LayerAlloc {
    /// Construct a `RawBlock` for the i-th block in this allocation.
    fn make_raw_block(&self, index: usize, block_size: usize) -> Arc<RawBlock> {
        match self {
            LayerAlloc::Split {
                k_allocation,
                v_allocation,
                k_base,
                v_base,
                padded_segment_size,
                ..
            } => {
                let half = block_size / 2;
                let k_ptr = k_base.add(index * padded_segment_size).host();
                let v_ptr = v_base.add(index * padded_segment_size).host();
                // Safety: pointers are within pinned allocations, validated during allocation.
                Arc::new(RawBlock::two_segments(
                    Segment::new(k_ptr, half, Arc::clone(k_allocation)),
                    Segment::new(v_ptr, half, Arc::clone(v_allocation)),
                ))
            }
            LayerAlloc::Contiguous {
                allocation, base, ..
            } => {
                let ptr = base.add(index * block_size).host();
                // Safety: pointer is within pinned allocation, validated during allocation.
                Arc::new(RawBlock::single_segment(Segment::new(
                    ptr,
                    block_size,
                    Arc::clone(allocation),
                )))
            }
        }
    }
}

/// Per-layer data for deferred RawBlock construction.
pub(crate) struct RawSaveLayer {
    pub slot_id: usize,
    /// Padded block size (SSD-aligned). Becomes `RawBlock.total_size` → `SlotMeta.total_size()`.
    pub padded_block_size: usize,
    /// Allocations for this layer. Vec length is 1 (batch) or num_blocks (blockwise).
    pub allocs: Vec<LayerAlloc>,
    /// Block hashes in allocation order.
    pub block_hashes: Vec<Vec<u8>>,
}

/// Deferred save batch: sent to insert worker after GPU copy completes.
pub(crate) struct RawSaveBatch {
    pub namespace: String,
    pub total_slots: usize,
    pub numa_node: NumaNode,
    pub layers: Vec<RawSaveLayer>,
}

/// Build insert entries from a raw batch (called by the insert worker).
///
/// Returns `(entries, total_bytes, total_blocks)` where entries is grouped
/// by hash: `Vec<(BlockKey, Vec<(slot_id, Arc<RawBlock>)>)>`.
pub(crate) fn build_insert_entries(batch: &RawSaveBatch) -> (InsertEntries, u64, usize) {
    use std::collections::HashMap;

    if let Some(entries) = build_ordered_insert_entries(batch) {
        return entries;
    }

    let mut hash_entries: HashMap<Vec<u8>, Vec<(usize, Arc<RawBlock>)>> = HashMap::new();
    let mut total_bytes: u64 = 0;
    let mut total_blocks: usize = 0;

    for layer in &batch.layers {
        let blockwise = layer.allocs.len() > 1;
        for (i, hash) in layer.block_hashes.iter().enumerate() {
            let (alloc_idx, offset_in_alloc) = if blockwise {
                (i, 0) // Each block has its own allocation
            } else {
                (0, i) // All blocks in one allocation
            };
            let block =
                layer.allocs[alloc_idx].make_raw_block(offset_in_alloc, layer.padded_block_size);
            hash_entries
                .entry(hash.clone())
                .or_default()
                .push((layer.slot_id, block));
        }
        let layer_blocks = layer.block_hashes.len();
        total_blocks += layer_blocks;
        total_bytes += (layer.padded_block_size as u64).saturating_mul(layer_blocks as u64);
    }

    let entries: InsertEntries = hash_entries
        .into_iter()
        .map(|(hash, slots)| (BlockKey::new(batch.namespace.clone(), hash), slots))
        .collect();

    (entries, total_bytes, total_blocks)
}

fn build_ordered_insert_entries(batch: &RawSaveBatch) -> Option<(InsertEntries, u64, usize)> {
    let first_layer = batch.layers.first()?;
    let num_entries = first_layer.block_hashes.len();
    if batch.layers.iter().any(|layer| {
        layer.block_hashes.len() != num_entries || layer.block_hashes != first_layer.block_hashes
    }) {
        return None;
    }

    let mut total_blocks = 0usize;
    let mut total_bytes = 0u64;
    for layer in &batch.layers {
        total_blocks += layer.block_hashes.len();
        total_bytes +=
            (layer.padded_block_size as u64).saturating_mul(layer.block_hashes.len() as u64);
    }

    let mut entries = Vec::with_capacity(num_entries);
    for (block_idx, hash) in first_layer.block_hashes.iter().enumerate() {
        let mut slots = Vec::with_capacity(batch.layers.len());
        for layer in &batch.layers {
            let blockwise = layer.allocs.len() > 1;
            let (alloc_idx, offset_in_alloc) = if blockwise {
                (block_idx, 0)
            } else {
                (0, block_idx)
            };
            let block =
                layer.allocs[alloc_idx].make_raw_block(offset_in_alloc, layer.padded_block_size);
            slots.push((layer.slot_id, block));
        }
        entries.push((BlockKey::new(batch.namespace.clone(), hash.clone()), slots));
    }

    Some((entries, total_bytes, total_blocks))
}

// ============================================================================
// Save methods on PegaEngine (moved from lib.rs)
// ============================================================================

impl PegaEngine {
    /// Batch save KV blocks from multiple layers.
    ///
    /// More efficient than calling `save_kv_blocks_from_ipc` in a loop as it
    /// reduces Python-Rust boundary crossings and batches GPU copies + storage
    /// operations across all layers for minimal lock overhead and a single
    /// CUDA stream synchronization.
    pub async fn batch_save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
    ) -> Result<(), EngineError> {
        self.batch_save_kv_blocks_from_ipc_with_numa_hint(
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            saves,
            None,
        )
        .await
    }

    /// Batch save KV blocks with an optional NUMA allocation override.
    ///
    /// `numa_hint` only changes the pinned-memory allocation node and the
    /// recorded slot NUMA metadata. CUDA reads still use the registered GPU
    /// context identified by `device_id`. Hints are accepted only when they
    /// target a registered NUMA node for this effective TP/PP group.
    #[allow(
        clippy::too_many_arguments,
        reason = "save requests carry the externally registered KV layout fields"
    )]
    pub async fn batch_save_kv_blocks_from_ipc_with_numa_hint(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
        numa_hint: Option<NumaNode>,
    ) -> Result<(), EngineError> {
        let batch_start = std::time::Instant::now();
        let total_layers = saves.len();
        self.query_leases.sweep_expired();
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace().to_string();
        let total_slots = instance.total_slots();

        // ── Phase 0: Resolve per-layer metadata and build valid_blocks ──
        trace_scope!("save.resolve_metadata", _s);

        /// Unified per-layer context for the save pipeline.
        /// Combines metadata, filtered blocks, and allocation results.
        struct LayerContext {
            layer_name: String,
            registration: KVCacheRegistration,
            slot_id: usize,
            padded_block_size: usize,
            /// Blocks to save: (block_idx, hash). Filtered in Phase 1.
            blocks_to_save: Vec<(usize, Vec<u8>)>,
            /// Allocation results, populated in Phase 2.
            /// Vec to support blockwise allocation (1 element for batch, N for blockwise).
            allocs: Vec<LayerAlloc>,
        }

        let gpu = instance.get_gpu_for_save_group(device_id, tp_rank, pp_rank)?;

        let mut layers: Vec<LayerContext> = Vec::with_capacity(saves.len());

        for LayerSave {
            layer_name,
            block_ids,
            block_hashes,
        } in saves
        {
            if block_ids.len() != block_hashes.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "block_ids length {} does not match block_hashes {} for layer {}",
                    block_ids.len(),
                    block_hashes.len(),
                    layer_name
                )));
            }

            let layer_id = instance.get_layer_id(&layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} unknown"))
            })?;

            let registration = gpu.get_registration(&layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} not registered on device"))
            })?;

            let slot_id = instance.get_slot_index(layer_id, tp_rank)?;
            if slot_id >= total_slots {
                return Err(EngineError::InvalidArgument(format!(
                    "slot_id {} out of range (total_slots {})",
                    slot_id, total_slots
                )));
            }

            let blocks_to_save: Vec<(usize, Vec<u8>)> = block_ids
                .into_iter()
                .zip(block_hashes)
                .filter(|(id, _)| *id >= 0)
                .filter_map(|(id, hash)| {
                    let idx = id as usize;
                    if idx < registration.num_blocks {
                        Some((idx, hash))
                    } else {
                        None
                    }
                })
                .collect();

            if blocks_to_save.is_empty() {
                continue;
            }

            let padded_block_size = registration.padded_block_size_bytes;
            layers.push(LayerContext {
                layer_name,
                registration,
                slot_id,
                padded_block_size,
                blocks_to_save,
                allocs: Vec::new(),
            });
        }
        trace_drop!(_s);

        if layers.is_empty() {
            info!(
                "save_batch skipped (no valid blocks): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // ── Phase 1: Union-filter hashes across all layers ──
        //
        // Layers may have different hash sets (heterogeneous). Compute the
        // union of all hashes, filter once against the cache, then per-layer
        // in-memory filter to determine which blocks each layer needs to save.

        trace_scope!("save.hash_filter", _s);

        let shared_hash_order = if let Some((first_layer, rest_layers)) = layers.split_first() {
            rest_layers.iter().all(|layer| {
                layer.blocks_to_save.len() == first_layer.blocks_to_save.len()
                    && layer
                        .blocks_to_save
                        .iter()
                        .zip(&first_layer.blocks_to_save)
                        .all(|((_, hash), (_, first_hash))| hash == first_hash)
            })
        } else {
            false
        };

        let mut hashes_to_save: HashSet<Vec<u8>> = HashSet::new();
        let hash_source_layers = if shared_hash_order {
            &layers[..1]
        } else {
            &layers[..]
        };
        for layer in hash_source_layers {
            for (_, hash) in &layer.blocks_to_save {
                hashes_to_save.insert(hash.clone());
            }
        }

        // Single in-place cache filter for all unique hashes
        self.storage
            .filter_hashes_not_in_cache_inplace(&namespace, &mut hashes_to_save);

        if hashes_to_save.is_empty() {
            trace_drop!(_s);
            debug!(
                "save_batch skipped (all cached): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // Per-layer filter: keep only blocks whose hash needs saving
        let mut total_blocks_to_save = 0usize;
        for layer in &mut layers {
            layer
                .blocks_to_save
                .retain(|(_, hash)| hashes_to_save.contains(hash.as_slice()));
            total_blocks_to_save += layer.blocks_to_save.len();
        }
        // Remove layers with no blocks to save
        layers.retain(|layer| !layer.blocks_to_save.is_empty());

        trace_drop!(_s, || {
            [
                ("unique_hashes", hashes_to_save.len().to_string()),
                ("to_save", total_blocks_to_save.to_string()),
            ]
        });

        if layers.is_empty() {
            debug!(
                "save_batch skipped (all filtered): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // ── Phase 2: Allocate pinned memory + build SaveBlocks for all layers ──

        trace_scope!("save.pinned_alloc", _s);
        let save_numa_node = match numa_hint {
            Some(hint) => instance.validate_save_numa_hint(tp_rank, pp_rank, hint)?,
            None => gpu.preferred_numa(),
        };
        let numa_node = Some(save_numa_node);
        let blockwise = self.storage.blockwise_alloc();

        let mut gpu_save_layers: Vec<SaveLayerData> = Vec::with_capacity(layers.len());

        for layer in &mut layers {
            let registration = &layer.registration;
            let num_blocks = layer.blocks_to_save.len();

            // Blockwise: allocate once per block; Batch: allocate once for all blocks
            let alloc_count = if blockwise { num_blocks } else { 1 };
            let blocks_per_alloc = if blockwise { 1 } else { num_blocks };

            let is_split = registration.segments == 2
                && registration.kv_stride_bytes > registration.bytes_per_block;

            if is_split {
                let padded_segment_size = registration.padded_bytes_per_block;

                for _ in 0..alloc_count {
                    let alloc_size = (padded_segment_size as u64)
                        .checked_mul(blocks_per_alloc as u64)
                        .and_then(NonZeroU64::new)
                        .ok_or_else(|| {
                            EngineError::Storage(
                                "allocation size overflow for K segments".to_string(),
                            )
                        })?;

                    let k_allocation =
                        self.storage
                            .allocate(alloc_size, numa_node)
                            .ok_or_else(|| {
                                EngineError::Storage(
                                    "pinned pool exhausted while allocating K segment buffer"
                                        .to_string(),
                                )
                            })?;
                    let v_allocation =
                        self.storage
                            .allocate(alloc_size, numa_node)
                            .ok_or_else(|| {
                                EngineError::Storage(
                                    "pinned pool exhausted while allocating V segment buffer"
                                        .to_string(),
                                )
                            })?;

                    let k_base = k_allocation.mapped_ptr();
                    let v_base = v_allocation.mapped_ptr();

                    layer.allocs.push(LayerAlloc::Split {
                        k_allocation,
                        v_allocation,
                        k_base,
                        v_base,
                        padded_segment_size,
                    });
                }

                // Build SaveBlocks from layer.allocs
                let save_blocks: Vec<SaveBlock> = layer
                    .blocks_to_save
                    .iter()
                    .enumerate()
                    .map(|(i, (block_idx, _))| {
                        let (alloc_idx, offset_in_alloc) = if blockwise { (i, 0) } else { (0, i) };
                        let (k_base, v_base) = match &layer.allocs[alloc_idx] {
                            LayerAlloc::Split { k_base, v_base, .. } => (*k_base, *v_base),
                            _ => unreachable!(),
                        };
                        let offset = offset_in_alloc * padded_segment_size;
                        let k_dst = k_base.add(offset);
                        let v_dst = v_base.add(offset);
                        SaveBlock {
                            block_idx: *block_idx,
                            k_dst,
                            v_dst: Some(v_dst),
                        }
                    })
                    .collect();

                gpu_save_layers.push(SaveLayerData {
                    layer_name: layer.layer_name.clone(),
                    registration: registration.clone(),
                    blocks: save_blocks,
                });
            } else {
                let padded_block_size = layer.padded_block_size;

                for _ in 0..alloc_count {
                    let alloc_size = (padded_block_size as u64)
                        .checked_mul(blocks_per_alloc as u64)
                        .and_then(NonZeroU64::new)
                        .ok_or_else(|| {
                            EngineError::Storage("allocation size overflow".to_string())
                        })?;

                    let allocation =
                        self.storage
                            .allocate(alloc_size, numa_node)
                            .ok_or_else(|| {
                                EngineError::Storage(
                                    "pinned pool exhausted while allocating contiguous block buffer"
                                        .to_string(),
                                )
                            })?;

                    let base = allocation.mapped_ptr();

                    layer
                        .allocs
                        .push(LayerAlloc::Contiguous { allocation, base });
                }

                // Build SaveBlocks from layer.allocs
                let save_blocks: Vec<SaveBlock> = layer
                    .blocks_to_save
                    .iter()
                    .enumerate()
                    .map(|(i, (block_idx, _))| {
                        let (alloc_idx, offset_in_alloc) = if blockwise { (i, 0) } else { (0, i) };
                        let base = match &layer.allocs[alloc_idx] {
                            LayerAlloc::Contiguous { base, .. } => *base,
                            _ => unreachable!(),
                        };
                        let offset = offset_in_alloc * padded_block_size;
                        let dst = base.add(offset);
                        SaveBlock {
                            block_idx: *block_idx,
                            k_dst: dst,
                            v_dst: None,
                        }
                    })
                    .collect();

                gpu_save_layers.push(SaveLayerData {
                    layer_name: layer.layer_name.clone(),
                    registration: registration.clone(),
                    blocks: save_blocks,
                });
            }
        }
        trace_drop!(_s);

        // ── Phase 3: Submit all GPU copies as one batch task (single sync) ──

        trace_future!(
            "save.gpu_copy",
            gpu.worker_pool().batch_save(gpu_save_layers)
        )
        .await?;

        // ── Phase 4 (deferred): Build RawBlocks + insert — sent to worker ──

        // Record metrics on the RPC path (cheap, measures RPC-visible latency)
        let metrics = core_metrics();
        let mut total_bytes = 0u64;
        for layer in &layers {
            let num_blocks = layer.blocks_to_save.len();
            let bytes = (layer.padded_block_size as u64)
                .checked_mul(num_blocks as u64)
                .unwrap_or(0);
            total_bytes += bytes;
        }
        if total_blocks_to_save > 0 {
            metrics.save_bytes.add(total_bytes, &[]);
            metrics
                .save_duration_seconds
                .record(batch_start.elapsed().as_secs_f64(), &[]);
        }

        debug!(
            "save_batch completed: instance_id={} tp_rank={} pp_rank={} device_id={} layers={} layers_saved={} blocks_saved={} bytes={} numa={} total_ms={:.2}",
            instance_id,
            tp_rank,
            pp_rank,
            device_id,
            total_layers,
            layers.len(),
            total_blocks_to_save,
            total_bytes,
            save_numa_node,
            batch_start.elapsed().as_secs_f64() * 1000.0
        );

        // Build RawSaveBatch and send to insert worker (fire-and-forget)
        let raw_layers: Vec<RawSaveLayer> = layers
            .into_iter()
            .map(|layer| {
                let block_hashes: Vec<Vec<u8>> = layer
                    .blocks_to_save
                    .into_iter()
                    .map(|(_, hash)| hash)
                    .collect();
                RawSaveLayer {
                    slot_id: layer.slot_id,
                    padded_block_size: layer.padded_block_size,
                    allocs: layer.allocs,
                    block_hashes,
                }
            })
            .collect();

        self.storage.send_raw_insert(RawSaveBatch {
            namespace,
            total_slots,
            numa_node: save_numa_node,
            layers: raw_layers,
        });

        Ok(())
    }
}
