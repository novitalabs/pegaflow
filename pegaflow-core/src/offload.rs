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

/// Grouped insert entries: each hash maps to its per-slot `RawBlock`s.
pub(crate) type InsertEntries = Vec<(BlockKey, Vec<(usize, RawBlock)>)>;
use crate::gpu_worker::{HostBlock, LayerTransferData, TransferBlock};
use crate::layout::KVCacheLayout;
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use crate::{EngineError, PegaEngine};
use pegaflow_common::NumaNode;

// ============================================================================
// Types sent to the insert worker (deferred Phase 4)
// ============================================================================

/// One layer's saved blocks, ready for cache insertion.
///
/// The `RawBlock`s are constructed at allocation time (Phase 2) and shared
/// with the GPU copy task; their segments own the pinned allocations, so no
/// separate allocation bookkeeping is needed.
pub(crate) struct RawSaveLayer {
    pub slot_id: usize,
    /// Padded block size (SSD-aligned). Becomes `RawBlock.total_size` → `SlotMeta.total_size()`.
    pub padded_block_size: usize,
    /// Saved blocks, parallel to `block_hashes`.
    pub blocks: Vec<RawBlock>,
    /// Block hashes in save order.
    pub block_hashes: Vec<Vec<u8>>,
}

/// Deferred save batch: sent to insert worker after GPU copy completes.
pub(crate) struct RawSaveBatch {
    pub namespace: String,
    pub total_slots: usize,
    pub numa_node: NumaNode,
    pub layers: Vec<RawSaveLayer>,
}

/// Unified per-layer context for the save pipeline.
/// Combines metadata, filtered blocks, and allocation results.
struct LayerContext {
    layer_name: String,
    layout: KVCacheLayout,
    slot_id: usize,
    /// Blocks to save: (block_idx, hash). Filtered in Phase 1.
    blocks_to_save: Vec<(usize, Vec<u8>)>,
    /// Host-side block images, parallel to `blocks_to_save`.
    /// Filled after the GPU save task returns the moved blocks.
    raw_blocks: Vec<RawBlock>,
}

/// Build insert entries from a raw batch (called by the insert worker).
///
/// Returns `(entries, total_bytes, total_blocks)` where entries is grouped
/// by hash: `Vec<(BlockKey, Vec<(slot_id, RawBlock)>)>`.
pub(crate) fn build_insert_entries(batch: RawSaveBatch) -> (InsertEntries, u64, usize) {
    let mut total_bytes: u64 = 0;
    let mut total_blocks: usize = 0;
    for layer in &batch.layers {
        let layer_blocks = layer.block_hashes.len();
        total_blocks += layer_blocks;
        total_bytes += (layer.padded_block_size as u64).saturating_mul(layer_blocks as u64);
    }

    let entries = if can_use_ordered_fast_path(&batch.layers) {
        build_ordered_insert_entries(batch.namespace, batch.layers)
    } else {
        build_hashed_insert_entries(batch.namespace, batch.layers)
    };
    (entries, total_bytes, total_blocks)
}

fn can_use_ordered_fast_path(layers: &[RawSaveLayer]) -> bool {
    let Some(first_layer) = layers.first() else {
        return false;
    };
    layers
        .iter()
        .all(|layer| layer.block_hashes == first_layer.block_hashes)
}

/// Fast path: all layers share one hash order, so entries can be built
/// block-by-block without a hash map.
fn build_ordered_insert_entries(namespace: String, layers: Vec<RawSaveLayer>) -> InsertEntries {
    let hashes = layers
        .first()
        .map(|layer| layer.block_hashes.clone())
        .unwrap_or_default();
    let num_blocks = hashes.len();
    let mut per_block_slots: Vec<Vec<(usize, RawBlock)>> =
        (0..num_blocks).map(|_| Vec::new()).collect();

    for layer in layers {
        let slot_id = layer.slot_id;
        for (block_idx, block) in layer.blocks.into_iter().enumerate() {
            per_block_slots[block_idx].push((slot_id, block));
        }
    }

    hashes
        .into_iter()
        .zip(per_block_slots)
        .map(|(hash, slots)| (BlockKey::new(namespace.clone(), hash), slots))
        .collect()
}

/// Fallback for heterogeneous per-layer hash sets: group via a hash map.
fn build_hashed_insert_entries(namespace: String, layers: Vec<RawSaveLayer>) -> InsertEntries {
    use std::collections::HashMap;

    let mut hash_entries: HashMap<Vec<u8>, Vec<(usize, RawBlock)>> = HashMap::new();
    for layer in layers {
        for (block, hash) in layer.blocks.into_iter().zip(layer.block_hashes) {
            hash_entries
                .entry(hash)
                .or_default()
                .push((layer.slot_id, block));
        }
    }

    hash_entries
        .into_iter()
        .map(|(hash, slots)| (BlockKey::new(namespace.clone(), hash), slots))
        .collect()
}

// ============================================================================
// Save methods on PegaEngine (moved from lib.rs)
// ============================================================================

impl PegaEngine {
    /// Page-first allocation: one contiguous pinned page per block, holding
    /// this writer's shard layers at their sealed shard-relative offsets. Each
    /// layer's GPU copy targets a sub-view into the pages; the pages themselves
    /// are sealed as the shard's stored slot (see Phase 4). Fills `pages` and
    /// `gpu_save_layers`.
    fn alloc_page_first_targets(
        &self,
        topology: &crate::instance::LayerTopology,
        layer_contexts: &[LayerContext],
        numa_node: Option<NumaNode>,
        pages: &mut Vec<Arc<PinnedAllocation>>,
        gpu_save_layers: &mut Vec<LayerTransferData>,
    ) -> Result<(), EngineError> {
        // One writer saves exactly one shard (its registered layers), so every
        // layer in this batch maps to the same shard/slot.
        let shard = layer_contexts[0].slot_id;
        let page_size = topology
            .shard_page_size(shard)
            .expect("page-first topology must have a shard page size");

        // A page holds all of the shard's layers, so every layer must save the
        // same blocks in the same order. Block hashes are content-derived
        // (identical across layers) and block_idx is the same physical KV slot
        // across layers, so this invariant holds for MLA saves.
        let reference = &layer_contexts[0].blocks_to_save;
        for ctx in layer_contexts {
            if ctx.layout.is_split() {
                return Err(EngineError::InvalidArgument(format!(
                    "page-first save does not support split K/V (layer {})",
                    ctx.layer_name
                )));
            }
            if ctx.slot_id != shard {
                return Err(EngineError::InvalidArgument(
                    "page-first save spans multiple shards (one writer must save one shard)".into(),
                ));
            }
            if ctx.blocks_to_save != *reference {
                return Err(EngineError::InvalidArgument(
                    "page-first save requires the same block set across all layers".into(),
                ));
            }
        }

        // A shard's page is sealed by this single save, so the save must carry
        // EVERY layer of the shard. A partial save (e.g. a pipeline stage that
        // holds only some of the shard's layers) would seal a page whose
        // missing-layer offsets are never written — they stay as stale pool
        // memory and load returns another block's KV. Reject it loudly.
        let mut layer_ids: Vec<usize> = layer_contexts
            .iter()
            .map(|ctx| topology.layer_id(&ctx.layer_name))
            .collect::<Result<_, _>>()?;
        layer_ids.sort_unstable();
        layer_ids.dedup();
        let shard_layers = topology
            .shard_layer_count(shard)
            .expect("page-first topology must have a shard layer count");
        if layer_ids.len() != shard_layers {
            return Err(EngineError::InvalidArgument(format!(
                "page-first save must cover all {shard_layers} layers of shard {shard}, \
                 got {} distinct (one worker must write a shard's entire page)",
                layer_ids.len()
            )));
        }

        // One page per block.
        let page_bytes = NonZeroU64::new(page_size as u64)
            .ok_or_else(|| EngineError::Storage("page size is zero".into()))?;
        for _ in 0..reference.len() {
            let page = self
                .storage
                .allocate(page_bytes, numa_node)
                .ok_or_else(|| {
                    EngineError::Storage("pinned pool exhausted while allocating page".into())
                })?;
            pages.push(page);
        }

        // Each layer copies its block into its slice of every page.
        for ctx in layer_contexts {
            let layer_id = topology.layer_id(&ctx.layer_name)?;
            let (offset, layer_bytes) = topology
                .page_placement(layer_id)
                .expect("page-first layer placement");
            let save_blocks: Vec<TransferBlock> = ctx
                .blocks_to_save
                .iter()
                .zip(pages.iter())
                .map(|((block_idx, _), page)| TransferBlock {
                    block_idx: *block_idx,
                    block: HostBlock::Owned(RawBlock::single_segment(Segment::new(
                        page.mapped_ptr().add(offset).host(),
                        layer_bytes,
                        Arc::clone(page),
                    ))),
                })
                .collect();
            gpu_save_layers.push(LayerTransferData {
                layer_name: ctx.layer_name.clone(),
                layout: ctx.layout.clone(),
                blocks: save_blocks,
            });
        }
        Ok(())
    }

    /// Batch save KV blocks from multiple layers.
    ///
    /// More efficient than calling `save_kv_blocks_from_ipc` in a loop as it
    /// reduces Python-Rust boundary crossings and batches GPU copies + storage
    /// operations across all layers for minimal lock overhead and a single
    /// CUDA stream synchronization.
    ///
    /// Pinned memory is allocated on the saving GPU's own NUMA node, keeping
    /// every D2H copy socket-local.
    pub async fn batch_save_kv_blocks_from_ipc(
        &self,
        instance_id: &str,
        tp_rank: usize,
        pp_rank: usize,
        device_id: i32,
        saves: Vec<LayerSave>,
    ) -> Result<(), EngineError> {
        self.query_leases.sweep_expired();

        let batch_start = std::time::Instant::now();
        let total_layers = saves.len();

        let instance = self.get_instance(instance_id)?;
        let topology = instance.sealed_topology()?;
        let namespace = instance.namespace().to_string();
        let total_slots = topology.total_slots();

        // ── Phase 0: Resolve per-layer metadata and build valid_blocks ──
        trace_scope!("save.resolve_metadata", _s);

        let gpu_context = instance.get_gpu_for_save_group(device_id, tp_rank, pp_rank)?;

        let mut layer_contexts: Vec<LayerContext> = Vec::with_capacity(saves.len());
        let mut hashes_to_save: HashSet<Vec<u8>> = HashSet::new();

        for LayerSave {
            layer_name,
            block_ids,
            block_hashes,
        } in saves
        {
            // Engine-level contract: in-process callers bypass the RPC-layer
            // validation in service.rs, and zip below would silently truncate.
            if block_ids.len() != block_hashes.len() {
                return Err(EngineError::InvalidArgument(format!(
                    "block_ids length {} does not match block_hashes {} for layer {}",
                    block_ids.len(),
                    block_hashes.len(),
                    layer_name
                )));
            }

            let layer_id = topology.layer_id(&layer_name)?;

            let layout = gpu_context.get_layout(&layer_name).ok_or_else(|| {
                EngineError::InvalidArgument(format!("layer {layer_name} not registered on device"))
            })?;

            let slot_id = topology.slot_index(layer_id, tp_rank)?;

            let num_blocks = layout.num_blocks();

            let blocks_to_save: Vec<(usize, Vec<u8>)> = block_ids
                .into_iter()
                .zip(block_hashes)
                .map(|(idx, hash)| {
                    if idx >= num_blocks {
                        return Err(EngineError::InvalidArgument(format!(
                            "block_id {} out of range (num_blocks={}) for layer {}",
                            idx, num_blocks, layer_name
                        )));
                    }
                    Ok((idx, hash))
                })
                .collect::<Result<_, _>>()?;

            if blocks_to_save.is_empty() {
                continue;
            }

            for (_, hash) in &blocks_to_save {
                hashes_to_save.insert(hash.clone());
            }

            layer_contexts.push(LayerContext {
                layer_name,
                layout,
                slot_id,
                blocks_to_save,
                raw_blocks: Vec::new(),
            });
        }
        trace_drop!(_s);

        if layer_contexts.is_empty() {
            info!(
                "save_batch skipped (no valid blocks): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // ── Phase 1: Filter hashes against cache ──

        trace_scope!("save.hash_filter", _s);

        // Single in-place cache filter for all unique hashes
        self.storage
            .filter_hashes_not_in_cache_inplace(&namespace, &mut hashes_to_save);

        if hashes_to_save.is_empty() {
            trace_drop!(_s);
            return Ok(());
        }

        // Per-layer filter: keep only blocks whose hash needs saving
        let mut total_blocks_to_save = 0usize;
        for ctx in &mut layer_contexts {
            ctx.blocks_to_save
                .retain(|(_, hash)| hashes_to_save.contains(hash.as_slice()));
            total_blocks_to_save += ctx.blocks_to_save.len();
        }
        // Remove layers with no blocks to save
        layer_contexts.retain(|ctx| !ctx.blocks_to_save.is_empty());

        trace_drop!(_s, || {
            [
                ("unique_hashes", hashes_to_save.len().to_string()),
                ("to_save", total_blocks_to_save.to_string()),
            ]
        });

        if layer_contexts.is_empty() {
            return Ok(());
        }

        // ── Phase 2: Allocate pinned memory + build SaveBlocks for all layers ──

        trace_scope!("save.pinned_alloc", _s);
        let save_numa_node = gpu_context.preferred_numa();
        let numa_node = Some(save_numa_node);
        let blockwise = self.storage.blockwise_alloc();
        let page_first = topology.is_page_first();

        let mut gpu_save_layers: Vec<LayerTransferData> = Vec::with_capacity(layer_contexts.len());
        // Page-first only: one contiguous page per block holds every layer.
        // Each layer's GPU copy targets a sub-view into these pages, and the
        // pages themselves become the single stored slot in Phase 4. Kept alive
        // across the GPU copy so the segments stay valid.
        let mut pages: Vec<Arc<PinnedAllocation>> = Vec::new();

        if page_first {
            self.alloc_page_first_targets(
                &topology,
                &layer_contexts,
                numa_node,
                &mut pages,
                &mut gpu_save_layers,
            )?;
        } else {
            for ctx in &mut layer_contexts {
                let layout = &ctx.layout;
                let num_blocks = ctx.blocks_to_save.len();

                // Blockwise: allocate once per block; Batch: allocate once for all blocks
                let alloc_count = if blockwise { num_blocks } else { 1 };
                let blocks_per_alloc = if blockwise { 1 } else { num_blocks };

                let alloc_pinned = |stride: usize, what: &str| {
                    let alloc_size = (stride as u64)
                        .checked_mul(blocks_per_alloc as u64)
                        .and_then(NonZeroU64::new)
                        .ok_or_else(|| {
                            EngineError::Storage(format!("allocation size overflow for {what}"))
                        })?;
                    self.storage.allocate(alloc_size, numa_node).ok_or_else(|| {
                        EngineError::Storage(format!(
                            "pinned pool exhausted while allocating {what}"
                        ))
                    })
                };

                // Allocate pinned memory and construct the host-side RawBlocks in
                // save order. Strides are padded (SSD-aligned); GPU copies later
                // use actual (unpadded) sizes, leaving the padding tail unused.
                let mut raw_blocks: Vec<RawBlock> = Vec::with_capacity(num_blocks);
                for _ in 0..alloc_count {
                    if layout.is_split() {
                        let stride = layout.padded_segment_bytes();
                        let k_alloc = alloc_pinned(stride, "K segment buffer")?;
                        let v_alloc = alloc_pinned(stride, "V segment buffer")?;
                        for i in 0..blocks_per_alloc {
                            let offset = i * stride;
                            // Safety: offsets are within the just-made allocations.
                            raw_blocks.push(RawBlock::two_segments(
                                Segment::new(
                                    k_alloc.mapped_ptr().add(offset).host(),
                                    stride,
                                    Arc::clone(&k_alloc),
                                ),
                                Segment::new(
                                    v_alloc.mapped_ptr().add(offset).host(),
                                    stride,
                                    Arc::clone(&v_alloc),
                                ),
                            ));
                        }
                    } else {
                        let stride = layout.padded_block_bytes();
                        let alloc = alloc_pinned(stride, "block buffer")?;
                        for i in 0..blocks_per_alloc {
                            // Safety: offset is within the just-made allocation.
                            raw_blocks.push(RawBlock::single_segment(Segment::new(
                                alloc.mapped_ptr().add(i * stride).host(),
                                stride,
                                Arc::clone(&alloc),
                            )));
                        }
                    }
                }

                let save_blocks: Vec<TransferBlock> = ctx
                    .blocks_to_save
                    .iter()
                    .zip(raw_blocks)
                    .map(|((block_idx, _), block)| TransferBlock {
                        block_idx: *block_idx,
                        block: HostBlock::Owned(block),
                    })
                    .collect();

                gpu_save_layers.push(LayerTransferData {
                    layer_name: ctx.layer_name.clone(),
                    layout: layout.clone(),
                    blocks: save_blocks,
                });
            }
        }
        trace_drop!(_s);

        // ── Phase 3: Submit all GPU copies as one batch task (single sync) ──

        let returned_layers = trace_future!(
            "save.gpu_copy",
            gpu_context.worker_pool().batch_save(gpu_save_layers)
        )
        .await?;

        if page_first {
            // The pages already hold the copied data; the returned sub-view
            // blocks share those pages via Arc and are dropped here.
            drop(returned_layers);
        } else {
            for (ctx, layer) in layer_contexts.iter_mut().zip(returned_layers) {
                ctx.raw_blocks = layer
                    .blocks
                    .into_iter()
                    .map(|transfer_block| match transfer_block.block {
                        HostBlock::Owned(block) => block,
                        HostBlock::Cached { .. } => {
                            panic!("save path must return HostBlock::Owned blocks")
                        }
                    })
                    .collect();
            }
        }

        // ── Phase 4 (deferred): Build RawBlocks + insert — sent to worker ──

        // Record metrics on the RPC path (cheap, measures RPC-visible latency)
        let metrics = core_metrics();
        let mut total_bytes = 0u64;
        for ctx in &layer_contexts {
            let num_blocks = ctx.blocks_to_save.len();
            let bytes = (ctx.layout.padded_block_bytes() as u64)
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
            layer_contexts.len(),
            total_blocks_to_save,
            total_bytes,
            save_numa_node,
            batch_start.elapsed().as_secs_f64() * 1000.0
        );

        // Build RawSaveBatch and send to insert worker (fire-and-forget).
        // Page-first collapses a shard's layers into one page slot per block, so
        // a whole shard is a single RawSaveLayer regardless of its layer count.
        let raw_layers: Vec<RawSaveLayer> = if page_first {
            let slot_id = layer_contexts[0].slot_id;
            let page_size = topology
                .shard_page_size(slot_id)
                .expect("page-first shard page size");
            let block_hashes: Vec<Vec<u8>> = layer_contexts[0]
                .blocks_to_save
                .iter()
                .map(|(_, hash)| hash.clone())
                .collect();
            let blocks: Vec<RawBlock> = pages
                .into_iter()
                .map(|page| {
                    let ptr = page.mapped_ptr().host();
                    RawBlock::single_segment(Segment::new(ptr, page_size, page))
                })
                .collect();
            vec![RawSaveLayer {
                slot_id,
                padded_block_size: page_size,
                blocks,
                block_hashes,
            }]
        } else {
            layer_contexts
                .into_iter()
                .map(|ctx| {
                    let block_hashes: Vec<Vec<u8>> = ctx
                        .blocks_to_save
                        .into_iter()
                        .map(|(_, hash)| hash)
                        .collect();
                    RawSaveLayer {
                        slot_id: ctx.slot_id,
                        padded_block_size: ctx.layout.padded_block_bytes(),
                        blocks: ctx.raw_blocks,
                        block_hashes,
                    }
                })
                .collect()
        };

        self.storage.send_raw_insert(RawSaveBatch {
            namespace,
            total_slots,
            numa_node: save_numa_node,
            layers: raw_layers,
        });

        Ok(())
    }
}
