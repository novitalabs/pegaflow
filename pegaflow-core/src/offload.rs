// ============================================================================
// Offload: GPU→CPU save path for KV cache blocks.
//
// Phases 0-3 (validation, hash filter, pinned allocation, GPU copy) run on the
// RPC path. Phase 4 (build LayerBlocks, group by hash, insert into inflight)
// is deferred to the storage insert worker via `RawSaveBatch`.
// ============================================================================

use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::Arc;

use log::{debug, info};

use crate::block::{BlockKey, LayerBlock, LayerSave};

/// Grouped insert entries: each hash maps to its per-slot `LayerBlock`s.
pub(crate) type InsertEntries = Vec<(BlockKey, Vec<(usize, Arc<LayerBlock>)>)>;
use crate::gpu_worker::{SaveBlock, SaveLayerData};
use crate::instance::KVCacheRegistration;
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::pinned_pool::PinnedAllocation;
use crate::{EngineError, PegaEngine};

// ============================================================================
// Types sent to the insert worker (deferred Phase 4)
// ============================================================================

/// How a layer's blocks are laid out in pinned memory after GPU copy.
pub(crate) enum LayerAlloc {
    Split {
        k_allocation: Arc<PinnedAllocation>,
        v_allocation: Arc<PinnedAllocation>,
        k_base: usize,
        v_base: usize,
        segment_size: usize,
    },
    Contiguous {
        allocation: Arc<PinnedAllocation>,
        base_addr: usize,
    },
}

impl LayerAlloc {
    /// Construct a `LayerBlock` for the i-th block in this allocation.
    fn make_layer_block(&self, index: usize, block_size: usize) -> Arc<LayerBlock> {
        match self {
            LayerAlloc::Split {
                k_allocation,
                v_allocation,
                k_base,
                v_base,
                segment_size,
            } => {
                let k_ptr = (k_base + index * segment_size) as *mut u8;
                let v_ptr = (v_base + index * segment_size) as *mut u8;
                Arc::new(LayerBlock::new_split(
                    k_ptr,
                    v_ptr,
                    block_size,
                    Arc::clone(k_allocation),
                    Arc::clone(v_allocation),
                ))
            }
            LayerAlloc::Contiguous {
                allocation,
                base_addr,
            } => {
                let ptr = (base_addr + index * block_size) as *mut u8;
                Arc::new(LayerBlock::new_contiguous(
                    ptr,
                    block_size,
                    Arc::clone(allocation),
                ))
            }
        }
    }
}

/// Per-layer data for deferred LayerBlock construction.
pub(crate) struct RawSaveLayer {
    pub slot_id: usize,
    pub block_size: usize,
    pub alloc: LayerAlloc,
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
/// by hash: `Vec<(BlockKey, Vec<(slot_id, Arc<LayerBlock>)>)>`.
pub(crate) fn build_insert_entries(batch: &RawSaveBatch) -> (InsertEntries, u64, usize) {
    use std::collections::HashMap;

    let mut hash_entries: HashMap<Vec<u8>, Vec<(usize, Arc<LayerBlock>)>> = HashMap::new();
    let mut total_bytes: u64 = 0;
    let mut total_blocks: usize = 0;

    for layer in &batch.layers {
        for (i, hash) in layer.block_hashes.iter().enumerate() {
            let block = layer.alloc.make_layer_block(i, layer.block_size);
            hash_entries
                .entry(hash.clone())
                .or_default()
                .push((layer.slot_id, block));
        }
        let layer_blocks = layer.block_hashes.len();
        total_blocks += layer_blocks;
        total_bytes += (layer.block_size as u64).saturating_mul(layer_blocks as u64);
    }

    let entries: InsertEntries = hash_entries
        .into_iter()
        .map(|(hash, slots)| (BlockKey::new(batch.namespace.clone(), hash), slots))
        .collect();

    (entries, total_bytes, total_blocks)
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
        device_id: i32,
        saves: Vec<LayerSave>,
    ) -> Result<(), EngineError> {
        let batch_start = std::time::Instant::now();
        let total_layers = saves.len();
        let instance = self.get_instance(instance_id)?;
        let namespace = instance.namespace().to_string();
        let total_slots = instance.total_slots();

        // ── Phase 0: Resolve per-layer metadata and build valid_blocks ──
        trace_scope!("save.resolve_metadata", _s);
        struct LayerMeta {
            registration: KVCacheRegistration,
            slot_id: usize,
            block_size: usize,
            valid_blocks: Vec<(usize, Vec<u8>)>,
        }

        let gpu = instance
            .get_gpu(device_id)
            .ok_or_else(|| EngineError::WorkerMissing(instance_id.to_string(), device_id))?;

        let mut layer_metas: Vec<LayerMeta> = Vec::with_capacity(saves.len());

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

            let valid_blocks: Vec<(usize, Vec<u8>)> = block_ids
                .into_iter()
                .zip(block_hashes.into_iter())
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

            if valid_blocks.is_empty() {
                continue;
            }

            let block_size = registration.block_size_bytes;
            layer_metas.push(LayerMeta {
                registration,
                slot_id,
                block_size,
                valid_blocks,
            });
        }
        trace_drop!(_s);

        if layer_metas.is_empty() {
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

        // Collect union of all unique hashes across layers
        let mut hashes_to_save: HashSet<Vec<u8>> = HashSet::new();
        for meta in &layer_metas {
            for (_, hash) in &meta.valid_blocks {
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
        struct LayerSavePrep {
            meta_idx: usize,
            blocks_to_save: Vec<(usize, Vec<u8>)>,
        }

        let mut layers_to_save: Vec<LayerSavePrep> = Vec::with_capacity(layer_metas.len());
        let mut total_blocks_to_save = 0usize;

        for (meta_idx, meta) in layer_metas.iter().enumerate() {
            let blocks_to_save: Vec<(usize, Vec<u8>)> = meta
                .valid_blocks
                .iter()
                .filter(|(_, hash)| hashes_to_save.contains(hash.as_slice()))
                .cloned()
                .collect();
            total_blocks_to_save += blocks_to_save.len();
            if !blocks_to_save.is_empty() {
                layers_to_save.push(LayerSavePrep {
                    meta_idx,
                    blocks_to_save,
                });
            }
        }

        trace_drop!(_s, || {
            [
                ("unique_hashes", hashes_to_save.len().to_string()),
                ("to_save", total_blocks_to_save.to_string()),
            ]
        });

        if layers_to_save.is_empty() {
            debug!(
                "save_batch skipped (all filtered): instance_id={} tp_rank={} device_id={} layers={}",
                instance_id, tp_rank, device_id, total_layers
            );
            return Ok(());
        }

        // ── Phase 2: Allocate pinned memory + build SaveBlocks for all layers ──

        trace_scope!("save.pinned_alloc", _s);
        let numa_node = Some(gpu.preferred_numa());

        let mut layer_allocs: Vec<LayerAlloc> = Vec::with_capacity(layers_to_save.len());
        let mut gpu_save_layers: Vec<SaveLayerData> = Vec::with_capacity(layers_to_save.len());

        for prep in &layers_to_save {
            let meta = &layer_metas[prep.meta_idx];
            let registration = &meta.registration;
            let num_blocks = prep.blocks_to_save.len();

            let is_split = registration.segments == 2
                && registration.kv_stride_bytes > registration.bytes_per_block;

            if is_split {
                let segment_size = registration.bytes_per_block;
                let alloc_size = (segment_size as u64)
                    .checked_mul(num_blocks as u64)
                    .and_then(NonZeroU64::new)
                    .ok_or_else(|| {
                        EngineError::Storage("allocation size overflow for K segments".to_string())
                    })?;

                let mut k_allocation =
                    self.storage
                        .allocate(alloc_size, numa_node)
                        .ok_or_else(|| {
                            EngineError::Storage(
                                "pinned pool exhausted while allocating K segment buffer"
                                    .to_string(),
                            )
                        })?;
                let mut v_allocation =
                    self.storage
                        .allocate(alloc_size, numa_node)
                        .ok_or_else(|| {
                            EngineError::Storage(
                                "pinned pool exhausted while allocating V segment buffer"
                                    .to_string(),
                            )
                        })?;

                let k_base = Arc::get_mut(&mut k_allocation)
                    .expect("k_allocation must be uniquely owned")
                    .as_mut_ptr() as usize;
                let v_base = Arc::get_mut(&mut v_allocation)
                    .expect("v_allocation must be uniquely owned")
                    .as_mut_ptr() as usize;

                let save_blocks: Vec<SaveBlock> = prep
                    .blocks_to_save
                    .iter()
                    .enumerate()
                    .map(|(i, (block_idx, _))| SaveBlock {
                        block_idx: *block_idx,
                        k_dst_ptr: (k_base + i * segment_size) as *mut u8,
                        v_dst_ptr: Some((v_base + i * segment_size) as *mut u8),
                    })
                    .collect();

                gpu_save_layers.push(SaveLayerData {
                    registration: registration.clone(),
                    blocks: save_blocks,
                });
                layer_allocs.push(LayerAlloc::Split {
                    k_allocation,
                    v_allocation,
                    k_base,
                    v_base,
                    segment_size,
                });
            } else {
                let block_size = meta.block_size;
                let alloc_size = (block_size as u64)
                    .checked_mul(num_blocks as u64)
                    .and_then(NonZeroU64::new)
                    .ok_or_else(|| EngineError::Storage("allocation size overflow".to_string()))?;

                let mut allocation =
                    self.storage
                        .allocate(alloc_size, numa_node)
                        .ok_or_else(|| {
                            EngineError::Storage(
                                "pinned pool exhausted while allocating contiguous block buffer"
                                    .to_string(),
                            )
                        })?;

                let base_addr = Arc::get_mut(&mut allocation)
                    .ok_or_else(|| {
                        EngineError::Storage("allocation shared unexpectedly".to_string())
                    })?
                    .as_mut_ptr() as usize;

                let save_blocks: Vec<SaveBlock> = prep
                    .blocks_to_save
                    .iter()
                    .enumerate()
                    .map(|(i, (block_idx, _))| SaveBlock {
                        block_idx: *block_idx,
                        k_dst_ptr: (base_addr + i * block_size) as *mut u8,
                        v_dst_ptr: None,
                    })
                    .collect();

                gpu_save_layers.push(SaveLayerData {
                    registration: registration.clone(),
                    blocks: save_blocks,
                });
                layer_allocs.push(LayerAlloc::Contiguous {
                    allocation,
                    base_addr,
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

        // ── Phase 4 (deferred): Build LayerBlocks + insert — sent to worker ──

        // Record metrics on the RPC path (cheap, measures RPC-visible latency)
        let metrics = core_metrics();
        let mut total_bytes = 0u64;
        for prep in &layers_to_save {
            let meta = &layer_metas[prep.meta_idx];
            let num_blocks = prep.blocks_to_save.len();
            let bytes = (meta.block_size as u64)
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
            "save_batch completed: instance_id={} tp_rank={} device_id={} layers={} layers_saved={} blocks_saved={} bytes={} total_ms={:.2}",
            instance_id,
            tp_rank,
            device_id,
            total_layers,
            layers_to_save.len(),
            total_blocks_to_save,
            total_bytes,
            batch_start.elapsed().as_secs_f64() * 1000.0
        );

        // Build RawSaveBatch and send to insert worker (fire-and-forget)
        let raw_layers: Vec<RawSaveLayer> = layers_to_save
            .into_iter()
            .zip(layer_allocs)
            .map(|(prep, alloc)| {
                let meta = &layer_metas[prep.meta_idx];
                let block_hashes: Vec<Vec<u8>> = prep
                    .blocks_to_save
                    .into_iter()
                    .map(|(_, hash)| hash)
                    .collect();
                RawSaveLayer {
                    slot_id: meta.slot_id,
                    block_size: meta.block_size,
                    alloc,
                    block_hashes,
                }
            })
            .collect();

        self.storage.send_raw_insert(RawSaveBatch {
            namespace,
            total_slots,
            numa_node: gpu.preferred_numa(),
            layers: raw_layers,
        });

        Ok(())
    }
}
