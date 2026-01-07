use futures::stream::{FuturesUnordered, StreamExt};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Instant;
use tracing::{debug, warn};

use crate::block::{BlockKey, LayerBlock, SealedBlock};
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use crate::seal_offload::SlotMeta;
use crate::uring::UringIoEngine;

/// Default prefetch IO depth (max concurrent read operations)
const DEFAULT_PREFETCH_IO_DEPTH: usize = 128;

/// Default write queue depth for SSD writer thread
const DEFAULT_WRITE_QUEUE_DEPTH: usize = 1024;

/// Result of a single prefetch operation: (key, begin_offset, block, duration_ms, block_size)
type PrefetchResult = (BlockKey, u64, Option<Arc<SealedBlock>>, f64, u64);

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the single-file SSD cache (logical ring).
#[derive(Debug, Clone)]
pub struct SsdCacheConfig {
    /// File path for the cache data file.
    pub cache_path: PathBuf,
    /// Total logical capacity of the cache (bytes).
    pub capacity_bytes: u64,
    /// Max pending write requests (blocks). New sealed blocks are dropped if the queue is full.
    pub write_queue_depth: usize,
    /// Prefetch IO depth: max concurrent read operations (blocks).
    pub prefetch_io_depth: usize,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_path: PathBuf::from("/tmp/pegaflow-ssd-cache/cache.bin"),
            capacity_bytes: 512 * 1024 * 1024 * 1024, // 512GB
            write_queue_depth: DEFAULT_WRITE_QUEUE_DEPTH,
            prefetch_io_depth: DEFAULT_PREFETCH_IO_DEPTH,
        }
    }
}

// ============================================================================
// Types for SSD operations
// ============================================================================

/// Metadata for a block stored in SSD cache
#[derive(Clone)]
pub struct SsdIndexEntry {
    /// Logical offset in the ring buffer (monotonically increasing)
    pub begin: u64,
    /// Logical end offset
    pub end: u64,
    /// Block size in bytes
    pub len: u64,
    /// Per-slot metadata for rebuilding SealedBlock
    pub slots: Vec<SlotMeta>,
}

/// Request to write a sealed block to SSD
pub struct SsdWriteRequest {
    pub key: BlockKey,
    pub block: Weak<SealedBlock>,
}

/// Pre-allocated slice from a contiguous allocation (for batched prefetch)
pub struct PreallocatedSlice {
    /// Parent allocation (shared via Arc)
    pub allocation: Arc<PinnedAllocation>,
    /// Offset within the parent allocation
    pub offset: usize,
}

/// Request to prefetch a block from SSD
pub struct PrefetchRequest {
    pub key: BlockKey,
    pub entry: SsdIndexEntry,
    /// Pre-allocated contiguous slice for this block
    pub preallocated: PreallocatedSlice,
}

// ============================================================================
// Storage handle (provided by StorageEngine)
// ============================================================================

/// Handle used by SSD workers to interact with storage.
pub struct SsdStorageHandle {
    prune_tail: Arc<dyn Fn(u64) + Send + Sync>,
    publish_write: Arc<dyn Fn(BlockKey, SsdIndexEntry, u64) + Send + Sync>,
    complete_prefetch: Arc<dyn Fn(BlockKey, Option<Arc<SealedBlock>>) + Send + Sync>,
    /// Check if a logical offset is still valid (not yet overwritten)
    is_offset_valid: Arc<dyn Fn(u64) -> bool + Send + Sync>,
}

impl SsdStorageHandle {
    pub fn new(
        prune_tail: impl Fn(u64) + Send + Sync + 'static,
        publish_write: impl Fn(BlockKey, SsdIndexEntry, u64) + Send + Sync + 'static,
        complete_prefetch: impl Fn(BlockKey, Option<Arc<SealedBlock>>) + Send + Sync + 'static,
        is_offset_valid: impl Fn(u64) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            prune_tail: Arc::new(prune_tail),
            publish_write: Arc::new(publish_write),
            complete_prefetch: Arc::new(complete_prefetch),
            is_offset_valid: Arc::new(is_offset_valid),
        }
    }

    #[inline]
    pub fn prune_tail(&self, new_tail: u64) {
        (self.prune_tail)(new_tail);
    }

    #[inline]
    pub fn publish_write(&self, key: BlockKey, entry: SsdIndexEntry, new_head: u64) {
        (self.publish_write)(key, entry, new_head);
    }

    #[inline]
    pub fn complete_prefetch(&self, key: BlockKey, block: Option<Arc<SealedBlock>>) {
        (self.complete_prefetch)(key, block);
    }

    #[inline]
    pub fn is_offset_valid(&self, begin: u64) -> bool {
        (self.is_offset_valid)(begin)
    }
}

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives sealed blocks and writes them to SSD.
pub async fn ssd_writer_loop(
    handle: Arc<SsdStorageHandle>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SsdWriteRequest>,
    io: Arc<UringIoEngine>,
    capacity: u64,
) {
    // Track logical head position (monotonically increasing)
    let mut head: u64 = 0;
    let mut pending: HashSet<BlockKey> = HashSet::new();
    let metrics = core_metrics();

    while let Some(req) = rx.recv().await {
        let key = req.key;
        if !pending.insert(key.clone()) {
            continue;
        }

        let Some(block) = req.block.upgrade() else {
            // Block was evicted before we could write it
            pending.remove(&key);
            continue;
        };

        let block_size = block.memory_footprint();

        if block_size > capacity || block_size == 0 {
            warn!(
                "SSD cache skipping block size is 0 or larger than capacity: {} > {}",
                block_size, capacity
            );
            pending.remove(&key);
            continue;
        }

        metrics.ssd_write_queue_pending.add(1, &[]);

        // Reserve space in ring buffer, avoiding wrap-around within a single block
        let phys = head % capacity;
        let space_until_end = capacity - phys;
        if block_size > space_until_end {
            head = head.saturating_add(space_until_end);
        }

        let begin = head;
        let end = head.saturating_add(block_size);
        let file_offset = begin % capacity;

        // Advance tail to evict old entries if needed
        let desired_tail = end.saturating_sub(capacity);
        handle.prune_tail(desired_tail);

        // Collect slot metadata for rebuilding
        let slots: Vec<SlotMeta> = block
            .slots()
            .iter()
            .map(|s| SlotMeta {
                is_split: s.v_ptr().is_some(),
                size: s.size() as u64,
            })
            .collect();

        // Write block data to SSD (with timing)
        let write_start = Instant::now();
        if let Err(e) = write_block_to_ssd(&io, file_offset, &block).await {
            warn!("SSD cache write failed: {}", e);
            pending.remove(&key);
            metrics.ssd_write_queue_pending.add(-1, &[]);
            continue;
        }
        let write_duration = write_start.elapsed();
        metrics
            .ssd_write_duration_ms
            .record(write_duration.as_secs_f64() * 1000.0, &[]);

        let entry = SsdIndexEntry {
            begin,
            end,
            len: block_size,
            slots,
        };

        handle.publish_write(key.clone(), entry, end);
        pending.remove(&key);
        metrics.ssd_write_queue_pending.add(-1, &[]);
        metrics.ssd_write_blocks.add(1, &[]);
        metrics.ssd_write_bytes.add(block_size, &[]);
        head = end;
    }

    debug!("SSD writer task exiting");
}

/// Write a sealed block to SSD file using writev.
///
/// Uses vectorized I/O to write all slots in a single syscall, reducing overhead
/// compared to writing each slot separately.
async fn write_block_to_ssd(
    io: &UringIoEngine,
    offset: u64,
    block: &SealedBlock,
) -> std::io::Result<()> {
    // Build iovecs and submit in a scope so raw pointers don't cross await
    let rx = {
        let mut iovecs = Vec::new();

        for slot in block.slots() {
            let size = slot.size();
            if let Some(v_ptr) = slot.v_ptr() {
                // Split layout: K then V
                let half = size / 2;
                iovecs.push((slot.k_ptr(), half));
                iovecs.push((v_ptr, half));
            } else {
                // Contiguous layout
                iovecs.push((slot.k_ptr(), size));
            }
        }

        // Submit vectorized write - iovecs is consumed here
        io.writev_at_async(iovecs, offset)?
    };

    // Now we can safely await (no raw pointers in scope)
    rx.await
        .map_err(|_| std::io::Error::other("writev recv failed"))??;

    Ok(())
}

// ============================================================================
// SSD Prefetch Loop
// ============================================================================

/// SSD prefetch worker: receives prefetch requests and loads blocks from SSD.
/// Uses FuturesUnordered to maintain concurrent IO operations.
pub async fn ssd_prefetch_loop(
    handle: Arc<SsdStorageHandle>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<PrefetchRequest>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    io_depth: usize,
) {
    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    let metrics = core_metrics();

    loop {
        tokio::select! {
            biased;  // Prioritize filling in_flight before processing completions

            // Accept new request if under IO depth limit
            Some(req) = rx.recv(), if in_flight.len() < io_depth => {
                let io = Arc::clone(&io);
                metrics.ssd_prefetch_inflight.add(1, &[]);
                in_flight.push(execute_prefetch(req, io, capacity));
            }
            // Process completed prefetch
            Some((key, begin, result, duration_ms, _block_size)) = in_flight.next() => {
                metrics.ssd_prefetch_inflight.add(-1, &[]);
                metrics.ssd_prefetch_duration_ms.record(duration_ms, &[]);

                // Validate data wasn't overwritten during read
                let result = if result.is_some() && !handle.is_offset_valid(begin) {
                    warn!("SSD prefetch: data overwritten during read, discarding");
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                } else if result.is_some() {
                    metrics.ssd_prefetch_success.add(1, &[]);
                    result
                } else {
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                };
                handle.complete_prefetch(key, result);
            }
            // Both channels exhausted
            else => break,
        }
    }

    debug!("SSD prefetch worker exiting");
}

/// Execute a single prefetch operation (async, does not block).
async fn execute_prefetch(
    req: PrefetchRequest,
    io: Arc<UringIoEngine>,
    capacity: u64,
) -> PrefetchResult {
    let start = Instant::now();
    let key = req.key;
    let begin = req.entry.begin;
    let allocation = req.preallocated.allocation;
    let alloc_offset = req.preallocated.offset;
    let len = req.entry.len as usize;
    let block_size = req.entry.len;

    // Calculate physical offset in SSD file
    let phys_offset = begin % capacity;
    if phys_offset + req.entry.len > capacity {
        warn!("SSD prefetch: block wraps around ring buffer");
        return (
            key,
            begin,
            None,
            start.elapsed().as_secs_f64() * 1000.0,
            block_size,
        );
    }

    // Build iovecs and submit IO in a scope so raw pointers don't cross await
    let read_result = {
        let base_ptr = allocation.as_ptr() as *mut u8;
        let mut iovecs = Vec::new();
        let mut current_offset = alloc_offset;

        for slot_meta in &req.entry.slots {
            let slot_size = slot_meta.size as usize;
            if slot_meta.is_split {
                // Split layout: K then V (same as write)
                let half = slot_size / 2;
                let k_ptr = unsafe { base_ptr.add(current_offset) };
                let v_ptr = unsafe { base_ptr.add(current_offset + half) };
                iovecs.push((k_ptr, half));
                iovecs.push((v_ptr, half));
            } else {
                // Contiguous layout
                let ptr = unsafe { base_ptr.add(current_offset) };
                iovecs.push((ptr, slot_size));
            }
            current_offset += slot_size;
        }

        io.readv_at_async(iovecs, phys_offset)
    };

    let duration_ms = || start.elapsed().as_secs_f64() * 1000.0;

    match read_result {
        Ok(rx) => match rx.await {
            Ok(Ok(bytes_read)) if bytes_read == len => {
                // Success - rebuild the block
                match rebuild_sealed_block_at_offset(
                    Arc::clone(&allocation),
                    alloc_offset,
                    &req.entry.slots,
                ) {
                    Ok(sealed) => (
                        key,
                        begin,
                        Some(Arc::new(sealed)),
                        duration_ms(),
                        block_size,
                    ),
                    Err(e) => {
                        warn!("SSD prefetch: failed to rebuild block: {}", e);
                        (key, begin, None, duration_ms(), block_size)
                    }
                }
            }
            Ok(Ok(bytes_read)) => {
                warn!("SSD prefetch: short read {} of {} bytes", bytes_read, len);
                (key, begin, None, duration_ms(), block_size)
            }
            Ok(Err(e)) => {
                warn!("SSD prefetch: read error: {}", e);
                (key, begin, None, duration_ms(), block_size)
            }
            Err(_) => {
                warn!("SSD prefetch: read channel closed");
                (key, begin, None, duration_ms(), block_size)
            }
        },
        Err(e) => {
            warn!("SSD prefetch: failed to submit read: {}", e);
            (key, begin, None, duration_ms(), block_size)
        }
    }
}

// ============================================================================
// Block Rebuilding
// ============================================================================

/// Rebuild a SealedBlock from a contiguous pinned allocation and slot metadata.
/// Used when loading blocks from SSD cache.
pub fn rebuild_sealed_block(
    allocation: Arc<PinnedAllocation>,
    slot_metas: &[SlotMeta],
) -> Result<SealedBlock, String> {
    rebuild_sealed_block_at_offset(allocation, 0, slot_metas)
}

/// Rebuild a SealedBlock from a shared allocation at a given offset.
/// Used for batched prefetch where multiple blocks share one contiguous allocation.
pub fn rebuild_sealed_block_at_offset(
    allocation: Arc<PinnedAllocation>,
    base_offset: usize,
    slot_metas: &[SlotMeta],
) -> Result<SealedBlock, String> {
    let mut layer_blocks = Vec::with_capacity(slot_metas.len());
    let base_ptr = allocation.as_ptr() as *mut u8;
    let mut current_offset = base_offset;

    for slot_meta in slot_metas {
        let slot_size = slot_meta.size as usize;

        let layer_block = if slot_meta.is_split {
            let half = slot_size / 2;
            let k_ptr = unsafe { base_ptr.add(current_offset) };
            let v_ptr = unsafe { base_ptr.add(current_offset + half) };

            Arc::new(LayerBlock::new_split(
                k_ptr,
                v_ptr,
                slot_size,
                Arc::clone(&allocation),
                Arc::clone(&allocation),
            ))
        } else {
            let ptr = unsafe { base_ptr.add(current_offset) };
            Arc::new(LayerBlock::new_contiguous(
                ptr,
                slot_size,
                Arc::clone(&allocation),
            ))
        };

        layer_blocks.push(layer_block);
        current_offset += slot_size;
    }

    Ok(SealedBlock::from_slots(layer_blocks))
}
