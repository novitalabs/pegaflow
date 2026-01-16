use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, warn};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Instant;

use crate::block::{BlockKey, LayerBlock, SealedBlock};
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use crate::seal_offload::SlotMeta;
use crate::uring::UringIoEngine;

/// Default write queue depth for SSD writer thread (blocks dropped if full)
pub const DEFAULT_SSD_WRITE_QUEUE_DEPTH: usize = 8;

/// Default prefetch queue depth (limits read tail latency)
pub const DEFAULT_SSD_PREFETCH_QUEUE_DEPTH: usize = 2;

/// Result of a single prefetch operation: (key, begin_offset, block, duration_secs, block_size)
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
    /// Max pending write batches. New sealed blocks are dropped if the queue is full.
    pub write_queue_depth: usize,
    /// Max pending prefetch batches (limits read tail latency).
    pub prefetch_queue_depth: usize,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_path: PathBuf::from("/tmp/pegaflow-ssd-cache/cache.bin"),
            capacity_bytes: 512 * 1024 * 1024 * 1024, // 512GB
            write_queue_depth: DEFAULT_SSD_WRITE_QUEUE_DEPTH,
            prefetch_queue_depth: DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
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

/// Batch of sealed blocks to write to SSD
pub struct SsdWriteBatch {
    pub blocks: Vec<(BlockKey, Weak<SealedBlock>)>,
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

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub struct PrefetchBatch {
    pub requests: Vec<PrefetchRequest>,
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
// Ring Buffer Allocator
// ============================================================================

/// Logical ring buffer space allocator.
/// Tracks head position and handles wrap-around for contiguous allocations.
struct RingAllocator {
    head: u64,
    capacity: u64,
}

impl RingAllocator {
    fn new(capacity: u64) -> Self {
        Self { head: 0, capacity }
    }

    /// Allocate contiguous space for a batch. Skips wrap-around gap if needed.
    /// Returns (logical_begin, file_offset).
    fn allocate(&mut self, size: u64) -> (u64, u64) {
        let phys = self.head % self.capacity;
        let space_until_end = self.capacity - phys;
        if size > space_until_end {
            // Skip to next wrap point
            self.head += space_until_end;
        }
        let begin = self.head;
        self.head += size;
        (begin, begin % self.capacity)
    }

    fn head(&self) -> u64 {
        self.head
    }
}

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// Prepared write entry for a single block within a batch
struct PreparedWrite {
    key: BlockKey,
    block: Arc<SealedBlock>,
    /// Offset within the batch's contiguous region
    offset_in_batch: u64,
    slots: Vec<SlotMeta>,
}

/// Batch of prepared writes with shared allocation info
struct PreparedBatch {
    writes: Vec<PreparedWrite>,
    /// Logical begin offset in ring buffer
    begin: u64,
    /// Physical file offset
    file_offset: u64,
    /// Total batch size
    total_size: u64,
}

impl PreparedBatch {
    fn end(&self) -> u64 {
        self.begin + self.total_size
    }

    fn make_entry(&self, w: &PreparedWrite) -> SsdIndexEntry {
        SsdIndexEntry {
            begin: self.begin + w.offset_in_batch,
            end: self.begin + w.offset_in_batch + w.block.memory_footprint(),
            len: w.block.memory_footprint(),
            slots: w.slots.clone(),
        }
    }
}

/// SSD writer task: receives batches of sealed blocks and writes them in parallel.
pub async fn ssd_writer_loop(
    handle: Arc<SsdStorageHandle>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
    io: Arc<UringIoEngine>,
    capacity: u64,
) {
    let mut ring = RingAllocator::new(capacity);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    let metrics = core_metrics();

    while let Some(batch) = rx.recv().await {
        // Dequeue: decrement pending count immediately
        metrics
            .ssd_write_queue_pending
            .add(-(batch.blocks.len() as i64), &[]);

        // Phase 1: Prepare - upgrade weak refs with prefix semantics (early break on failure)
        let prepared = match prepare_batch(&batch, &mut seen, &mut ring, capacity) {
            Some(p) => p,
            None => continue,
        };

        if prepared.writes.is_empty() {
            continue;
        }

        // Phase 2: Prune tail for the entire batch
        handle.prune_tail(prepared.end().saturating_sub(capacity));

        // Phase 3: Parallel IO
        let write_start = Instant::now();
        let mut total_bytes_written: u64 = 0;

        for chunk in prepared.writes.chunks(prepared.writes.len()) {
            let futures: FuturesUnordered<_> = chunk
                .iter()
                .map(|w| {
                    let io = Arc::clone(&io);
                    let block = Arc::clone(&w.block);
                    let offset = prepared.file_offset + w.offset_in_batch;
                    let slots = w.slots.clone();
                    async move { write_block_to_ssd(&io, offset, &block, &slots).await }
                })
                .collect();

            let results: Vec<_> = futures.collect().await;

            // Phase 4: Publish results
            for (w, result) in chunk.iter().zip(results) {
                seen.remove(&w.key);

                match result {
                    Ok(()) => {
                        handle.publish_write(w.key.clone(), prepared.make_entry(w), ring.head());
                        let block_bytes = w.block.memory_footprint();
                        metrics.ssd_write_bytes.add(block_bytes, &[]);
                        total_bytes_written += block_bytes;
                    }
                    Err(e) => {
                        warn!("SSD cache write failed for {:?}: {}", w.key, e);
                    }
                }
            }
        }

        let duration = write_start.elapsed();
        let duration_secs = duration.as_secs_f64();
        metrics
            .ssd_write_duration_seconds
            .record(duration_secs, &[]);

        // Record throughput in bytes/s
        if duration_secs > 0.0 && total_bytes_written > 0 {
            let throughput_bytes_per_second = (total_bytes_written as f64) / duration_secs;
            metrics
                .ssd_write_throughput_bytes_per_second
                .record(throughput_bytes_per_second, &[]);
        }
    }

    debug!("SSD writer task exiting");
}

/// Prepare a batch for writing: upgrade weak refs, compute sizes, allocate ring space.
/// Uses prefix semantics: if any block fails to upgrade, the entire batch is skipped.
fn prepare_batch(
    batch: &SsdWriteBatch,
    seen: &mut HashSet<BlockKey>,
    ring: &mut RingAllocator,
    capacity: u64,
) -> Option<PreparedBatch> {
    // First pass: upgrade all weak refs and compute total size
    let mut blocks: Vec<(BlockKey, Arc<SealedBlock>, Vec<SlotMeta>)> = Vec::new();
    let mut total_size: u64 = 0;

    for (key, weak_block) in &batch.blocks {
        // Skip duplicates
        if seen.contains(key) {
            continue;
        }

        // Prefix semantics: any upgrade failure means skip entire batch
        let block = weak_block.upgrade()?;

        let block_size = block.memory_footprint();
        if block_size == 0 || block_size > capacity {
            warn!(
                "SSD cache skipping batch: block size {} (capacity {})",
                block_size, capacity
            );
            return None;
        }

        let slots: Vec<SlotMeta> = block
            .slots()
            .iter()
            .map(|s| SlotMeta {
                is_split: s.v_ptr().is_some(),
                size: s.size() as u64,
            })
            .collect();

        total_size += block_size;
        blocks.push((key.clone(), block, slots));
    }

    if blocks.is_empty() {
        return Some(PreparedBatch {
            writes: Vec::new(),
            begin: 0,
            file_offset: 0,
            total_size: 0,
        });
    }

    // Allocate contiguous space for entire batch
    let (begin, file_offset) = ring.allocate(total_size);

    // Second pass: compute per-block offsets within the batch
    let mut offset_in_batch: u64 = 0;
    let writes: Vec<PreparedWrite> = blocks
        .into_iter()
        .map(|(key, block, slots)| {
            let w = PreparedWrite {
                key: key.clone(),
                block: Arc::clone(&block),
                offset_in_batch,
                slots,
            };
            seen.insert(key);
            offset_in_batch += block.memory_footprint();
            w
        })
        .collect();

    Some(PreparedBatch {
        writes,
        begin,
        file_offset,
        total_size,
    })
}

/// Write a sealed block to SSD file using writev.
///
/// Uses vectorized I/O to write all slots in a single syscall, reducing overhead
/// compared to writing each slot separately.
async fn write_block_to_ssd(
    io: &UringIoEngine,
    offset: u64,
    block: &SealedBlock,
    slots_meta: &[SlotMeta],
) -> std::io::Result<()> {
    // Build iovecs from slot metadata
    let rx = {
        let iovecs: Vec<_> = slots_meta
            .iter()
            .zip(block.slots())
            .flat_map(|(meta, slot)| meta.write_iovecs(slot))
            .collect();

        io.writev_at_async(iovecs, offset)?
    };

    rx.await
        .map_err(|_| std::io::Error::other("writev recv failed"))??;

    Ok(())
}

// ============================================================================
// SSD Prefetch Loop
// ============================================================================

/// SSD prefetch worker: receives batches of prefetch requests and loads blocks from SSD.
/// Processes all requests in a batch concurrently, then waits for completion before next batch.
pub async fn ssd_prefetch_loop(
    handle: Arc<SsdStorageHandle>,
    mut rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    io: Arc<UringIoEngine>,
    capacity: u64,
) {
    let metrics = core_metrics();

    while let Some(batch) = rx.recv().await {
        if batch.requests.is_empty() {
            continue;
        }

        let batch_start = Instant::now();
        let batch_bytes: u64 = batch.requests.iter().map(|r| r.entry.len).sum();

        // Process all requests in the batch concurrently
        let futures: FuturesUnordered<_> = batch
            .requests
            .into_iter()
            .map(|req| {
                let io = Arc::clone(&io);
                metrics.ssd_prefetch_inflight.add(1, &[]);
                execute_prefetch(req, io, capacity)
            })
            .collect();

        let results: Vec<PrefetchResult> = futures.collect().await;

        // Complete all prefetches
        for (key, begin, result, duration_secs, _block_size) in results {
            metrics.ssd_prefetch_inflight.add(-1, &[]);
            metrics
                .ssd_prefetch_duration_seconds
                .record(duration_secs, &[]);

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

        // Record batch throughput in bytes/s
        let duration_secs = batch_start.elapsed().as_secs_f64();
        if duration_secs > 0.0 && batch_bytes > 0 {
            let throughput_bytes_per_second = (batch_bytes as f64) / duration_secs;
            metrics
                .ssd_prefetch_throughput_bytes_per_second
                .record(throughput_bytes_per_second, &[]);
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
    let duration_secs = || start.elapsed().as_secs_f64();
    let fail = |key, begin, size| (key, begin, None, duration_secs(), size);

    let key = req.key;
    let begin = req.entry.begin;
    let block_size = req.entry.len;

    // Calculate physical offset in SSD file
    let phys_offset = begin % capacity;
    if phys_offset + block_size > capacity {
        warn!("SSD prefetch: block wraps around ring buffer");
        return fail(key, begin, block_size);
    }

    // Build iovecs from slot metadata
    let read_result = {
        let base_ptr = req.preallocated.allocation.as_ptr() as *mut u8;
        let mut current_offset = req.preallocated.offset;
        let iovecs: Vec<_> = req
            .entry
            .slots
            .iter()
            .flat_map(|meta| {
                // SAFETY: preallocated slice is sized to fit all slots
                let iov = unsafe { meta.read_iovecs(base_ptr, current_offset) };
                current_offset += meta.size as usize;
                iov
            })
            .collect();

        io.readv_at_async(iovecs, phys_offset)
    };

    // Await IO result and rebuild block
    let expected_len = req.entry.len as usize;
    match read_result {
        Ok(rx) => match rx.await {
            Ok(Ok(bytes_read)) if bytes_read == expected_len => {
                match rebuild_sealed_block_at_offset(
                    req.preallocated.allocation,
                    req.preallocated.offset,
                    &req.entry.slots,
                ) {
                    Ok(sealed) => (
                        key,
                        begin,
                        Some(Arc::new(sealed)),
                        duration_secs(),
                        block_size,
                    ),
                    Err(e) => {
                        warn!("SSD prefetch: failed to rebuild block: {}", e);
                        fail(key, begin, block_size)
                    }
                }
            }
            Ok(Ok(n)) => {
                warn!("SSD prefetch: short read {} of {} bytes", n, expected_len);
                fail(key, begin, block_size)
            }
            Ok(Err(e)) => {
                warn!("SSD prefetch: read error: {}", e);
                fail(key, begin, block_size)
            }
            Err(_) => {
                warn!("SSD prefetch: read channel closed");
                fail(key, begin, block_size)
            }
        },
        Err(e) => {
            warn!("SSD prefetch: failed to submit read: {}", e);
            fail(key, begin, block_size)
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
