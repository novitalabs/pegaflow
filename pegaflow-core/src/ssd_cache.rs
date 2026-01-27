use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
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

/// Default max concurrent writes (not critical path, keep low)
pub const DEFAULT_SSD_WRITE_INFLIGHT: usize = 2;

/// Default max concurrent prefetches
pub const DEFAULT_SSD_PREFETCH_INFLIGHT: usize = 16;

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
    /// Max concurrent block writes (not critical path, keep low).
    pub write_inflight: usize,
    /// Max concurrent block prefetches.
    pub prefetch_inflight: usize,
}

impl Default for SsdCacheConfig {
    fn default() -> Self {
        Self {
            cache_path: PathBuf::from("/tmp/pegaflow-ssd-cache/cache.bin"),
            capacity_bytes: 512 * 1024 * 1024 * 1024, // 512GB
            write_queue_depth: DEFAULT_SSD_WRITE_QUEUE_DEPTH,
            prefetch_queue_depth: DEFAULT_SSD_PREFETCH_QUEUE_DEPTH,
            write_inflight: DEFAULT_SSD_WRITE_INFLIGHT,
            prefetch_inflight: DEFAULT_SSD_PREFETCH_INFLIGHT,
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

/// Request to prefetch a block from SSD (metadata only, allocation done in worker)
pub struct PrefetchRequest {
    pub key: BlockKey,
    pub entry: SsdIndexEntry,
}

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub struct PrefetchBatch {
    pub requests: Vec<PrefetchRequest>,
}

/// Internal: single block prefetch task with allocated memory
struct PrefetchTask {
    key: BlockKey,
    entry: SsdIndexEntry,
    allocation: Arc<PinnedAllocation>,
    alloc_offset: usize,
}

/// Internal: single block write task
struct WriteTask {
    key: BlockKey,
    block: Arc<SealedBlock>,
    file_offset: u64,
    slots: Vec<SlotMeta>,
    entry: SsdIndexEntry,
    new_head: u64,
}

/// Result of a single write operation: (key, result, entry, new_head, duration_secs, block_size)
type WriteResult = (BlockKey, std::io::Result<()>, SsdIndexEntry, u64, f64, u64);

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
    /// Allocate pinned memory for prefetch
    allocate: Arc<dyn Fn(u64) -> Option<Arc<PinnedAllocation>> + Send + Sync>,
}

impl SsdStorageHandle {
    pub fn new(
        prune_tail: impl Fn(u64) + Send + Sync + 'static,
        publish_write: impl Fn(BlockKey, SsdIndexEntry, u64) + Send + Sync + 'static,
        complete_prefetch: impl Fn(BlockKey, Option<Arc<SealedBlock>>) + Send + Sync + 'static,
        is_offset_valid: impl Fn(u64) -> bool + Send + Sync + 'static,
        allocate: impl Fn(u64) -> Option<Arc<PinnedAllocation>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            prune_tail: Arc::new(prune_tail),
            publish_write: Arc::new(publish_write),
            complete_prefetch: Arc::new(complete_prefetch),
            is_offset_valid: Arc::new(is_offset_valid),
            allocate: Arc::new(allocate),
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

    #[inline]
    pub fn allocate(&self, size: u64) -> Option<Arc<PinnedAllocation>> {
        (self.allocate)(size)
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
}

/// SSD writer task: receives batches of sealed blocks and writes them in parallel.
///
/// Uses FuturesOrdered to maintain publish order while allowing cross-batch pipelining.
/// This ensures new_head values are published in monotonically increasing order,
/// avoiding race conditions where a later batch's writes complete before an earlier batch.
pub async fn ssd_writer_loop(
    handle: Arc<SsdStorageHandle>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    write_inflight: usize,
) {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    type WriteFuture = Pin<Box<dyn Future<Output = WriteResult> + Send>>;

    let mut ring = RingAllocator::new(capacity);
    let mut seen: HashSet<BlockKey> = HashSet::new();
    let metrics = core_metrics();
    let max_inflight = write_inflight.max(1);

    let mut pending: VecDeque<WriteTask> = VecDeque::new();
    let mut inflight: FuturesOrdered<WriteFuture> = FuturesOrdered::new();

    loop {
        tokio::select! {
            biased;

            // Priority 1: Complete writes in push order (FuturesOrdered guarantees ordering)
            Some((key, result, entry, new_head, duration_secs, block_size)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_write_inflight.add(-1, &[]);
                seen.remove(&key);

                match result {
                    Ok(()) => {
                        handle.publish_write(key, entry, new_head);
                        metrics.ssd_write_bytes.add(block_size, &[]);
                        let throughput = block_size as f64 / duration_secs;
                        metrics.ssd_write_throughput_bytes_per_second.record(throughput, &[]);
                    }
                    Err(e) => {
                        warn!("SSD cache write failed for {:?}: {}", key, e);
                    }
                }
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, io.clone())));
            }

            // Priority 3: Receive new batch (can pipeline with in-flight IO)
            batch = rx.recv(), if pending.is_empty() => {
                match batch {
                    Some(b) => {
                        // Dequeue: decrement pending count immediately
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);

                        // Prepare batch
                        let prepared = match prepare_batch(&b, &mut seen, &mut ring, capacity) {
                            Some(p) if !p.writes.is_empty() => p,
                            _ => continue,
                        };

                        // Prune tail for the entire batch
                        handle.prune_tail(prepared.end().saturating_sub(capacity));

                        // Convert to WriteTask and add to pending
                        let new_head = ring.head();
                        for w in prepared.writes {
                            let entry = SsdIndexEntry {
                                begin: prepared.begin + w.offset_in_batch,
                                end: prepared.begin + w.offset_in_batch + w.block.memory_footprint(),
                                len: w.block.memory_footprint(),
                                slots: w.slots.clone(),
                            };
                            pending.push_back(WriteTask {
                                key: w.key,
                                block: w.block,
                                file_offset: prepared.file_offset + w.offset_in_batch,
                                slots: w.slots,
                                entry,
                                new_head,
                            });
                        }
                    }
                    None => break,
                }
            }
        }
    }

    // Drain remaining inflight writes
    while let Some((key, result, entry, new_head, duration_secs, block_size)) =
        inflight.next().await
    {
        metrics.ssd_write_inflight.add(-1, &[]);
        seen.remove(&key);

        match result {
            Ok(()) => {
                handle.publish_write(key, entry, new_head);
                metrics.ssd_write_bytes.add(block_size, &[]);
                let throughput = block_size as f64 / duration_secs;
                metrics
                    .ssd_write_throughput_bytes_per_second
                    .record(throughput, &[]);
            }
            Err(e) => {
                warn!("SSD cache write failed for {:?}: {}", key, e);
            }
        }
    }

    debug!("SSD writer task exiting");
}

/// Execute a single block write to SSD.
async fn execute_write(task: WriteTask, io: Arc<UringIoEngine>) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let entry = task.entry;
    let new_head = task.new_head;
    let block_size = task.block.memory_footprint();

    let result = write_block_to_ssd(&io, task.file_offset, &task.block, &task.slots).await;

    let duration_secs = start.elapsed().as_secs_f64();
    (key, result, entry, new_head, duration_secs, block_size)
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
// SSD Prefetch Pipeline (Dispatcher + Worker)
// ============================================================================

/// SSD prefetch entry point. Spawns dispatcher + worker pipeline internally.
pub async fn ssd_prefetch_loop(
    handle: Arc<SsdStorageHandle>,
    rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    prefetch_inflight: usize,
) {
    let prefetch_inflight = prefetch_inflight.max(1);

    // Bounded channel: capacity = max inflight tasks
    let (task_tx, task_rx) = tokio::sync::mpsc::channel(prefetch_inflight);

    // Spawn dispatcher and worker
    let dispatcher = tokio::spawn(ssd_prefetch_dispatcher(handle.clone(), rx, task_tx));
    let worker = tokio::spawn(ssd_prefetch_worker(
        handle,
        task_rx,
        io,
        capacity,
        prefetch_inflight,
    ));

    // Wait for both to complete
    let _ = dispatcher.await;
    let _ = worker.await;

    debug!("SSD prefetch pipeline exiting");
}

/// Dispatcher: receives batches, allocates memory, splits into block-level tasks.
async fn ssd_prefetch_dispatcher(
    handle: Arc<SsdStorageHandle>,
    mut batch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    task_tx: tokio::sync::mpsc::Sender<PrefetchTask>,
) {
    while let Some(batch) = batch_rx.recv().await {
        if batch.requests.is_empty() {
            continue;
        }

        // Batch-level allocation (preserves memory contiguity for GPU transfers)
        let total_size: u64 = batch.requests.iter().map(|r| r.entry.len).sum();
        let allocation = match handle.allocate(total_size) {
            Some(alloc) => alloc,
            None => {
                warn!(
                    "SSD prefetch dispatcher: alloc failed for {} bytes ({} blocks)",
                    total_size,
                    batch.requests.len()
                );
                for req in batch.requests {
                    handle.complete_prefetch(req.key, None);
                }
                continue;
            }
        };

        // Split into block-level tasks
        let mut offset: usize = 0;
        for req in batch.requests {
            let alloc_offset = offset;
            offset += req.entry.len as usize;

            let task = PrefetchTask {
                key: req.key,
                entry: req.entry,
                allocation: allocation.clone(),
                alloc_offset,
            };

            // Bounded send: blocks when channel full (natural backpressure)
            if task_tx.send(task).await.is_err() {
                debug!("SSD prefetch dispatcher: worker channel closed");
                return;
            }
        }
    }

    debug!("SSD prefetch dispatcher exiting");
}

/// Worker: maintains FuturesUnordered with max_inflight concurrent I/O operations.
async fn ssd_prefetch_worker(
    handle: Arc<SsdStorageHandle>,
    mut task_rx: tokio::sync::mpsc::Receiver<PrefetchTask>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    max_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type PrefetchFuture = Pin<Box<dyn Future<Output = PrefetchResult> + Send>>;

    let metrics = core_metrics();
    let mut inflight: FuturesUnordered<PrefetchFuture> = FuturesUnordered::new();

    loop {
        tokio::select! {
            biased;

            // Complete finished tasks first (priority)
            Some((key, begin, result, duration_secs, block_size)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_prefetch_inflight.add(-1, &[]);

                // Validate data wasn't overwritten during read
                let result = if result.is_some() && !handle.is_offset_valid(begin) {
                    warn!("SSD prefetch: data overwritten during read, discarding");
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                } else if result.is_some() {
                    metrics.ssd_prefetch_success.add(1, &[]);
                    metrics.ssd_prefetch_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_prefetch_throughput_bytes_per_second.record(throughput, &[]);
                    result
                } else {
                    metrics.ssd_prefetch_failures.add(1, &[]);
                    None
                };
                handle.complete_prefetch(key, result);
            }

            // Accept new task if below limit
            task = task_rx.recv(), if inflight.len() < max_inflight => {
                match task {
                    Some(t) => {
                        metrics.ssd_prefetch_inflight.add(1, &[]);
                        inflight.push(Box::pin(execute_prefetch(t, io.clone(), capacity)));
                    }
                    None => {
                        // Channel closed, drain remaining
                        break;
                    }
                }
            }
        }
    }

    // Drain remaining inflight tasks
    while let Some((key, begin, result, duration_secs, block_size)) = inflight.next().await {
        metrics.ssd_prefetch_inflight.add(-1, &[]);

        let result = if result.is_some() && !handle.is_offset_valid(begin) {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        } else if result.is_some() {
            metrics.ssd_prefetch_success.add(1, &[]);
            metrics.ssd_prefetch_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_prefetch_throughput_bytes_per_second
                .record(throughput, &[]);
            result
        } else {
            metrics.ssd_prefetch_failures.add(1, &[]);
            None
        };
        handle.complete_prefetch(key, result);
    }

    debug!("SSD prefetch worker exiting");
}

/// Execute a single prefetch operation.
async fn execute_prefetch(
    task: PrefetchTask,
    io: Arc<UringIoEngine>,
    capacity: u64,
) -> PrefetchResult {
    let start = Instant::now();
    let duration_secs = || start.elapsed().as_secs_f64();
    let fail = |key, begin, size| (key, begin, None, duration_secs(), size);

    let key = task.key;
    let begin = task.entry.begin;
    let block_size = task.entry.len;

    // Calculate physical offset in SSD file
    let phys_offset = begin % capacity;
    if phys_offset + block_size > capacity {
        warn!("SSD prefetch: block wraps around ring buffer");
        return fail(key, begin, block_size);
    }

    // Build iovecs from slot metadata
    let read_result = {
        let base_ptr = task.allocation.as_ptr() as *mut u8;
        let mut current_offset = task.alloc_offset;
        let iovecs: Vec<_> = task
            .entry
            .slots
            .iter()
            .flat_map(|meta| {
                // SAFETY: allocation is sized to fit all slots
                let iov = unsafe { meta.read_iovecs(base_ptr, current_offset) };
                current_offset += meta.size as usize;
                iov
            })
            .collect();

        io.readv_at_async(iovecs, phys_offset)
    };

    // Await IO result and rebuild block
    let expected_len = task.entry.len as usize;
    match read_result {
        Ok(rx) => match rx.await {
            Ok(Ok(bytes_read)) if bytes_read == expected_len => {
                match rebuild_sealed_block_at_offset(
                    task.allocation,
                    task.alloc_offset,
                    &task.entry.slots,
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
