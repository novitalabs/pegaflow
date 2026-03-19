use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
use log::{debug, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;
use tokio::sync::oneshot;

use super::ssd::SsdBackingStore;
use super::uring::UringIoEngine;
use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use crate::seal_offload::{self, SlotMeta};
use pegaflow_common::NumaNode;
use smallvec::SmallVec;

/// SSD I/O alignment requirement (O_DIRECT requires 512-byte aligned I/O)
pub const SSD_ALIGNMENT: usize = 512;

/// Default max blocks allowed in prefetching state (backpressure for SSD prefetch).
/// ~15GB assuming 10MB per block.
pub const DEFAULT_MAX_PREFETCH_BLOCKS: usize = 1500;

/// Default write queue depth for SSD writer thread (blocks dropped if full)
pub const DEFAULT_SSD_WRITE_QUEUE_DEPTH: usize = 8;

/// Default prefetch queue depth (limits read tail latency)
pub const DEFAULT_SSD_PREFETCH_QUEUE_DEPTH: usize = 2;

/// Default max concurrent writes (not critical path, keep low)
pub const DEFAULT_SSD_WRITE_INFLIGHT: usize = 2;

/// Default max concurrent prefetches
pub const DEFAULT_SSD_PREFETCH_INFLIGHT: usize = 16;

/// Result of a single prefetch I/O: (key, begin_offset, block, duration_secs, block_size, ctx)
type SinglePrefetchResult = (
    BlockKey,
    u64,
    Option<Arc<SealedBlock>>,
    f64,
    u64,
    Arc<BatchContext>,
);

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
pub(super) struct SsdIndexEntry {
    /// Logical offset in the ring buffer (monotonically increasing)
    pub begin: u64,
    /// Block size in bytes
    pub len: u64,
    /// Physical file offset for IO
    pub file_offset: u64,
    /// Per-slot metadata for rebuilding SealedBlock
    pub slots: Vec<SlotMeta>,
}

/// State of an SSD index entry (two-phase commit)
#[derive(Clone)]
pub(super) enum SsdEntryState {
    /// IO in progress, not yet readable
    Writing(SsdIndexEntry),
    /// IO completed, readable
    Committed(SsdIndexEntry),
}

impl SsdEntryState {
    #[inline]
    fn begin(&self) -> u64 {
        match self {
            Self::Writing(e) | Self::Committed(e) => e.begin,
        }
    }
}

/// SSD ring buffer: unified state for space allocation + block index.
///
/// Combines head/tail pointers with FIFO index. Maintains insertion order
/// for O(k) tail pruning while preserving O(1) lookup via HashMap.
///
/// Two-phase commit: prepare_batch inserts Writing state, commit transitions
/// to Committed (or removes on failure). Only Committed entries are readable.
pub(super) struct SsdRingBuffer {
    /// Ring buffer capacity in bytes
    capacity: u64,
    /// Next write position (logical, monotonically increasing)
    head: u64,
    /// Oldest valid position (logical); entries with begin < tail are invalid
    tail: u64,
    /// Keys ordered by insertion time (oldest at front, may contain stale keys)
    order: VecDeque<BlockKey>,
    /// Fast lookup: key -> state (Writing or Committed)
    entries: HashMap<BlockKey, SsdEntryState>,
}

impl SsdRingBuffer {
    /// Create a new ring buffer with given capacity.
    pub(super) fn new(capacity: u64) -> Self {
        Self {
            capacity,
            head: 0,
            tail: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Check if key has a valid Committed entry (not yet overwritten).
    #[cfg(test)]
    pub(super) fn has_valid_entry(&self, key: &BlockKey) -> bool {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) => e.begin >= self.tail,
            _ => false,
        }
    }

    /// Lookup a Committed entry by key, returning None if Writing or expired.
    pub(super) fn get(&self, key: &BlockKey) -> Option<&SsdIndexEntry> {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) if e.begin >= self.tail => Some(e),
            _ => None,
        }
    }

    /// Check if a logical offset is still valid (not yet overwritten).
    #[inline]
    pub(super) fn is_offset_valid(&self, begin: u64) -> bool {
        begin >= self.tail
    }

    /// Allocate contiguous space for a batch and advance tail.
    /// Returns (logical_begin, file_offset). Skips wrap-around gap if needed.
    fn allocate_contiguous(&mut self, size: u64) -> (u64, u64) {
        let phys = self.head % self.capacity;
        let space_until_end = self.capacity - phys;
        if size > space_until_end {
            // Skip to next wrap point
            self.head += space_until_end;
        }
        let begin = self.head;
        self.head += size;

        // Advance tail to maintain invariant: head - tail <= capacity
        let new_tail = self.head.saturating_sub(self.capacity);
        self.advance_tail(new_tail);

        (begin, begin % self.capacity)
    }

    /// Advance tail and prune expired entries (FIFO order).
    /// Handles both Writing and Committed states uniformly.
    fn advance_tail(&mut self, new_tail: u64) {
        if new_tail <= self.tail {
            return;
        }
        self.tail = new_tail;

        while let Some(key) = self.order.front() {
            match self.entries.get(key) {
                // Valid entry (Writing or Committed) with begin >= new_tail -> stop
                Some(state) if state.begin() >= new_tail => break,
                // Expired or already removed (aborted) -> clean up
                _ => {
                    let key = self.order.pop_front().unwrap();
                    self.entries.remove(&key);
                }
            }
        }
    }

    /// Commit a write: success=true transitions Writing→Committed, success=false removes.
    /// Returns false if entry was already expired or missing.
    pub(super) fn commit(&mut self, key: &BlockKey, success: bool) -> bool {
        let Some(state) = self.entries.get(key) else {
            // Already removed by advance_tail or previous abort
            return false;
        };

        // Only process Writing state
        let entry = match state {
            SsdEntryState::Writing(e) => e,
            SsdEntryState::Committed(_) => {
                warn!("SSD commit: key already committed, ignoring");
                return true;
            }
        };

        // Check if expired (eviction faster than write)
        if entry.begin < self.tail {
            warn!("SSD commit: entry expired before IO completed");
            self.entries.remove(key);
            return false;
        }

        if success {
            // Writing → Committed
            let entry = entry.clone();
            self.entries
                .insert(key.clone(), SsdEntryState::Committed(entry));
            true
        } else {
            // Write failed, remove entry (order will be cleaned by advance_tail)
            self.entries.remove(key);
            false
        }
    }

    /// Prepare a batch for writing: filter, allocate space, advance tail, insert Writing.
    /// Returns list of blocks to write with their allocated offsets.
    pub(super) fn prepare_batch(
        &mut self,
        candidates: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> PreparedBatch {
        // 1. Filter: skip keys that already exist (Writing or Committed)
        let to_write: Vec<_> = candidates
            .into_iter()
            .filter(|(k, _)| !self.entries.contains_key(k))
            .collect();

        if to_write.is_empty() {
            return PreparedBatch::empty();
        }

        // 2. Allocate contiguous space
        let total_size: u64 = to_write.iter().map(|(_, b)| b.memory_footprint()).sum();
        let (batch_begin, batch_file_offset) = self.allocate_contiguous(total_size);

        // 3. Insert Writing state and build WriteInfo
        let mut offset = 0u64;
        let writes = to_write
            .into_iter()
            .map(|(key, block)| {
                let size = block.memory_footprint();
                let slot_numas = block.slot_numas();
                debug_assert!(
                    slot_numas.is_empty() || slot_numas.len() == block.slots().len(),
                    "slot_numas length mismatch: {} vs {}",
                    slot_numas.len(),
                    block.slots().len(),
                );
                let slots: Vec<SlotMeta> = block
                    .slots()
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        let segment_sizes: SmallVec<[u64; 4]> = (0..s.num_segments())
                            .map(|idx| s.segment_size(idx).unwrap() as u64)
                            .collect();
                        SlotMeta {
                            segment_sizes,
                            numa_node: slot_numas.get(i).copied().unwrap_or(NumaNode::UNKNOWN),
                        }
                    })
                    .collect();

                let file_offset = batch_file_offset + offset;
                let entry = SsdIndexEntry {
                    begin: batch_begin + offset,
                    len: size,
                    file_offset,
                    slots,
                };

                // Insert Writing state
                self.entries
                    .insert(key.clone(), SsdEntryState::Writing(entry.clone()));
                self.order.push_back(key.clone());

                let info = WriteInfo { key, block, entry };
                offset += size;
                info
            })
            .collect();

        PreparedBatch { writes }
    }
}

impl Default for SsdRingBuffer {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Info for a single block write within a batch.
pub(super) struct WriteInfo {
    pub key: BlockKey,
    pub block: Arc<SealedBlock>,
    pub entry: SsdIndexEntry,
}

/// Prepared batch ready for IO.
pub(super) struct PreparedBatch {
    pub writes: Vec<WriteInfo>,
}

impl PreparedBatch {
    pub(super) fn empty() -> Self {
        Self { writes: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

/// Batch of sealed blocks to write to SSD
pub(super) struct SsdWriteBatch {
    pub blocks: Vec<(BlockKey, Weak<SealedBlock>)>,
}

/// Request to prefetch a block from SSD (metadata only, allocation done in worker)
pub(super) struct PrefetchRequest {
    pub key: BlockKey,
    pub entry: SsdIndexEntry,
}

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub(super) struct PrefetchBatch {
    pub requests: Vec<PrefetchRequest>,
    pub done_tx: oneshot::Sender<super::PrefetchResult>,
}

/// Shared context for a batch of prefetch operations.
/// Collects successful blocks and delivers them as a batch when all reads finish.
pub(super) struct BatchContext {
    results: Mutex<super::PrefetchResult>,
    remaining: AtomicUsize,
    done_tx: Mutex<Option<oneshot::Sender<super::PrefetchResult>>>,
}

impl BatchContext {
    fn new(count: usize, done_tx: oneshot::Sender<super::PrefetchResult>) -> Self {
        Self {
            results: Mutex::new(Vec::with_capacity(count)),
            remaining: AtomicUsize::new(count),
            done_tx: Mutex::new(Some(done_tx)),
        }
    }

    fn complete_one(&self, key: BlockKey, block: Option<Arc<SealedBlock>>) {
        if let Some(block) = block {
            self.results.lock().push((key, block));
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(tx) = self.done_tx.lock().take()
        {
            let results = std::mem::take(&mut *self.results.lock());
            let _ = tx.send(results);
        }
    }
}

/// Per-slot allocation reference: which allocation and offset within it.
struct SlotAlloc {
    allocation: Arc<PinnedAllocation>,
    offset: usize,
}

/// Internal: single block prefetch task with per-slot allocated memory.
struct PrefetchTask {
    key: BlockKey,
    entry: SsdIndexEntry,
    /// One per slot (parallel to `entry.slots`), each from the correct NUMA pool.
    slot_allocs: Vec<SlotAlloc>,
    /// Shared batch context: per-block callback + completion counter.
    ctx: Arc<BatchContext>,
}

/// Internal: single block write task
struct WriteTask {
    key: BlockKey,
    block: Arc<SealedBlock>,
    entry: SsdIndexEntry,
}

/// Result of a single write operation: (key, success, duration_secs, block_size)
type WriteResult = (BlockKey, bool, f64, u64);

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives batches of sealed blocks and writes them.
pub(super) async fn ssd_writer_loop(
    store: Weak<SsdBackingStore>,
    mut rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
    io: Arc<UringIoEngine>,
    write_inflight: usize,
) {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;

    type WriteFuture = Pin<Box<dyn Future<Output = WriteResult> + Send>>;

    let metrics = core_metrics();
    let max_inflight = write_inflight.max(1);

    let mut pending: VecDeque<WriteTask> = VecDeque::new();
    let mut inflight: FuturesOrdered<WriteFuture> = FuturesOrdered::new();

    loop {
        tokio::select! {
            biased;

            // Priority 1: Complete writes
            Some((key, success, duration_secs, block_size)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_write_inflight.add(-1, &[]);

                // Commit result to ring buffer (Writing→Committed or remove)
                if let Some(s) = store.upgrade() {
                    s.commit_write(&key, success);
                }

                if success {
                    metrics.ssd_write_bytes.add(block_size, &[]);
                    let throughput = block_size as f64 / duration_secs;
                    metrics.ssd_write_throughput_bytes_per_second.record(throughput, &[]);
                } else {
                    warn!("SSD cache write failed for {:?}", key);
                }
            }

            // Priority 2: Submit pending writes if inflight has room
            _ = std::future::ready(()), if inflight.len() < max_inflight && !pending.is_empty() => {
                let task = pending.pop_front().unwrap();
                metrics.ssd_write_inflight.add(1, &[]);
                inflight.push_back(Box::pin(execute_write(task, io.clone())));
            }

            // Priority 3: Receive new batch
            batch = rx.recv(), if pending.is_empty() => {
                match batch {
                    Some(b) => {
                        // Dequeue metric
                        metrics.ssd_write_queue_pending.add(-(b.blocks.len() as i64), &[]);

                        // Upgrade weak refs (per-block, not prefix semantics)
                        let candidates: Vec<_> = b.blocks
                            .into_iter()
                            .filter_map(|(k, w)| w.upgrade().map(|b| (k, b)))
                            .collect();

                        if candidates.is_empty() {
                            continue;
                        }

                        // Prepare batch: filter + allocate + insert Writing
                        let Some(s) = store.upgrade() else { continue };
                        let prepared = s.prepare_batch(candidates);

                        if prepared.is_empty() {
                            continue;
                        }

                        // Convert to WriteTask
                        for w in prepared.writes {
                            pending.push_back(WriteTask {
                                key: w.key,
                                block: w.block,
                                entry: w.entry,
                            });
                        }
                    }
                    None => break,
                }
            }
        }
    }

    // Drain remaining inflight writes
    while let Some((key, success, duration_secs, block_size)) = inflight.next().await {
        metrics.ssd_write_inflight.add(-1, &[]);

        if let Some(s) = store.upgrade() {
            s.commit_write(&key, success);
        }

        if success {
            metrics.ssd_write_bytes.add(block_size, &[]);
            let throughput = block_size as f64 / duration_secs;
            metrics
                .ssd_write_throughput_bytes_per_second
                .record(throughput, &[]);
        } else {
            warn!("SSD cache write failed for {:?}", key);
        }
    }

    debug!("SSD writer task exiting");
}

/// Execute a single block write to SSD.
async fn execute_write(task: WriteTask, io: Arc<UringIoEngine>) -> WriteResult {
    let start = Instant::now();
    let key = task.key;
    let block_size = task.block.memory_footprint();

    let result = write_block_to_ssd(&io, task.entry.file_offset, &task.block).await;

    let duration_secs = start.elapsed().as_secs_f64();
    (key, result.is_ok(), duration_secs, block_size)
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
    // Build iovecs from RawBlock segments (layout-agnostic)
    let rx = {
        let iovecs: Vec<_> = block
            .slots()
            .iter()
            .flat_map(|slot| seal_offload::write_iovecs(slot))
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
pub(super) async fn ssd_prefetch_loop(
    store: Weak<SsdBackingStore>,
    rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    prefetch_inflight: usize,
) {
    let prefetch_inflight = prefetch_inflight.max(1);

    // Bounded channel: capacity = max inflight tasks
    let (task_tx, task_rx) = tokio::sync::mpsc::channel(prefetch_inflight);

    // Spawn dispatcher and worker
    let dispatcher = tokio::spawn(ssd_prefetch_dispatcher(store.clone(), rx, task_tx));
    let worker = tokio::spawn(ssd_prefetch_worker(
        store,
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

/// Dispatcher: receives batches, allocates per-slot memory grouped by NUMA,
/// then splits into block-level tasks.
async fn ssd_prefetch_dispatcher(
    store: Weak<SsdBackingStore>,
    mut batch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
    task_tx: tokio::sync::mpsc::Sender<PrefetchTask>,
) {
    while let Some(batch) = batch_rx.recv().await {
        if batch.requests.is_empty() {
            let _ = batch.done_tx.send(Vec::new());
            continue;
        }

        let Some(s) = store.upgrade() else { break };
        if !dispatch_prefetch_batch(&s, &task_tx, batch).await {
            return;
        }
    }

    debug!("SSD prefetch dispatcher exiting");
}

/// Slot reference for NUMA grouping during allocation.
struct SlotRef {
    block_idx: usize,
    slot_idx: usize,
    size: u64,
}

/// Group all slots across all blocks by NUMA node.
///
/// Returns a map from NUMA key (None = global/unknown) to the list of slot
/// references that should share a single contiguous allocation.
fn group_slots_by_numa(
    is_numa: bool,
    requests: &[PrefetchRequest],
) -> HashMap<Option<NumaNode>, Vec<SlotRef>> {
    let mut groups: HashMap<Option<NumaNode>, Vec<SlotRef>> = HashMap::new();
    for (block_idx, req) in requests.iter().enumerate() {
        for (slot_idx, meta) in req.entry.slots.iter().enumerate() {
            let numa_key = if is_numa {
                let numa = meta.numa_node;
                if numa.is_unknown() { None } else { Some(numa) }
            } else {
                None
            };
            groups.entry(numa_key).or_default().push(SlotRef {
                block_idx,
                slot_idx,
                size: meta.total_size(),
            });
        }
    }
    groups
}

/// Allocate per-slot memory grouped by NUMA, build PrefetchTasks, and enqueue.
/// Returns false if the task channel is closed (should exit).
async fn dispatch_prefetch_batch(
    store: &SsdBackingStore,
    task_tx: &tokio::sync::mpsc::Sender<PrefetchTask>,
    batch: PrefetchBatch,
) -> bool {
    let PrefetchBatch { requests, done_tx } = batch;

    // 1. Group all slots across all blocks by NUMA node
    let numa_groups = group_slots_by_numa(store.is_numa(), &requests);

    // 2. Allocate per NUMA group, assign per-slot (allocation, offset).
    //    All blocks in a batch share the same slot layout, so a single NUMA
    //    group failure means every block is missing slots → fail the whole batch.
    let mut slot_allocs: Vec<Vec<Option<SlotAlloc>>> = requests
        .iter()
        .map(|r| (0..r.entry.slots.len()).map(|_| None).collect())
        .collect();

    for (numa_node, refs) in &numa_groups {
        let total_size: u64 = refs.iter().map(|r| r.size).sum();
        let allocation = match store.allocate_prefetch(total_size, *numa_node) {
            Some(alloc) => alloc,
            None => {
                warn!(
                    "SSD prefetch dispatcher: alloc failed for {} bytes ({} slots) numa={:?}, failing entire batch",
                    total_size,
                    refs.len(),
                    numa_node
                );
                let _ = done_tx.send(Vec::new());
                return true;
            }
        };

        let mut offset = 0usize;
        for r in refs {
            slot_allocs[r.block_idx][r.slot_idx] = Some(SlotAlloc {
                allocation: allocation.clone(),
                offset,
            });
            offset += r.size as usize;
        }
    }

    // 3. Build BatchContext + PrefetchTasks and enqueue
    let ctx = Arc::new(BatchContext::new(requests.len(), done_tx));

    for (block_idx, req) in requests.into_iter().enumerate() {
        let allocs: Vec<SlotAlloc> = slot_allocs[block_idx]
            .drain(..)
            .map(|opt| opt.expect("all slots must have allocations"))
            .collect();

        let task = PrefetchTask {
            key: req.key,
            entry: req.entry,
            slot_allocs: allocs,
            ctx: Arc::clone(&ctx),
        };

        if task_tx.send(task).await.is_err() {
            debug!("SSD prefetch dispatcher: worker channel closed");
            return false;
        }
    }

    true
}

/// Worker: maintains FuturesUnordered with max_inflight concurrent I/O operations.
async fn ssd_prefetch_worker(
    store: Weak<SsdBackingStore>,
    mut task_rx: tokio::sync::mpsc::Receiver<PrefetchTask>,
    io: Arc<UringIoEngine>,
    capacity: u64,
    max_inflight: usize,
) {
    use std::future::Future;
    use std::pin::Pin;

    type PrefetchFuture = Pin<Box<dyn Future<Output = SinglePrefetchResult> + Send>>;

    let metrics = core_metrics();
    let mut inflight: FuturesUnordered<PrefetchFuture> = FuturesUnordered::new();

    loop {
        tokio::select! {
            biased;

            // Complete finished tasks first (priority)
            Some((key, begin, result, duration_secs, block_size, ctx)) = inflight.next(), if !inflight.is_empty() => {
                metrics.ssd_prefetch_inflight.add(-1, &[]);

                // Validate data wasn't overwritten during read
                let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(begin));
                let result = if result.is_some() && !valid {
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
                ctx.complete_one(key, result);
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
    while let Some((key, begin, result, duration_secs, block_size, ctx)) = inflight.next().await {
        metrics.ssd_prefetch_inflight.add(-1, &[]);

        let valid = store.upgrade().is_some_and(|s| s.is_offset_valid(begin));
        let result = if result.is_some() && !valid {
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
        ctx.complete_one(key, result);
    }

    debug!("SSD prefetch worker exiting");
}

/// Execute a single prefetch operation.
async fn execute_prefetch(
    task: PrefetchTask,
    io: Arc<UringIoEngine>,
    capacity: u64,
) -> SinglePrefetchResult {
    let start = Instant::now();
    let duration_secs = || start.elapsed().as_secs_f64();

    let key = task.key;
    let begin = task.entry.begin;
    let block_size = task.entry.len;
    let ctx = task.ctx;

    // Calculate physical offset in SSD file
    let phys_offset = begin % capacity;
    if phys_offset + block_size > capacity {
        warn!("SSD prefetch: block wraps around ring buffer");
        return (key, begin, None, duration_secs(), block_size, ctx);
    }

    // Build iovecs from per-slot allocations
    let read_result = {
        let iovecs: Vec<_> = task
            .entry
            .slots
            .iter()
            .zip(&task.slot_allocs)
            .flat_map(|(meta, alloc)| {
                let base_ptr = alloc.allocation.as_ptr() as *mut u8;
                // SAFETY: each allocation is sized to fit its NUMA group's slots
                unsafe { seal_offload::read_iovecs(meta, base_ptr, alloc.offset) }
            })
            .collect();

        io.readv_at_async(iovecs, phys_offset)
    };

    // Await IO result and rebuild block
    let expected_len = task.entry.len as usize;
    let block = match read_result {
        Ok(rx) => match rx.await {
            Ok(Ok(bytes_read)) if bytes_read == expected_len => {
                match rebuild_sealed_block_per_slot(task.slot_allocs, &task.entry.slots) {
                    Ok(sealed) => Some(Arc::new(sealed)),
                    Err(e) => {
                        warn!("SSD prefetch: failed to rebuild block: {}", e);
                        None
                    }
                }
            }
            Ok(Ok(n)) => {
                warn!("SSD prefetch: short read {} of {} bytes", n, expected_len);
                None
            }
            Ok(Err(e)) => {
                warn!("SSD prefetch: read error: {}", e);
                None
            }
            Err(_) => {
                warn!("SSD prefetch: read channel closed");
                None
            }
        },
        Err(e) => {
            warn!("SSD prefetch: failed to submit read: {}", e);
            None
        }
    };

    (key, begin, block, duration_secs(), block_size, ctx)
}

// ============================================================================
// Block Rebuilding
// ============================================================================

/// Rebuild a SealedBlock from per-slot allocations (consumed).
///
/// Each slot may reside in a different NUMA-local allocation. Takes ownership
/// of `SlotAlloc`s to move (not clone) the `Arc<PinnedAllocation>` references.
fn rebuild_sealed_block_per_slot(
    slot_allocs: Vec<SlotAlloc>,
    slot_metas: &[SlotMeta],
) -> Result<SealedBlock, String> {
    let raw_blocks: Vec<_> = slot_metas
        .iter()
        .zip(slot_allocs)
        .map(|(meta, alloc)| unsafe {
            seal_offload::reconstruct_raw_block(meta, alloc.allocation, alloc.offset)
        })
        .collect();

    Ok(SealedBlock::from_slots(raw_blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::smallvec;

    fn make_key(n: u8) -> BlockKey {
        BlockKey::new("test".to_string(), vec![n])
    }

    impl SsdRingBuffer {
        /// Insert a Committed entry for testing. Returns the key.
        fn insert_committed(&mut self, n: u8, begin: u64, len: u64) -> BlockKey {
            let key = make_key(n);
            let entry = SsdIndexEntry {
                begin,
                len,
                file_offset: begin % self.capacity.max(1),
                slots: vec![],
            };
            self.entries
                .insert(key.clone(), SsdEntryState::Committed(entry));
            self.order.push_back(key.clone());
            key
        }

        /// Insert a Writing entry for testing. Returns the key.
        fn insert_writing(&mut self, n: u8, begin: u64, len: u64) -> BlockKey {
            let key = make_key(n);
            let entry = SsdIndexEntry {
                begin,
                len,
                file_offset: begin % self.capacity.max(1),
                slots: vec![],
            };
            self.entries
                .insert(key.clone(), SsdEntryState::Writing(entry));
            self.order.push_back(key.clone());
            key
        }
    }

    // ========================================================================
    // allocate_contiguous tests
    // ========================================================================

    #[test]
    fn test_allocate_contiguous_basic() {
        let mut ring = SsdRingBuffer::new(1000);

        let (begin, offset) = ring.allocate_contiguous(100);
        assert_eq!((begin, offset, ring.head, ring.tail), (0, 0, 100, 0));

        let (begin, offset) = ring.allocate_contiguous(200);
        assert_eq!((begin, offset, ring.head, ring.tail), (100, 100, 300, 0));
    }

    #[test]
    fn test_allocate_contiguous_wrap_around() {
        let mut ring = SsdRingBuffer::new(1000);
        ring.allocate_contiguous(900);

        // 200 bytes doesn't fit in remaining 100, skips to wrap point
        let (begin, offset) = ring.allocate_contiguous(200);
        assert_eq!(begin, 1000); // skipped to wrap point
        assert_eq!(offset, 0); // wraps to file start
        assert_eq!(ring.tail, 200); // head(1200) - capacity(1000)
    }

    #[test]
    fn test_allocate_contiguous_prunes_expired() {
        let mut ring = SsdRingBuffer::new(1000);
        let key = ring.insert_committed(1, 0, 100);

        ring.allocate_contiguous(600);
        ring.allocate_contiguous(600); // head=1200, tail=200

        assert!(!ring.entries.contains_key(&key));
        assert!(ring.order.is_empty());
    }

    // ========================================================================
    // is_offset_valid tests
    // ========================================================================

    #[test]
    fn test_is_offset_valid() {
        let mut ring = SsdRingBuffer::new(1000);
        assert!(ring.is_offset_valid(0));

        ring.tail = 50;
        assert!(!ring.is_offset_valid(49));
        assert!(ring.is_offset_valid(50));
    }

    // ========================================================================
    // commit tests
    // ========================================================================

    #[test]
    fn test_commit_writing_to_committed() {
        let mut ring = SsdRingBuffer::new(1000);
        let key = ring.insert_writing(1, 100, 50);

        assert!(ring.commit(&key, true));
        assert!(matches!(
            ring.entries.get(&key),
            Some(SsdEntryState::Committed(_))
        ));
    }

    #[test]
    fn test_commit_failure_removes_entry() {
        let mut ring = SsdRingBuffer::new(1000);
        let key = ring.insert_writing(1, 100, 50);

        assert!(!ring.commit(&key, false));
        assert!(!ring.entries.contains_key(&key));
        assert_eq!(ring.order.len(), 1); // order cleaned by advance_tail later
    }

    #[test]
    fn test_commit_expired_entry() {
        let mut ring = SsdRingBuffer::new(1000);
        let key = ring.insert_writing(1, 100, 50);
        ring.tail = 200; // expire it

        assert!(!ring.commit(&key, true));
        assert!(!ring.entries.contains_key(&key));
    }

    #[test]
    fn test_commit_edge_cases() {
        let mut ring = SsdRingBuffer::new(1000);

        // Missing key
        assert!(!ring.commit(&make_key(99), true));

        // Already committed (idempotent)
        let key = ring.insert_committed(1, 100, 50);
        assert!(ring.commit(&key, true));
    }

    // ========================================================================
    // get / has_valid_entry tests
    // ========================================================================

    #[test]
    fn test_get_and_has_valid_entry() {
        let mut ring = SsdRingBuffer::new(1000);
        let k_writing = ring.insert_writing(1, 100, 50);
        let k_committed = ring.insert_committed(2, 200, 50);

        // Writing: not readable
        assert!(ring.get(&k_writing).is_none());
        assert!(!ring.has_valid_entry(&k_writing));

        // Committed: readable
        let entry = ring.get(&k_committed).unwrap();
        assert_eq!((entry.begin, entry.len), (200, 50));
        assert!(ring.has_valid_entry(&k_committed));

        // Expired: not readable
        ring.tail = 250;
        assert!(ring.get(&k_committed).is_none());
        assert!(!ring.has_valid_entry(&k_committed));
    }

    // ========================================================================
    // advance_tail tests
    // ========================================================================

    #[test]
    fn test_advance_tail_prunes_expired() {
        let mut ring = SsdRingBuffer::new(1000);
        ring.insert_committed(0, 0, 50);
        ring.insert_committed(1, 100, 50);
        ring.insert_committed(2, 200, 50);

        ring.advance_tail(150);

        assert_eq!(ring.entries.len(), 1);
        assert!(ring.get(&make_key(2)).is_some());
    }

    #[test]
    fn test_advance_tail_cleans_ghost_entries() {
        let mut ring = SsdRingBuffer::new(1000);
        // Ghost: in order but not in entries (aborted write)
        ring.order.push_back(make_key(1));
        ring.insert_committed(2, 200, 50);

        ring.advance_tail(100);

        assert_eq!(ring.order.len(), 1);
        assert_eq!(ring.order.front(), Some(&make_key(2)));
    }

    #[test]
    fn test_advance_tail_preserves_valid_writing() {
        let mut ring = SsdRingBuffer::new(1000);
        ring.insert_writing(1, 100, 50);

        ring.advance_tail(50); // tail < begin, should preserve

        assert_eq!(ring.entries.len(), 1);
    }

    // ========================================================================
    // Duplicate key filtering
    // ========================================================================

    #[test]
    fn test_duplicate_key_filtered() {
        let mut ring = SsdRingBuffer::new(1000);
        let key = ring.insert_writing(1, 100, 50);

        let filtered: Vec<_> = vec![key]
            .into_iter()
            .filter(|k| !ring.entries.contains_key(k))
            .collect();

        assert!(filtered.is_empty());
    }

    // ========================================================================
    // group_slots_by_numa tests
    // ========================================================================

    fn make_slot(numa_node: NumaNode, size: u64) -> SlotMeta {
        SlotMeta {
            segment_sizes: smallvec![size],
            numa_node,
        }
    }

    fn make_prefetch_request(n: u8, slots: Vec<SlotMeta>) -> PrefetchRequest {
        let total_size: u64 = slots.iter().map(|s| s.total_size()).sum();
        PrefetchRequest {
            key: make_key(n),
            entry: SsdIndexEntry {
                begin: 0,
                len: total_size,
                file_offset: 0,
                slots,
            },
        }
    }

    #[test]
    fn test_group_slots_non_numa_single_group() {
        // When !is_numa, all slots collapse into None regardless of numa_node
        let requests = vec![
            make_prefetch_request(
                1,
                vec![make_slot(NumaNode(0), 100), make_slot(NumaNode(1), 100)],
            ),
            make_prefetch_request(2, vec![make_slot(NumaNode(0), 200)]),
        ];

        let groups = group_slots_by_numa(false, &requests);
        assert_eq!(groups.len(), 1);
        assert!(groups.contains_key(&None));
        assert_eq!(groups[&None].len(), 3); // all 3 slots in one group
    }

    #[test]
    fn test_group_slots_numa_tp8_split() {
        // TP8: 4 slots on NUMA0, 4 on NUMA1, 2 blocks
        let slots = |n0, n1| {
            vec![
                make_slot(NumaNode(0), 64),
                make_slot(NumaNode(0), 64),
                make_slot(NumaNode(0), 64),
                make_slot(NumaNode(0), 64),
                make_slot(NumaNode(n0), 64),
                make_slot(NumaNode(n0), 64),
                make_slot(NumaNode(n1), 64),
                make_slot(NumaNode(n1), 64),
            ]
        };
        let requests = vec![
            make_prefetch_request(1, slots(1, 1)),
            make_prefetch_request(2, slots(1, 1)),
        ];

        let groups = group_slots_by_numa(true, &requests);
        assert_eq!(groups.len(), 2); // NUMA0 and NUMA1

        let numa0 = &groups[&Some(NumaNode(0))];
        let numa1 = &groups[&Some(NumaNode(1))];

        // 4 slots/block * 2 blocks = 8 per NUMA
        assert_eq!(numa0.len(), 8);
        assert_eq!(numa1.len(), 8);

        // Total size: 8 * 64 = 512 per group
        let total_0: u64 = numa0.iter().map(|r| r.size).sum();
        let total_1: u64 = numa1.iter().map(|r| r.size).sum();
        assert_eq!(total_0, 512);
        assert_eq!(total_1, 512);

        // Verify slots reference correct slot indices
        assert!(numa0.iter().all(|r| r.slot_idx < 4));
        assert!(numa1.iter().all(|r| r.slot_idx >= 4));
    }

    #[test]
    fn test_group_slots_unknown_maps_to_none() {
        let requests = vec![make_prefetch_request(
            1,
            vec![
                make_slot(NumaNode(0), 100),
                make_slot(NumaNode::UNKNOWN, 100),
            ],
        )];

        let groups = group_slots_by_numa(true, &requests);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[&Some(NumaNode(0))].len(), 1);
        assert_eq!(groups[&None].len(), 1); // UNKNOWN → None
    }
}
