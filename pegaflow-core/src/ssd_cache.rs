use futures::stream::{FuturesOrdered, FuturesUnordered, StreamExt};
use log::{debug, warn};
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Instant;

use crate::block::{BlockKey, LayerBlock, SealedBlock};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
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
pub(crate) struct SsdIndexEntry {
    /// Logical offset in the ring buffer (monotonically increasing)
    pub begin: u64,
    /// Block size in bytes
    pub len: u64,
    /// Physical file offset for IO
    pub file_offset: u64,
    /// Per-slot metadata for rebuilding SealedBlock
    pub slots: Vec<SlotMeta>,
    /// NUMA node affinity for the originating GPU
    pub numa_node: NumaNode,
}

/// State of an SSD index entry (two-phase commit)
#[derive(Clone)]
pub(crate) enum SsdEntryState {
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
pub(crate) struct SsdRingBuffer {
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
    pub(crate) fn new(capacity: u64) -> Self {
        Self {
            capacity,
            head: 0,
            tail: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Check if key has a valid Committed entry (not yet overwritten).
    #[inline]
    pub(crate) fn has_valid_entry(&self, key: &BlockKey) -> bool {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) => e.begin >= self.tail,
            _ => false,
        }
    }

    /// Lookup a Committed entry by key, returning None if Writing or expired.
    pub(crate) fn get(&self, key: &BlockKey) -> Option<&SsdIndexEntry> {
        match self.entries.get(key) {
            Some(SsdEntryState::Committed(e)) if e.begin >= self.tail => Some(e),
            _ => None,
        }
    }

    /// Check if a logical offset is still valid (not yet overwritten).
    #[inline]
    pub(crate) fn is_offset_valid(&self, begin: u64) -> bool {
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
    pub(crate) fn commit(&mut self, key: &BlockKey, success: bool) -> bool {
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
    pub(crate) fn prepare_batch(
        &mut self,
        candidates: Vec<(BlockKey, Arc<SealedBlock>)>,
        numa_node: NumaNode,
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
                let slots: Vec<SlotMeta> = block
                    .slots()
                    .iter()
                    .map(|s| SlotMeta {
                        is_split: s.v_ptr().is_some(),
                        size: s.size() as u64,
                    })
                    .collect();

                let file_offset = batch_file_offset + offset;
                let entry = SsdIndexEntry {
                    begin: batch_begin + offset,
                    len: size,
                    file_offset,
                    slots,
                    numa_node,
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
pub(crate) struct WriteInfo {
    pub key: BlockKey,
    pub block: Arc<SealedBlock>,
    pub entry: SsdIndexEntry,
}

/// Prepared batch ready for IO.
pub(crate) struct PreparedBatch {
    pub writes: Vec<WriteInfo>,
}

impl PreparedBatch {
    pub fn empty() -> Self {
        Self { writes: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.writes.is_empty()
    }
}

/// Batch of sealed blocks to write to SSD
pub(crate) struct SsdWriteBatch {
    pub blocks: Vec<(BlockKey, Weak<SealedBlock>)>,
    pub numa_node: NumaNode,
}

/// Request to prefetch a block from SSD (metadata only, allocation done in worker)
pub(crate) struct PrefetchRequest {
    pub key: BlockKey,
    pub entry: SsdIndexEntry,
}

/// Batch of prefetch requests (sent as a unit to limit queue depth)
pub(crate) struct PrefetchBatch {
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
    entry: SsdIndexEntry,
}

/// Result of a single write operation: (key, success, duration_secs, block_size)
type WriteResult = (BlockKey, bool, f64, u64);

// ============================================================================
// Storage handle (provided by StorageEngine, used by prefetch worker)
// ============================================================================

/// Type alias for prepare batch callback.
pub(crate) type PrepareBatchFn =
    Arc<dyn Fn(Vec<(BlockKey, Arc<SealedBlock>)>, NumaNode) -> PreparedBatch + Send + Sync>;

/// Type alias for commit write callback.
pub(crate) type CommitWriteFn = Arc<dyn Fn(&BlockKey, bool) + Send + Sync>;

/// Type alias for prefetch allocation callback.
pub(crate) type AllocatePrefetchFn =
    Arc<dyn Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>> + Send + Sync>;

/// Handle used by SSD workers to interact with storage.
/// Both writer and prefetcher use callbacks to access StorageInner state.
pub(crate) struct SsdStorageHandle {
    /// Complete prefetch operation (insert into cache)
    complete_prefetch: Arc<dyn Fn(BlockKey, Option<Arc<SealedBlock>>) + Send + Sync>,
    /// Check if a logical offset is still valid (not yet overwritten)
    is_valid: Arc<dyn Fn(u64) -> bool + Send + Sync>,
    /// Allocate pinned memory for prefetch
    allocate: AllocatePrefetchFn,
    /// Prepare batch for SSD write (filter, allocate space, insert Writing)
    prepare: PrepareBatchFn,
    /// Commit write result (success: Writing→Committed, failure: remove)
    commit: CommitWriteFn,
    is_numa: bool,
}

impl SsdStorageHandle {
    pub(crate) fn new(
        complete_prefetch: impl Fn(BlockKey, Option<Arc<SealedBlock>>) + Send + Sync + 'static,
        is_valid: impl Fn(u64) -> bool + Send + Sync + 'static,
        allocate: impl Fn(u64, Option<NumaNode>) -> Option<Arc<PinnedAllocation>>
        + Send
        + Sync
        + 'static,
        prepare: impl Fn(Vec<(BlockKey, Arc<SealedBlock>)>, NumaNode) -> PreparedBatch
        + Send
        + Sync
        + 'static,
        commit: impl Fn(&BlockKey, bool) + Send + Sync + 'static,
        is_numa: bool,
    ) -> Self {
        Self {
            complete_prefetch: Arc::new(complete_prefetch),
            is_valid: Arc::new(is_valid),
            allocate: Arc::new(allocate),
            prepare: Arc::new(prepare),
            commit: Arc::new(commit),
            is_numa,
        }
    }

    #[inline]
    pub(crate) fn complete_prefetch(&self, key: BlockKey, block: Option<Arc<SealedBlock>>) {
        (self.complete_prefetch)(key, block);
    }

    #[inline]
    pub(crate) fn is_valid(&self, begin: u64) -> bool {
        (self.is_valid)(begin)
    }

    #[inline]
    pub(crate) fn allocate(
        &self,
        size: u64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        (self.allocate)(size, numa_node)
    }

    #[inline]
    pub(crate) fn prepare(
        &self,
        candidates: Vec<(BlockKey, Arc<SealedBlock>)>,
        numa_node: NumaNode,
    ) -> PreparedBatch {
        (self.prepare)(candidates, numa_node)
    }

    #[inline]
    pub(crate) fn commit(&self, key: &BlockKey, success: bool) {
        (self.commit)(key, success)
    }

    #[inline]
    pub(crate) fn is_numa(&self) -> bool {
        self.is_numa
    }
}

// ============================================================================
// SSD Writer Loop
// ============================================================================

/// SSD writer task: receives batches of sealed blocks and writes them.
///
/// Uses `SsdStorageHandle` to interact with storage state. All ring buffer
/// operations go through callbacks that acquire StorageInner lock.
pub(crate) async fn ssd_writer_loop(
    handle: Arc<SsdStorageHandle>,
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
                handle.commit(&key, success);

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

                        // Prepare batch: filter + allocate + insert Writing (via handle)
                        let prepared = handle.prepare(candidates, b.numa_node);

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

        handle.commit(&key, success);

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

    let result =
        write_block_to_ssd(&io, task.entry.file_offset, &task.block, &task.entry.slots).await;

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
pub(crate) async fn ssd_prefetch_loop(
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

        let groups = build_allocation_groups(handle.is_numa(), batch.requests);
        for group in groups {
            let total_size = total_request_size(&group.requests);
            let allocation = match handle.allocate(total_size, group.numa_node) {
                Some(alloc) => alloc,
                None => {
                    warn!(
                        "SSD prefetch dispatcher: alloc failed for {} bytes ({} blocks) numa={:?}",
                        total_size,
                        group.requests.len(),
                        group.numa_node
                    );
                    for req in group.requests {
                        handle.complete_prefetch(req.key, None);
                    }
                    continue;
                }
            };

            if !enqueue_prefetch_tasks(&task_tx, allocation, group.requests).await {
                return;
            }
        }
    }

    debug!("SSD prefetch dispatcher exiting");
}

fn total_request_size(requests: &[PrefetchRequest]) -> u64 {
    requests.iter().map(|r| r.entry.len).sum()
}

struct AllocationGroup {
    numa_node: Option<NumaNode>,
    requests: Vec<PrefetchRequest>,
}

fn build_allocation_groups(is_numa: bool, requests: Vec<PrefetchRequest>) -> Vec<AllocationGroup> {
    if !is_numa {
        return vec![AllocationGroup {
            numa_node: None,
            requests,
        }];
    }

    let mut groups: HashMap<Option<NumaNode>, Vec<PrefetchRequest>> = HashMap::new();
    for req in requests {
        let key = if req.entry.numa_node.is_unknown() {
            None
        } else {
            Some(req.entry.numa_node)
        };
        groups.entry(key).or_default().push(req);
    }

    groups
        .into_iter()
        .map(|(numa_node, requests)| AllocationGroup {
            numa_node,
            requests,
        })
        .collect()
}

async fn enqueue_prefetch_tasks(
    task_tx: &tokio::sync::mpsc::Sender<PrefetchTask>,
    allocation: Arc<PinnedAllocation>,
    requests: Vec<PrefetchRequest>,
) -> bool {
    // Split into block-level tasks
    let mut offset: usize = 0;
    for req in requests {
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
            return false;
        }
    }

    true
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
                let result = if result.is_some() && !handle.is_valid(begin) {
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

        let result = if result.is_some() && !handle.is_valid(begin) {
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

/// Rebuild a SealedBlock from a shared allocation at a given offset.
/// Used for batched prefetch where multiple blocks share one contiguous allocation.
pub(crate) fn rebuild_sealed_block_at_offset(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(n: u8) -> BlockKey {
        BlockKey::new("test".to_string(), vec![n])
    }

    fn make_request(n: u8, numa_node: NumaNode) -> PrefetchRequest {
        let key = make_key(n);
        let entry = SsdIndexEntry {
            begin: n as u64,
            len: 8,
            file_offset: 0,
            slots: vec![],
            numa_node,
        };
        PrefetchRequest { key, entry }
    }

    #[test]
    fn test_build_allocation_groups_non_numa_single_group() {
        let requests = vec![
            make_request(1, NumaNode::UNKNOWN),
            make_request(2, NumaNode::UNKNOWN),
            make_request(3, NumaNode::UNKNOWN),
        ];

        let groups = build_allocation_groups(false, requests);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].numa_node.is_none());
        assert_eq!(groups[0].requests.len(), 3);
    }

    #[test]
    fn test_build_allocation_groups_numa_split_and_unknown() {
        let requests = vec![
            make_request(1, NumaNode(0)),
            make_request(2, NumaNode(1)),
            make_request(3, NumaNode(0)),
        ];

        let groups = build_allocation_groups(true, requests);
        let mut counts: HashMap<Option<NumaNode>, usize> = HashMap::new();
        for group in groups {
            counts.insert(group.numa_node, group.requests.len());
        }

        assert_eq!(counts.get(&Some(NumaNode(0))), Some(&2));
        assert_eq!(counts.get(&Some(NumaNode(1))), Some(&1));
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
                numa_node: NumaNode::UNKNOWN,
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
                numa_node: NumaNode::UNKNOWN,
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
}
