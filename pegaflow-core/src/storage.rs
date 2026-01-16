// ============================================================================
// StorageEngine: Two-phase block storage with separate write and read paths.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Architecture:
// - Single Mutex<StorageInner> for all state (inflight, prefetching, cache, ssd_index)
// - Allocator: PinnedMemoryPool for pinned memory allocation
// - Prefetch worker: background io_uring reads from SSD
//
// Eviction only targets the cache; inflight blocks are never evicted.
// ============================================================================
use bytesize::ByteSize;
use log::{debug, error, info, warn};
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, Weak};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::block::{
    BlockHash, BlockInsertError, BlockKey, InflightBlock, LayerBlock, PrefetchStatus, SealedBlock,
};
use crate::cache::TinyLfuCache;
use crate::metrics::core_metrics;
use crate::pinned_pool::{PinnedAllocation, PinnedMemoryPool};
use crate::ssd_cache::{
    ssd_prefetch_loop, ssd_writer_loop, PreallocatedSlice, PrefetchBatch, PrefetchRequest,
    SsdCacheConfig, SsdIndexEntry, SsdStorageHandle, SsdWriteBatch,
};

// ============================================================================
// Constants
// ============================================================================

/// Number of LRU blocks to evict per iteration when reclaiming memory
const RECLAIM_BATCH_SIZE: usize = 64;

/// SSD I/O alignment requirement (O_DIRECT requires 512-byte aligned I/O)
pub const SSD_ALIGNMENT: usize = 512;

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity)
    pub hint_value_size_bytes: Option<usize>,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
            ssd_cache_config: None,
        }
    }
}

// A "slot" in this file refers to a specific position in the flattened logical storage,
// calculated as `layer_id * tp_size + tp_rank`.
// vLLM/Connectors report the total topology (layers * tp_size) via registration,
// and this count is immutable for the lifetime of the Instance.

// ============================================================================
// StorageEngine
// ============================================================================

/// Notification sent when a block is sealed (for SSD offload, etc.)
pub type SealNotification = (BlockKey, Weak<SealedBlock>);

/// SSD cache runtime state (file + io_uring + channels)
struct SsdState {
    config: SsdCacheConfig,
    /// RAII guard: keeps the file descriptor alive for io_uring operations.
    /// UringIoEngine only holds the raw fd, so dropping this would invalidate all IO.
    _file: Arc<std::fs::File>,
    io: Arc<crate::uring::UringIoEngine>,
    /// Channel to SSD writer task (bounded to limit queue depth)
    writer_tx: tokio::sync::mpsc::Sender<SsdWriteBatch>,
    /// Channel to prefetch worker (bounded to limit queue depth)
    prefetch_tx: tokio::sync::mpsc::Sender<PrefetchBatch>,
    /// Logical head pointer (next write position)
    head: std::sync::atomic::AtomicU64,
}

/// Receivers for SSD workers (separated so they can be moved to workers)
struct SsdReceivers {
    writer_rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
    prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
}

/// Inner state protected by a single mutex
struct StorageInner {
    /// Write path: blocks being filled (not yet sealed)
    inflight: HashMap<BlockKey, InflightBlock>,
    /// Blocks currently being prefetched from SSD
    prefetching: HashSet<BlockKey>,
    /// Read path: sealed blocks available for lookup (TinyLFU admission + LRU eviction)
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    /// SSD cache index (moved from SsdCache for unified locking)
    ssd_index: HashMap<BlockKey, SsdIndexEntry>,
    /// Logical tail pointer for SSD ring buffer eviction
    ssd_tail: u64,
    /// Pinned blocks between query and load (prevents eviction race)
    /// Key: (instance_id, block_key), Value: (block, ref_count)
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
}

pub struct StorageEngine {
    /// Pinned memory allocator
    pinned_pool: Arc<PinnedMemoryPool>,

    /// All mutable state under one lock
    inner: Mutex<StorageInner>,

    /// Channel to notify consumers when blocks are sealed (for SSD offload)
    seal_notify_tx: Option<UnboundedSender<SealNotification>>,

    /// SSD cache file handle and io_uring engine (if configured)
    ssd_state: Option<SsdState>,
}

impl StorageEngine {
    /// Create a new StorageEngine with optional seal notification channel.
    /// Returns (engine, receiver) where receiver gets notified of sealed blocks.
    pub fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: impl Into<StorageConfig>,
    ) -> (Arc<Self>, UnboundedReceiver<SealNotification>) {
        let config = config.into();
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let pinned_pool = Arc::new(PinnedMemoryPool::new(
            capacity_bytes,
            use_hugepages,
            value_size_hint.and_then(|size| NonZeroU64::new(size as u64)),
        ));

        let cache = TinyLfuCache::new_unbounded(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
        );

        let inner = Mutex::new(StorageInner {
            inflight: HashMap::new(),
            prefetching: HashSet::new(),
            cache,
            ssd_index: HashMap::new(),
            ssd_tail: 0,
            pinned_for_load: HashMap::new(),
        });

        // Create unbounded channel for seal notifications
        let (seal_notify_tx, seal_notify_rx) = mpsc::unbounded_channel();

        // Initialize SSD cache if configured (file + io_uring + channels)
        let (ssd_state, ssd_receivers) = match config.ssd_cache_config {
            Some(ssd_cfg) => match Self::init_ssd_state(ssd_cfg) {
                Ok((state, receivers)) => (Some(state), Some(receivers)),
                Err(e) => {
                    error!("Failed to initialize SSD cache: {}", e);
                    (None, None)
                }
            },
            None => (None, None),
        };

        let engine = Arc::new(Self {
            pinned_pool,
            inner,
            seal_notify_tx: Some(seal_notify_tx),
            ssd_state,
        });

        // Spawn SSD workers after Arc is created (they need callbacks into storage)
        if let Some(receivers) = ssd_receivers {
            if let Some(handle) = Self::make_ssd_handle(&engine) {
                Self::spawn_ssd_workers(&engine, handle, receivers);
            } else {
                warn!("SSD cache configured but ssd_state missing; skipping workers");
            }
        }

        (engine, seal_notify_rx)
    }

    /// Initialize SSD cache state (file + io_uring + channels, no workers yet)
    fn init_ssd_state(config: SsdCacheConfig) -> std::io::Result<(SsdState, SsdReceivers)> {
        use std::fs::{self, OpenOptions};
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::io::AsRawFd;

        if let Some(parent) = config.cache_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(&config.cache_path)?;
        file.set_len(config.capacity_bytes)?;

        let io = Arc::new(crate::uring::UringIoEngine::new(
            file.as_raw_fd(),
            crate::uring::UringConfig::default(),
        )?);

        let (writer_tx, writer_rx) = tokio::sync::mpsc::channel(config.write_queue_depth);
        let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(config.prefetch_queue_depth);

        info!(
            "SSD cache initialized at {} (capacity {})",
            config.cache_path.display(),
            ByteSize(config.capacity_bytes)
        );

        let state = SsdState {
            config,
            _file: Arc::new(file),
            io,
            writer_tx,
            prefetch_tx,
            head: std::sync::atomic::AtomicU64::new(0),
        };

        let receivers = SsdReceivers {
            writer_rx,
            prefetch_rx,
        };

        Ok((state, receivers))
    }

    /// Spawn SSD writer thread and prefetch worker (requires Arc<Self>)
    fn spawn_ssd_workers(
        engine: &Arc<Self>,
        handle: Arc<SsdStorageHandle>,
        receivers: SsdReceivers,
    ) {
        let SsdReceivers {
            writer_rx,
            prefetch_rx,
        } = receivers;

        // Get references to SSD state
        let ssd_state = engine.ssd_state.as_ref().expect("ssd_state must exist");
        let io = Arc::clone(&ssd_state.io);
        let capacity = ssd_state.config.capacity_bytes;

        // Spawn writer task
        let writer_handle = Arc::clone(&handle);
        let writer_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_writer_loop(writer_handle, writer_rx, writer_io, capacity).await;
        });

        // Spawn prefetch task
        let prefetch_handle = Arc::clone(&handle);
        let prefetch_io = Arc::clone(&ssd_state.io);
        let prefetch_capacity = capacity;
        tokio::spawn(async move {
            ssd_prefetch_loop(prefetch_handle, prefetch_rx, prefetch_io, prefetch_capacity).await;
        });

        debug!("SSD workers spawned");
    }

    /// Build the SSD storage handle capturing a weak pointer to StorageEngine.
    fn make_ssd_handle(engine: &Arc<Self>) -> Option<Arc<SsdStorageHandle>> {
        engine.ssd_state.as_ref()?;

        let weak = Arc::downgrade(engine);
        let weak_prune = Weak::clone(&weak);
        let weak_publish = Weak::clone(&weak);
        let weak_complete = Weak::clone(&weak);
        let weak_valid = Weak::clone(&weak);

        Some(Arc::new(SsdStorageHandle::new(
            move |new_tail| {
                if let Some(engine) = weak_prune.upgrade() {
                    engine.ssd_prune_tail(new_tail);
                }
            },
            move |key, entry, new_head| {
                if let Some(engine) = weak_publish.upgrade() {
                    engine.ssd_publish_write(key, entry, new_head);
                }
            },
            move |key, block| {
                if let Some(engine) = weak_complete.upgrade() {
                    engine.complete_prefetch(key, block);
                }
            },
            move |begin| {
                weak_valid
                    .upgrade()
                    .map(|engine| engine.is_ssd_offset_valid(begin))
                    .unwrap_or(false)
            },
        )))
    }

    /// Returns true if SSD cache is enabled.
    pub fn is_ssd_enabled(&self) -> bool {
        self.ssd_state.is_some()
    }

    // ========================================================================
    // Allocation
    // ========================================================================

    pub fn allocate(&self, size: NonZeroU64) -> Option<Arc<PinnedAllocation>> {
        let requested_bytes = size.get();

        loop {
            if let Some(allocation) = self.pinned_pool.allocate(size) {
                return Some(Arc::new(allocation));
            }

            let (freed_blocks, freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes);

            if largest_free >= requested_bytes {
                continue;
            }

            let (used, total) = self.pinned_pool.usage();
            error!(
                "Pinned memory pool exhausted; cannot satisfy allocation: requested={} used={} total={} largest_free={} freed_blocks={} freed_bytes={}",
                ByteSize(requested_bytes),
                ByteSize(used),
                ByteSize(total),
                ByteSize(largest_free),
                freed_blocks,
                ByteSize(freed_bytes)
            );
            core_metrics().pool_alloc_failures.add(1, &[]);
            return None;
        }
    }

    // ========================================================================
    // Write path (inflight)
    // ========================================================================

    /// Filter blocks that need to be saved (not in cache and slot not in inflight).
    /// Returns indices of blocks that need saving. Single lock acquisition.
    pub fn filter_blocks_to_save(
        &self,
        namespace: &str,
        block_hashes: &[Vec<u8>],
        slot_id: usize,
    ) -> Vec<usize> {
        let mut inner = self.inner.lock().unwrap();
        block_hashes
            .iter()
            .enumerate()
            .filter(|(_, hash)| {
                let key = BlockKey::new(namespace.to_string(), hash.to_vec());
                // Skip if already sealed in cache
                if inner.cache.get(&key).is_some() {
                    return false;
                }
                // Skip if slot already exists in inflight
                if let Some(block) = inner.inflight.get(&key) {
                    if block.slot_exists(slot_id) {
                        return false;
                    }
                }
                true
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Insert a slot into a block. Handles:
    /// - Skip if already in cache (sealed)
    /// - Skip if slot already exists in inflight
    /// - Create inflight block if needed
    /// - Seal when complete (but does NOT auto-admit to cache)
    ///
    /// Returns:
    /// - `Ok(None)` if slot was skipped (already exists) or block not yet complete
    /// - `Ok(Some((key, block)))` if block just completed and was sealed
    ///
    /// Caller is responsible for calling `cache_admit` to insert into cache.
    pub fn insert_slot(
        &self,
        namespace: &str,
        block_hash: BlockHash,
        slot_id: usize,
        block: Arc<LayerBlock>,
        total_slots: usize,
    ) -> Result<Option<(BlockKey, Arc<SealedBlock>)>, BlockInsertError> {
        let key = BlockKey::new(namespace.to_string(), block_hash);

        let mut inner = self.inner.lock().unwrap();

        // Fast path: already sealed in cache
        if inner.cache.contains_key(&key) {
            return Ok(None);
        }

        // Get or create inflight block
        let inflight_block = match inner.inflight.entry(key.clone()) {
            Entry::Vacant(v) => {
                core_metrics().inflight_blocks.add(1, &[]);
                v.insert(InflightBlock::new(total_slots))
            }
            Entry::Occupied(o) => o.into_mut(),
        };

        // Check if slot already exists
        if inflight_block.slot_exists(slot_id) {
            return Ok(None);
        }

        let completed = inflight_block.insert_slot(slot_id, block, total_slots)?;

        if completed {
            // Remove from inflight and seal
            let inflight_block = inner.inflight.remove(&key).expect("just checked");
            core_metrics().inflight_blocks.add(-1, &[]);
            let sealed = Arc::new(inflight_block.seal());

            // Notify external consumers (fire-and-forget)
            drop(inner); // Release lock before sending to channel
            if let Some(tx) = &self.seal_notify_tx {
                let _ = tx.send((key.clone(), Arc::downgrade(&sealed)));
            }

            return Ok(Some((key, sealed)));
        }

        Ok(None)
    }

    /// Attempt to admit a sealed block into cache using TinyLFU policy.
    /// Returns true if admitted, false if rejected.
    pub fn cache_admit(&self, key: BlockKey, block: Arc<SealedBlock>) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.cache.insert(key, block) {
            core_metrics().cache_block_insertions.add(1, &[]);
            true
        } else {
            core_metrics().cache_block_admission_rejections.add(1, &[]);
            false
        }
    }

    /// Send a batch of sealed blocks to SSD writer for async persistence.
    /// Called after sealing a batch of blocks from seal_offload.
    /// Drops the batch if write queue is full (backpressure).
    pub fn send_ssd_batch(&self, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
        let Some(ref ssd_state) = self.ssd_state else {
            return;
        };

        if blocks.is_empty() {
            return;
        }

        let batch = SsdWriteBatch {
            blocks: blocks
                .iter()
                .map(|(k, b)| (k.clone(), Arc::downgrade(b)))
                .collect(),
        };

        if ssd_state.writer_tx.try_send(batch).is_ok() {
            core_metrics()
                .ssd_write_queue_pending
                .add(blocks.len() as i64, &[]);
        } else {
            // Queue full - drop the batch (backpressure)
            warn!("SSD write queue full, dropping {} blocks", blocks.len());
            core_metrics()
                .ssd_write_queue_full
                .add(blocks.len() as u64, &[]);
        }
    }

    // ========================================================================
    // Read path (cache)
    // ========================================================================

    /// Lookup multiple blocks for load operation.
    /// Consumes pinned blocks (removes from pinned_for_load).
    pub fn cache_lookup_many(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> Result<Vec<Arc<SealedBlock>>, String> {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();

        let mut inner = self.inner.lock().unwrap();
        let mut result: Vec<Arc<SealedBlock>> = Vec::with_capacity(keys.len());

        for (idx, key) in keys.into_iter().enumerate() {
            let pin_key = (instance_id.to_string(), key.clone());

            // Consume pinned_for_load (ref_count -1, remove if 0)
            if let Some((block, count)) = inner.pinned_for_load.get_mut(&pin_key) {
                let cloned = Arc::clone(block);
                *count -= 1;
                if *count == 0 {
                    inner.pinned_for_load.remove(&pin_key);
                }
                result.push(cloned);
            } else {
                error!(
                    "missing pinned KV block: instance={} idx={} hash_len={}",
                    instance_id,
                    idx,
                    key.hash.len()
                );
                return Err(format!(
                    "missing pinned KV block at index {} (namespace={}, hash_len={})",
                    idx,
                    key.namespace,
                    key.hash.len()
                ));
            }
        }

        Ok(result)
    }

    // ========================================================================
    // Eviction (cache only)
    // ========================================================================

    fn reclaim_until_allocator_can_allocate(&self, required_bytes: u64) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (0, 0, self.pinned_pool.largest_free_allocation());
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.pinned_pool.largest_free_allocation();

        while largest_free < required_bytes {
            // Collect evicted blocks under lock, then drop outside lock
            let evicted: Vec<_> = {
                let mut inner = self.inner.lock().unwrap();
                (0..RECLAIM_BATCH_SIZE)
                    .map_while(|_| inner.cache.remove_lru())
                    .collect()
            };

            if evicted.is_empty() {
                break;
            }

            for (_key, block) in evicted {
                freed_bytes = freed_bytes.saturating_add(block.memory_footprint());
                freed_blocks += 1;
            }

            largest_free = self.pinned_pool.largest_free_allocation();
        }

        if freed_blocks > 0 {
            debug!(
                "Reclaimed cache blocks toward allocator request: freed_blocks={} freed_bytes={} largest_free={} required={}",
                freed_blocks,
                ByteSize(freed_bytes),
                ByteSize(largest_free),
                ByteSize(required_bytes)
            );
            core_metrics()
                .cache_block_evictions
                .add(freed_blocks as u64, &[]);
        }

        (freed_blocks, freed_bytes, largest_free)
    }

    /// Remove stale inflight blocks that have been stuck for longer than `max_age`.
    ///
    /// This is a safety net for rare race conditions where partial blocks can never
    /// complete (e.g., cache eviction between layer checks). In normal operation,
    /// this should clean very few or zero blocks.
    ///
    /// Returns the number of cleaned blocks.
    pub fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let before = inner.inflight.len();

        inner.inflight.retain(|key, block| {
            let age = block.age();
            if age > max_age {
                warn!(
                    "GC: removing stale inflight block: namespace={} hash_len={} filled={} total={} age_secs={}",
                    key.namespace,
                    key.hash.len(),
                    block.filled_count(),
                    block.total_slots(),
                    age.as_secs()
                );
                // Decrement the inflight gauge for each removed block
                core_metrics().inflight_blocks.add(-1, &[]);
                false
            } else {
                true
            }
        });

        let cleaned = before - inner.inflight.len();
        if cleaned > 0 {
            core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
            info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
        }
        cleaned
    }

    /// Check prefix blocks and trigger prefetch for blocks in SSD.
    /// Returns status indicating whether caller should retry.
    /// Hit blocks are pinned to prevent eviction before load.
    pub fn check_prefix_and_prefetch(
        &self,
        instance_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
    ) -> PrefetchStatus {
        let mut hit = 0usize;
        let mut loading = 0usize;
        let mut missing = 0usize;
        let mut to_prefetch: Vec<BlockKey> = Vec::new();

        {
            let mut inner = self.inner.lock().unwrap();

            let mut cpu_hit_keys: Vec<BlockKey> = Vec::new();

            for hash in hashes {
                let key = BlockKey::new(namespace.to_string(), hash.clone());

                // Then check cache
                if inner.cache.get(&key).is_some() {
                    hit += 1;
                    cpu_hit_keys.push(key);
                    continue;
                }

                if inner.prefetching.contains(&key) {
                    loading += 1;
                    continue;
                }

                // Check SSD index
                if let Some(entry) = inner.ssd_index.get(&key) {
                    if inner.ssd_tail <= entry.begin {
                        // Block is in SSD, schedule prefetch
                        to_prefetch.push(key);
                        loading += 1;
                        continue;
                    }
                }

                // Block not found anywhere - this is a miss
                // For prefix matching, first miss means remaining blocks are also missing
                missing = hashes.len() - hit - loading;
                break;
            }

            // Pin hit blocks when returning Done (no loading in progress)
            if loading == 0 {
                for key in cpu_hit_keys {
                    let pin_key = (instance_id.to_string(), key.clone());
                    let block = inner.cache.get(&key).unwrap();
                    inner
                        .pinned_for_load
                        .entry(pin_key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert_with(|| (Arc::clone(&block), 1));
                }
            }
        }

        // Trigger prefetch for blocks in SSD (outside lock)
        if !to_prefetch.is_empty() {
            self.trigger_prefetch(to_prefetch);
        }

        if loading > 0 {
            PrefetchStatus::Loading { hit, loading }
        } else {
            PrefetchStatus::Done { hit, missing }
        }
    }

    /// Mark blocks as prefetching and send batch to prefetch worker.
    /// Pre-allocates a contiguous memory region for all blocks to ensure
    /// CPU memory locality for subsequent GPU transfers.
    /// If prefetch queue is full, drops the request (treats as cache miss).
    fn trigger_prefetch(&self, keys: Vec<BlockKey>) {
        let ssd_state = match &self.ssd_state {
            Some(state) => state,
            None => return,
        };

        // Collect valid entries and calculate total size
        let mut valid_requests: Vec<(BlockKey, SsdIndexEntry)> = Vec::with_capacity(keys.len());
        let mut total_size: u64 = 0;

        {
            let mut inner = self.inner.lock().unwrap();

            for key in keys {
                // Skip if already prefetching
                if inner.prefetching.contains(&key) {
                    continue;
                }

                // Get SSD index entry
                let entry = match inner.ssd_index.get(&key) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                // Check not evicted
                if inner.ssd_tail > entry.begin {
                    continue;
                }

                total_size = total_size.saturating_add(entry.len);
                valid_requests.push((key, entry));
            }

            // Mark all as prefetching before releasing lock
            for (key, _) in &valid_requests {
                inner.prefetching.insert(key.clone());
            }
        }

        if valid_requests.is_empty() {
            return;
        }

        // Allocate contiguous memory for all blocks
        let Some(alloc_size) = NonZeroU64::new(total_size) else {
            warn!("SSD prefetch: total size is zero");
            return;
        };

        let Some(allocation) = self.allocate(alloc_size) else {
            warn!(
                "SSD prefetch: failed to allocate {} for {} blocks",
                ByteSize(total_size),
                valid_requests.len()
            );
            // Remove from prefetching set on failure
            let mut inner = self.inner.lock().unwrap();
            for (key, _) in &valid_requests {
                inner.prefetching.remove(key);
            }
            return;
        };

        // Build batch with pre-allocated slices
        let mut requests = Vec::with_capacity(valid_requests.len());
        let mut offset: usize = 0;
        let keys_for_cleanup: Vec<_> = valid_requests.iter().map(|(k, _)| k.clone()).collect();

        for (key, entry) in valid_requests {
            let slice = PreallocatedSlice {
                allocation: Arc::clone(&allocation),
                offset,
            };
            offset += entry.len as usize;

            requests.push(PrefetchRequest {
                key,
                entry,
                preallocated: slice,
            });
        }

        // Send batch (non-blocking, drop if queue full)
        let batch = PrefetchBatch { requests };
        if ssd_state.prefetch_tx.try_send(batch).is_err() {
            // Queue full - treat as cache miss, clean up prefetching set
            warn!(
                "SSD prefetch queue full, dropping {} blocks",
                keys_for_cleanup.len()
            );
            core_metrics()
                .ssd_prefetch_queue_full
                .add(keys_for_cleanup.len() as u64, &[]);
            let mut inner = self.inner.lock().unwrap();
            for key in keys_for_cleanup {
                inner.prefetching.remove(&key);
            }
        }
    }

    /// Called by prefetch worker when a block is loaded from SSD.
    pub fn complete_prefetch(&self, key: BlockKey, block: Option<Arc<SealedBlock>>) {
        let mut inner = self.inner.lock().unwrap();
        inner.prefetching.remove(&key);

        if let Some(block) = block {
            if inner.cache.insert(key, block) {
                core_metrics().cache_block_insertions.add(1, &[]);
            }
        }
    }

    /// Update SSD tail pointer and evict stale index entries.
    /// Called by SSD writer thread before overwriting the ring buffer.
    pub(crate) fn ssd_prune_tail(&self, new_tail: u64) {
        let mut inner = self.inner.lock().unwrap();
        if new_tail <= inner.ssd_tail {
            return;
        }
        inner.ssd_tail = new_tail;

        // Remove evicted entries from index
        inner.ssd_index.retain(|_, entry| new_tail <= entry.begin);
    }

    /// Check if a logical SSD offset is still valid (not yet overwritten).
    pub(crate) fn is_ssd_offset_valid(&self, begin: u64) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.ssd_tail <= begin
    }

    /// Publish a completed SSD write by updating the index and head pointer.
    pub(crate) fn ssd_publish_write(&self, key: BlockKey, entry: SsdIndexEntry, new_head: u64) {
        let mut inner = self.inner.lock().unwrap();
        inner.ssd_index.insert(key, entry);

        if let Some(ref ssd_state) = self.ssd_state {
            ssd_state
                .head
                .store(new_head, std::sync::atomic::Ordering::Release);
        }
    }
}
