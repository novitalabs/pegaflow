// ============================================================================
// StorageEngine: Two-phase block storage with separate write and read paths.
//
// Lifecycle: Allocate → Write (inflight) → Seal → Cache (read-only) → Evict
//
// Key invariant: Sealing is a one-way gate. Once sealed, a block is immutable.
//
// Architecture:
// - Mutex<StorageInner>: prefetching, cache, pinned state
// - Insert Worker (dedicated thread): owns inflight HashMap, receives
//   RawSaveBatch messages via channel, builds LayerBlocks and seals completed blocks
// - RwLock<SsdRingBuffer>: SSD head/tail/index (shared by writer and readers)
// - Allocator: PinnedMemoryPool for pinned memory allocation
// - Prefetch worker: background io_uring reads from SSD
//
// Eviction only targets the cache; inflight blocks are never evicted.
// ============================================================================
use bytesize::ByteSize;
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::num::NonZeroU64;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Weak};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::sync::oneshot;

use pegaflow_proto::proto::engine::InsertBlockHashesRequest;
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use tonic::transport::Channel;

use crate::block::{BlockKey, InflightBlock, PrefetchStatus, SealedBlock, SlotInsertResult};
use crate::cache::{CacheInsertOutcome, TinyLfuCache};
use crate::metrics::core_metrics;
use crate::numa::NumaNode;
use crate::offload::InsertEntries;
use crate::pinned_pool::{PinnedAllocation, PinnedAllocator};
use crate::ssd_cache::{
    PrefetchBatch, PrefetchRequest, SsdCacheConfig, SsdIndexEntry, SsdRingBuffer, SsdStorageHandle,
    SsdWriteBatch, ssd_prefetch_loop, ssd_writer_loop,
};

// ============================================================================
// Constants
// ============================================================================

/// Number of LRU blocks to evict per iteration when reclaiming memory
const RECLAIM_BATCH_SIZE: usize = 64;

/// SSD I/O alignment requirement (O_DIRECT requires 512-byte aligned I/O)
pub const SSD_ALIGNMENT: usize = 512;

/// Max blocks allowed in prefetching state (backpressure for SSD prefetch)
/// ~15GB assuming 10MB per block
const MAX_PREFETCH_BLOCKS: usize = 1500;

// ============================================================================
// Metrics helpers (keep insert/evict logic together for easy audit)
// ============================================================================

/// Records metrics when bytes are added to inflight blocks.
fn record_inflight_bytes_added(bytes: u64) {
    if let Ok(v) = i64::try_from(bytes) {
        core_metrics().inflight_bytes.add(v, &[]);
    }
}

/// Records metrics when bytes are removed from inflight blocks (seal or gc).
fn record_inflight_bytes_removed(bytes: u64) {
    if let Ok(v) = i64::try_from(bytes) {
        core_metrics().inflight_bytes.add(-v, &[]);
    }
}

/// Records metrics for a new cache insertion.
fn record_cache_insert_new(footprint_bytes: u64) {
    let m = core_metrics();
    m.cache_block_insertions.add(1, &[]);
    if let Ok(v) = i64::try_from(footprint_bytes) {
        m.cache_resident_bytes.add(v, &[]);
    }
}

/// Records metrics for a cache eviction.
fn record_cache_eviction(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().cache_resident_bytes.add(-v, &[]);
    }
}

/// Records metrics when a new unique block is pinned.
fn record_pin_unique_added(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().pinned_for_load_unique_bytes.add(v, &[]);
    }
}

/// Records metrics when the last reference to a unique block is unpinned.
fn record_pin_unique_removed(footprint_bytes: u64) {
    if let Ok(v) = i64::try_from(footprint_bytes) {
        core_metrics().pinned_for_load_unique_bytes.add(-v, &[]);
    }
}

/// Configuration for cache + storage behavior.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub enable_lfu_admission: bool,
    /// Optional hint for expected value size in bytes (tunes cache + allocator granularity)
    pub hint_value_size_bytes: Option<usize>,
    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch).
    /// ~15GB assuming 10MB per block.
    pub max_prefetch_blocks: usize,
    /// Optional SSD cache for sealed blocks (single-node, FIFO).
    pub ssd_cache_config: Option<SsdCacheConfig>,
    /// Enable NUMA-aware memory allocation. When true on multi-NUMA systems,
    /// PegaEngine auto-detects topology and creates per-node pinned pools.
    pub enable_numa_affinity: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
            max_prefetch_blocks: MAX_PREFETCH_BLOCKS,
            ssd_cache_config: None,
            enable_numa_affinity: true,
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
}

/// Receivers for SSD workers (separated so they can be moved to workers)
struct SsdReceivers {
    writer_rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
    prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
}

// ============================================================================
// Insert Worker (actor model for inflight block management)
// ============================================================================

/// Command sent to the insert worker.
enum InsertWorkerCommand {
    /// Deferred save: build LayerBlocks + insert into inflight.
    RawInsert(crate::offload::RawSaveBatch),
    /// GC stale inflight blocks older than max_age.
    Gc {
        max_age: std::time::Duration,
        reply: oneshot::Sender<usize>,
    },
}

/// Inner state protected by a single mutex (no longer contains inflight)
struct StorageInner {
    /// Blocks currently being prefetched from SSD
    prefetching: HashSet<BlockKey>,
    /// Read path: sealed blocks available for lookup (TinyLFU admission + LRU eviction)
    cache: TinyLfuCache<BlockKey, Arc<SealedBlock>>,
    /// Pinned blocks between query and load (prevents eviction race)
    /// Key: (instance_id, block_key), Value: (block, ref_count)
    pinned_for_load: HashMap<(String, BlockKey), (Arc<SealedBlock>, usize)>,
    /// Aggregated pinned_for_load refcounts by block key (for attribution metrics).
    /// Value: (footprint_bytes, total_refcount)
    pinned_for_load_by_key: HashMap<BlockKey, (u64, usize)>,
    /// SSD ring buffer: unified head/tail/index (managed via SsdStorageHandle callbacks)
    ssd_ring: Option<SsdRingBuffer>,
}

pub struct StorageEngine {
    /// Unified pinned memory allocator (handles both global and NUMA modes)
    allocator: Arc<PinnedAllocator>,

    /// Mutable state under one lock (cache, prefetching, pinned_for_load)
    inner: Mutex<StorageInner>,

    /// Channel to the insert worker thread (owns inflight HashMap)
    insert_tx: Sender<InsertWorkerCommand>,

    /// Channel to notify consumers when blocks are sealed (for SSD offload)
    seal_notify_tx: Option<UnboundedSender<SealNotification>>,

    /// SSD cache file handle and io_uring engine (if configured)
    ssd_state: Option<SsdState>,

    /// Max blocks allowed in prefetching state (backpressure for SSD prefetch)
    max_prefetch_blocks: usize,

    /// Channel to the metaserver insert worker (set by `PegaEngine::set_metaserver_client`)
    metaserver_tx: Mutex<Option<UnboundedSender<crate::MetaserverInsertCmd>>>,
}

impl StorageEngine {
    /// Create a new StorageEngine with optional seal notification channel.
    /// Returns (engine, receiver) where receiver gets notified of sealed blocks.
    ///
    /// # Arguments
    /// * `capacity_bytes` - Total pinned memory pool capacity
    /// * `use_hugepages` - Use 2MB huge pages for allocation
    /// * `config` - Storage behavior configuration
    /// * `numa_nodes` - NUMA nodes for per-node pools (empty = single global pool)
    pub fn new_with_config(
        capacity_bytes: usize,
        use_hugepages: bool,
        config: impl Into<StorageConfig>,
        numa_nodes: &[NumaNode],
    ) -> (Arc<Self>, UnboundedReceiver<SealNotification>) {
        let config = config.into();
        let value_size_hint = config.hint_value_size_bytes.filter(|size| *size > 0);
        let unit_hint = value_size_hint.and_then(|size| NonZeroU64::new(size as u64));

        // Create unified allocator based on NUMA configuration
        let allocator = if !numa_nodes.is_empty() {
            info!(
                "Creating NUMA-aware pinned pools for {} nodes",
                numa_nodes.len()
            );
            Arc::new(PinnedAllocator::new_numa(
                capacity_bytes,
                numa_nodes,
                use_hugepages,
                unit_hint,
            ))
        } else {
            info!("Creating global pinned pool (NUMA affinity disabled)");
            Arc::new(PinnedAllocator::new_global(
                capacity_bytes,
                use_hugepages,
                unit_hint,
            ))
        };

        let cache = TinyLfuCache::new_unbounded(
            capacity_bytes,
            config.enable_lfu_admission,
            value_size_hint,
        );

        // Initialize SSD cache if configured (file + io_uring + channels)
        let (ssd_state, ssd_receivers, ssd_ring) = match config.ssd_cache_config {
            Some(ssd_cfg) => {
                let capacity = ssd_cfg.capacity_bytes;
                match Self::init_ssd_state(ssd_cfg) {
                    Ok((state, receivers)) => {
                        let ring = SsdRingBuffer::new(capacity);
                        (Some(state), Some(receivers), Some(ring))
                    }
                    Err(e) => {
                        error!("Failed to initialize SSD cache: {}", e);
                        (None, None, None)
                    }
                }
            }
            None => (None, None, None),
        };

        let inner = Mutex::new(StorageInner {
            prefetching: HashSet::new(),
            cache,
            pinned_for_load: HashMap::new(),
            pinned_for_load_by_key: HashMap::new(),
            ssd_ring,
        });

        // Create unbounded channel for seal notifications
        let (seal_notify_tx, seal_notify_rx) = mpsc::unbounded_channel();

        // Create insert worker channel (std::sync::mpsc — worker is a dedicated OS thread)
        let (insert_tx, insert_rx) = std::sync::mpsc::channel();

        let engine = Arc::new(Self {
            allocator,
            inner,
            insert_tx,
            seal_notify_tx: Some(seal_notify_tx),
            ssd_state,
            max_prefetch_blocks: config.max_prefetch_blocks,
            metaserver_tx: Mutex::new(None),
        });

        // Spawn insert worker on a dedicated OS thread (CPU-bound work)
        {
            let weak_engine = Arc::downgrade(&engine);
            std::thread::Builder::new()
                .name("pegaflow-insert".into())
                .spawn(move || insert_worker_loop(insert_rx, weak_engine))
                .expect("failed to spawn insert worker thread");
        }

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

    /// Set the MetaServer client for cross-node block hash registry.
    ///
    /// Spawns a background worker that batches and sends insert requests.
    pub(crate) fn set_metaserver_client(
        &self,
        client: MetaServerClient<Channel>,
        node_url: String,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(metaserver_worker_loop(rx, client, node_url.clone()));
        *self.metaserver_tx.lock() = Some(tx);
        info!(
            "MetaServer client configured for block hash registry (node_url={})",
            node_url
        );
    }

    /// Send block hashes to the metaserver insert worker (fire-and-forget).
    pub(crate) fn send_metaserver_insert(&self, namespace: String, block_hashes: Vec<Vec<u8>>) {
        if let Some(tx) = self.metaserver_tx.lock().as_ref() {
            let _ = tx.send(crate::MetaserverInsertCmd {
                namespace,
                block_hashes,
            });
        }
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
        let write_inflight = ssd_state.config.write_inflight;
        let prefetch_inflight = ssd_state.config.prefetch_inflight;

        // Spawn writer task
        let writer_handle = Arc::clone(&handle);
        let writer_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_writer_loop(writer_handle, writer_rx, writer_io, write_inflight).await;
        });

        // Spawn prefetch task
        let prefetch_handle = Arc::clone(&handle);
        let prefetch_io = Arc::clone(&ssd_state.io);
        let prefetch_capacity = capacity;
        tokio::spawn(async move {
            ssd_prefetch_loop(
                prefetch_handle,
                prefetch_rx,
                prefetch_io,
                prefetch_capacity,
                prefetch_inflight,
            )
            .await;
        });

        debug!("SSD workers spawned");
    }

    /// Build the SSD storage handle capturing a weak pointer to StorageEngine.
    /// Used by both writer (prepare, commit) and prefetch worker.
    fn make_ssd_handle(engine: &Arc<Self>) -> Option<Arc<SsdStorageHandle>> {
        engine.ssd_state.as_ref()?;

        let weak_complete = Arc::downgrade(engine);
        let weak_valid = Arc::downgrade(engine);
        let weak_alloc = Arc::downgrade(engine);
        let weak_prepare = Arc::downgrade(engine);
        let weak_commit = Arc::downgrade(engine);

        Some(Arc::new(SsdStorageHandle::new(
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
            move |size, numa_node| {
                weak_alloc
                    .upgrade()
                    .and_then(|engine| engine.allocate(NonZeroU64::new(size)?, numa_node))
            },
            move |candidates| {
                weak_prepare
                    .upgrade()
                    .map(|engine| {
                        let mut inner = engine.inner.lock();
                        inner
                            .ssd_ring
                            .as_mut()
                            .map(|ring| ring.prepare_batch(candidates))
                            .unwrap_or_else(crate::ssd_cache::PreparedBatch::empty)
                    })
                    .unwrap_or_else(crate::ssd_cache::PreparedBatch::empty)
            },
            move |key, success| {
                if let Some(engine) = weak_commit.upgrade() {
                    let mut inner = engine.inner.lock();
                    if let Some(ring) = inner.ssd_ring.as_mut() {
                        ring.commit(key, success);
                    }
                }
            },
            engine.is_numa_enabled(),
        )))
    }

    /// Returns true if SSD cache is enabled.
    pub fn is_ssd_enabled(&self) -> bool {
        self.ssd_state.is_some()
    }

    /// Returns true if NUMA-aware allocation is enabled.
    pub fn is_numa_enabled(&self) -> bool {
        self.allocator.is_numa()
    }

    // ========================================================================
    // Allocation
    // ========================================================================

    /// Allocate pinned memory, optionally from a specific NUMA node's pool.
    ///
    /// If `numa_node` is `Some` and NUMA pools are configured, allocates from
    /// that NUMA node's pool. Otherwise uses the global pool.
    ///
    /// Returns `None` if the pool is exhausted after eviction attempts.
    pub fn allocate(
        &self,
        size: NonZeroU64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        let requested_bytes = size.get();
        let node = numa_node.unwrap_or(NumaNode::UNKNOWN);

        loop {
            // Try to allocate from the unified allocator
            if let Some(alloc) = self.allocator.allocate(size, node) {
                return Some(alloc);
            }

            // Allocation failed, try to reclaim memory
            let (freed_blocks, freed_bytes, largest_free) =
                self.reclaim_until_allocator_can_allocate(requested_bytes);

            if largest_free >= requested_bytes {
                continue;
            }

            // Still can't allocate, report error
            let (used, total) = self.allocator.usage();
            error!(
                "Pinned memory pool exhausted; cannot satisfy allocation: requested={} used={} total={} largest_free={} freed_blocks={} freed_bytes={} numa={:?}",
                ByteSize(requested_bytes),
                ByteSize(used),
                ByteSize(total),
                ByteSize(largest_free),
                freed_blocks,
                ByteSize(freed_bytes),
                numa_node
            );
            core_metrics().pool_alloc_failures.add(1, &[]);
            return None;
        }
    }

    /// Get aggregate pool usage: (used_bytes, total_bytes)
    fn pool_usage(&self) -> (u64, u64) {
        self.allocator.usage()
    }

    /// Get largest free allocation across all pools.
    fn largest_free_allocation(&self) -> u64 {
        self.allocator.largest_free_allocation()
    }

    // ========================================================================
    // Write path (inflight)
    // ========================================================================

    /// Fire-and-forget raw insert: send a deferred save batch to the insert worker.
    ///
    /// The worker builds `LayerBlock` objects, groups by hash, and inserts
    /// into inflight. The caller does NOT need to wait — once GPU→CPU copy
    /// is done the pinned memory is reference-counted.
    pub(crate) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    }

    /// In-place filter for hashes that are NOT already sealed in cache.
    ///
    /// After return, `hashes` only contains entries that need saving.
    /// Since cache membership is hash-based (not layer-specific), this only
    /// needs to be called once for all layers sharing the same namespace.
    pub(crate) fn filter_hashes_not_in_cache_inplace(
        &self,
        namespace: &str,
        hashes: &mut HashSet<Vec<u8>>,
    ) {
        let namespace = namespace.to_string();
        let inner = self.inner.lock();
        hashes.retain(|hash| {
            let key = BlockKey::new(namespace.clone(), hash.clone());
            !inner.cache.contains_key(&key)
        });
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
    #[cfg_attr(
        feature = "tracing",
        fastrace::trace(name = "storage.cache_lookup_many")
    )]
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
        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .map(|key| (instance_id.to_string(), key.clone()))
            .collect();

        let mut inner = self.inner.lock();
        let mut result: Vec<Arc<SealedBlock>> = Vec::with_capacity(keys.len());

        for (idx, (key, pin_key)) in keys.into_iter().zip(pin_keys.into_iter()).enumerate() {
            // Consume pinned_for_load (ref_count -1, remove if 0)
            if let Entry::Occupied(mut entry) = inner.pinned_for_load.entry(pin_key) {
                let (block, count) = entry.get_mut();
                let cloned = Arc::clone(block);
                *count -= 1;

                if *count == 0 {
                    entry.remove();
                }

                // Track unique block removal
                let mut unique_bytes_to_remove: Option<u64> = None;
                if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                    *total = total.saturating_sub(1);
                    if *total == 0 {
                        unique_bytes_to_remove = Some(*bytes);
                    }
                } else {
                    error!(
                        "BUG: pinned_for_load_by_key missing key during consume: namespace={} hash_len={}",
                        key.namespace,
                        key.hash.len()
                    );
                }

                if let Some(bytes) = unique_bytes_to_remove {
                    inner.pinned_for_load_by_key.remove(&key);
                    record_pin_unique_removed(bytes);
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

    /// Unpin blocks that were pinned during query.
    /// This decrements the ref_count and removes the entry when it reaches 0.
    /// Returns the number of blocks that were successfully unpinned.
    pub fn unpin_blocks(
        &self,
        instance_id: &str,
        namespace: &str,
        block_hashes: &[Vec<u8>],
    ) -> usize {
        let keys: Vec<BlockKey> = block_hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let pin_keys: Vec<(String, BlockKey)> = keys
            .iter()
            .map(|key| (instance_id.to_string(), key.clone()))
            .collect();

        let mut inner = self.inner.lock();
        let mut unpinned = 0usize;

        for (key, pin_key) in keys.into_iter().zip(pin_keys.into_iter()) {
            if let Some((_, count)) = inner.pinned_for_load.get_mut(&pin_key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    inner.pinned_for_load.remove(&pin_key);
                }
                unpinned += 1;

                // Track unique block removal
                let mut unique_bytes_to_remove: Option<u64> = None;
                if let Some((bytes, total)) = inner.pinned_for_load_by_key.get_mut(&key) {
                    *total = total.saturating_sub(1);
                    if *total == 0 {
                        unique_bytes_to_remove = Some(*bytes);
                    }
                } else {
                    error!(
                        "BUG: pinned_for_load_by_key missing key during unpin: namespace={} hash_len={}",
                        key.namespace,
                        key.hash.len()
                    );
                }

                if let Some(bytes) = unique_bytes_to_remove {
                    inner.pinned_for_load_by_key.remove(&key);
                    record_pin_unique_removed(bytes);
                }
            }
        }

        unpinned
    }

    // ========================================================================
    // Eviction (cache only)
    // ========================================================================

    fn reclaim_until_allocator_can_allocate(&self, required_bytes: u64) -> (usize, u64, u64) {
        if required_bytes == 0 {
            return (0, 0, self.largest_free_allocation());
        }

        let mut freed_blocks = 0usize;
        let mut freed_bytes = 0u64;
        let mut largest_free = self.largest_free_allocation();

        while largest_free < required_bytes {
            let used_before = self.pool_usage().0;

            // Collect evicted blocks under lock, then drop outside lock
            let evicted: Vec<_> = {
                let mut inner = self.inner.lock();
                (0..RECLAIM_BATCH_SIZE)
                    .map_while(|_| inner.cache.remove_lru())
                    .collect()
            };

            if evicted.is_empty() {
                break;
            }

            let mut batch_bytes = 0u64;
            let mut still_referenced = 0u64;
            for (_key, block) in evicted.iter() {
                let b = block.memory_footprint();
                batch_bytes = batch_bytes.saturating_add(b);
                if Arc::strong_count(block) > 1 {
                    still_referenced += 1;
                }
                record_cache_eviction(b);
            }

            if still_referenced > 0 {
                core_metrics()
                    .cache_block_evictions_still_referenced
                    .add(still_referenced, &[]);
            }

            freed_bytes = freed_bytes.saturating_add(batch_bytes);
            freed_blocks += evicted.len();

            drop(evicted); // allow allocation drops to run before sampling allocator usage
            let used_after = self.pool_usage().0;
            let reclaimed = used_before.saturating_sub(used_after);
            if reclaimed > 0 {
                core_metrics()
                    .cache_eviction_reclaimed_bytes
                    .add(reclaimed, &[]);
            }

            largest_free = self.largest_free_allocation();
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
    /// Sends a GC command to the insert worker, which owns the inflight HashMap.
    /// This is async because it waits for the worker's reply.
    ///
    /// Returns the number of cleaned blocks.
    pub async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self
            .insert_tx
            .send(InsertWorkerCommand::Gc {
                max_age,
                reply: reply_tx,
            })
            .is_err()
        {
            return 0;
        }
        reply_rx.await.unwrap_or(0)
    }

    /// Pure memory-only prefix check. Returns `(hit, missing)` counts.
    ///
    /// No SSD prefetch, no pinning — suitable for lightweight query RPCs.
    pub fn check_prefix_memory_only(&self, namespace: &str, hashes: &[Vec<u8>]) -> (usize, usize) {
        let mut hit = 0usize;

        {
            let mut inner = self.inner.lock();

            for hash in hashes {
                let key = BlockKey::new(namespace.to_string(), hash.clone());
                if inner.cache.get(&key).is_some() {
                    hit += 1;
                } else {
                    break;
                }
            }
        }

        let missing = hashes.len() - hit;
        (hit, missing)
    }

    /// Check prefix blocks and prefetch from SSD if needed.
    ///
    /// Scans blocks in prefix order, checking cache, prefetching set, and SSD index.
    /// Pin hit blocks when returning Done (no loading in progress).
    pub fn check_prefix_and_prefetch(
        &self,
        instance_id: &str,
        namespace: &str,
        hashes: &[Vec<u8>],
        num_workers: usize,
    ) -> PrefetchStatus {
        let keys: Vec<BlockKey> = hashes
            .iter()
            .map(|hash| BlockKey::new(namespace.to_string(), hash.clone()))
            .collect();
        let instance_id_owned = instance_id.to_string();

        let mut hit = 0usize;
        let mut loading = 0usize;
        let mut missing = 0usize;
        let mut to_prefetch: Vec<BlockKey> = Vec::new();
        let mut backpressure_missing = 0usize;

        // Blocks to pin: (key, block, footprint_bytes)
        let mut blocks_to_pin: Vec<(BlockKey, Arc<SealedBlock>, u64)> = Vec::new();

        {
            let mut inner = self.inner.lock();

            for key in &keys {
                // First check cache
                if let Some(block) = inner.cache.get(key) {
                    hit += 1;
                    blocks_to_pin.push((key.clone(), Arc::clone(&block), block.memory_footprint()));
                    continue;
                }

                if inner.prefetching.contains(key) {
                    loading += 1;
                    continue;
                }

                // Backpressure: stop scheduling if too many blocks are prefetching
                if inner.prefetching.len() >= self.max_prefetch_blocks {
                    backpressure_missing = hashes.len() - hit - loading;
                    missing = backpressure_missing;
                    break;
                }

                // Check SSD index (ssd_ring is now in StorageInner)
                let in_ssd = inner
                    .ssd_ring
                    .as_ref()
                    .is_some_and(|ring| ring.has_valid_entry(key));

                if in_ssd {
                    // Block is in SSD, schedule prefetch
                    to_prefetch.push(key.clone());
                    loading += 1;
                    continue;
                }

                // Block not found anywhere - this is a miss
                // For prefix matching, first miss means remaining blocks are also missing
                missing = hashes.len() - hit - loading;
                break;
            }

            // Pin hit blocks when returning Done (no loading in progress)
            // Increment ref_count by num_workers so each worker can consume the pin once
            if loading == 0 {
                for (key, block, footprint_bytes) in blocks_to_pin {
                    let pin_key = (instance_id_owned.clone(), key.clone());

                    match inner.pinned_for_load.entry(pin_key) {
                        Entry::Occupied(mut o) => {
                            o.get_mut().1 += num_workers;
                        }
                        Entry::Vacant(v) => {
                            v.insert((block, num_workers));
                        }
                    }

                    match inner.pinned_for_load_by_key.entry(key) {
                        Entry::Occupied(mut o) => {
                            o.get_mut().1 += num_workers;
                        }
                        Entry::Vacant(v) => {
                            v.insert((footprint_bytes, num_workers));
                            record_pin_unique_added(footprint_bytes);
                        }
                    }
                }
            }
        }

        if backpressure_missing > 0 {
            core_metrics()
                .ssd_prefetch_backpressure_blocks
                .add(backpressure_missing as u64, &[]);
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
    /// Memory allocation is handled by the prefetch dispatcher for better pipelining.
    /// If prefetch queue is full, drops the request (treats as cache miss).
    fn trigger_prefetch(&self, keys: Vec<BlockKey>) {
        let ssd_state = match &self.ssd_state {
            Some(state) => state,
            None => return,
        };

        // Collect valid entries (ssd_ring is now in StorageInner)
        let mut valid_requests: Vec<(BlockKey, SsdIndexEntry)> = Vec::with_capacity(keys.len());

        {
            let mut inner = self.inner.lock();
            let ring = match inner.ssd_ring.as_ref() {
                Some(r) => r,
                None => return,
            };

            for key in keys {
                // Skip if already prefetching
                if inner.prefetching.contains(&key) {
                    continue;
                }

                // Get SSD index entry (includes validity check)
                let entry = match ring.get(&key) {
                    Some(e) => e.clone(),
                    None => continue,
                };

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

        // Build batch (memory allocation moved to prefetch dispatcher)
        let keys_for_cleanup: Vec<_> = valid_requests.iter().map(|(k, _)| k.clone()).collect();
        let requests: Vec<_> = valid_requests
            .into_iter()
            .map(|(key, entry)| PrefetchRequest { key, entry })
            .collect();

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
            let mut inner = self.inner.lock();
            for key in keys_for_cleanup {
                inner.prefetching.remove(&key);
            }
        }
    }

    /// Called by prefetch worker when a block is loaded from SSD.
    pub fn complete_prefetch(&self, key: BlockKey, block: Option<Arc<SealedBlock>>) {
        let footprint_bytes = block.as_ref().map(|b| b.memory_footprint());

        let mut inner = self.inner.lock();
        inner.prefetching.remove(&key);

        if let Some(block) = block {
            match inner.cache.insert(key, block) {
                CacheInsertOutcome::InsertedNew => {
                    if let Some(bytes) = footprint_bytes {
                        record_cache_insert_new(bytes);
                    }
                }
                CacheInsertOutcome::AlreadyExists => {
                    // No overwrite, no-op for metrics.
                }
                CacheInsertOutcome::Rejected => {
                    core_metrics().cache_block_admission_rejections.add(1, &[]);
                }
            }
        }
    }

    /// Check if a logical SSD offset is still valid (not yet overwritten).
    /// Used by prefetch worker to validate reads.
    pub(crate) fn is_ssd_offset_valid(&self, begin: u64) -> bool {
        let inner = self.inner.lock();
        inner
            .ssd_ring
            .as_ref()
            .is_some_and(|ring| ring.is_offset_valid(begin))
    }
}

// ============================================================================
// Insert Worker (dedicated thread, owns inflight HashMap)
// ============================================================================

/// Dedicated insert worker task. Owns the inflight HashMap exclusively,
/// eliminating lock contention on the hot insert path. Sealed blocks are
/// admitted to cache via brief `StorageInner` lock acquisitions.
fn insert_worker_loop(rx: Receiver<InsertWorkerCommand>, engine: Weak<StorageEngine>) {
    let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

    while let Ok(cmd) = rx.recv() {
        // Drain additional commands for batching
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        for cmd in cmds {
            match cmd {
                InsertWorkerCommand::RawInsert(batch) => {
                    process_raw_save_batch(&mut inflight, &engine, batch);
                }
                InsertWorkerCommand::Gc { max_age, reply } => {
                    let cleaned = gc_inflight(&mut inflight, max_age);
                    let _ = reply.send(cleaned);
                }
            }
        }
    }

    info!(
        "Insert worker shutting down, {} inflight blocks remaining",
        inflight.len()
    );
}

/// Process a deferred raw save batch: build LayerBlocks, then delegate to
/// `process_insert_batch` for inflight/seal/cache logic.
fn process_raw_save_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    engine: &Weak<StorageEngine>,
    batch: crate::offload::RawSaveBatch,
) {
    let phase4_start = std::time::Instant::now();
    let namespace = batch.namespace.clone();
    let numa_node = batch.numa_node;
    let total_slots = batch.total_slots;

    let (entries, _total_bytes, _total_blocks) = crate::offload::build_insert_entries(&batch);

    process_insert_batch(
        inflight,
        engine,
        entries,
        total_slots,
        numa_node,
        &namespace,
    );

    debug!(
        "insert_worker phase4: blocks={} bytes={} ms={:.2}",
        _total_blocks,
        _total_bytes,
        phase4_start.elapsed().as_secs_f64() * 1000.0,
    );
}

/// Process a single insert batch (fire-and-forget).
/// Inflight HashMap is owned exclusively by the worker (no lock needed).
/// Cache insertion + metaserver announcement + SSD offload handled internally.
fn process_insert_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    engine: &Weak<StorageEngine>,
    entries: InsertEntries,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
) {
    let mut sealed_blocks: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();
    let mut inflight_bytes_added: u64 = 0;
    let mut inflight_bytes_removed: u64 = 0;

    for (key, slots) in entries {
        // Get or create inflight block (no lock — worker-exclusive HashMap)
        let inflight_block = match inflight.entry(key.clone()) {
            Entry::Vacant(v) => v.insert(InflightBlock::new(total_slots)),
            Entry::Occupied(o) => {
                let ib = o.into_mut();
                if ib.total_slots() != total_slots {
                    error!(
                        "insert worker: slot count mismatch: key namespace={} expected={} got={}",
                        namespace,
                        ib.total_slots(),
                        total_slots
                    );
                    continue;
                }
                ib
            }
        };

        // Insert all slots for this hash
        let mut completed = false;
        for (slot_id, block) in slots {
            match inflight_block.insert_slot(slot_id, block, numa_node) {
                SlotInsertResult::Inserted {
                    completed: c,
                    footprint_added,
                } => {
                    inflight_bytes_added = inflight_bytes_added.saturating_add(footprint_added);
                    completed = c;
                    if completed {
                        break;
                    }
                }
                SlotInsertResult::Duplicate => {}
            }
        }

        if completed {
            let inflight_block = inflight.remove(&key).expect("just inserted");
            let total_footprint = inflight_block.footprint();
            inflight_bytes_removed = inflight_bytes_removed.saturating_add(total_footprint);
            let sealed = Arc::new(inflight_block.seal());

            // Brief lock: admit sealed block to cache
            if let Some(engine) = engine.upgrade() {
                let mut inner = engine.inner.lock();
                match inner.cache.insert(key.clone(), Arc::clone(&sealed)) {
                    CacheInsertOutcome::InsertedNew => {
                        record_cache_insert_new(total_footprint);
                    }
                    CacheInsertOutcome::AlreadyExists => {}
                    CacheInsertOutcome::Rejected => {
                        core_metrics().cache_block_admission_rejections.add(1, &[]);
                    }
                }
                drop(inner);

                // Seal notification (for SSD offload)
                if let Some(tx) = &engine.seal_notify_tx {
                    let _ = tx.send((key.clone(), Arc::downgrade(&sealed)));
                }
            }

            sealed_blocks.push((key, sealed));
        }
    }

    if inflight_bytes_added > 0 {
        record_inflight_bytes_added(inflight_bytes_added);
    }
    if inflight_bytes_removed > 0 {
        record_inflight_bytes_removed(inflight_bytes_removed);
    }

    // Send block hashes to metaserver worker (batched, fire-and-forget)
    if !sealed_blocks.is_empty()
        && let Some(engine) = engine.upgrade()
    {
        let metaserver_hashes: Vec<Vec<u8>> = sealed_blocks
            .iter()
            .map(|(key, _)| key.hash.clone())
            .collect();
        engine.send_metaserver_insert(namespace.to_owned(), metaserver_hashes);
    }

    // SSD offload (fire-and-forget internally)
    if !sealed_blocks.is_empty()
        && let Some(engine) = engine.upgrade()
    {
        engine.send_ssd_batch(&sealed_blocks);
    }
}

/// GC stale inflight blocks within the insert worker.
fn gc_inflight(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    max_age: std::time::Duration,
) -> usize {
    let before = inflight.len();

    inflight.retain(|key, block| {
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
            record_inflight_bytes_removed(block.footprint());
            false
        } else {
            true
        }
    });

    let cleaned = before - inflight.len();
    if cleaned > 0 {
        core_metrics().inflight_gc_cleaned.add(cleaned as u64, &[]);
        info!("GC cleaned stale inflight blocks: cleaned={}", cleaned);
    }
    cleaned
}

// ============================================================================
// Metaserver Worker (dedicated task, batches block hash inserts)
// ============================================================================

/// Background worker that receives block hash insert commands and batches them
/// into MetaServer gRPC calls, grouped by namespace.
async fn metaserver_worker_loop(
    mut rx: UnboundedReceiver<crate::MetaserverInsertCmd>,
    mut client: MetaServerClient<Channel>,
    node_url: String,
) {
    while let Some(cmd) = rx.recv().await {
        // Drain additional commands for batching
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        // Merge hashes by namespace
        let mut by_namespace: HashMap<String, Vec<Vec<u8>>> = HashMap::new();
        for cmd in cmds {
            by_namespace
                .entry(cmd.namespace)
                .or_default()
                .extend(cmd.block_hashes);
        }

        for (namespace, block_hashes) in by_namespace {
            let count = block_hashes.len();
            let req = InsertBlockHashesRequest {
                namespace,
                block_hashes,
                node: node_url.clone(),
            };
            match client.insert_block_hashes(req).await {
                Ok(response) => {
                    debug!(
                        "MetaServer insert: sent {} hashes, inserted {}",
                        count,
                        response.into_inner().inserted_count
                    );
                }
                Err(err) => {
                    warn!("MetaServer insert failed: {}", err);
                }
            }
        }
    }

    info!("Metaserver worker shutting down");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_keeps_only_uncached() {
        let (storage, _rx) =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let namespace = "ns";
        let cached_hash = vec![1, 1, 1];
        let uncached_hash = vec![2, 2, 2];

        storage.complete_prefetch(
            BlockKey::new(namespace.to_string(), cached_hash.clone()),
            Some(Arc::new(SealedBlock::from_slots(Vec::new()))),
        );

        let mut hashes = HashSet::from([cached_hash, uncached_hash.clone()]);
        storage.filter_hashes_not_in_cache_inplace(namespace, &mut hashes);

        assert_eq!(hashes.len(), 1);
        assert!(hashes.contains(&uncached_hash));
    }

    #[tokio::test]
    async fn filter_hashes_not_in_cache_inplace_handles_empty_input() {
        let (storage, _rx) =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let mut hashes: HashSet<Vec<u8>> = HashSet::new();

        storage.filter_hashes_not_in_cache_inplace("ns", &mut hashes);
        assert!(hashes.is_empty());
    }
}
