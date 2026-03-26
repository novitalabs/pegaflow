use std::sync::{Arc, Weak};

use bytesize::ByteSize;
use log::{debug, error, info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use pegaflow_common::NumaNode;

use super::SsdCacheConfig;
use super::ssd_cache::{
    PrefetchBatch, PrefetchRequest, PreparedBatch, SsdRingBuffer, SsdWriteBatch, ssd_prefetch_loop,
    ssd_writer_loop,
};
use super::uring::{UringConfig, UringIoEngine};
use super::{AllocateFn, PrefetchResult};

struct SsdInner {
    ring: SsdRingBuffer,
}

pub(crate) struct SsdBackingStore {
    /// Keeps the file descriptor alive for io_uring operations.
    _file: std::fs::File,
    io: Arc<UringIoEngine>,
    write_tx: tokio::sync::mpsc::Sender<SsdWriteBatch>,
    prefetch_tx: tokio::sync::mpsc::Sender<PrefetchBatch>,
    inner: Mutex<SsdInner>,
    allocate_fn: AllocateFn,
    is_numa: bool,
}

impl SsdBackingStore {
    pub(super) fn new(
        config: SsdCacheConfig,
        allocate_fn: AllocateFn,
        is_numa: bool,
    ) -> std::io::Result<Arc<Self>> {
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

        let io = Arc::new(UringIoEngine::new(
            file.as_raw_fd(),
            UringConfig::default(),
        )?);

        let (write_tx, write_rx) = tokio::sync::mpsc::channel(config.write_queue_depth);
        let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(config.prefetch_queue_depth);

        let capacity = config.capacity_bytes;
        let write_inflight = config.write_inflight;
        let prefetch_inflight = config.prefetch_inflight;

        info!(
            "SSD cache initialized at {} (capacity {})",
            config.cache_path.display(),
            ByteSize(capacity)
        );

        let store = Arc::new(Self {
            _file: file,
            io: Arc::clone(&io),
            write_tx,
            prefetch_tx,
            inner: Mutex::new(SsdInner {
                ring: SsdRingBuffer::new(capacity),
            }),
            allocate_fn,
            is_numa,
        });

        Self::spawn_workers(
            &store,
            write_rx,
            prefetch_rx,
            capacity,
            write_inflight,
            prefetch_inflight,
        );

        Ok(store)
    }

    pub(super) fn is_offset_valid(&self, begin: u64) -> bool {
        self.inner.lock().ring.is_offset_valid(begin)
    }

    pub(super) fn allocate_prefetch(
        &self,
        size: u64,
        numa_node: Option<NumaNode>,
    ) -> Option<Arc<PinnedAllocation>> {
        (self.allocate_fn)(size, numa_node)
    }

    pub(super) fn prepare_batch(
        &self,
        candidates: Vec<(BlockKey, Arc<SealedBlock>)>,
    ) -> PreparedBatch {
        self.inner.lock().ring.prepare_batch(candidates)
    }

    pub(super) fn commit_write(&self, key: &BlockKey, success: bool) {
        self.inner.lock().ring.commit(key, success);
    }

    pub(super) fn is_numa(&self) -> bool {
        self.is_numa
    }

    fn spawn_workers(
        store: &Arc<Self>,
        write_rx: tokio::sync::mpsc::Receiver<SsdWriteBatch>,
        prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
        capacity: u64,
        write_inflight: usize,
        prefetch_inflight: usize,
    ) {
        let io = Arc::clone(&store.io);

        let writer_weak = Arc::downgrade(store);
        let writer_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_writer_loop(writer_weak, write_rx, writer_io, write_inflight).await;
        });

        let prefetch_weak = Arc::downgrade(store);
        let prefetch_io = Arc::clone(&io);
        tokio::spawn(async move {
            ssd_prefetch_loop(
                prefetch_weak,
                prefetch_rx,
                prefetch_io,
                capacity,
                prefetch_inflight,
            )
            .await;
        });

        debug!("SSD backing store workers spawned");
    }

    /// Fire-and-forget write.
    ///
    /// `blocks` holds `Weak` references so the backing store cannot prevent
    /// cache eviction from freeing the pinned memory before the write completes.
    pub(crate) fn ingest_batch(&self, blocks: Vec<(BlockKey, Weak<SealedBlock>)>) {
        if blocks.is_empty() {
            return;
        }
        let len = blocks.len();
        let batch = SsdWriteBatch { blocks };
        if self.write_tx.try_send(batch).is_ok() {
            core_metrics().ssd_write_queue_pending.add(len as i64, &[]);
        } else {
            warn!("SSD write queue full, dropping {} blocks", len);
            core_metrics().ssd_write_queue_full.add(len as u64, &[]);
        }
    }

    /// Submit prefix reads: scan `keys` in order, submit reads for consecutive hits, stop at first miss.
    ///
    /// Returns `(submitted, done_rx)` where `done_rx` delivers completed blocks.
    pub(crate) fn submit_prefix(
        &self,
        keys: Vec<BlockKey>,
    ) -> (usize, oneshot::Receiver<PrefetchResult>) {
        let (done_tx, done_rx) = oneshot::channel();

        // Prefix-scan the ring buffer: stop at first miss.
        let requests: Vec<PrefetchRequest> = {
            let inner = self.inner.lock();
            keys.into_iter()
                .map_while(|key| {
                    let entry = inner.ring.get(&key)?.clone();
                    Some(PrefetchRequest { key, entry })
                })
                .collect()
        };

        let found = requests.len();
        if found == 0 {
            let _ = done_tx.send(Vec::new());
            return (0, done_rx);
        }

        let batch = PrefetchBatch { requests, done_tx };

        if let Err(e) = self.prefetch_tx.try_send(batch) {
            let batch = e.into_inner();
            let count = batch.requests.len();
            warn!("SSD prefetch queue full, dropping {} reads", count);
            core_metrics()
                .ssd_prefetch_queue_full
                .add(count as u64, &[]);
            let _ = batch.done_tx.send(Vec::new());
        }

        (found, done_rx)
    }
}

/// Returns `None` if the SSD cache cannot be initialised (logs the error).
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: AllocateFn,
    is_numa: bool,
) -> Option<Arc<SsdBackingStore>> {
    match SsdBackingStore::new(config, allocate_fn, is_numa) {
        Ok(b) => Some(b),
        Err(e) => {
            error!("Failed to initialise SSD backing store: {}", e);
            None
        }
    }
}
