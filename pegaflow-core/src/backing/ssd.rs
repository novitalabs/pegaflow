use std::path::PathBuf;
use std::sync::{Arc, Weak};

use bytesize::ByteSize;
use log::{debug, info, warn};
use mea::oneshot;
use parking_lot::Mutex;

use crate::block::{BlockKey, SealedBlock};
use crate::metrics::core_metrics;
use crate::pinned_pool::PinnedAllocation;
use pegaflow_common::NumaNode;

use super::SsdCacheConfig;
use super::ssd_cache::{
    PrefetchBatch, PrefetchRequest, PreparedBatch, SsdRingBuffer, SsdWriteBatch, SsdWriteCommand,
    ssd_prefetch_loop, ssd_writer_loop,
};
use super::uring::{UringConfig, UringIoEngine};
use super::{AllocateFn, PrefetchResult};

struct SsdInner {
    ring: SsdRingBuffer,
}

pub(crate) struct SsdBackingStore {
    /// Keeps file descriptors alive for io_uring operations.
    _files: Vec<std::fs::File>,
    io: Arc<UringIoEngine>,
    write_tx: tokio::sync::mpsc::Sender<SsdWriteCommand>,
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
        use std::fs::OpenOptions;
        use std::os::unix::io::AsRawFd;

        let shards_per_path = config.shards.get();
        let total_shards = config.cache_paths.len() * shards_per_path;
        let shard_capacity = aligned_shard_capacity(config.capacity_bytes, total_shards)?;
        let files = open_cache_files(
            &config.cache_paths,
            shards_per_path,
            shard_capacity,
            &mut OpenOptions::new(),
        )?;
        let fds: Vec<_> = files.iter().map(|file| file.as_raw_fd()).collect();

        let io = Arc::new(UringIoEngine::new_multi(fds, UringConfig::default())?);

        let (write_tx, write_rx) = tokio::sync::mpsc::channel(config.write_queue_depth);
        let (prefetch_tx, prefetch_rx) = tokio::sync::mpsc::channel(config.prefetch_queue_depth);

        let capacity = shard_capacity * total_shards as u64;
        let write_inflight = config.write_inflight;
        let prefetch_inflight = config.prefetch_inflight;

        info!(
            "SSD cache initialized at {} (capacity {}, shards {}, shard capacity {})",
            config
                .cache_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            ByteSize(capacity),
            total_shards,
            ByteSize(shard_capacity)
        );

        let store = Arc::new(Self {
            _files: files,
            io: Arc::clone(&io),
            write_tx,
            prefetch_tx,
            inner: Mutex::new(SsdInner {
                ring: SsdRingBuffer::new_sharded(vec![shard_capacity; total_shards]),
            }),
            allocate_fn,
            is_numa,
        });

        Self::spawn_workers(
            &store,
            write_rx,
            prefetch_rx,
            write_inflight,
            prefetch_inflight,
        );

        Ok(store)
    }

    pub(super) fn is_offset_valid(&self, entry: &super::ssd_cache::SsdIndexEntry) -> bool {
        self.inner.lock().ring.is_offset_valid(entry)
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
        write_rx: tokio::sync::mpsc::Receiver<SsdWriteCommand>,
        prefetch_rx: tokio::sync::mpsc::Receiver<PrefetchBatch>,
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
            ssd_prefetch_loop(prefetch_weak, prefetch_rx, prefetch_io, prefetch_inflight).await;
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
        if self
            .write_tx
            .try_send(SsdWriteCommand::Write(batch))
            .is_ok()
        {
            core_metrics().ssd_write_queue_pending.add(len as i64, &[]);
        } else {
            warn!("SSD write queue full, dropping {} blocks", len);
            core_metrics().ssd_write_queue_full.add(len as u64, &[]);
        }
    }

    /// Flush the SSD writer: waits until all enqueued writes complete.
    pub(crate) async fn flush(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if self.write_tx.send(SsdWriteCommand::Flush(tx)).await.is_ok() {
            let _ = rx.await;
        }
    }

    /// Count consecutive SSD-resident keys from the start of `keys`.
    pub(crate) fn prefix_len(&self, keys: &[BlockKey]) -> usize {
        let inner = self.inner.lock();
        keys.iter()
            .map_while(|key| inner.ring.get(key).map(|_| ()))
            .count()
    }

    /// Submit prefix reads: scan `keys` in order, submit reads for consecutive hits, stop at first miss.
    ///
    /// Returns `(submitted, done_rx)` where `done_rx` delivers completed blocks.
    fn submit_prefix(&self, keys: Vec<BlockKey>) -> (usize, oneshot::Receiver<PrefetchResult>) {
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

    /// Prefetch prefix reads and await completion.
    pub(crate) async fn prefetch_prefix(&self, keys: Vec<BlockKey>) -> (usize, PrefetchResult) {
        let (found, done_rx) = self.submit_prefix(keys);
        if found == 0 {
            return (0, Vec::new());
        }

        match done_rx.await {
            Ok(blocks) => (found, blocks),
            Err(_) => {
                warn!("SSD prefetch completion channel closed");
                (found, Vec::new())
            }
        }
    }
}

fn aligned_shard_capacity(capacity_bytes: u64, shard_count: usize) -> std::io::Result<u64> {
    let shard_count = u64::try_from(shard_count).expect("usize fits into u64");
    let raw = capacity_bytes / shard_count;
    let alignment = super::SSD_ALIGNMENT as u64;
    let capacity = raw / alignment * alignment;
    if capacity == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache capacity is too small for the requested shard count",
        ));
    }
    Ok(capacity)
}

fn open_cache_files(
    cache_paths: &[PathBuf],
    shards_per_path: usize,
    shard_capacity: u64,
    options: &mut std::fs::OpenOptions,
) -> std::io::Result<Vec<std::fs::File>> {
    use std::fs;
    use std::os::unix::fs::OpenOptionsExt;

    options
        .create(true)
        .truncate(true)
        .read(true)
        .write(true)
        .custom_flags(libc::O_DIRECT);

    if cache_paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "SSD cache paths cannot be empty",
        ));
    }

    let total_shards = cache_paths.len() * shards_per_path;

    // Backward compatibility: single path + single shard = single file.
    if total_shards == 1 {
        if let Some(parent) = cache_paths[0].parent() {
            fs::create_dir_all(parent)?;
        }
        let file = options.open(&cache_paths[0])?;
        file.set_len(shard_capacity)?;
        return Ok(vec![file]);
    }

    // Multi-path or multi-shard: each path must be a directory.
    for path in cache_paths {
        if path.exists() && !path.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "SSD cache path {} must be a directory when using multiple paths or shards",
                    path.display()
                ),
            ));
        }
        fs::create_dir_all(path)?;
    }

    let mut files = Vec::with_capacity(total_shards);
    for (path_id, path) in cache_paths.iter().enumerate() {
        for local_shard in 0..shards_per_path {
            let global_shard_id = path_id * shards_per_path + local_shard;
            let file_path = path.join(format!("shard-{global_shard_id:06}.dat"));
            let file = options.open(&file_path)?;
            file.set_len(shard_capacity)?;
            files.push(file);
        }
    }

    Ok(files)
}

/// Creates the SSD backing store, failing startup if it cannot be initialised.
pub(crate) fn new_ssd(
    config: SsdCacheConfig,
    allocate_fn: AllocateFn,
    is_numa: bool,
) -> Arc<SsdBackingStore> {
    SsdBackingStore::new(config, allocate_fn, is_numa)
        .unwrap_or_else(|e| panic!("failed to initialise SSD backing store: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_open_cache_files_single_path_single_shard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_path = temp_dir.path().join("cache.bin");
        let mut options = std::fs::OpenOptions::new();
        let files =
            open_cache_files(std::slice::from_ref(&cache_path), 1, 4096, &mut options).unwrap();
        assert_eq!(files.len(), 1);
        assert!(cache_path.is_file());
        assert_eq!(fs::metadata(&cache_path).unwrap().len(), 4096);
    }

    #[test]
    fn test_open_cache_files_single_path_multi_shard() {
        let temp_dir = tempfile::tempdir().unwrap();
        let cache_path = temp_dir.path().join("cache");
        let mut options = std::fs::OpenOptions::new();
        let files =
            open_cache_files(std::slice::from_ref(&cache_path), 4, 4096, &mut options).unwrap();
        assert_eq!(files.len(), 4);
        assert!(cache_path.is_dir());
        for shard_id in 0..4 {
            let shard = cache_path.join(format!("shard-{shard_id:06}.dat"));
            assert!(shard.is_file());
            assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
        }
    }

    #[test]
    fn test_open_cache_files_multi_path_per_path_shards() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path0 = temp_dir.path().join("ssd0");
        let path1 = temp_dir.path().join("ssd1");
        let mut options = std::fs::OpenOptions::new();
        // 2 paths * 2 shards_per_path = 4 total shards
        let files =
            open_cache_files(&[path0.clone(), path1.clone()], 2, 4096, &mut options).unwrap();
        assert_eq!(files.len(), 4);
        assert!(path0.is_dir());
        assert!(path1.is_dir());
        // path0 gets shards 0,1 ; path1 gets shards 2,3
        for path_id in 0..2 {
            for local_shard in 0..2 {
                let global_shard_id = path_id * 2 + local_shard;
                let expected_path = if path_id == 0 { &path0 } else { &path1 };
                let shard = expected_path.join(format!("shard-{global_shard_id:06}.dat"));
                assert!(
                    shard.is_file(),
                    "shard {global_shard_id} should be at {}",
                    shard.display()
                );
                assert_eq!(fs::metadata(&shard).unwrap().len(), 4096);
            }
        }
    }

    #[test]
    fn test_open_cache_files_multi_path_single_shard_uses_all_paths() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path0 = temp_dir.path().join("ssd0");
        let path1 = temp_dir.path().join("ssd1");
        let mut options = std::fs::OpenOptions::new();
        // 2 paths * 1 shard_per_path = 2 total shards, one on each path
        let files =
            open_cache_files(&[path0.clone(), path1.clone()], 1, 4096, &mut options).unwrap();
        assert_eq!(files.len(), 2);
        assert!(path0.is_dir());
        assert!(path1.is_dir());
        assert!(path0.join("shard-000000.dat").is_file());
        assert!(path1.join("shard-000001.dat").is_file());
    }

    #[test]
    fn test_open_cache_files_empty_paths_fails() {
        let mut options = std::fs::OpenOptions::new();
        let result = open_cache_files(&[], 1, 4096, &mut options);
        assert!(result.is_err());
    }
}
