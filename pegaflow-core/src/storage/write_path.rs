use std::collections::{HashMap, hash_map::Entry};
use std::sync::{Arc, Weak};

use log::{debug, error, info, warn};
use std::sync::mpsc::{Receiver, Sender};
use tokio::sync::oneshot;

use crate::backing::SsdBackingStore;
use crate::block::{BlockKey, InflightBlock, SealedBlock, SlotInsertResult};
use crate::internode::MetaServerClient;
use crate::metrics::core_metrics;
use crate::offload::InsertEntries;
use pegaflow_common::NumaNode;

use super::read_cache::ReadCache;

pub(super) enum InsertWorkerCommand {
    RawInsert(crate::offload::RawSaveBatch),
    Gc {
        max_age: std::time::Duration,
        reply: oneshot::Sender<usize>,
    },
}

pub(super) struct WritePipeline {
    insert_tx: Sender<InsertWorkerCommand>,
}

impl WritePipeline {
    pub(super) fn new() -> (Self, Receiver<InsertWorkerCommand>) {
        let (insert_tx, insert_rx) = std::sync::mpsc::channel();
        (Self { insert_tx }, insert_rx)
    }

    pub(super) fn send_raw_insert(&self, batch: crate::offload::RawSaveBatch) {
        let _ = self.insert_tx.send(InsertWorkerCommand::RawInsert(batch));
    }

    pub(super) async fn gc_stale_inflight(&self, max_age: std::time::Duration) -> usize {
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
}

pub(super) struct InsertDeps {
    pub(super) read_cache: Arc<ReadCache>,
    pub(super) ssd_store: Option<Arc<SsdBackingStore>>,
    pub(super) metaserver_client: Option<Arc<MetaServerClient>>,
}

pub(super) fn insert_worker_loop(rx: Receiver<InsertWorkerCommand>, deps: Weak<InsertDeps>) {
    let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

    while let Ok(cmd) = rx.recv() {
        let mut cmds = vec![cmd];
        while let Ok(more) = rx.try_recv() {
            cmds.push(more);
        }

        for cmd in cmds {
            match cmd {
                InsertWorkerCommand::RawInsert(batch) => {
                    process_raw_save_batch(&mut inflight, &deps, batch);
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

fn process_raw_save_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    batch: crate::offload::RawSaveBatch,
) {
    let start = std::time::Instant::now();
    let namespace = &batch.namespace;
    let numa_node = batch.numa_node;
    let total_slots = batch.total_slots;

    let (entries, total_bytes, total_blocks) = crate::offload::build_insert_entries(&batch);

    process_insert_batch(inflight, deps, entries, total_slots, numa_node, namespace);

    debug!(
        "insert_worker: batch sealed blocks={} bytes={} ms={:.2}",
        total_blocks,
        total_bytes,
        start.elapsed().as_secs_f64() * 1000.0,
    );
}

fn process_insert_batch(
    inflight: &mut HashMap<BlockKey, InflightBlock>,
    deps: &Weak<InsertDeps>,
    entries: InsertEntries,
    total_slots: usize,
    numa_node: NumaNode,
    namespace: &str,
) {
    let mut sealed_blocks: Vec<(BlockKey, Arc<SealedBlock>)> = Vec::new();
    let mut inflight_bytes_added: u64 = 0;
    let mut inflight_bytes_removed: u64 = 0;

    for (key, slots) in entries {
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

            if let Some(deps) = deps.upgrade() {
                deps.read_cache
                    .batch_insert(vec![(key.clone(), Arc::clone(&sealed))]);
            }

            sealed_blocks.push((key, sealed));
        }
    }

    if inflight_bytes_added > 0 {
        core_metrics()
            .inflight_bytes
            .add(inflight_bytes_added as i64, &[]);
    }
    if inflight_bytes_removed > 0 {
        core_metrics()
            .inflight_bytes
            .add(-(inflight_bytes_removed as i64), &[]);
    }

    if !sealed_blocks.is_empty()
        && let Some(deps) = deps.upgrade()
    {
        send_backing_batches(&deps, &sealed_blocks);
    }
}

fn send_backing_batches(deps: &InsertDeps, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
    if blocks.is_empty() {
        return;
    }
    let weak_blocks: Vec<(BlockKey, Weak<SealedBlock>)> = blocks
        .iter()
        .map(|(k, b)| (k.clone(), Arc::downgrade(b)))
        .collect();
    if let Some(ssd) = &deps.ssd_store {
        ssd.ingest_batch(weak_blocks);
    }
    if let Some(client) = &deps.metaserver_client {
        register_block_hashes(client, blocks);
    }
}

fn register_block_hashes(client: &MetaServerClient, blocks: &[(BlockKey, Arc<SealedBlock>)]) {
    let entries: Vec<(String, Vec<u8>)> = blocks
        .iter()
        .map(|(key, _)| (key.namespace.clone(), key.hash.clone()))
        .collect();
    client.try_register(entries);
}

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
            core_metrics().inflight_bytes.add(-(block.footprint() as i64), &[]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU64;

    use crate::block::RawBlock;
    use crate::storage::{StorageConfig, StorageEngine};

    fn make_raw_block(engine: &StorageEngine, size: u64) -> Arc<RawBlock> {
        use crate::block::Segment;
        let mut alloc = engine
            .allocate(NonZeroU64::new(size).unwrap(), None)
            .expect("test pool should have space");
        let ptr = Arc::get_mut(&mut alloc).unwrap().as_mut_ptr();
        let ptr = std::ptr::NonNull::new(ptr).expect("test alloc pointer must be non-null");
        Arc::new(RawBlock::new(vec![Segment::new(ptr, size as usize, alloc)]))
    }

    fn make_deps(engine: &StorageEngine) -> Arc<InsertDeps> {
        Arc::new(InsertDeps {
            read_cache: engine.read_cache.clone(),
            ssd_store: engine.ssd_store.clone(),
            metaserver_client: None,
        })
    }

    #[tokio::test]
    async fn single_slot_seals_immediately() {
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let deps = make_deps(&engine);
        let weak_deps = Arc::downgrade(&deps);

        let key = BlockKey::new("ns".into(), vec![1, 2, 3]);
        let block = make_raw_block(&engine, 64);

        let entries: InsertEntries = vec![(key.clone(), vec![(0, block)])];
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            1,
            NumaNode::UNKNOWN,
            "ns",
        );

        assert!(inflight.is_empty(), "block should have been sealed");
        assert!(
            engine.read_cache.contains_keys(std::slice::from_ref(&key))[0],
            "sealed block should be in cache"
        );
    }

    #[tokio::test]
    async fn multi_slot_partial_then_complete() {
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let deps = make_deps(&engine);
        let weak_deps = Arc::downgrade(&deps);

        let key = BlockKey::new("ns".into(), vec![1]);
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

        let block0 = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(0, block0)])];
        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            3,
            NumaNode::UNKNOWN,
            "ns",
        );
        assert_eq!(inflight.len(), 1, "block should still be inflight");
        assert!(!engine.read_cache.contains_keys(std::slice::from_ref(&key))[0]);

        let block1 = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(1, block1)])];
        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            3,
            NumaNode::UNKNOWN,
            "ns",
        );
        assert_eq!(inflight.len(), 1, "still inflight after 2/3 slots");

        let block2 = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(2, block2)])];
        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            3,
            NumaNode::UNKNOWN,
            "ns",
        );
        assert!(
            inflight.is_empty(),
            "block should be sealed after 3/3 slots"
        );
        assert!(engine.read_cache.contains_keys(std::slice::from_ref(&key))[0]);
    }

    #[tokio::test]
    async fn duplicate_slot_is_idempotent() {
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let deps = make_deps(&engine);
        let weak_deps = Arc::downgrade(&deps);

        let key = BlockKey::new("ns".into(), vec![1]);
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

        let block_a = make_raw_block(&engine, 64);
        let block_b = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(0, block_a), (0, block_b)])];

        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            2,
            NumaNode::UNKNOWN,
            "ns",
        );

        assert_eq!(inflight.len(), 1);
        let inflight_block = inflight.get(&key).unwrap();
        assert_eq!(inflight_block.filled_count(), 1);
    }

    #[tokio::test]
    async fn slot_count_mismatch_skips_key() {
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let deps = make_deps(&engine);
        let weak_deps = Arc::downgrade(&deps);

        let key = BlockKey::new("ns".into(), vec![1]);
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

        let block0 = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(0, block0)])];
        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            2,
            NumaNode::UNKNOWN,
            "ns",
        );
        assert_eq!(inflight.len(), 1);

        let block1 = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(1, block1)])];
        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            4, // mismatch
            NumaNode::UNKNOWN,
            "ns",
        );

        let inflight_block = inflight.get(&key).unwrap();
        assert_eq!(inflight_block.filled_count(), 1);
    }

    #[tokio::test]
    async fn gc_inflight_removes_old_blocks() {
        let key = BlockKey::new("ns".into(), vec![1]);
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();
        inflight.insert(key, InflightBlock::new(2));

        let cleaned = gc_inflight(&mut inflight, std::time::Duration::from_secs(60));
        assert_eq!(cleaned, 0);
        assert_eq!(inflight.len(), 1);

        let cleaned = gc_inflight(&mut inflight, std::time::Duration::ZERO);
        assert_eq!(cleaned, 1);
        assert!(inflight.is_empty());
    }

    #[tokio::test]
    async fn send_backing_batches_no_stores_is_noop() {
        let engine =
            StorageEngine::new_with_config(1 << 20, false, StorageConfig::default(), &[]);
        let deps = make_deps(&engine);
        let weak_deps = Arc::downgrade(&deps);

        let key = BlockKey::new("ns".into(), vec![7]);
        let block = make_raw_block(&engine, 64);
        let entries: InsertEntries = vec![(key.clone(), vec![(0, block)])];
        let mut inflight: HashMap<BlockKey, InflightBlock> = HashMap::new();

        process_insert_batch(
            &mut inflight,
            &weak_deps,
            entries,
            1,
            NumaNode::UNKNOWN,
            "ns",
        );

        assert!(inflight.is_empty());
        assert!(engine.read_cache.contains_keys(std::slice::from_ref(&key))[0]);
    }
}
