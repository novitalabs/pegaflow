use std::{
    collections::HashSet,
    sync::{Arc, Barrier, mpsc},
    thread,
    time::Duration,
};

use super::*;

fn key(id: usize) -> BlockKey {
    BlockKey::new("concurrent".into(), id.to_le_bytes().to_vec())
}
fn block() -> Arc<SealedBlock> {
    Arc::new(SealedBlock::from_slots(Vec::new()))
}
fn cache() -> ReadCache {
    ReadCache::new(4096, false, Some(1))
}

fn insert_sized_metadata(cache: &ReadCache, key: BlockKey) {
    // Account nonzero, synthetic footprints without invoking a CUDA allocator.
    // Publication, arbitration and all later operations use the real cache.
    let mut resident = Resident::new(key, cache.ticket(), block(), ResidentClass::Retained);
    resident.bytes = 128;
    let resident = Arc::new(resident);
    cache
        .retained
        .push(resident.generation, Arc::clone(&resident), &epoch::pin());
    cache.finish_insert(&resident);
}

fn queue_contains(cache: &ReadCache, resident: &Arc<Resident>, class: ResidentClass) -> bool {
    let guard = &epoch::pin();
    cache.clock(class).iter(guard).any(|entry| {
        Arc::ptr_eq(entry.value(), resident)
            && entry.ticket() == resident.ticket(class).load(Ordering::Acquire)
            && !resident.is_removed()
    })
}

fn assert_empty_queues(cache: &ReadCache) {
    cache.prune_queues(&epoch::pin());
    assert!(cache.retained.iter(&epoch::pin()).next().is_none());
    assert!(cache.reclaimable.iter(&epoch::pin()).next().is_none());
}

fn paused(
    worker: impl FnOnce(mpsc::SyncSender<()>, mpsc::Receiver<()>) + Send,
    foreground: impl FnOnce(),
) {
    let (ready_tx, ready_rx) = mpsc::sync_channel(0);
    let (resume_tx, resume_rx) = mpsc::sync_channel(0);
    thread::scope(|scope| {
        let task = scope.spawn(|| worker(ready_tx, resume_rx));
        ready_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("worker must reach pause point");
        foreground();
        resume_tx.send(()).unwrap();
        task.join().unwrap();
    });
}

fn wait_for_resume(ready: mpsc::SyncSender<()>, resume: mpsc::Receiver<()>) {
    ready.send(()).unwrap();
    resume
        .recv_timeout(Duration::from_secs(10))
        .expect("foreground operations must not wait for paused worker");
}

#[test]
fn lookup_reference_and_eviction_pause_windows() {
    for stage in 0..3 {
        let cache = cache();
        let value = block();
        let weak = Arc::downgrade(&value);
        cache.batch_insert(vec![(key(0), value), (key(1), block())]);
        paused(
            |ready, resume| {
                let entry = cache.lookup(&key(0)).unwrap();
                match stage {
                    0 => {
                        // The index metadata is protected, but no block access
                        // has been registered. Eviction can still free the block.
                        wait_for_resume(ready, resume);
                        assert!(entry.value().read(false).is_none());
                    }
                    1 => {
                        let reader = entry.value().read(false).unwrap();
                        wait_for_resume(ready, resume);
                        assert!(Arc::strong_count(&reader.clone_block()) >= 2);
                    }
                    _ => {
                        let held = entry.value().read(false).unwrap().clone_block();
                        wait_for_resume(ready, resume);
                        assert_eq!(Arc::strong_count(&held), 2);
                    }
                }
            },
            || {
                // Other readers of the same key must complete, not just readers
                // of an unrelated bucket. Drop their pins before pressure.
                assert_eq!(cache.get_prefix_blocks(&[key(0)]).0, 1);
                let evicted = cache.remove_clock_batch(2);
                let keys: HashSet<_> = evicted.iter().map(|(key, _)| key.clone()).collect();
                assert!(
                    keys.contains(&key(1)),
                    "paused reader must not stop other eviction"
                );
                assert_eq!(keys.contains(&key(0)), stage == 0);
                drop(evicted);
                assert_eq!(weak.upgrade().is_none(), stage == 0);
            },
        );
        drop(cache.remove_clock_batch(2));
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn pending_publication_is_helped_only_after_index_arbitration() {
    for indexed in [false, true] {
        let cache = cache();
        let pending = Arc::new(Resident::new(
            key(0),
            cache.ticket(),
            block(),
            ResidentClass::Retained,
        ));
        paused(
            |ready, resume| {
                cache
                    .retained
                    .push(pending.generation, Arc::clone(&pending), &epoch::pin());
                if indexed {
                    cache.index.get_or_insert(&pending, &epoch::pin());
                }
                wait_for_resume(ready, resume);
                if !indexed {
                    assert_eq!(
                        cache.finish_insert(&pending),
                        CacheInsertOutcome::InsertedNew
                    );
                }
            },
            || {
                cache.batch_insert(vec![(key(1), block())]);
                assert_eq!(cache.get_prefix_blocks(&[key(0)]).0, usize::from(indexed));
                let evicted = cache.remove_clock_batch(2);
                assert!(evicted.iter().any(|(key, _)| *key == self::key(1)));
                assert_eq!(evicted.iter().any(|(key, _)| *key == self::key(0)), indexed);
                if !indexed {
                    assert!(!pending.is_live());
                    assert!(queue_contains(&cache, &pending, ResidentClass::Retained));
                }
            },
        );
        drop(cache.remove_all());
        assert_empty_queues(&cache);
    }
}

#[test]
fn losing_insert_releases_payload_while_metadata_is_still_held() {
    let cache = cache();
    let value = block();
    let weak = Arc::downgrade(&value);
    let loser = Arc::new(Resident::new(
        key(0),
        cache.ticket(),
        value,
        ResidentClass::Retained,
    ));
    paused(
        |ready, resume| {
            cache
                .retained
                .push(loser.generation, Arc::clone(&loser), &epoch::pin());
            wait_for_resume(ready, resume);
            assert_eq!(
                cache.finish_insert(&loser),
                CacheInsertOutcome::AlreadyExists
            );
        },
        || {
            cache.batch_insert(vec![(key(0), block())]);
            assert!(cache.contains_keys(&[key(0)])[0]);
        },
    );
    assert!(loser.is_removed());
    assert!(
        weak.upgrade().is_none(),
        "retired metadata must not retain rejected KV payload"
    );
    assert!(cache.contains_keys(&[key(0)])[0]);
}

#[test]
fn paused_demotion_keeps_destination_and_can_be_helped() {
    let cache = cache();
    cache.batch_insert(vec![(key(0), block()), (key(1), block())]);
    let entry = cache.lookup(&key(0)).unwrap();
    let resident = entry.value();
    let pin = cache.peek_blocks(&[key(0)]);
    paused(
        |ready, resume| {
            cache.publish_demotion(resident);
            wait_for_resume(ready, resume);
            assert!(
                cache.finish_demotion(resident).is_none(),
                "another demoter completed it"
            );
        },
        || {
            assert_eq!(cache.remove_clock_batch(2)[0].0, key(1));
            assert!(
                queue_contains(&cache, resident, ResidentClass::Reclaimable),
                "pending destination must not be discarded"
            );
            assert_eq!(
                cache.mark_reclaimable_keys(&[key(0)], DemotionReason::TransferRelease),
                1
            );
            assert!(cache.is_reclaimable_for_test(&key(0)));
        },
    );
    drop(pin);
    assert_eq!(cache.remove_clock_batch(1)[0].0, key(0));
    assert_empty_queues(&cache);
}

#[test]
fn rotation_publication_survives_cleanup_and_class_migration() {
    for (stage, cleanup) in [(0, true), (1, true), (1, false)] {
        let cache = cache();
        cache.batch_insert(vec![(key(0), block()), (key(1), block())]);
        let old = cache.lookup(&key(0)).unwrap();
        let resident = old.value();
        paused(
            |ready, resume| {
                let guard = &epoch::pin();
                let entry = cache
                    .retained
                    .iter(guard)
                    .find(|entry| Arc::ptr_eq(entry.value(), resident))
                    .unwrap();
                let reader = resident.read(false).unwrap();
                cache.rotate_with(&entry, ResidentClass::Retained, &reader, |step| {
                    if step == stage {
                        ready.send(()).unwrap();
                        resume.recv_timeout(Duration::from_secs(10)).unwrap();
                    }
                });
            },
            || {
                assert!(cache.contains_keys(&[key(0)])[0]);
                if cleanup {
                    drop(cache.remove_all());
                    cache.batch_insert(vec![(key(0), block())]);
                    assert_ne!(
                        cache.resident_generations(&[key(0)])[0].1,
                        resident.generation
                    );
                } else {
                    assert_eq!(
                        cache.mark_reclaimable_keys(&[key(0)], DemotionReason::TransferRelease),
                        1
                    );
                    assert!(
                        cache
                            .remove_clock_batch(2)
                            .iter()
                            .any(|(k, _)| *k == key(1))
                    );
                }
            },
        );
        cache.prune_queues(&epoch::pin());
        assert!(!queue_contains(&cache, resident, ResidentClass::Retained));
        assert!(cache.contains_keys(&[key(0)])[0]);
        if !cleanup {
            assert!(queue_contains(&cache, resident, ResidentClass::Reclaimable));
        }
        drop(cache.remove_all());
        assert_empty_queues(&cache);
    }
}

#[test]
fn delayed_demotion_publication_does_not_restore_rotated_ticket() {
    let cache = cache();
    cache.batch_insert(vec![(key(0), block())]);
    let indexed = cache.lookup(&key(0)).unwrap();
    let resident = indexed.value();
    paused(
        |ready, resume| {
            cache.publish_demotion_with(resident, || wait_for_resume(ready, resume));
        },
        || {
            let ticket = cache.publish_demotion(resident);
            assert!(cache.finish_demotion(resident).is_some());
            let guard = &epoch::pin();
            let entry = cache
                .reclaimable
                .iter(guard)
                .find(|entry| entry.ticket() == ticket)
                .unwrap();
            let reader = resident.read(false).unwrap();
            cache.rotate(&entry, ResidentClass::Reclaimable, &reader);
            assert_ne!(resident.reclaim_ticket.load(Ordering::Acquire), ticket);
        },
    );
    cache.prune_queues(&epoch::pin());
    assert_eq!(cache.reclaimable.iter(&epoch::pin()).count(), 1);
    assert!(queue_contains(&cache, resident, ResidentClass::Reclaimable));
    assert!(cache.contains_keys(&[key(0)])[0]);
    assert_eq!(cache.remove_clock_batch(1)[0].0, key(0));
    assert_empty_queues(&cache);
}

#[test]
fn stale_metadata_deletion_cannot_remove_reinserted_key() {
    let cache = cache();
    cache.batch_insert(vec![(key(0), block())]);
    let old = cache.lookup(&key(0)).unwrap();
    let block = old
        .value()
        .read(false)
        .unwrap()
        .evict(ResidentClass::Retained)
        .unwrap();
    // A new insert helps unlink the retired old index before publishing.
    cache.batch_insert(vec![(key(0), self::block())]);
    let new_generation = cache.resident_generations(&[key(0)])[0].1;
    cache.detach(old.value());
    old.remove();
    assert_ne!(new_generation, old.value().generation);
    assert_eq!(cache.resident_generations(&[key(0)])[0].1, new_generation);
    drop(block);
}

#[test]
fn paused_reader_survives_cleanup_and_same_key_reinsertion() {
    let cache = cache();
    let old_block = block();
    let weak = Arc::downgrade(&old_block);
    cache.batch_insert(vec![(key(0), old_block)]);
    let old_generation = cache.resident_generations(&[key(0)])[0].1;
    paused(
        |ready, resume| {
            let old = cache.lookup(&key(0)).unwrap();
            let reader = old.value().read(false).unwrap();
            wait_for_resume(ready, resume);
            assert!(old.value().is_removed());
            assert!(Arc::ptr_eq(&reader.clone_block(), &weak.upgrade().unwrap()));
        },
        || {
            drop(cache.remove_all());
            assert!(
                weak.upgrade().is_some(),
                "paused reader still owns access to old payload"
            );
            insert_sized_metadata(&cache, key(0));
            assert_ne!(cache.resident_generations(&[key(0)])[0].1, old_generation);
            assert_eq!(cache.get_prefix_blocks(&[key(0)]).0, 1);
            assert_eq!(cache.class_bytes().retained, 128);
        },
    );
    assert!(
        weak.upgrade().is_none(),
        "last old reader must release the old payload"
    );
    assert!(cache.contains_keys(&[key(0)])[0]);
    assert_eq!(cache.class_bytes().retained, 128);
    drop(cache.remove_all());
    assert_eq!(cache.class_bytes(), ClassBytes::default());
}

#[test]
fn concurrent_evictors_return_each_resident_at_most_once() {
    let cache = cache();
    const BLOCKS: usize = 2048;
    cache.batch_insert((0..BLOCKS).map(|id| (key(id), block())).collect());
    let start = Barrier::new(8);
    let all = thread::scope(|scope| {
        let tasks: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    let mut keys = Vec::new();
                    for _ in 0..64 {
                        keys.extend(cache.remove_clock_batch(32).into_iter().map(|(key, _)| key));
                    }
                    keys
                })
            })
            .collect();
        tasks
            .into_iter()
            .flat_map(|task| task.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(all.len(), BLOCKS);
    assert_eq!(all.iter().collect::<HashSet<_>>().len(), BLOCKS);
    assert_empty_queues(&cache);
}

#[test]
fn mixed_operations_preserve_quiescent_index_fifo_and_accounting() {
    let cache = cache();
    let keys: Vec<_> = (0..128).map(key).collect();
    let start = Barrier::new(8);
    thread::scope(|scope| {
        for worker in 0..8 {
            let keys = &keys;
            let cache = &cache;
            let start = &start;
            scope.spawn(move || {
                start.wait();
                for i in 0..2000 {
                    let key = &keys[(i * 17 + worker * 31) % keys.len()];
                    match (i + worker) % 7 {
                        0 | 1 => {
                            insert_sized_metadata(cache, key.clone());
                        }
                        2 => {
                            drop(cache.get_blocks(std::slice::from_ref(key)));
                        }
                        3 => {
                            cache.mark_reclaimable_keys(
                                std::slice::from_ref(key),
                                DemotionReason::ReclaimHint,
                            );
                        }
                        4 => {
                            drop(cache.remove_clock_batch(3));
                        }
                        5 => {
                            cache.backup_candidates(u64::MAX, 4);
                        }
                        _ => {
                            if i % 31 == 0 {
                                drop(cache.remove_all());
                            }
                        }
                    }
                }
            });
        }
    });
    let mut generations = HashSet::new();
    let mut expected = ClassBytes::default();
    let guard = &epoch::pin();
    for resident in cache.index.iter(guard) {
        assert!(resident.is_live());
        assert!(generations.insert(resident.generation));
        match resident.class() {
            ResidentClass::Retained => {
                expected.retained += resident.bytes;
            }
            ResidentClass::Reclaimable => {
                expected.reclaimable += resident.bytes;
            }
        };
        assert!(queue_contains(&cache, resident, resident.class()));
    }
    assert_eq!(cache.class_bytes(), expected);
    assert_eq!(
        cache.reclaimable_bytes.load(Ordering::Relaxed),
        expected.reclaimable as i64
    );
    assert_eq!(
        cache.retained_bytes.load(Ordering::Relaxed),
        expected.retained as i64
    );
    cache.prune_queues(guard);
    let queued: HashSet<_> = [ResidentClass::Reclaimable, ResidentClass::Retained]
        .into_iter()
        .flat_map(|class| {
            cache
                .clock(class)
                .iter(guard)
                .map(|entry| entry.value().generation)
        })
        .collect();
    assert_eq!(queued, generations);
    drop(cache.remove_all());
    assert!(cache.index.iter(guard).next().is_none());
    assert_empty_queues(&cache);
}

#[test]
fn membership_and_peek_do_not_update_lfu_frequency() {
    let cache = ReadCache::new(4096, true, Some(1));
    cache.batch_insert(vec![(key(0), block())]);
    let frequency = cache.frequency.as_ref().unwrap();
    let before = frequency.get(key(0));
    assert!(cache.contains_keys(&[key(0)])[0]);
    drop(cache.peek_blocks(&[key(0)]));
    assert_eq!(frequency.get(key(0)), before);
    drop(cache.get_blocks(&[key(0)]));
    assert!(frequency.get(key(0)) > before);
}

#[test]
fn bounded_backup_prunes_stale_positions_behind_a_cold_head() {
    let cache = cache();
    let keys: Vec<_> = (0..66).map(key).collect();
    cache.batch_insert(keys.iter().map(|key| (key.clone(), block())).collect());
    cache.mark_reclaimable_keys(&keys[1..65], DemotionReason::TransferRelease);
    let mut found_tail = false;
    for _ in 0..16 {
        let candidates = cache.backup_candidates(u64::MAX, 2);
        assert_eq!(candidates[0].0, keys[0], "cold head must not move");
        if candidates.iter().any(|(key, _)| *key == keys[65]) {
            found_tail = true;
            break;
        }
    }
    assert!(
        found_tail,
        "bounded scans must progress through stale interior records"
    );
    let removed: Vec<_> = cache
        .remove_clock_batch(66)
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    assert_eq!(removed[..64], keys[1..65]);
    assert_eq!(removed[64..], [keys[0].clone(), keys[65].clone()]);
}
