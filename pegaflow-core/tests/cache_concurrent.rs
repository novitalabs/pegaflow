//! Concurrent tests for TinyLfuCache.
//!
//! Tests thread safety of cache operations including:
//! - Concurrent insert/get operations
//! - TinyLFU frequency counter concurrent updates
//! - Concurrent eviction behavior

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use pegaflow_core::block::{BlockKey, LayerBlock, SealedBlock};
use pegaflow_core::pinned_pool::PinnedMemoryPool;

use common::{block_hash_from_index, run_concurrent_with_barrier, shared_barrier, TEST_NAMESPACE};

/// Create a mock SealedBlock for testing cache operations.
fn create_mock_sealed_block(pool: &Arc<PinnedMemoryPool>, size: usize) -> Arc<SealedBlock> {
    use std::num::NonZeroU64;

    let allocation = pool
        .allocate(NonZeroU64::new(size as u64).unwrap())
        .expect("allocation should succeed");
    let ptr = allocation.as_ptr() as *mut u8;
    let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, size, allocation));
    Arc::new(SealedBlock::from_slots(vec![layer_block]))
}

/// Test concurrent insert and get operations on TinyLfuCache via StorageEngine.
#[test]
fn test_concurrent_cache_insert_get() {
    use pegaflow_core::storage::{StorageConfig, StorageEngine};

    let config = StorageConfig::default();
    let (storage, _rx) = StorageEngine::new_with_config(
        64 * 1024 * 1024, // 64MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let num_threads = 4;
    let ops_per_thread = 100;
    let block_size = 4096usize;

    let barrier = shared_barrier(num_threads);

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let storage = Arc::clone(&storage);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                let mut inserts = 0usize;
                let mut hits = 0usize;

                barrier.wait();

                for i in 0..ops_per_thread {
                    let block_idx = thread_id * ops_per_thread + i;
                    let block_hash = block_hash_from_index(block_idx);

                    // Insert
                    if let Some(allocation) = storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
                        let ptr = allocation.as_ptr() as *mut u8;
                        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                        let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                        let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash.clone());

                        if storage.cache_admit(key, sealed) {
                            inserts += 1;
                        }
                    }

                    // Try to get a random previously inserted block
                    if i > 0 {
                        let lookup_idx = thread_id * ops_per_thread + (i % ops_per_thread);
                        let lookup_hash = block_hash_from_index(lookup_idx);
                        if storage.cache_contains(TEST_NAMESPACE, &lookup_hash) {
                            hits += 1;
                        }
                    }
                }

                (inserts, hits)
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let total_inserts: usize = results.iter().map(|(i, _)| i).sum();
    let total_hits: usize = results.iter().map(|(_, h)| h).sum();

    println!("Total inserts: {}, total hits: {}", total_inserts, total_hits);
    assert!(total_inserts > 0, "Some inserts should succeed");
}

/// Test concurrent frequency counter updates.
/// TinyLFU uses atomic counters that should handle concurrent updates correctly.
#[test]
fn test_concurrent_frequency_updates() {
    use pegaflow_core::storage::{StorageConfig, StorageEngine};

    let config = StorageConfig {
        enable_lfu_admission: true,
        ..Default::default()
    };
    let (storage, _rx) = StorageEngine::new_with_config(
        32 * 1024 * 1024, // 32MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 4096usize;

    // Create a set of blocks that will be repeatedly accessed
    let num_hot_blocks = 10;
    for i in 0..num_hot_blocks {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    let num_threads = 8;
    let accesses_per_thread = 1000;
    let access_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let storage = Arc::clone(&storage);
            let access_count = Arc::clone(&access_count);

            thread::spawn(move || {
                for i in 0..accesses_per_thread {
                    // Access hot blocks repeatedly to increase their frequency
                    let block_idx = i % num_hot_blocks;
                    let block_hash = block_hash_from_index(block_idx);

                    // cache_contains triggers frequency update via get
                    if storage.cache_contains(TEST_NAMESPACE, &block_hash) {
                        access_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let total_accesses = access_count.load(Ordering::Relaxed);
    println!("Total successful cache accesses: {}", total_accesses);

    // Most accesses should hit since we're accessing existing blocks
    let expected_min = num_threads * accesses_per_thread / 2;
    assert!(
        total_accesses >= expected_min,
        "Expected at least {} accesses, got {}",
        expected_min,
        total_accesses
    );
}

/// Test that hot blocks survive eviction due to TinyLFU.
#[test]
fn test_hot_blocks_survive_eviction() {
    use pegaflow_core::storage::{StorageConfig, StorageEngine};

    let config = StorageConfig {
        enable_lfu_admission: true,
        ..Default::default()
    };
    let (storage, _rx) = StorageEngine::new_with_config(
        4 * 1024 * 1024, // 4MB - small to force eviction
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB blocks

    // Insert "hot" blocks and access them multiple times
    let num_hot_blocks = 5;
    for i in 0..num_hot_blocks {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash.clone());
            storage.cache_admit(key, sealed);
        }

        // Access multiple times to increase frequency
        for _ in 0..10 {
            storage.cache_contains(TEST_NAMESPACE, &block_hash);
        }
    }

    // Insert "cold" blocks that should be evicted first
    for i in 100..200 {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // Check that hot blocks are still present
    let mut hot_blocks_present = 0;
    for i in 0..num_hot_blocks {
        let block_hash = block_hash_from_index(i);
        if storage.cache_contains(TEST_NAMESPACE, &block_hash) {
            hot_blocks_present += 1;
        }
    }

    println!("Hot blocks still present: {}/{}", hot_blocks_present, num_hot_blocks);

    // With TinyLFU, hot blocks should have higher survival rate
    // Note: This is a probabilistic test, some blocks might be evicted
    assert!(
        hot_blocks_present >= num_hot_blocks / 2,
        "At least half of hot blocks should survive"
    );
}

/// Test concurrent eviction (LRU removal) behavior.
#[test]
fn test_concurrent_eviction() {
    use pegaflow_core::storage::{StorageConfig, StorageEngine};

    let config = StorageConfig::default();
    let (storage, _rx) = StorageEngine::new_with_config(
        2 * 1024 * 1024, // 2MB - very small to force frequent eviction
        false,
        config,
    );
    let storage = Arc::new(storage);

    let num_threads = 4;
    let inserts_per_thread = 50;
    let block_size = 64 * 1024usize; // 64KB

    let barrier = shared_barrier(num_threads);
    let eviction_triggered = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let storage = Arc::clone(&storage);
            let barrier = Arc::clone(&barrier);
            let eviction_triggered = Arc::clone(&eviction_triggered);

            thread::spawn(move || {
                barrier.wait();

                for i in 0..inserts_per_thread {
                    let block_idx = thread_id * 1000 + i;
                    let block_hash = block_hash_from_index(block_idx);

                    // Allocation might fail when pool is exhausted (eviction in progress)
                    match storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
                        Some(allocation) => {
                            let ptr = allocation.as_ptr() as *mut u8;
                            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);

                            // cache_admit returns false if TinyLFU rejects the block
                            // This is normal behavior, not an error
                            let _ = storage.cache_admit(key, sealed);
                        }
                        None => {
                            // Pool exhausted, eviction is happening
                            eviction_triggered.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let evictions = eviction_triggered.load(Ordering::Relaxed);
    println!("Allocation failures (pool exhausted): {}", evictions);

    // Test passes if no deadlocks or panics occurred
    // Some eviction-related allocation failures are expected with a small pool
}

/// Test that cache operations don't corrupt data under concurrent access.
#[test]
fn test_data_integrity_under_concurrent_access() {
    use pegaflow_core::storage::{StorageConfig, StorageEngine};

    let config = StorageConfig::default();
    let (storage, _rx) = StorageEngine::new_with_config(
        32 * 1024 * 1024, // 32MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 4096usize;
    let num_blocks = 100;

    // Insert blocks with known patterns
    for i in 0..num_blocks {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(std::num::NonZeroU64::new(block_size as u64).unwrap()) {
            // Write pattern: first byte = block index
            unsafe {
                let ptr = allocation.as_ptr() as *mut u8;
                *ptr = (i & 0xFF) as u8;
            }

            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    let num_threads = 8;
    let reads_per_thread = 500;
    let corruption_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let storage = Arc::clone(&storage);
            let corruption_count = Arc::clone(&corruption_count);

            thread::spawn(move || {
                for i in 0..reads_per_thread {
                    let block_idx = i % num_blocks;
                    let block_hash = block_hash_from_index(block_idx);

                    if let Ok(blocks) = storage.cache_lookup_many(TEST_NAMESPACE, &[block_hash]) {
                        if let Some(block) = blocks.into_iter().next() {
                            if let Some(slot) = block.get_slot(0) {
                                let ptr = slot.k_ptr();
                                unsafe {
                                    let first_byte = *ptr;
                                    if first_byte != (block_idx & 0xFF) as u8 {
                                        corruption_count.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let corruptions = corruption_count.load(Ordering::Relaxed);
    assert_eq!(corruptions, 0, "No data corruption should occur");
}
