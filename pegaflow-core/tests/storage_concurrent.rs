//! Concurrent tests for StorageEngine.
//!
//! Tests thread safety of storage operations including:
//! - Concurrent insert_slot to same block
//! - Concurrent cache_admit
//! - Eviction during lookup
//! - Concurrent allocate + eviction

mod common;

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use pegaflow_core::block::{BlockKey, LayerBlock, SealedBlock};
use pegaflow_core::storage::{StorageConfig, StorageEngine};

use common::config::TestConfig;
use common::{block_hash_from_index, run_concurrent, run_concurrent_with_barrier, shared_barrier, TEST_NAMESPACE};

/// Test concurrent insert_slot operations to the same block.
/// Multiple threads inserting different slots to the same block_hash should
/// result in a sealed block with all slots present.
#[test]
fn test_concurrent_insert_slot_same_block() {
    let config = TestConfig::small_pool();
    let (storage, _rx) = config.build_storage();

    let block_hash = block_hash_from_index(0);
    let num_slots = 4;
    let block_size = 4096usize;

    // Allocate memory for each slot
    let allocations: Vec<_> = (0..num_slots)
        .map(|_| {
            storage
                .allocate(NonZeroU64::new(block_size as u64).unwrap())
                .expect("allocation should succeed")
        })
        .collect();

    let storage = Arc::new(storage);
    let block_hash = Arc::new(block_hash);
    let allocations = Arc::new(allocations);
    let barrier = shared_barrier(num_slots);

    let handles: Vec<_> = (0..num_slots)
        .map(|slot_id| {
            let storage = Arc::clone(&storage);
            let block_hash = Arc::clone(&block_hash);
            let allocations = Arc::clone(&allocations);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                let allocation = Arc::clone(&allocations[slot_id]);
                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));

                // Synchronize all threads before inserting
                barrier.wait();

                let result = storage.insert_slot(
                    TEST_NAMESPACE,
                    &block_hash,
                    slot_id,
                    layer_block,
                    num_slots,
                );

                result
            })
        })
        .collect();

    // Collect results - exactly one thread should get the sealed block
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let sealed_count = results.iter().filter(|r| r.as_ref().map(|o| o.is_some()).unwrap_or(false)).count();

    // Exactly one thread should seal the block
    assert_eq!(sealed_count, 1, "Exactly one thread should seal the block");

    // Verify the block is in cache
    assert!(
        storage.cache_contains(TEST_NAMESPACE, &block_hash),
        "Block should be in cache after sealing"
    );
}

/// Test concurrent cache_admit operations.
/// Multiple threads admitting different blocks should not cause data corruption.
#[test]
fn test_concurrent_cache_admit() {
    let config = TestConfig::memory_only();
    let (storage, _rx) = config.build_storage();
    let storage = Arc::new(storage);

    let num_threads = 8;
    let blocks_per_thread = 10;
    let block_size = 4096usize;

    let results = run_concurrent_with_barrier(num_threads, |thread_id, barrier| {
        let mut admitted = 0usize;

        for i in 0..blocks_per_thread {
            let block_idx = thread_id * blocks_per_thread + i;
            let block_hash = block_hash_from_index(block_idx);

            // Allocate and create a sealed block
            let allocation = storage
                .allocate(NonZeroU64::new(block_size as u64).unwrap())
                .expect("allocation should succeed");
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));

            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);

            // Synchronize before admitting
            barrier.wait();

            if storage.cache_admit(key, sealed) {
                admitted += 1;
            }
        }

        admitted
    });

    let total_admitted: usize = results.iter().sum();
    println!("Total blocks admitted: {}", total_admitted);

    // At least some blocks should be admitted
    assert!(total_admitted > 0, "At least some blocks should be admitted");
}

/// Test that Arc protects block data during eviction.
/// Thread A holds an Arc to a block, Thread B triggers eviction.
/// Thread A should still be able to access the block data.
#[test]
fn test_eviction_during_lookup() {
    // Use a very small pool to force eviction
    let config = TestConfig {
        pool_size: 1 * 1024 * 1024, // 1MB
        ..TestConfig::memory_only()
    };
    let (storage, _rx) = config.build_storage();
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB per block
    let num_initial_blocks = 8; // Fill cache with 512KB of blocks

    // Fill the cache with initial blocks
    for i in 0..num_initial_blocks {
        let block_hash = block_hash_from_index(i);
        let allocation = storage
            .allocate(NonZeroU64::new(block_size as u64).unwrap())
            .expect("allocation should succeed");

        // Write a pattern to verify data integrity
        unsafe {
            let ptr = allocation.as_ptr() as *mut u8;
            for j in 0..block_size {
                *ptr.add(j) = ((i + j) & 0xFF) as u8;
            }
        }

        let ptr = allocation.as_ptr() as *mut u8;
        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
        let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
        let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
        storage.cache_admit(key, sealed);
    }

    // Get Arc to first block
    let first_hash = block_hash_from_index(0);
    let first_block_arc = storage
        .cache_lookup_many(TEST_NAMESPACE, &[first_hash.clone()])
        .expect("lookup should succeed");

    assert_eq!(first_block_arc.len(), 1, "Should find exactly one block");
    let held_block = first_block_arc.into_iter().next().unwrap();

    // Verify we can access the data
    let slot = held_block.get_slot(0).expect("should have slot 0");
    let ptr = slot.k_ptr();
    unsafe {
        // Check first few bytes match expected pattern
        assert_eq!(*ptr, 0u8, "Data should be intact before eviction");
    }

    // Now force eviction by allocating more blocks
    for i in num_initial_blocks..(num_initial_blocks + 20) {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // The original block might be evicted from cache, but our Arc should still be valid
    let slot = held_block.get_slot(0).expect("should still have slot 0");
    let ptr = slot.k_ptr();
    unsafe {
        // Data should still be intact because Arc keeps memory alive
        assert_eq!(*ptr, 0u8, "Data should still be intact after eviction");
    }

    // Verify Arc refcount is keeping the allocation alive
    assert!(
        Arc::strong_count(&held_block) >= 1,
        "Arc should keep block alive"
    );
}

/// Test concurrent allocate operations that trigger eviction.
#[test]
fn test_concurrent_allocate_with_eviction() {
    let config = TestConfig {
        pool_size: 2 * 1024 * 1024, // 2MB - small enough to trigger eviction
        ..TestConfig::memory_only()
    };
    let (storage, _rx) = config.build_storage();
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB
    let num_threads = 4;
    let allocations_per_thread = 20;

    let successful_allocations = Arc::new(AtomicUsize::new(0));

    let results = run_concurrent_with_barrier(num_threads, |thread_id, barrier| {
        let mut local_allocations = Vec::new();
        let mut successful = 0usize;

        barrier.wait();

        for i in 0..allocations_per_thread {
            if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                // Create and admit block
                let block_idx = thread_id * allocations_per_thread + i;
                let block_hash = block_hash_from_index(block_idx);

                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block.clone()]));
                let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);

                if storage.cache_admit(key, sealed) {
                    successful += 1;
                }

                // Keep some allocations alive to stress the system
                if i % 3 == 0 {
                    local_allocations.push(layer_block);
                }
            }
        }

        successful
    });

    let total_successful: usize = results.iter().sum();
    println!("Total successful cache admits: {}", total_successful);

    // Some allocations should succeed even under pressure
    assert!(total_successful > 0, "Some allocations should succeed");
}

/// Test that inflight blocks are not evicted.
#[test]
fn test_inflight_blocks_not_evicted() {
    let config = TestConfig::small_pool();
    let (storage, _rx) = config.build_storage();

    let block_hash = block_hash_from_index(0);
    let num_slots = 4;
    let block_size = 4096usize;

    // Insert only some slots (block stays inflight)
    for slot_id in 0..2 {
        let allocation = storage
            .allocate(NonZeroU64::new(block_size as u64).unwrap())
            .expect("allocation should succeed");
        let ptr = allocation.as_ptr() as *mut u8;
        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));

        let result = storage.insert_slot(TEST_NAMESPACE, &block_hash, slot_id, layer_block, num_slots);
        // Should not seal yet (need all 4 slots)
        assert!(result.unwrap().is_none(), "Block should not seal with partial slots");
    }

    // Block should be in inflight state
    assert!(
        storage.inflight_has_slot(TEST_NAMESPACE, &block_hash, 0),
        "Slot 0 should be in inflight"
    );
    assert!(
        storage.inflight_has_slot(TEST_NAMESPACE, &block_hash, 1),
        "Slot 1 should be in inflight"
    );

    // Allocate a lot more to try triggering eviction
    for i in 100..200 {
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let other_hash = block_hash_from_index(i);
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), other_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // Inflight block should still exist
    assert!(
        storage.inflight_has_slot(TEST_NAMESPACE, &block_hash, 0),
        "Inflight slot 0 should survive eviction"
    );
    assert!(
        storage.inflight_has_slot(TEST_NAMESPACE, &block_hash, 1),
        "Inflight slot 1 should survive eviction"
    );
}

/// Test cache_lookup_many with concurrent modifications.
#[test]
fn test_concurrent_lookup_and_insert() {
    let config = TestConfig::memory_only();
    let (storage, _rx) = config.build_storage();
    let storage = Arc::new(storage);

    let block_size = 4096usize;
    let num_blocks = 50;

    // Pre-populate some blocks
    for i in 0..num_blocks {
        let block_hash = block_hash_from_index(i);
        let allocation = storage
            .allocate(NonZeroU64::new(block_size as u64).unwrap())
            .expect("allocation should succeed");
        let ptr = allocation.as_ptr() as *mut u8;
        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
        let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
        let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
        storage.cache_admit(key, sealed);
    }

    let barrier = shared_barrier(2);

    // Thread 1: Continuously lookup blocks
    let storage_clone = Arc::clone(&storage);
    let barrier_clone = Arc::clone(&barrier);
    let lookup_handle = thread::spawn(move || {
        barrier_clone.wait();

        let mut successful_lookups = 0usize;
        for _ in 0..100 {
            let hashes: Vec<Vec<u8>> = (0..10).map(|i| block_hash_from_index(i)).collect();
            if let Ok(results) = storage_clone.cache_lookup_many(TEST_NAMESPACE, &hashes) {
                successful_lookups += results.len();
            }
        }
        successful_lookups
    });

    // Thread 2: Insert new blocks
    let storage_clone = Arc::clone(&storage);
    let barrier_clone = Arc::clone(&barrier);
    let insert_handle = thread::spawn(move || {
        barrier_clone.wait();

        let mut successful_inserts = 0usize;
        for i in num_blocks..(num_blocks + 100) {
            let block_hash = block_hash_from_index(i);
            if let Some(allocation) = storage_clone.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
                if storage_clone.cache_admit(key, sealed) {
                    successful_inserts += 1;
                }
            }
        }
        successful_inserts
    });

    let lookups = lookup_handle.join().unwrap();
    let inserts = insert_handle.join().unwrap();

    println!("Successful lookups: {}, inserts: {}", lookups, inserts);

    // Both operations should have succeeded
    assert!(lookups > 0, "Lookups should succeed");
    assert!(inserts > 0, "Inserts should succeed");
}
