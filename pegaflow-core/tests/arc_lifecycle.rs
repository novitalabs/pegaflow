//! Arc lifecycle tests for block memory safety.
//!
//! Tests that Arc reference counting properly protects block data:
//! - Load during eviction
//! - Multi-layer Arc sharing
//! - SealedBlock cross-thread transfer
//! - Memory lifetime guarantees

mod common;

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use pegaflow_core::block::{BlockKey, LayerBlock, SealedBlock};
use pegaflow_core::pinned_pool::PinnedMemoryPool;
use pegaflow_core::storage::{StorageConfig, StorageEngine};

use common::{block_hash_from_index, shared_barrier, TEST_NAMESPACE};

/// Skip test if CUDA is not available.
macro_rules! skip_without_cuda {
    () => {
        if cudarc::driver::CudaContext::new(0).is_err() {
            eprintln!("Skipping test: CUDA not available");
            return;
        }
    };
}

/// Test that holding an Arc prevents memory from being freed during eviction.
#[test]
fn test_arc_prevents_eviction_free() {
    skip_without_cuda!();

    let config = StorageConfig::default();
    let (storage, _rx) = StorageEngine::new_with_config(
        2 * 1024 * 1024, // 2MB - small to force eviction
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB
    let pattern_byte = 0xAB_u8;

    // Insert a block with a known pattern
    let target_hash = block_hash_from_index(0);
    {
        let allocation = storage
            .allocate(NonZeroU64::new(block_size as u64).unwrap())
            .expect("allocation should succeed");

        // Write pattern
        unsafe {
            let ptr = allocation.as_ptr() as *mut u8;
            for i in 0..block_size {
                *ptr.add(i) = pattern_byte;
            }
        }

        let ptr = allocation.as_ptr() as *mut u8;
        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
        let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
        let key = BlockKey::new(TEST_NAMESPACE.to_string(), target_hash.clone());
        storage.cache_admit(key, sealed);
    }

    // Lookup and hold the Arc
    let held_blocks = storage
        .cache_lookup_many(TEST_NAMESPACE, &[target_hash.clone()])
        .expect("lookup should succeed");
    assert_eq!(held_blocks.len(), 1);
    let held_block = held_blocks.into_iter().next().unwrap();

    // Verify pattern before eviction
    {
        let slot = held_block.get_slot(0).unwrap();
        let ptr = slot.k_ptr();
        unsafe {
            assert_eq!(*ptr, pattern_byte, "Pattern should be intact before eviction");
        }
    }

    // Force eviction by allocating many more blocks
    for i in 100..200 {
        let hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), hash);
            storage.cache_admit(key, sealed);
        }
    }

    // The target block may have been evicted from cache
    let still_in_cache = storage.cache_contains(TEST_NAMESPACE, &target_hash);
    println!("Target block still in cache: {}", still_in_cache);

    // But our held Arc should still be valid!
    {
        let slot = held_block.get_slot(0).unwrap();
        let ptr = slot.k_ptr();
        unsafe {
            assert_eq!(
                *ptr, pattern_byte,
                "Pattern should be intact after eviction - Arc protects memory"
            );
        }
    }

    // Verify refcount
    let refcount = Arc::strong_count(&held_block);
    assert!(refcount >= 1, "Arc should keep block alive");
}

/// Test that multiple layers can share the same PinnedAllocation.
#[test]
fn test_multi_layer_arc_sharing() {
    skip_without_cuda!();

    let pool = Arc::new(PinnedMemoryPool::new(
        16 * 1024 * 1024, // 16MB
        false,
        None,
    ));

    let block_size = 8192usize;
    let num_layers = 4;

    // Allocate a single piece of memory
    let allocation = Arc::new(
        pool.allocate(NonZeroU64::new(block_size as u64).unwrap())
            .expect("allocation should succeed"),
    );

    // Write a pattern
    unsafe {
        let ptr = (*allocation).as_ptr() as *mut u8;
        for i in 0..block_size {
            *ptr.add(i) = (i & 0xFF) as u8;
        }
    }

    // Create multiple LayerBlocks sharing the same allocation
    let layer_blocks: Vec<Arc<LayerBlock>> = (0..num_layers)
        .map(|_| {
            let ptr = (*allocation).as_ptr() as *mut u8;
            Arc::new(LayerBlock::new_contiguous(
                ptr,
                block_size,
                Arc::clone(&allocation),
            ))
        })
        .collect();

    // Create a SealedBlock from these layers
    let sealed = Arc::new(SealedBlock::from_slots(layer_blocks.clone()));

    // Verify allocation refcount (original + num_layers in LayerBlocks + 1 in SealedBlock creation)
    let alloc_refcount = Arc::strong_count(&allocation);
    println!("Allocation refcount with {} layers: {}", num_layers, alloc_refcount);
    assert!(alloc_refcount >= num_layers + 1, "Each layer should hold a reference");

    // Drop the sealed block
    drop(sealed);

    // Drop layer_blocks one by one
    for (i, lb) in layer_blocks.into_iter().enumerate() {
        drop(lb);
        let remaining = Arc::strong_count(&allocation);
        println!("After dropping layer {}: refcount = {}", i, remaining);
    }

    // Only our original reference should remain
    let final_refcount = Arc::strong_count(&allocation);
    assert_eq!(final_refcount, 1, "Only original reference should remain");

    // Verify memory is still valid
    unsafe {
        let ptr = (*allocation).as_ptr() as *const u8;
        assert_eq!(*ptr, 0, "Memory should still be accessible");
    }
}

/// Test SealedBlock can be safely transferred between threads.
#[test]
fn test_sealed_block_cross_thread_transfer() {
    skip_without_cuda!();

    let pool = Arc::new(PinnedMemoryPool::new(
        8 * 1024 * 1024,
        false,
        None,
    ));

    let block_size = 4096usize;
    let pattern = 0xCD_u8;

    // Create a sealed block in main thread
    let allocation = pool
        .allocate(NonZeroU64::new(block_size as u64).unwrap())
        .expect("allocation should succeed");

    unsafe {
        let ptr = allocation.as_ptr() as *mut u8;
        for i in 0..block_size {
            *ptr.add(i) = pattern;
        }
    }

    let ptr = allocation.as_ptr() as *mut u8;
    let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
    let sealed: Arc<SealedBlock> = Arc::new(SealedBlock::from_slots(vec![layer_block]));

    // Transfer to another thread
    let sealed_clone = Arc::clone(&sealed);
    let handle = thread::spawn(move || {
        // Verify data in other thread
        let slot = sealed_clone.get_slot(0).unwrap();
        let ptr = slot.k_ptr();

        for i in 0..block_size {
            unsafe {
                let byte = *ptr.add(i);
                assert_eq!(byte, pattern, "Data should be intact in other thread at offset {}", i);
            }
        }

        // Return the Arc back
        sealed_clone
    });

    // Get it back
    let returned = handle.join().unwrap();

    // Verify it's still the same block
    assert!(Arc::ptr_eq(&sealed, &returned), "Should be the same Arc");

    // Verify data in main thread again
    let slot = sealed.get_slot(0).unwrap();
    let ptr = slot.k_ptr();
    unsafe {
        assert_eq!(*ptr, pattern, "Data should be intact after round-trip");
    }
}

/// Test concurrent Arc access from multiple threads.
#[test]
fn test_concurrent_arc_access() {
    skip_without_cuda!();

    let pool = Arc::new(PinnedMemoryPool::new(
        8 * 1024 * 1024,
        false,
        None,
    ));

    let block_size = 4096usize;

    // Create a sealed block
    let allocation = pool
        .allocate(NonZeroU64::new(block_size as u64).unwrap())
        .expect("allocation should succeed");

    // Write index-based pattern
    unsafe {
        let ptr = allocation.as_ptr() as *mut u8;
        for i in 0..block_size {
            *ptr.add(i) = (i & 0xFF) as u8;
        }
    }

    let ptr = allocation.as_ptr() as *mut u8;
    let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
    let sealed: Arc<SealedBlock> = Arc::new(SealedBlock::from_slots(vec![layer_block]));

    let num_threads = 8;
    let reads_per_thread = 1000;
    let error_count = Arc::new(AtomicUsize::new(0));
    let barrier = shared_barrier(num_threads);

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let sealed = Arc::clone(&sealed);
            let error_count = Arc::clone(&error_count);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                for _ in 0..reads_per_thread {
                    let slot = sealed.get_slot(0).unwrap();
                    let ptr = slot.k_ptr();

                    // Verify pattern at random offsets
                    for offset in [0, 100, 1000, 4000].iter() {
                        if *offset < block_size {
                            unsafe {
                                let expected = (*offset & 0xFF) as u8;
                                let actual = *ptr.add(*offset);
                                if actual != expected {
                                    error_count.fetch_add(1, Ordering::Relaxed);
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

    let errors = error_count.load(Ordering::Relaxed);
    assert_eq!(errors, 0, "No data corruption should occur during concurrent reads");

    // Verify final refcount
    let final_refcount = Arc::strong_count(&sealed);
    assert_eq!(final_refcount, 1, "Only main thread reference should remain");
}

/// Test that dropping one reference doesn't affect others.
#[test]
fn test_independent_arc_lifetimes() {
    skip_without_cuda!();

    let pool = Arc::new(PinnedMemoryPool::new(
        8 * 1024 * 1024,
        false,
        None,
    ));

    let block_size = 4096usize;

    let allocation = pool
        .allocate(NonZeroU64::new(block_size as u64).unwrap())
        .expect("allocation should succeed");

    unsafe {
        let ptr = allocation.as_ptr() as *mut u8;
        *ptr = 0x42;
    }

    let ptr = allocation.as_ptr() as *mut u8;
    let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
    let sealed: Arc<SealedBlock> = Arc::new(SealedBlock::from_slots(vec![layer_block]));

    // Create multiple clones
    let clone1 = Arc::clone(&sealed);
    let clone2 = Arc::clone(&sealed);
    let clone3 = Arc::clone(&sealed);

    assert_eq!(Arc::strong_count(&sealed), 4);

    // Drop clones in different order
    drop(clone2);
    assert_eq!(Arc::strong_count(&sealed), 3);

    // Remaining clones should still work
    {
        let slot = clone1.get_slot(0).unwrap();
        let ptr = slot.k_ptr();
        unsafe {
            assert_eq!(*ptr, 0x42, "clone1 should still access data");
        }
    }

    drop(clone1);
    assert_eq!(Arc::strong_count(&sealed), 2);

    {
        let slot = clone3.get_slot(0).unwrap();
        let ptr = slot.k_ptr();
        unsafe {
            assert_eq!(*ptr, 0x42, "clone3 should still access data");
        }
    }

    drop(clone3);
    assert_eq!(Arc::strong_count(&sealed), 1);

    // Original should still work
    {
        let slot = sealed.get_slot(0).unwrap();
        let ptr = slot.k_ptr();
        unsafe {
            assert_eq!(*ptr, 0x42, "original should still access data");
        }
    }
}

/// Test Arc protection during simulated load operation.
#[test]
fn test_simulated_load_during_eviction() {
    skip_without_cuda!();

    let config = StorageConfig::default();
    let (storage, _rx) = StorageEngine::new_with_config(
        4 * 1024 * 1024, // 4MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize;
    let num_blocks = 20;

    // Insert blocks
    for i in 0..num_blocks {
        let hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            // Write block index as pattern
            unsafe {
                let ptr = allocation.as_ptr() as *mut u8;
                *ptr = (i & 0xFF) as u8;
            }

            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), hash);
            storage.cache_admit(key, sealed);
        }
    }

    let storage_for_load = Arc::clone(&storage);
    let storage_for_evict = Arc::clone(&storage);

    let load_done = Arc::new(AtomicBool::new(false));
    let load_done_clone = Arc::clone(&load_done);

    // Thread simulating "load" - holds Arc to a block
    let load_handle = thread::spawn(move || {
        // Lookup and hold block 0
        let hash = block_hash_from_index(0);
        let blocks = storage_for_load
            .cache_lookup_many(TEST_NAMESPACE, &[hash])
            .expect("lookup should succeed");

        if let Some(block) = blocks.into_iter().next() {
            // Simulate long load operation
            thread::sleep(Duration::from_millis(100));

            // Verify data is still valid
            let slot = block.get_slot(0).unwrap();
            let ptr = slot.k_ptr();
            let value = unsafe { *ptr };

            load_done_clone.store(true, Ordering::Release);

            assert_eq!(value, 0, "Data should be protected by Arc during load");
        }
    });

    // Thread forcing eviction
    let evict_handle = thread::spawn(move || {
        // Wait a bit then force eviction
        thread::sleep(Duration::from_millis(10));

        for i in 100..200 {
            let hash = block_hash_from_index(i);
            if let Some(allocation) = storage_for_evict.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                let key = BlockKey::new(TEST_NAMESPACE.to_string(), hash);
                storage_for_evict.cache_admit(key, sealed);
            }
        }
    });

    load_handle.join().unwrap();
    evict_handle.join().unwrap();

    assert!(load_done.load(Ordering::Acquire), "Load should complete successfully");
}
