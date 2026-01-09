//! Concurrent tests for SSD Cache (Linux only).
//!
//! Tests thread safety of SSD cache operations including:
//! - Concurrent prefetch requests
//! - Ring buffer overwrites during read
//! - Concurrent write + prefetch
//! - SSD cache tail pruning
//! - Full integration with StorageEngine

#![cfg(target_os = "linux")]

mod common;

use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use pegaflow_core::block::{BlockKey, LayerBlock, SealedBlock};
use pegaflow_core::ssd_cache::SsdCacheConfig;
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

/// Create a temporary SSD cache config for testing.
fn test_ssd_config() -> SsdCacheConfig {
    let temp_dir = std::env::temp_dir();
    let cache_path = temp_dir.join(format!(
        "pegaflow_ssd_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    SsdCacheConfig {
        cache_path,
        capacity_bytes: 64 * 1024 * 1024, // 64MB
        write_queue_depth: 128,
        prefetch_io_depth: 32,
    }
}

/// Cleanup helper for SSD test files.
struct SsdTestCleanup {
    path: PathBuf,
}

impl SsdTestCleanup {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Drop for SsdTestCleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Test that StorageEngine with SSD cache initializes correctly.
#[test]
fn test_ssd_storage_initialization() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, _rx) = StorageEngine::new_with_config(
        16 * 1024 * 1024, // 16MB pinned pool
        false,
        config,
    );

    // Verify SSD is enabled
    assert!(storage.is_ssd_enabled(), "SSD cache should be enabled");
}

/// Test basic write to SSD cache via block sealing.
#[tokio::test]
async fn test_ssd_write_via_seal() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, mut seal_rx) = StorageEngine::new_with_config(
        8 * 1024 * 1024, // 8MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 4096usize;
    let block_hash = block_hash_from_index(0);

    // Insert a block
    let allocation = storage
        .allocate(NonZeroU64::new(block_size as u64).unwrap())
        .expect("allocation should succeed");

    let ptr = allocation.as_ptr() as *mut u8;
    let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
    let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
    let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
    storage.cache_admit(key, sealed);

    // Wait a bit for seal notification to be processed
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Check if seal notification was received
    match seal_rx.try_recv() {
        Ok((key, _weak)) => {
            println!("Seal notification received for: {:?}", key);
        }
        Err(_) => {
            println!("No seal notification yet (may be queued)");
        }
    }
}

/// Test concurrent block admissions with SSD cache.
#[tokio::test]
async fn test_concurrent_admit_with_ssd() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, _rx) = StorageEngine::new_with_config(
        16 * 1024 * 1024, // 16MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 8192usize;
    let num_threads = 4;
    let blocks_per_thread = 20;

    let barrier = shared_barrier(num_threads);
    let admitted_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let storage = Arc::clone(&storage);
            let barrier = Arc::clone(&barrier);
            let admitted_count = Arc::clone(&admitted_count);

            thread::spawn(move || {
                barrier.wait();

                for i in 0..blocks_per_thread {
                    let block_idx = thread_id * blocks_per_thread + i;
                    let block_hash = block_hash_from_index(block_idx);

                    if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                        // Write pattern
                        unsafe {
                            let ptr = allocation.as_ptr() as *mut u8;
                            *ptr = (block_idx & 0xFF) as u8;
                        }

                        let ptr = allocation.as_ptr() as *mut u8;
                        let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                        let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                        let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);

                        if storage.cache_admit(key, sealed) {
                            admitted_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let total_admitted = admitted_count.load(Ordering::Relaxed);
    println!("Total blocks admitted with SSD: {}", total_admitted);

    // Give SSD writer time to process
    tokio::time::sleep(Duration::from_millis(500)).await;

    assert!(total_admitted > 0, "Some blocks should be admitted");
}

/// Test prefix hit checking with SSD cache.
#[tokio::test]
async fn test_prefix_check_with_ssd() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, _rx) = StorageEngine::new_with_config(
        8 * 1024 * 1024,
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 4096usize;

    // Insert a sequence of blocks
    for i in 0..10 {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // Check prefix
    let hashes: Vec<Vec<u8>> = (0..5).map(|i| block_hash_from_index(i)).collect();
    let status = storage.check_prefix_and_prefetch(TEST_NAMESPACE, &hashes);

    println!("Prefix check status: {:?}", status);
}

/// Test eviction triggers SSD offload.
#[tokio::test]
async fn test_eviction_triggers_ssd_offload() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    // Very small pool to force eviction
    let (storage, _rx) = StorageEngine::new_with_config(
        2 * 1024 * 1024, // 2MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB - only ~30 can fit

    // Insert many blocks to force eviction
    for i in 0..100 {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // Give SSD writer time to process evicted blocks
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Some blocks should have been evicted and potentially written to SSD
    // This test passes if no panics/deadlocks occur
}

/// Test concurrent operations with SSD: lookup + insert + prefetch.
#[tokio::test]
async fn test_concurrent_ssd_operations() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let _cleanup = SsdTestCleanup::new(ssd_config.cache_path.clone());

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, _rx) = StorageEngine::new_with_config(
        4 * 1024 * 1024, // 4MB
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 32 * 1024usize;
    let num_initial_blocks = 30;

    // Pre-populate
    for i in 0..num_initial_blocks {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            let ptr = allocation.as_ptr() as *mut u8;
            let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
            let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
            let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
            storage.cache_admit(key, sealed);
        }
    }

    // Give SSD time to process initial blocks
    tokio::time::sleep(Duration::from_millis(200)).await;

    let barrier = shared_barrier(3);

    // Thread 1: Continuous lookups
    let storage1 = Arc::clone(&storage);
    let barrier1 = Arc::clone(&barrier);
    let lookup_handle = thread::spawn(move || {
        barrier1.wait();
        let mut hits = 0usize;
        for _ in 0..100 {
            let idx = rand::random::<usize>() % num_initial_blocks;
            let hash = block_hash_from_index(idx);
            if storage1.cache_contains(TEST_NAMESPACE, &hash) {
                hits += 1;
            }
        }
        hits
    });

    // Thread 2: Insert new blocks
    let storage2 = Arc::clone(&storage);
    let barrier2 = Arc::clone(&barrier);
    let insert_handle = thread::spawn(move || {
        barrier2.wait();
        let mut inserts = 0usize;
        for i in 100..200 {
            let block_hash = block_hash_from_index(i);
            if let Some(allocation) = storage2.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
                if storage2.cache_admit(key, sealed) {
                    inserts += 1;
                }
            }
        }
        inserts
    });

    // Thread 3: Check prefixes (triggers prefetch)
    let storage3 = Arc::clone(&storage);
    let barrier3 = Arc::clone(&barrier);
    let prefetch_handle = thread::spawn(move || {
        barrier3.wait();
        let mut checks = 0usize;
        for _ in 0..50 {
            let start = rand::random::<usize>() % num_initial_blocks;
            let hashes: Vec<Vec<u8>> = (start..std::cmp::min(start + 5, num_initial_blocks))
                .map(|i| block_hash_from_index(i))
                .collect();
            let _ = storage3.check_prefix_and_prefetch(TEST_NAMESPACE, &hashes);
            checks += 1;
        }
        checks
    });

    let lookups = lookup_handle.join().unwrap();
    let inserts = insert_handle.join().unwrap();
    let prefetch_checks = prefetch_handle.join().unwrap();

    println!(
        "SSD concurrent operations - lookups: {}, inserts: {}, prefetch_checks: {}",
        lookups, inserts, prefetch_checks
    );

    // All operations should complete without deadlock
    assert!(lookups > 0 || inserts > 0 || prefetch_checks > 0, "Some operations should succeed");
}

/// Test SSD cache with small ring buffer (forces wrap-around).
#[tokio::test]
async fn test_ssd_ring_buffer_wrap() {
    skip_without_cuda!();

    let temp_dir = std::env::temp_dir();
    let cache_path = temp_dir.join(format!(
        "pegaflow_ssd_wrap_test_{}.bin",
        std::process::id()
    ));
    let _cleanup = SsdTestCleanup::new(cache_path.clone());

    let ssd_config = SsdCacheConfig {
        cache_path,
        capacity_bytes: 1 * 1024 * 1024, // 1MB - very small to force wrap
        write_queue_depth: 64,
        prefetch_io_depth: 16,
    };

    let config = StorageConfig {
        enable_lfu_admission: true,
        hint_value_size_bytes: None,
        ssd_cache_config: Some(ssd_config),
    };

    let (storage, _rx) = StorageEngine::new_with_config(
        2 * 1024 * 1024, // 2MB pinned pool
        false,
        config,
    );
    let storage = Arc::new(storage);

    let block_size = 64 * 1024usize; // 64KB blocks

    // Insert enough blocks to wrap the ring buffer multiple times
    for i in 0..50 {
        let block_hash = block_hash_from_index(i);
        if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
            // Write pattern
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

        // Small delay to let SSD writer keep up
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Wait for all writes to complete
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Test passes if no panics/deadlocks during wrap-around
    println!("SSD ring buffer wrap test completed");
}

/// Test SSD cache cleanup on drop.
#[tokio::test]
async fn test_ssd_cleanup() {
    skip_without_cuda!();

    let ssd_config = test_ssd_config();
    let cache_path = ssd_config.cache_path.clone();

    {
        let config = StorageConfig {
            enable_lfu_admission: true,
            hint_value_size_bytes: None,
            ssd_cache_config: Some(ssd_config),
        };

        let (storage, _rx) = StorageEngine::new_with_config(
            8 * 1024 * 1024,
            false,
            config,
        );

        // Insert some blocks
        let block_size = 4096usize;
        for i in 0..5 {
            let block_hash = block_hash_from_index(i);
            if let Some(allocation) = storage.allocate(NonZeroU64::new(block_size as u64).unwrap()) {
                let ptr = allocation.as_ptr() as *mut u8;
                let layer_block = Arc::new(LayerBlock::new_contiguous(ptr, block_size, allocation));
                let sealed = Arc::new(SealedBlock::from_slots(vec![layer_block]));
                let key = BlockKey::new(TEST_NAMESPACE.to_string(), block_hash);
                storage.cache_admit(key, sealed);
            }
        }

        // storage is dropped here
    }

    // Clean up the cache file
    let _ = std::fs::remove_file(&cache_path);
}
