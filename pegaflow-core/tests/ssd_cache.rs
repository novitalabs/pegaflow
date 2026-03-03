//! SSD cache integration tests.
//!
//! Verifies that KV blocks are persisted to the SSD cache file and that
//! the full save → load round-trip works with SSD-backed storage.

mod common;

use std::fs;

use common::*;
use pegaflow_core::*;

/// SSD smoke test: initialize cache file in temp dir, save at least one block,
/// verify SSD file receives non-zero bytes, then complete normal roundtrip.
#[tokio::test]
async fn ssd_smoke_roundtrip_with_temp_dir() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    const SSD_CAPACITY: u64 = 64 * 1024 * 1024;

    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");

    let harness = RoundtripHarness::new(
        HarnessConfig::new("test-ssd-smoke", "test-ns-ssd", NUM_BLOCKS, BLOCK_SIZE)
            .with_hash_salt(44)
            .with_storage_config(StorageConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: None,
                max_prefetch_blocks: 100,
                ssd_cache_config: Some(SsdCacheConfig {
                    cache_path: cache_path.clone(),
                    capacity_bytes: SSD_CAPACITY,
                    ..SsdCacheConfig::default()
                }),
                enable_numa_affinity: false,
            }),
    );

    let file_meta = fs::metadata(&cache_path).expect("SSD cache file should be created");
    assert_eq!(
        file_meta.len(),
        SSD_CAPACITY,
        "SSD cache file should be preallocated"
    );

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;
    wait_for_ssd_nonzero(&cache_path, CACHE_WAIT_TIMEOUT).await;

    harness.zero_gpu_and_assert();
    harness.expect_query_prefetch_done_all();
    harness.load_all_and_wait().await.expect("batch_load");
    harness.assert_gpu_matches_host();
}
