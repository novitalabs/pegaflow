//! SSD cache integration tests.
//!
//! Verifies that KV blocks are persisted to the SSD cache file and that
//! the full save -> load round-trip works with SSD-backed storage.

mod common;

use std::fs;

use common::*;
use pegaflow_core::*;

/// SSD smoke test: initialize cache file in temp dir, save at least one block,
/// verify SSD file receives non-zero bytes, then complete normal roundtrip.
#[tokio::test]
async fn ssd_smoke_roundtrip_with_temp_dir() {
    const SSD_CAPACITY: u64 = 64 * 1024 * 1024;

    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");

    let env = TestEnvBuilder::new("test-ssd-smoke", "test-ns-ssd")
        .layer("layer_0", 4, 1024)
        .storage(StorageConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_path: cache_path.clone(),
                capacity_bytes: SSD_CAPACITY,
                ..SsdCacheConfig::default()
            }),
            ..StorageConfig::default()
        })
        .build();

    let file_meta = fs::metadata(&cache_path).expect("SSD cache file should be created");
    assert_eq!(
        file_meta.len(),
        SSD_CAPACITY,
        "SSD cache file should be preallocated"
    );

    let hashes = env.hashes(44);
    env.save_and_wait(&hashes).await;
    wait_for_ssd_nonzero(&cache_path, CACHE_WAIT_TIMEOUT).await;

    env.data().zero_gpu();
    env.assert_all_hit_and_pin(&hashes).await;
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();
}
