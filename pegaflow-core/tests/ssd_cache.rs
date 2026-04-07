//! SSD cache integration tests.
//!
//! Verifies the complete SSD lifecycle: write persistence, prefetch from SSD
//! after memory eviction, and data integrity through the full round-trip.

mod common;

use common::*;
use pegaflow_core::*;

const BLOCK_SIZE: usize = 4096;
const NUM_BLOCKS: usize = 4;
/// Pool fits one batch with headroom but not two — forces eviction.
const POOL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE * 2;
const SSD_CAPACITY: u64 = 64 * 1024 * 1024;

fn ssd_env(instance_id: &'static str) -> (TestEnv, std::path::PathBuf) {
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let env = TestEnvBuilder::new(instance_id, "test-ns-ssd")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(StorageConfig {
            ssd_cache_config: Some(SsdCacheConfig {
                cache_path: cache_path.clone(),
                capacity_bytes: SSD_CAPACITY,
                ..SsdCacheConfig::default()
            }),
            ..StorageConfig::default()
        })
        .build();
    // Keep temp_dir alive so the cache file isn't deleted.
    std::mem::forget(temp_dir);
    (env, cache_path)
}

/// Save blocks, flush to SSD, verify the cache file contains non-zero bytes.
#[tokio::test]
async fn ssd_write_persists_to_file() {
    let (env, cache_path) = ssd_env("test-ssd-write");

    let file_meta = std::fs::metadata(&cache_path).expect("SSD cache file should be created");
    assert_eq!(
        file_meta.len(),
        SSD_CAPACITY,
        "SSD cache file should be preallocated"
    );

    let hashes = env.hashes(44);
    env.save_and_wait(&hashes).await;
    env.engine.flush_all().await;

    // After flush_all, SSD file must contain written data.
    let mut f = std::fs::File::open(&cache_path).expect("open SSD cache file");
    let mut buf = vec![0u8; 4096];
    std::io::Read::read(&mut f, &mut buf).expect("read SSD cache file");
    assert!(
        buf.iter().any(|&b| b != 0),
        "SSD cache file should contain non-zero data after flush"
    );
}

/// Full SSD prefetch round-trip: save → flush to SSD → evict from memory →
/// query (triggers SSD prefetch) → load → verify data integrity.
#[tokio::test]
async fn ssd_prefetch_roundtrip_after_eviction() {
    let (env, _cache_path) = ssd_env("test-ssd-prefetch");

    // Phase 1: Save target blocks and ensure they're persisted to SSD.
    let target = env.hashes(1);
    env.save_and_wait(&target).await;
    env.engine.flush_all().await;

    // Phase 2: Evict target blocks from memory by saving filler that overflows the pool.
    let filler = make_block_hashes(NUM_BLOCKS, 99);
    env.save_layer(0, &filler).await;
    env.engine.flush_saves().await;

    // Phase 3: Query the original hashes — should trigger SSD prefetch.
    // First query may return Loading (SSD read in flight), poll until Done.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let status = env.query(&target).await;
        match status {
            PrefetchStatus::Done { hit, .. } => {
                if hit > 0 {
                    env.unpin(&target[..hit]);
                }
                if hit == target.len() {
                    break;
                }
            }
            PrefetchStatus::Loading { .. } => {}
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for SSD prefetch to complete"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    // Phase 4: Pin, load to GPU, and verify data integrity.
    env.assert_all_hit_and_pin(&target).await;
    env.data().zero_gpu();
    env.load_to_gpu(&target).await;
    env.data().assert_gpu_matches_expected();
}
