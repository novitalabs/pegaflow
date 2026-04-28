//! Memory-pressure eviction tests (pure CPU memory, no SSD).
//!
//! Verifies LRU eviction behavior when the pinned memory pool is exhausted:
//! new blocks push out old ones, pinned blocks survive pressure, and loaded
//! data after eviction is from the correct batch.

mod common;

use common::*;
use pegaflow_core::StorageConfig;

const BLOCK_SIZE: usize = 4096;
const NUM_BLOCKS: usize = 4;
/// Pool fits one batch with headroom for metadata, but not two.
const POOL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE * 2;

/// Save more blocks than the pool can hold — old blocks get evicted,
/// new blocks round-trip with correct (overwritten) data.
#[tokio::test]
async fn eviction_reclaims_old_blocks_for_new() {
    let mut env = TestEnvBuilder::new("inst-evict", "ns-evict")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .build();

    let old = env.hashes(1);
    env.save_and_wait(&old).await;

    // Overwrite GPU data, save second batch — triggers eviction.
    env.data_mut().overwrite(42);
    let new = env.hashes(2);
    env.save_and_wait(&new).await;

    // Load new batch — must get the overwritten data.
    env.data().zero_gpu();
    env.assert_all_hit_and_pin(&new).await;
    env.load_to_gpu(&new).await;
    env.data().assert_gpu_matches_expected();
}

/// Pinned blocks survive memory pressure: prepare-load pins prevent LRU eviction.
#[tokio::test]
async fn pinned_blocks_survive_eviction_pressure() {
    let env = TestEnvBuilder::new("inst-pin", "ns-pin")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .build();

    let hashes = env.hashes(10);
    env.save_and_wait(&hashes).await;
    env.assert_all_hit_and_pin(&hashes).await; // pins first batch

    // Second batch creates eviction pressure while first is pinned.
    let pressure = make_block_hashes(NUM_BLOCKS, 20);
    env.save_layer(0, &pressure).await;

    // First batch is still pinned — load must succeed.
    env.data().zero_gpu();
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();
}

/// Eviction works identically with pool_shards > 1.
#[tokio::test]
async fn eviction_works_with_sharded_pool() {
    let mut env = TestEnvBuilder::new("inst-shard", "ns-shard")
        .layer("layer_0", NUM_BLOCKS, BLOCK_SIZE)
        .pool_size(POOL_SIZE)
        .storage(StorageConfig {
            enable_lfu_admission: false,
            pool_shards: 2,
            ..StorageConfig::default()
        })
        .build();

    let old = env.hashes(30);
    env.save_and_wait(&old).await;

    env.data_mut().overwrite(99);
    let new = env.hashes(31);
    env.save_and_wait(&new).await;

    env.data().zero_gpu();
    env.assert_all_hit_and_pin(&new).await;
    env.load_to_gpu(&new).await;
    env.data().assert_gpu_matches_expected();
}
