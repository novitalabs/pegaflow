//! Basic PegaEngine integration tests.
//!
//! Verifies the core lifecycle: data integrity through GPU<>CPU transfers,
//! block deduplication, multi-layer completeness, and namespace isolation.

mod common;

use common::*;
use pegaflow_core::StorageConfig;

/// Full save -> query -> load round-trip with data integrity check.
#[tokio::test]
async fn save_query_load_roundtrip() {
    let env = TestEnvBuilder::new("test-roundtrip", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    env.assert_all_hit_and_pin(&hashes).await;
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();
}

/// Round-trip with NUMA-aware allocation enabled.
#[tokio::test]
async fn save_query_load_roundtrip_with_numa() {
    let env = TestEnvBuilder::new("test-roundtrip-numa", "test-ns")
        .layer("layer_0", 4, 1024)
        .storage(StorageConfig {
            enable_numa_affinity: true,
            ..StorageConfig::default()
        })
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    env.assert_all_hit_and_pin(&hashes).await;
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();
}

/// Save path deduplicates blocks already in cache.
#[tokio::test]
async fn save_deduplicates_cached_blocks() {
    let env = TestEnvBuilder::new("test-dedup", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.save_layer(0, &hashes).await; // second save — should dedup

    assert_eq!(env.count_hits_then_unpin(&hashes).await, 4);
}

/// Multi-layer: hash becomes hittable only after ALL layers are saved.
#[tokio::test]
async fn multi_layer_partial_save_no_hit() {
    let env = TestEnvBuilder::new("test-ml", "test-ns")
        .layer("layer_0", 4, 1024)
        .layer("layer_1", 4, 1024)
        .build();
    let hashes = make_block_hashes(4, 55);

    // Only layer 0 saved — should be 0 hits.
    env.save_layer(0, &hashes).await;
    env.wait_cached().await;
    assert_eq!(
        env.count_hits_then_unpin(&hashes).await,
        0,
        "partial save must not expose as hit"
    );

    // Save layer 1 — now all hit.
    env.save_layer(1, &hashes).await;
    env.wait_cached().await;
    assert_eq!(env.count_hits_then_unpin(&hashes).await, 4);
}

/// Namespace isolation: blocks in one namespace are invisible to another.
#[tokio::test]
async fn namespace_isolation() {
    let a = TestEnvBuilder::new("inst-a", "ns-alpha")
        .layer("layer_0", 2, 1024)
        .build();
    let b = TestEnvBuilder::new("inst-b", "ns-beta")
        .layer("layer_0", 2, 1024)
        .build();
    let hashes = make_block_hashes(2, 11);

    a.save_and_wait(&hashes).await;
    assert_eq!(
        b.count_hits_then_unpin(&hashes).await,
        0,
        "namespace isolation violated"
    );
}
