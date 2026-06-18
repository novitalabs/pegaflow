//! Basic PegaEngine integration tests.
//!
//! Verifies the core lifecycle: data integrity through GPU<>CPU transfers,
//! block deduplication, multi-layer completeness, and namespace isolation.

mod common;

use common::*;
use pegaflow_core::{LayerSave, StorageConfig, TransferMode};

/// Full save -> query -> load round-trip with data integrity check.
#[tokio::test]
async fn save_query_load_roundtrip() {
    let env = TestEnvBuilder::new("test-roundtrip", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Page-first round-trip: all layers of a block live in one host page slot,
/// saved in a single batch and loaded back. Each layer is given DISTINCT
/// content AND a DISTINCT (512-aligned) size, so the page offsets are an
/// uneven prefix sum (0, 512, 1536). A wrong offset or a mishandled prefix
/// sum makes a layer read/write another layer's slice and fails the assert.
#[tokio::test]
async fn page_first_multi_layer_roundtrip() {
    let mut env = TestEnvBuilder::new("test-page-first", "test-ns")
        .page_first()
        .layer("layer_a", 4, 512)
        .layer("layer_b", 4, 1024)
        .layer("layer_c", 4, 1536)
        .build();

    // Make every layer's content unique so layer offsets are actually checked.
    env.layers[1].data.overwrite(0x40);
    env.layers[2].data.overwrite(0x80);

    let hashes = make_block_hashes(4, 0xAB);

    env.save_all_layers_one_batch(&hashes).await;
    for layer in &env.layers {
        layer.data.zero_gpu();
    }

    let lease = env.assert_all_hit_lease(&hashes).await;
    env.load_to_gpu(lease, hashes.len()).await;

    for layer in &env.layers {
        layer.data.assert_gpu_matches_expected();
    }
}

/// Page-first seals a block's single slot on the first save, so one save must
/// carry EVERY layer of the page. A save covering only some layers (e.g. a
/// pipeline stage holding a subset) would seal a page whose other offsets are
/// never written — stale pool memory that load would return as another block's
/// KV. The engine must reject such a partial-layer page-first save loudly.
#[tokio::test]
async fn page_first_rejects_partial_layer_save() {
    let env = TestEnvBuilder::new("test-page-first-partial", "test-ns")
        .page_first()
        .layer("layer_a", 4, 512)
        .layer("layer_b", 4, 512)
        .layer("layer_c", 4, 512)
        .build();
    let hashes = make_block_hashes(4, 0xCD);

    // Submit only layer_a — a partial page that must not seal.
    let partial = vec![LayerSave {
        layer_name: env.layers[0].name.clone(),
        block_ids: (0..hashes.len()).collect(),
        block_hashes: hashes.clone(),
    }];
    let err = env
        .engine
        .batch_save_kv_blocks_from_ipc(&env.instance_id, 0, 0, 0, partial)
        .await
        .expect_err("partial page-first save must be rejected");
    assert!(
        err.to_string().contains("must cover all"),
        "unexpected error: {err}"
    );
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
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
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

    assert_eq!(env.count_hits_then_release(&hashes).await, 4);
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
        env.count_hits_then_release(&hashes).await,
        0,
        "partial save must not expose as hit"
    );

    // Save layer 1 — now all hit.
    env.save_layer(1, &hashes).await;
    env.wait_cached().await;
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);
}

/// Split-storage (K/V separated) round-trip with data integrity check.
#[tokio::test]
async fn save_query_load_roundtrip_split_storage() {
    // segment_size=512, kv_stride=4096 (stride > segment → split path)
    let env = TestEnvBuilder::new("test-split", "test-ns")
        .split_layer("layer_0", 4, 512, 4096)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
}

/// Kernel backend round-trip over real mapped pinned allocations.
///
/// Split storage exercises both K and V segment descriptors, so this covers the
/// production path from `PinnedAllocation` through `RawBlock`/`LayerAlloc` into
/// `CopyDesc.host_device` for save and load.
#[tokio::test]
async fn kernel_backend_roundtrip_split_storage() {
    let env = TestEnvBuilder::new("test-kernel-split", "test-ns")
        .split_layer("layer_0", 2, 4096, 16384)
        .transfer_mode(TransferMode::Kernel)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;
    env.data().zero_gpu();
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();
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
        b.count_hits_then_release(&hashes).await,
        0,
        "namespace isolation violated"
    );
}
