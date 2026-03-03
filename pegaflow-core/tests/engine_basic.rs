//! Basic PegaEngine integration tests.
//!
//! Verifies the core lifecycle: data integrity through GPU↔CPU transfers,
//! block deduplication, and namespace isolation.

mod common;

use std::time::Duration;

use common::*;
use cudarc::driver::CudaContext;
use pegaflow_core::StorageConfig;

/// Full save → query → load round-trip with data integrity check.
///
/// 1. Allocate GPU memory, fill with known pattern
/// 2. register_context_layer
/// 3. batch_save (GPU→CPU)
/// 4. Wait for blocks to appear in cache
/// 5. Zero GPU memory
/// 6. Pin blocks via query_prefetch (scheduler step)
/// 7. batch_load (CPU→GPU, worker step)
/// 8. Wait for load completion
/// 9. Verify GPU data matches original
#[tokio::test]
async fn save_query_load_roundtrip() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    let harness = RoundtripHarness::new(HarnessConfig::new(
        "test-roundtrip",
        "test-ns",
        NUM_BLOCKS,
        BLOCK_SIZE,
    ));

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;
    harness.zero_gpu_and_assert();
    harness.expect_query_prefetch_done_all();
    harness.load_all_and_wait().await.expect("batch_load");
    harness.assert_gpu_matches_host();
}

/// Round-trip should work when NUMA-aware allocation is enabled.
#[tokio::test]
async fn save_query_load_roundtrip_with_numa_affinity_enabled() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    let harness = RoundtripHarness::new(
        HarnessConfig::new("test-roundtrip-numa-on", "test-ns", NUM_BLOCKS, BLOCK_SIZE)
            .with_storage_config(StorageConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: None,
                max_prefetch_blocks: 100,
                ssd_cache_config: None,
                enable_numa_affinity: true,
            }),
    );

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;
    harness.zero_gpu_and_assert();
    harness.expect_query_prefetch_done_all();
    harness.load_all_and_wait().await.expect("batch_load");
    harness.assert_gpu_matches_host();
}

/// Save path skips blocks already in cache (dedup).
#[tokio::test]
async fn save_deduplicates_cached_blocks() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    let harness = RoundtripHarness::new(HarnessConfig::new(
        "test-dedup",
        "test-ns",
        NUM_BLOCKS,
        BLOCK_SIZE,
    ));

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;
    harness.save_all().await;

    let (hit, missing) = harness.query_hits(harness.block_hashes());
    assert_eq!(hit, NUM_BLOCKS);
    assert_eq!(missing, 0);
}

/// Multi-layer completeness: a hash should become hittable only after all layers are saved.
#[tokio::test]
async fn multi_layer_blocks_hit_only_after_all_layers_saved() {
    const DEVICE_ID: i32 = 0;
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    const TOTAL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;
    const NUM_LAYERS: usize = 2;
    const INSTANCE_ID: &str = "inst-multi-layer-hit";
    const NAMESPACE: &str = "ns-multi-layer-hit";
    const LAYER_1: &str = "layer_1";

    let _ctx = CudaContext::new(0).expect("CUDA init");

    let gpu_l0 = GpuBuffer::alloc(TOTAL_SIZE);
    let mut data_l0 = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut data_l0, BLOCK_SIZE);
    gpu_l0.copy_from_host(&data_l0);

    let gpu_l1 = GpuBuffer::alloc(TOTAL_SIZE);
    let mut data_l1 = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut data_l1, BLOCK_SIZE);
    for b in &mut data_l1 {
        *b = b.wrapping_add(17);
    }
    gpu_l1.copy_from_host(&data_l1);

    let engine = test_engine();
    register_single_layer(
        &engine,
        INSTANCE_ID,
        NAMESPACE,
        DEFAULT_LAYER,
        gpu_l0.as_u64(),
        TOTAL_SIZE,
        NUM_BLOCKS,
        BLOCK_SIZE,
        DEVICE_ID,
        0,
        1,
        1,
        NUM_LAYERS,
    );
    register_single_layer(
        &engine,
        INSTANCE_ID,
        NAMESPACE,
        LAYER_1,
        gpu_l1.as_u64(),
        TOTAL_SIZE,
        NUM_BLOCKS,
        BLOCK_SIZE,
        DEVICE_ID,
        0,
        1,
        1,
        NUM_LAYERS,
    );

    let block_ids = make_block_ids(NUM_BLOCKS);
    let block_hashes = make_block_hashes(NUM_BLOCKS, 55);

    save_single_layer(
        &engine,
        INSTANCE_ID,
        0,
        DEVICE_ID,
        DEFAULT_LAYER,
        block_ids.clone(),
        block_hashes.clone(),
    )
    .await
    .expect("save layer_0");

    // Save insertion into cache is async; give insert worker a chance to process partial data.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let (hit, missing) = engine
        .count_prefix_hit_blocks(INSTANCE_ID, &block_hashes)
        .expect("query after layer_0 save");
    assert_eq!(
        hit, 0,
        "partial layer save must not expose hash as cache hit"
    );
    assert_eq!(missing, NUM_BLOCKS);

    save_single_layer(
        &engine,
        INSTANCE_ID,
        0,
        DEVICE_ID,
        LAYER_1,
        block_ids,
        block_hashes.clone(),
    )
    .await
    .expect("save layer_1");

    wait_for_cache(
        &engine,
        INSTANCE_ID,
        &block_hashes,
        NUM_BLOCKS,
        CACHE_WAIT_TIMEOUT,
    )
    .await;

    let (hit, missing) = engine
        .count_prefix_hit_blocks(INSTANCE_ID, &block_hashes)
        .expect("query after all layers saved");
    assert_eq!(hit, NUM_BLOCKS);
    assert_eq!(missing, 0);
}

/// Namespace isolation: blocks saved under one namespace are invisible to another.
#[tokio::test]
async fn namespace_isolation_roundtrip() {
    const DEVICE_ID: i32 = 0;
    const BLOCK_SIZE: usize = 1024;
    const NUM_BLOCKS: usize = 2;
    const TOTAL_SIZE: usize = NUM_BLOCKS * BLOCK_SIZE;

    let _ctx = CudaContext::new(0).expect("CUDA init");

    let gpu_a = GpuBuffer::alloc(TOTAL_SIZE);
    let mut data_a = vec![0u8; TOTAL_SIZE];
    fill_test_pattern(&mut data_a, BLOCK_SIZE);
    gpu_a.copy_from_host(&data_a);

    let engine = test_engine();

    let register = |id, ns| {
        register_single_layer(
            &engine,
            id,
            ns,
            DEFAULT_LAYER,
            gpu_a.as_u64(),
            TOTAL_SIZE,
            NUM_BLOCKS,
            BLOCK_SIZE,
            DEVICE_ID,
            0,
            1,
            1,
            1,
        );
    };
    register("inst-a", "ns-alpha");
    register("inst-b", "ns-beta");

    let hashes = make_block_hashes(NUM_BLOCKS, 11);

    // Save under inst-a (namespace "ns-alpha")
    save_single_layer(
        &engine,
        "inst-a",
        0,
        DEVICE_ID,
        DEFAULT_LAYER,
        make_block_ids(NUM_BLOCKS),
        hashes.clone(),
    )
    .await
    .expect("save inst-a");

    wait_for_cache(&engine, "inst-a", &hashes, NUM_BLOCKS, CACHE_WAIT_TIMEOUT).await;

    // inst-b should see zero hits (different namespace)
    let (hit, missing) = engine
        .count_prefix_hit_blocks("inst-b", &hashes)
        .expect("query inst-b");
    assert_eq!(hit, 0, "namespace isolation violated");
    assert_eq!(missing, NUM_BLOCKS);
}
