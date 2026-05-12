//! Reserve-load protocol tests.
//!
//! Verifies the scheduler->worker contract: reserve_load must reserve blocks
//! before workers can load, and each query's reservation budget is consumed
//! exactly once per worker.

mod common;

use common::*;
use pegaflow_core::LoadState;

/// vLLM worker must not load before scheduler reserve_load creates a lease.
#[tokio::test]
async fn load_requires_reserve_load() {
    let env = TestEnvBuilder::new("test-load-needs-query", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;

    // Reserve and immediately release — no reservation held.
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);

    // Load without held lease should fail.
    env.expect_load_error(&hashes, "missing load lease");
}

/// One scheduler query pins each block with ref_count=world_size; each worker consumes once.
#[tokio::test]
async fn reserve_then_load_consumes_reservation_budget() {
    let env = TestEnvBuilder::new("test-world-size-lease", "test-ns")
        .layer("layer_0", 4, 1024)
        .world_size(2)
        .build();
    let hashes = env.hashes(22);

    env.save_and_wait(&hashes).await;
    env.assert_all_hit_and_reserve(&hashes).await; // reserves with ref_count=2

    // First worker load (consumes one lease reference).
    env.data().zero_gpu();
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();

    // Second worker load (consumes last lease reference).
    env.data().zero_gpu();
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();

    // Third load — lease budget exhausted.
    env.expect_load_error(&hashes, "missing load lease");
}

#[tokio::test]
async fn failed_load_validation_does_not_consume_lease() {
    let env = TestEnvBuilder::new("test-lease-validation-retry", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(33);

    env.save_and_wait(&hashes).await;
    env.assert_all_hit_and_reserve(&hashes).await;
    let lease_id = env.pending_load_lease();

    let load_state = LoadState::new().expect("create LoadState");
    let shm_name = load_state.shm_name().to_string();
    let bad_block_ids = vec![0, 1, 2];
    let bad_leases = vec![(lease_id.as_str(), bad_block_ids)];
    let err = env
        .engine
        .batch_load_kv_blocks_multi_layer(
            &env.instance_id,
            0,
            0,
            &shm_name,
            &["layer_0"],
            &bad_leases,
        )
        .expect_err("mismatched lease block count should fail before consume");
    assert!(err.to_string().contains("request provided 3 block_ids"));

    env.data().zero_gpu();
    let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
    env.load_to_gpu_with_lease(&lease_id, &block_ids, &["layer_0"])
        .await;
    env.data().assert_gpu_matches_expected();
}
