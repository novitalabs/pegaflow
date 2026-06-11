//! Query lease protocol tests.
//!
//! Verifies the scheduler->worker contract: query_prefetch returns an opaque
//! lease that owns ready blocks, and load consumes one lease share per worker.

mod common;

use common::*;
use pegaflow_core::LoadState;

/// vLLM worker must not load after the scheduler releases the query lease.
#[tokio::test]
async fn load_requires_query_prefetch() {
    let env = TestEnvBuilder::new("test-load-needs-query", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;

    // Query and immediately release — no lease held.
    assert_eq!(env.count_hits_then_release(&hashes).await, 4);

    let released = env.assert_all_hit_lease(&hashes).await;
    env.release(&released);
    env.expect_load_error(released, hashes.len(), "query lease is unknown or expired");
}

/// One scheduler query lease is consumed once per registered world-size worker.
#[tokio::test]
async fn query_then_load_consumes_reservation_budget() {
    if !has_cuda_devices(2) {
        eprintln!("skipping query_then_load_consumes_reservation_budget: needs >= 2 CUDA devices");
        return;
    }

    let env = TestEnvBuilder::new("test-query-lease", "test-ns")
        .layer("layer_0", 4, 1024)
        .world_size(2)
        .build();
    let hashes = env.hashes(22);

    env.save_and_wait(&hashes).await;
    let lease = env.assert_all_hit_lease(&hashes).await;

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();

    env.data().zero_gpu();
    env.load_to_gpu(lease, hashes.len()).await;
    env.data().assert_gpu_matches_expected();

    let block_ids: Vec<i32> = (0..hashes.len() as i32).collect();
    let layer_names: Vec<&str> = env.layers.iter().map(|l| l.name.as_str()).collect();
    let load_state = LoadState::new().expect("create LoadState");
    let err = env
        .engine
        .batch_load_kv_blocks_multi_layer(
            &env.instance_id,
            0,
            0,
            load_state.shm_name(),
            &layer_names,
            &[(lease, block_ids)],
        )
        .expect_err("third load should fail");
    assert!(
        err.to_string()
            .contains("query lease is unknown or expired")
    );
}
