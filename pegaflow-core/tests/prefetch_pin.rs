//! Prefetch-pin protocol tests.
//!
//! Verifies the scheduler->worker contract: query_prefetch must pin blocks
//! before workers can load, and each query's reservation budget is consumed
//! exactly once per worker.

mod common;

use common::*;

/// vLLM worker must not load before scheduler query_prefetch pins blocks.
#[tokio::test]
async fn load_requires_query_prefetch() {
    let env = TestEnvBuilder::new("test-load-needs-query", "test-ns")
        .layer("layer_0", 4, 1024)
        .build();
    let hashes = env.hashes(0);

    env.save_and_wait(&hashes).await;

    // Query and immediately unpin — no reservation held.
    assert_eq!(env.count_hits_then_unpin(&hashes).await, 4);

    // Load without held pin should fail.
    env.expect_load_error(&hashes, "missing pinned KV block");
}

/// One scheduler query pins each block with ref_count=world_size; each worker consumes once.
#[tokio::test]
async fn query_then_load_consumes_reservation_budget() {
    let env = TestEnvBuilder::new("test-world-size-pin", "test-ns")
        .layer("layer_0", 4, 1024)
        .world_size(2)
        .build();
    let hashes = env.hashes(22);

    env.save_and_wait(&hashes).await;
    env.assert_all_hit_and_pin(&hashes).await; // pins with ref_count=2

    // First worker load (consumes one pin reference).
    env.data().zero_gpu();
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();

    // Second worker load (consumes last pin reference).
    env.data().zero_gpu();
    env.load_to_gpu(&hashes).await;
    env.data().assert_gpu_matches_expected();

    // Third load — pin budget exhausted.
    env.expect_load_error(&hashes, "missing pinned KV block");
}
