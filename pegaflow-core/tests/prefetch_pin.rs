//! Prefetch-pin protocol tests.
//!
//! Verifies the scheduler→worker contract: query_prefetch must pin blocks
//! before workers can load, and each query's reservation budget is consumed
//! exactly once per worker.

mod common;

use common::*;

/// vLLM worker must not load before scheduler query_prefetch pins blocks.
#[tokio::test]
async fn load_requires_query_prefetch() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    let harness = RoundtripHarness::new(HarnessConfig::new(
        "test-load-needs-query",
        "test-ns",
        NUM_BLOCKS,
        BLOCK_SIZE,
    ));

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;

    let (hit, missing) = harness.query_hits(harness.block_hashes()).await;
    assert_eq!(hit, NUM_BLOCKS, "blocks should already be cached");
    assert_eq!(missing, 0);

    harness.expect_load_submit_error(
        harness.block_ids(),
        harness.block_hashes(),
        "missing pinned KV block",
    );
}

/// One scheduler query pins each block with ref_count=world_size; each worker consumes once.
#[tokio::test]
async fn query_then_load_consumes_reservation_budget() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;
    let harness = RoundtripHarness::new(
        HarnessConfig::new("test-world-size-pin", "test-ns", NUM_BLOCKS, BLOCK_SIZE)
            .with_world_size(2)
            .with_hash_salt(22),
    );

    harness.save_all().await;
    harness.assert_cache_eventually_all().await;
    harness.expect_query_prefetch_done_all().await;

    // First worker load (consumes one pin reference per block).
    harness.zero_gpu_and_assert();
    harness.load_all_and_wait().await.expect("first load");
    harness.assert_gpu_matches_host();

    // Second worker load (consumes second and last pin reference per block).
    harness.zero_gpu_and_assert();
    harness.load_all_and_wait().await.expect("second load");
    harness.assert_gpu_matches_host();

    // Third load without re-query should fail because pin budget is exhausted.
    harness.expect_load_submit_error(
        harness.block_ids(),
        harness.block_hashes(),
        "missing pinned KV block",
    );
}
