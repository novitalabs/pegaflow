//! Prefix-hit semantics tests.
//!
//! Verifies that `count_prefix_hit_blocks_with_prefetch` returns the longest
//! contiguous prefix of cached blocks — gaps and missing leading blocks
//! correctly terminate the prefix.

mod common;

use common::*;

/// Save first 3 of 5 blocks; prefix scan should stop at block 3.
#[tokio::test]
async fn partial_prefix_reports_contiguous_hit_count() {
    let env = TestEnvBuilder::new("test-partial-prefix", "test-ns")
        .layer("layer_0", 3, 1024)
        .build();

    let save_hashes = make_block_hashes(3, 10);
    let query_hashes = make_block_hashes(5, 10); // first 3 match save_hashes

    env.save_layer_and_flush(0, &save_hashes).await;

    let hit_blocks = env.query(&query_hashes).await;
    assert_eq!(hit_blocks, 3);
    assert_eq!(query_hashes.len() - hit_blocks, 2);
    env.unpin(&query_hashes[..3]);
}

/// Cache holds h0, h2, h3 but not h1. Prefix scan should stop at h1.
#[tokio::test]
async fn gap_in_cached_blocks_breaks_prefix() {
    let env = TestEnvBuilder::new("test-gap-prefix", "test-ns")
        .layer("layer_0", 3, 1024)
        .build();

    let all_hashes = make_block_hashes(4, 20);
    // Save h0, h2, h3 — skip h1. Block IDs [0,1,2] in GPU, hashes don't need
    // to be contiguous in index space, only in the query order.
    let save_hashes = vec![
        all_hashes[0].clone(),
        all_hashes[2].clone(),
        all_hashes[3].clone(),
    ];

    env.save_layer_and_flush(0, &save_hashes).await;

    let hit_blocks = env.query(&all_hashes).await;
    assert_eq!(hit_blocks, 1, "prefix should stop at first gap");
    assert_eq!(all_hashes.len() - hit_blocks, 3);
    env.unpin(&all_hashes[..1]);
}

/// Cache holds h1, h2, h3 but not h0. Hit count should be 0.
#[tokio::test]
async fn first_block_missing_yields_zero_prefix_hit() {
    let env = TestEnvBuilder::new("test-no-prefix", "test-ns")
        .layer("layer_0", 3, 1024)
        .build();

    let all_hashes = make_block_hashes(4, 30);
    let save_hashes = vec![
        all_hashes[1].clone(),
        all_hashes[2].clone(),
        all_hashes[3].clone(),
    ];

    env.save_layer_and_flush(0, &save_hashes).await;

    let hit_blocks = env.query(&all_hashes).await;
    assert_eq!(hit_blocks, 0);
    assert_eq!(all_hashes.len() - hit_blocks, 4);
    // hit_blocks=0, nothing pinned
}

/// Empty hash list → zero hits, zero missing.
#[tokio::test]
async fn empty_query_returns_zero() {
    let env = TestEnvBuilder::new("test-empty-query", "test-ns")
        .layer("layer_0", 1, 1024)
        .build();

    let hit_blocks = env.query(&[]).await;
    assert_eq!(hit_blocks, 0);
}
