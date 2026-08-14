//! Hybrid-cache (mamba/HMA) storage groups.
//!
//! Attention groups save every block; the recurrent group saves only
//! checkpoint blocks (the state *after* a block's tokens). Each group seals
//! blocks against its own slot space, so a final-block-only recurrent save
//! becomes visible without waiting for attention's per-block cadence — and
//! one lease per group loads both halves back into the GPU.

mod common;

use common::*;
use pegaflow_core::*;

/// The HMA bull's-eye: recurrent group seals a tail-only checkpoint save,
/// queries isolate groups by hash, and a two-lease load restores both the
/// attention prefix and the recurrent state block.
#[tokio::test]
async fn recurrent_group_seals_final_block_save() {
    if !has_cuda_devices(1) {
        eprintln!("skipping recurrent_group_seals_final_block_save: needs >= 1 CUDA device");
        return;
    }

    const ATTN_BLOCK: usize = 1024;
    const RECUR_BLOCK: usize = 2048;

    let env = TestEnvBuilder::new("hybrid-hma", "hybrid-hma-ns")
        .grouped_layer("attn_0", 8, ATTN_BLOCK, 0)
        .grouped_layer("attn_1", 8, ATTN_BLOCK, 0)
        .grouped_layer("recurrent_state", 8, RECUR_BLOCK, 1)
        .pool_size(64 << 20)
        .build();

    // Same hash namespace for both groups on purpose: group encoding, not the
    // connector, must keep attention and recurrent blocks apart.
    let hashes = make_block_hashes(8, 10);
    let prefix_hashes = hashes[0..3].to_vec();

    // Attention saves blocks 0..2 for both layers; the recurrent group saves
    // ONLY the block-2 checkpoint (state after block 2's tokens), mirroring
    // the connector's final-save. The recurrent checkpoint physically lives
    // in GPU block 5 — state blocks are per-request slots, not prefix cells.
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![
                LayerSave {
                    layer_name: "attn_0".into(),
                    block_ids: vec![0, 1, 2],
                    block_hashes: prefix_hashes.clone(),
                },
                LayerSave {
                    layer_name: "attn_1".into(),
                    block_ids: vec![0, 1, 2],
                    block_hashes: prefix_hashes.clone(),
                },
            ],
        )
        .await
        .expect("save attention prefix");
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: "recurrent_state".into(),
                block_ids: vec![5],
                block_hashes: vec![hashes[2].clone()],
            }],
        )
        .await
        .expect("save recurrent checkpoint");
    env.engine.flush_saves().await;

    // Membership is per group: group 0 holds all three prefix blocks...
    let attn_hits = env
        .engine
        .query_group_membership(&env.instance_id, 0, &prefix_hashes)
        .expect("query group 0 membership");
    assert_eq!(
        attn_hits.iter().map(|b| b.is_some()).collect::<Vec<_>>(),
        vec![true, true, true]
    );

    // ...while group 1 holds ONLY the block-2 checkpoint, even though the
    // identical hash bytes for blocks 0/1 exist in group 0 (isolation).
    let recur_hits = env
        .engine
        .query_group_membership(&env.instance_id, 1, &prefix_hashes)
        .expect("query group 1 membership");
    assert_eq!(
        recur_hits.iter().map(|b| b.is_some()).collect::<Vec<_>>(),
        vec![false, false, true],
        "recurrent group must seal the tail-only checkpoint"
    );

    // The classic prefix query (group 0) is untouched by the recurrent save.
    match env.query(&prefix_hashes).await {
        PrefetchStatus::Ready { blocks, missing } => {
            assert_eq!(blocks.len(), 3);
            assert_eq!(missing, 0);
        }
        other => panic!("expected Ready, got {other:?}"),
    }

    // Reconcile (connector-side): rightmost recurrent checkpoint within the
    // attention prefix → hit = 2 + 1 = 3 blocks.
    let attn_lease = match env.query(&prefix_hashes).await {
        PrefetchStatus::Ready { blocks, .. } => env
            .engine
            .create_query_lease(&env.instance_id, blocks)
            .expect("attention lease"),
        other => panic!("expected Ready, got {other:?}"),
    };
    let recur_block = recur_hits[2]
        .as_ref()
        .expect("rightmost recurrent checkpoint")
        .clone();
    let recur_lease = env
        .engine
        .create_query_lease(&env.instance_id, vec![recur_block])
        .expect("recurrent lease");

    // Snapshot expected bytes, then wipe GPU memory so the load proves the
    // roundtrip instead of re-reading its own source.
    let attn_expected: Vec<Vec<u8>> = env.layers[0..2]
        .iter()
        .map(|l| l.data.expected_bytes().to_vec())
        .collect();
    let recur_expected = env.layers[2].data.expected_bytes().to_vec();
    for layer in &env.layers {
        layer.data.zero_gpu();
    }

    // One load, two leases, two layer groups: attention restores the [0,3)
    // prefix; the recurrent group has exactly one physical target (the
    // request's live state slot, block 7) and None elsewhere.
    let layer_groups: Vec<Vec<&str>> = vec![vec!["attn_0", "attn_1"], vec!["recurrent_state"]];
    let load_state = LoadState::new().expect("create LoadState");
    let shm_name = load_state.shm_name().to_string();
    env.engine
        .batch_load_kv_blocks_multi_layer(
            &env.instance_id,
            0,
            0,
            &shm_name,
            &layer_groups,
            &[
                (
                    attn_lease,
                    vec![vec![Some(0), Some(1), Some(2)], vec![None, None, None]],
                ),
                (recur_lease, vec![vec![None], vec![Some(7)]]),
            ],
        )
        .expect("submit hybrid load");
    wait_for_load(&load_state, LOAD_WAIT_TIMEOUT).await;

    // Attention blocks 0..2 restored, the rest still zero.
    for (layer_idx, expected) in attn_expected.iter().enumerate() {
        let mut want = vec![0u8; expected.len()];
        want[..3 * ATTN_BLOCK].copy_from_slice(&expected[..3 * ATTN_BLOCK]);
        env.layers[layer_idx].data.assert_gpu_matches(&want);
    }
    // Recurrent: only block 7 holds the checkpoint copied from block 5.
    let mut want = vec![0u8; recur_expected.len()];
    want[7 * RECUR_BLOCK..8 * RECUR_BLOCK]
        .copy_from_slice(&recur_expected[5 * RECUR_BLOCK..6 * RECUR_BLOCK]);
    env.layers[2].data.assert_gpu_matches(&want);
}

/// Group 0 must behave bit-identically to classic single-group storage when
/// nothing declares a nonzero group: prefix query, per-hash isolation, and
/// slot counts all follow the historical layout.
#[tokio::test]
async fn single_group_instances_are_unaffected() {
    if !has_cuda_devices(1) {
        eprintln!("skipping single_group_instances_are_unaffected: needs >= 1 CUDA device");
        return;
    }

    let env = TestEnvBuilder::new("hybrid-compat", "hybrid-compat-ns")
        .layer("layer_0", 4, 1024)
        .layer("layer_1", 4, 1024)
        .build();

    let hashes = make_block_hashes(4, 7);
    env.save_and_wait(&hashes[0..2]).await;

    let hits = env
        .engine
        .query_group_membership(&env.instance_id, 0, &hashes[0..4])
        .expect("membership");
    assert_eq!(
        hits.iter().map(|b| b.is_some()).collect::<Vec<_>>(),
        vec![true, true, false, false]
    );

    // A nonsensical group on a single-group instance is a hard error, not an
    // all-miss answer.
    assert!(
        env.engine
            .query_group_membership(&env.instance_id, 7, &hashes[0..1])
            .is_err()
    );
}

/// The prefill/decode handoff shape: `query_group_membership_with_fetch`
/// treats the hash list as an exact want-set with all-or-nothing semantics.
/// A fully saved recurrent checkpoint set answers Ready and complete; a
/// want-set with any absent member (no remote tier configured here) comes
/// back short, which callers must read as a miss. Group isolation holds: a
/// hash saved only under the attention group never satisfies the recurrent
/// want-set.
#[tokio::test]
async fn recurrent_membership_fetch_is_all_or_nothing() {
    if !has_cuda_devices(1) {
        eprintln!("skipping recurrent_membership_fetch_is_all_or_nothing: needs >= 1 CUDA device");
        return;
    }

    const ATTN_BLOCK: usize = 1024;
    const RECUR_BLOCK: usize = 2048;

    let env = TestEnvBuilder::new("hybrid-pd", "hybrid-pd-ns")
        .grouped_layer("attn_0", 8, ATTN_BLOCK, 0)
        .grouped_layer("recurrent_state", 8, RECUR_BLOCK, 1)
        .pool_size(64 << 20)
        .build();

    let hashes = make_block_hashes(8, 21);

    // Attention saves blocks 0..2; the recurrent group saves the block-1 and
    // block-2 checkpoints (a two-member set, like a multi-component state).
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: "attn_0".into(),
                block_ids: vec![0, 1, 2],
                block_hashes: hashes[0..3].to_vec(),
            }],
        )
        .await
        .expect("save attention prefix");
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: "recurrent_state".into(),
                block_ids: vec![4, 5],
                block_hashes: vec![hashes[1].clone(), hashes[2].clone()],
            }],
        )
        .await
        .expect("save recurrent checkpoints");
    env.engine.flush_saves().await;

    // The exact want-set resolves Ready and complete, in requested order.
    let want = vec![hashes[1].clone(), hashes[2].clone()];
    match env
        .engine
        .query_group_membership_with_fetch(&env.instance_id, "pd-req-full", 1, &want)
        .await
        .expect("want-set query")
    {
        PrefetchStatus::Ready { blocks, missing } => {
            assert_eq!(blocks.len(), 2, "full want-set must resolve completely");
            assert_eq!(missing, 0);
            let _lease = env
                .engine
                .create_query_lease(&env.instance_id, blocks)
                .expect("lease over the fetched set");
        }
        other => panic!("expected Ready, got {other:?}"),
    }

    // A want-set with an absent member comes back short (no remote tier is
    // configured in this harness): the caller must treat that as a miss.
    let short = vec![hashes[1].clone(), hashes[5].clone(), hashes[2].clone()];
    match env
        .engine
        .query_group_membership_with_fetch(&env.instance_id, "pd-req-short", 1, &short)
        .await
        .expect("short want-set query")
    {
        PrefetchStatus::Ready { blocks, missing } => {
            assert!(
                blocks.len() < short.len(),
                "an incomplete want-set must not report as complete"
            );
            assert!(missing > 0);
        }
        other => panic!("expected Ready, got {other:?}"),
    }

    // Group isolation: hashes[0] exists in the attention group only, so the
    // recurrent want-set containing it cannot complete.
    let cross = vec![hashes[0].clone()];
    match env
        .engine
        .query_group_membership_with_fetch(&env.instance_id, "pd-req-cross", 1, &cross)
        .await
        .expect("cross-group query")
    {
        PrefetchStatus::Ready { blocks, missing } => {
            assert_eq!(blocks.len(), 0);
            assert_eq!(missing, 1);
        }
        other => panic!("expected Ready, got {other:?}"),
    }
}
