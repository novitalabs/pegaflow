//! Save-path input validation tests.
//!
//! Verifies that PegaEngine rejects malformed save requests with clear errors.

mod common;

use common::*;
use pegaflow_core::*;

/// Save request must validate per-layer block_ids/block_hashes shape before copy.
#[tokio::test]
async fn save_rejects_mismatched_block_id_and_hash_lengths() {
    let env = TestEnvBuilder::new("test-save-args", "test-ns")
        .layer("layer_0", 2, 1024)
        .build();

    let err = env
        .engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: "layer_0".to_string(),
                block_ids: vec![0, 1],
                block_hashes: vec![vec![9u8]],
            }],
        )
        .await
        .expect_err("save should reject mismatched lengths");

    match err {
        EngineError::InvalidArgument(msg) => {
            assert!(
                msg.contains("does not match"),
                "unexpected invalid-argument message: {msg}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[tokio::test]
async fn save_rejects_device_group_mismatch() {
    let env = TestEnvBuilder::new("test-save-group", "test-ns")
        .layer("layer_0", 2, 1024)
        .build();
    let hashes = env.hashes(7);

    env.engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: "layer_0".to_string(),
                block_ids: vec![0],
                block_hashes: vec![hashes[0].clone()],
            }],
        )
        .await
        .expect("save with registered group should pass");

    let err = env
        .engine
        .batch_save_kv_blocks_from_ipc(
            &env.instance_id,
            0,
            1,
            0,
            vec![LayerSave {
                layer_name: "layer_0".to_string(),
                block_ids: vec![1],
                block_hashes: vec![hashes[1].clone()],
            }],
        )
        .await
        .expect_err("save should reject mismatched pp_rank");

    match err {
        EngineError::InvalidArgument(msg) => {
            assert!(
                msg.contains("device_id 0") && msg.contains("pp_rank 1"),
                "unexpected invalid-argument message: {msg}"
            );
        }
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}
