//! Save-path input validation tests.
//!
//! Verifies that PegaEngine rejects malformed save requests with clear errors.

mod common;

use common::*;
use pegaflow_core::EngineError;

/// Save request must validate per-layer block_ids/block_hashes shape before copy.
#[tokio::test]
async fn save_rejects_mismatched_block_id_and_hash_lengths() {
    let harness = RoundtripHarness::new(HarnessConfig::new("test-save-args", "test-ns", 2, 1024));

    let err = harness
        .save_layer(vec![0, 1], vec![vec![9u8]])
        .await
        .expect_err("save should reject mismatched block_ids/block_hashes lengths");

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
