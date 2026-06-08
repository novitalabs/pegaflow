//! Instance teardown safety under concurrent transfers.
//!
//! These cover the contract that `unregister` exists to provide: shutting an
//! instance down drains and joins its GPU worker threads, so any save that was
//! in flight finishes touching GPU memory *before* unregister returns. A caller
//! can then release the CUDA tensors without a use-after-unmap. The race is set
//! up so that every interleaving is valid — a save either completes or is
//! rejected cleanly; a CUDA fault or panic is the regression.

mod common;

use std::sync::Arc;

use common::*;
use pegaflow_core::{EngineError, LayerSave};

/// A save error that is acceptable when it loses the race against teardown:
/// the instance was removed, or its worker channel was already closed.
fn is_teardown_error(msg: &str) -> bool {
    msg.contains("not found") || msg.contains("channel closed") || msg.contains("shut down")
}

/// Save one layer through the public engine API. Owns its inputs so the future
/// is `'static` and can be `tokio::spawn`ed.
async fn save_one_layer(
    env: Arc<TestEnv>,
    instance_id: String,
    layer: String,
    num_blocks: usize,
    salt: u8,
) -> Result<(), EngineError> {
    let block_ids: Vec<i32> = (0..num_blocks as i32).collect();
    env.engine
        .batch_save_kv_blocks_from_ipc(
            &instance_id,
            0,
            0,
            0,
            vec![LayerSave {
                layer_name: layer,
                block_ids,
                block_hashes: make_block_hashes(num_blocks, salt),
            }],
        )
        .await
}

/// Saves racing with `unregister_instance` never crash, and teardown wins
/// cleanly: unregister succeeds (workers drained + joined) and the instance is
/// gone afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unregister_while_saving_is_safe() {
    let env = Arc::new(
        TestEnvBuilder::new("test-unreg-race", "test-ns")
            .layer("layer_0", 8, 4096)
            .build(),
    );

    // Distinct salts → distinct blocks → real GPU->CPU copies (no dedup short
    // circuit), so the worker threads are genuinely busy during teardown.
    let savers: Vec<_> = (0..16u8)
        .map(|salt| {
            let env = Arc::clone(&env);
            tokio::spawn(save_one_layer(
                env,
                "test-unreg-race".to_string(),
                "layer_0".to_string(),
                8,
                salt,
            ))
        })
        .collect();

    // Tear the instance down. Mirror the server: the blocking join runs off the
    // runtime workers via spawn_blocking.
    let unreg_env = Arc::clone(&env);
    let unreg = tokio::task::spawn_blocking(move || {
        unreg_env.engine.unregister_instance("test-unreg-race")
    });

    unreg
        .await
        .expect("unregister task panicked")
        .expect("unregister must succeed: it only joins worker threads");

    // Across runs this races both ways: some saves complete (submitted, then
    // drained during the join) and some are rejected cleanly (lost the race).
    // Either is valid; a panic or CUDA fault is the use-after-unmap regression.
    for saver in savers {
        if let Err(err) = saver.await.expect("save task panicked") {
            let msg = err.to_string();
            assert!(
                is_teardown_error(&msg),
                "save failed for an unexpected reason: {msg}"
            );
        }
    }

    assert!(
        !env.engine
            .list_instance_ids()
            .contains(&"test-unreg-race".to_string()),
        "instance must be gone after unregister"
    );
}

/// `unregister_all` drains every instance's workers even under concurrent save
/// load, returns all removed IDs, and leaves the registry empty — no single
/// instance can hold the others hostage.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unregister_all_drains_every_instance_under_load() {
    let env = Arc::new(
        TestEnvBuilder::new("inst-a", "test-ns")
            .layer("layer_0", 4, 4096)
            .build(),
    );

    // Two more instances in the same engine. Reuse the GPU buffer: this test
    // exercises teardown, not data correctness, and concurrent D2H reads of the
    // same region are fine.
    let layer_infos = vec![LayerInfo {
        name: "layer_0".to_string(),
        gpu_ptr: env.data().ptr(),
        total_size: env.data().total_size(),
        num_blocks: 4,
        block_size: 4096,
        kv_stride: 0,
        segments: 1,
    }];
    for id in ["inst-b", "inst-c"] {
        register_layers(&env.engine, id, "test-ns", &layer_infos, 0, 0, 0, 1, 1, 1);
    }

    let instances = ["inst-a", "inst-b", "inst-c"];
    let mut savers = Vec::new();
    for (i, id) in instances.iter().enumerate() {
        for salt in 0..4u8 {
            let env = Arc::clone(&env);
            savers.push(tokio::spawn(save_one_layer(
                env,
                id.to_string(),
                "layer_0".to_string(),
                4,
                (i as u8) * 16 + salt,
            )));
        }
    }

    let unreg_env = Arc::clone(&env);
    let mut removed =
        tokio::task::spawn_blocking(move || unreg_env.engine.unregister_all_instances())
            .await
            .expect("unregister-all task panicked");
    removed.sort();
    assert_eq!(removed, vec!["inst-a", "inst-b", "inst-c"]);

    for saver in savers {
        if let Err(err) = saver.await.expect("save task panicked") {
            let msg = err.to_string();
            assert!(
                is_teardown_error(&msg),
                "save failed for an unexpected reason: {msg}"
            );
        }
    }

    assert!(
        env.engine.list_instance_ids().is_empty(),
        "registry must be empty after unregister_all"
    );
}
