//! Mock vLLM RPC E2E test.
//!
//! This keeps the real GPU/engine boundary, but replaces vLLM with a thin
//! Rust client that calls the same gRPC methods used by the connector.

mod common;

use common::{
    BLOCK_COUNT, INSTANCE_ID, LAYER_NAME, MockVllmRpcHarness, SECOND_INSTANCE_ID,
    cuda_device_count, make_block_hashes,
};
use pegaflow_server::proto::engine::{QueryReady, query_response};

#[tokio::test]
async fn mock_vllm_save_query_load_roundtrip_over_rpc() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 17);

    let cold_query = harness
        .query_prefetch("mock-vllm-cold-query", &hashes)
        .await;
    assert_eq!(cold_query.request.instance_id, INSTANCE_ID);
    assert_eq!(cold_query.request.block_hashes, hashes);
    assert_eq!(cold_query.request.req_id, "mock-vllm-cold-query");
    let cold_ready = expect_query_ready(cold_query.response.outcome, "cold query");
    assert_eq!(cold_ready.num_hit_blocks, 0);
    assert!(cold_ready.lease.is_empty());

    let save = harness.save_blocks(&hashes).await;
    assert_eq!(save.request.instance_id, INSTANCE_ID);
    assert_eq!(save.request.tp_rank, 0);
    assert_eq!(save.request.device_id, 0);
    assert_eq!(save.request.pp_rank, 0);
    assert_eq!(save.request.saves[0].layer_name, LAYER_NAME);
    assert_eq!(save.request.saves[0].block_ids, vec![0, 1, 2, 3]);
    assert_eq!(save.request.saves[0].block_hashes, hashes);
    assert_response_ok(save.response.status.as_ref(), "save");

    let hit_query = harness.query_prefetch("mock-vllm-hit-query", &hashes).await;
    assert_eq!(hit_query.request.instance_id, INSTANCE_ID);
    assert_eq!(hit_query.request.block_hashes, hashes);
    assert_eq!(hit_query.request.req_id, "mock-vllm-hit-query");
    let hit_ready = expect_query_ready(hit_query.response.outcome, "hit query");
    assert_eq!(hit_ready.num_hit_blocks as usize, hashes.len());
    assert!(!hit_ready.lease.is_empty());

    harness.gpus[0].zero();
    let load = harness.submit_load(hit_ready.lease, hashes.len()).await;
    assert_eq!(load.request.instance_id, INSTANCE_ID);
    assert_eq!(load.request.tp_rank, 0);
    assert_eq!(load.request.device_id, 0);
    assert!(!load.request.load_state_shm.is_empty());
    assert_eq!(load.request.layer_names, vec![LAYER_NAME.to_string()]);
    assert_eq!(load.request.loads[0].block_ids, vec![0, 1, 2, 3]);
    assert_response_ok(load.response.status.as_ref(), "load");

    harness.wait_for_load(&load.state).await;
    harness.gpus[0].assert_matches_expected();
}

#[tokio::test]
async fn mock_vllm_empty_req_id_query_returns_invalid_argument() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 31);

    let save = harness.save_blocks(&hashes).await;
    assert_response_ok(save.response.status.as_ref(), "save");

    let err = match harness.try_query_prefetch("", &hashes).await {
        Ok(_) => panic!("empty req_id query should fail validation"),
        Err(err) => err,
    };
    assert_eq!(err.request.instance_id, INSTANCE_ID);
    assert_eq!(err.request.block_hashes, hashes);
    assert_eq!(err.request.req_id, "");
    assert_eq!(err.status.code(), tonic::Code::InvalidArgument);
    assert_eq!(err.status.message(), "req_id must not be empty");
}

#[tokio::test]
async fn mock_vllm_query_lease_is_bound_to_instance_id() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 37);

    let save = harness.save_blocks(&hashes).await;
    assert_response_ok(save.response.status.as_ref(), "save");

    let query = harness
        .query_prefetch("mock-vllm-instance-bound", &hashes)
        .await;
    let ready = expect_query_ready(query.response.outcome, "instance-bound query");
    assert_eq!(ready.num_hit_blocks as usize, hashes.len());
    assert!(!ready.lease.is_empty());

    harness.register_second_instance();
    let wrong_instance_load = match harness
        .try_submit_load_for_instance_worker(SECOND_INSTANCE_ID, 0, ready.lease, hashes.len())
        .await
    {
        Ok(_) => panic!("load with lease from another instance should fail"),
        Err(err) => err,
    };
    assert_eq!(wrong_instance_load.request.instance_id, SECOND_INSTANCE_ID);
    assert!(
        wrong_instance_load
            .status
            .message()
            .contains("query lease belongs to instance"),
        "unexpected error: {}",
        wrong_instance_load.status
    );
    assert!(wrong_instance_load.state.get() < 0);
}

#[tokio::test]
async fn mock_vllm_release_unknown_lease_returns_error() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 39);

    let save = harness.save_blocks(&hashes).await;
    assert_response_ok(save.response.status.as_ref(), "save");

    let query = harness
        .query_prefetch("mock-vllm-release-once", &hashes)
        .await;
    let ready = expect_query_ready(query.response.outcome, "release once query");
    assert_eq!(ready.num_hit_blocks as usize, hashes.len());
    assert!(!ready.lease.is_empty());

    let released = match harness.try_release_lease(ready.lease.clone()).await {
        Ok(exchange) => exchange,
        Err(err) => panic!("first release should succeed: {}", err.status),
    };
    assert_eq!(released.request.lease, ready.lease);

    let duplicate_release = match harness.try_release_lease(ready.lease).await {
        Ok(_) => panic!("second release should fail"),
        Err(err) => err,
    };
    assert!(!duplicate_release.request.lease.is_empty());
    assert_eq!(
        duplicate_release.status.code(),
        tonic::Code::FailedPrecondition
    );
    assert!(
        duplicate_release
            .status
            .message()
            .contains("query lease is unknown or expired"),
        "unexpected error: {}",
        duplicate_release.status
    );
}

#[tokio::test]
async fn mock_vllm_prefix_partial_query_returns_prefix_lease() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 41);
    let prefix = &hashes[..2];

    let save = harness.save_blocks(prefix).await;
    assert_eq!(save.request.saves[0].block_ids, vec![0, 1]);
    assert_eq!(save.request.saves[0].block_hashes, prefix);
    assert_response_ok(save.response.status.as_ref(), "prefix save");

    let query = harness
        .query_prefetch("mock-vllm-prefix-partial", &hashes)
        .await;
    assert_eq!(query.request.block_hashes, hashes);
    let ready = expect_query_ready(query.response.outcome, "prefix partial query");
    assert_eq!(ready.num_hit_blocks, 2);
    assert!(!ready.lease.is_empty());

    harness.gpus[0].zero();
    let load = harness.submit_load(ready.lease, 2).await;
    assert_eq!(load.request.loads[0].block_ids, vec![0, 1]);
    assert_response_ok(load.response.status.as_ref(), "prefix load");
    harness.wait_for_load(&load.state).await;
    harness.gpus[0].assert_prefix_loaded_and_suffix_zero(2);
}

#[tokio::test]
async fn mock_vllm_load_rejects_lease_block_count_mismatch() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 43);
    let prefix = &hashes[..2];

    let save = harness.save_blocks(prefix).await;
    assert_response_ok(save.response.status.as_ref(), "prefix save");

    let query = harness
        .query_prefetch("mock-vllm-load-count-mismatch", &hashes)
        .await;
    let ready = expect_query_ready(query.response.outcome, "load count mismatch query");
    assert_eq!(ready.num_hit_blocks, 2);
    assert!(!ready.lease.is_empty());

    let mismatch = match harness
        .try_submit_load_for_worker(0, ready.lease, hashes.len())
        .await
    {
        Ok(_) => panic!("load block count mismatch should fail"),
        Err(err) => err,
    };
    assert_eq!(mismatch.request.loads[0].block_ids, vec![0, 1, 2, 3]);
    assert_eq!(mismatch.status.code(), tonic::Code::InvalidArgument);
    assert!(
        mismatch
            .status
            .message()
            .contains("query lease block count 2 does not match destination block count 4"),
        "unexpected error: {}",
        mismatch.status
    );
    assert!(mismatch.state.get() < 0);
}

#[tokio::test]
async fn mock_vllm_session_disconnect_cleans_instance() {
    let mut harness = MockVllmRpcHarness::new().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 47);

    let session = harness.open_session().await;
    assert_eq!(session.request.instance_id, INSTANCE_ID);
    assert_eq!(session.request.namespace, "mock-vllm");
    assert_eq!(session.request.tp_size, 1);
    assert_eq!(session.request.world_size, 1);

    let save = harness.save_blocks(&hashes).await;
    assert_response_ok(save.response.status.as_ref(), "save before session cleanup");

    let query = harness
        .query_prefetch("mock-vllm-session-cleanup-hit", &hashes)
        .await;
    let ready = expect_query_ready(query.response.outcome, "session cleanup hit query");
    assert_eq!(ready.num_hit_blocks as usize, hashes.len());
    assert!(!ready.lease.is_empty());

    drop(session.stream);
    let cleanup_probe = harness.wait_for_instance_cleanup(&hashes).await;
    assert_eq!(cleanup_probe.request.instance_id, INSTANCE_ID);
    assert_eq!(cleanup_probe.request.block_hashes, hashes);
    assert_eq!(cleanup_probe.status.code(), tonic::Code::FailedPrecondition);
    assert!(
        cleanup_probe
            .status
            .message()
            .contains("instance mock-vllm-rpc-e2e not found"),
        "unexpected error: {}",
        cleanup_probe.status
    );

    let save_after_cleanup = match harness.try_save_blocks_for_worker(0, &hashes).await {
        Ok(_) => panic!("save after session cleanup should fail"),
        Err(err) => err,
    };
    assert_eq!(save_after_cleanup.request.instance_id, INSTANCE_ID);
    assert_eq!(
        save_after_cleanup.status.code(),
        tonic::Code::FailedPrecondition
    );
    assert!(
        save_after_cleanup
            .status
            .message()
            .contains("instance mock-vllm-rpc-e2e not found"),
        "unexpected error: {}",
        save_after_cleanup.status
    );

    let load_after_cleanup = match harness
        .try_submit_load_for_worker(0, ready.lease, hashes.len())
        .await
    {
        Ok(_) => panic!("load with pre-cleanup lease should fail after session cleanup"),
        Err(err) => err,
    };
    assert_eq!(load_after_cleanup.request.instance_id, INSTANCE_ID);
    assert_eq!(
        load_after_cleanup.status.code(),
        tonic::Code::FailedPrecondition
    );
    assert!(
        load_after_cleanup
            .status
            .message()
            .contains("instance mock-vllm-rpc-e2e not found"),
        "unexpected error: {}",
        load_after_cleanup.status
    );
    assert!(load_after_cleanup.state.get() < 0);
}

#[tokio::test]
async fn mock_vllm_naive_tp2_load_roundtrip_over_rpc() {
    if cuda_device_count() < 2 {
        eprintln!("skipping naive TP=2 RPC E2E: requires at least 2 CUDA devices");
        return;
    }

    let mut harness = MockVllmRpcHarness::naive_tp2().await;
    let hashes = make_block_hashes(BLOCK_COUNT, 23);

    // TP0 alone must not make the hash prefix visible; TP1 below completes the
    // full TP group and uses the wait-for-visibility helper.
    let save_tp0 = harness.save_blocks_for_worker_no_wait(0, &hashes).await;
    assert_eq!(save_tp0.request.tp_rank, 0);
    assert_eq!(save_tp0.request.device_id, 0);
    assert_eq!(save_tp0.request.saves[0].block_hashes, hashes);
    assert_response_ok(save_tp0.response.status.as_ref(), "save tp0");

    let partial_query = harness
        .query_prefetch("mock-vllm-naive-tp2-before-all-slots", &hashes)
        .await;
    let partial_ready = expect_query_ready(
        partial_query.response.outcome,
        "naive tp2 query before all TP slots are saved",
    );
    assert_eq!(partial_ready.num_hit_blocks, 0);
    assert!(partial_ready.lease.is_empty());

    let save_tp1 = harness.save_blocks_for_worker(1, &hashes).await;
    assert_eq!(save_tp1.request.tp_rank, 1);
    assert_eq!(save_tp1.request.device_id, 1);
    assert_eq!(save_tp1.request.saves[0].block_hashes, hashes);
    assert_response_ok(save_tp1.response.status.as_ref(), "save tp1");

    let hit_query = harness
        .query_prefetch("mock-vllm-naive-tp2-hit", &hashes)
        .await;
    assert_eq!(hit_query.request.instance_id, INSTANCE_ID);
    assert_eq!(hit_query.request.block_hashes, hashes);
    assert_eq!(hit_query.request.req_id, "mock-vllm-naive-tp2-hit");
    let hit_ready = expect_query_ready(hit_query.response.outcome, "naive tp2 hit query");
    assert_eq!(hit_ready.num_hit_blocks as usize, hashes.len());
    assert!(!hit_ready.lease.is_empty());

    harness.gpus[0].zero();
    harness.gpus[1].zero();

    let load_tp0 = harness
        .submit_load_for_worker(0, hit_ready.lease.clone(), hashes.len())
        .await;
    assert_eq!(load_tp0.request.tp_rank, 0);
    assert_eq!(load_tp0.request.device_id, 0);
    assert_eq!(load_tp0.request.loads[0].block_ids, vec![0, 1, 2, 3]);
    assert_response_ok(load_tp0.response.status.as_ref(), "load tp0");

    let load_tp1 = harness
        .submit_load_for_worker(1, hit_ready.lease.clone(), hashes.len())
        .await;
    assert_eq!(load_tp1.request.tp_rank, 1);
    assert_eq!(load_tp1.request.device_id, 1);
    assert_eq!(load_tp1.request.loads[0].block_ids, vec![0, 1, 2, 3]);
    assert_response_ok(load_tp1.response.status.as_ref(), "load tp1");

    harness.wait_for_load(&load_tp0.state).await;
    harness.wait_for_load(&load_tp1.state).await;
    harness.gpus[0].assert_matches_expected();
    harness.gpus[1].assert_matches_expected();

    let over_consumed = match harness
        .try_submit_load_for_worker(0, hit_ready.lease, hashes.len())
        .await
    {
        Ok(_) => panic!("third TP load should exhaust query lease"),
        Err(err) => err,
    };
    assert_eq!(over_consumed.request.tp_rank, 0);
    assert_eq!(over_consumed.request.device_id, 0);
    assert!(
        over_consumed
            .status
            .message()
            .contains("query lease is unknown or expired"),
        "unexpected error: {}",
        over_consumed.status
    );
    assert!(over_consumed.state.get() < 0);
}

fn expect_query_ready(outcome: Option<query_response::Outcome>, context: &str) -> QueryReady {
    match outcome {
        Some(query_response::Outcome::Ready(ready)) => ready,
        Some(query_response::Outcome::Loading(_)) => {
            panic!("{context}: memory-only query should not return Loading")
        }
        None => panic!("{context}: QueryResponse missing outcome"),
    }
}

fn assert_response_ok(status: Option<&pegaflow_server::proto::engine::ResponseStatus>, rpc: &str) {
    let status = status.unwrap_or_else(|| panic!("{rpc} response missing status"));
    assert!(status.ok, "{rpc} failed: {}", status.message);
    assert!(status.message.is_empty());
}
