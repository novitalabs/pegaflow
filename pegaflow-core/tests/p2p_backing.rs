//! P2P backing store integration tests.
//!
//! Verifies that block hashes are registered with the MetaServer coordinator
//! and that the full save → load roundtrip works with P2P-backed storage.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use pegaflow_core::metaserver::{BlockHashStore, GrpcMetaService};
use pegaflow_core::*;
use pegaflow_proto::proto::engine::meta_server_client::MetaServerClient;
use pegaflow_proto::proto::engine::meta_server_server::MetaServerServer;
use pegaflow_proto::proto::engine::{
    HealthRequest, InsertBlockHashesRequest, QueryBlockHashesRequest,
};
use pegaflow_transfer::MooncakeTransferEngine;
use tokio::sync::Notify;
use tonic::transport::Server;

/// Start an in-process MetaServer on a random port.
///
/// Returns `(endpoint_url, store)` so the test can both configure the P2P
/// backing store and verify registered block hashes directly.
async fn start_test_metaserver() -> (String, Arc<BlockHashStore>) {
    let store = Arc::new(BlockHashStore::with_capacity_and_ttl(10 * 1024 * 1024, 5));
    let shutdown = Arc::new(Notify::new());
    let service = GrpcMetaService::new(Arc::clone(&store), shutdown);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind random port");
    let addr = listener.local_addr().expect("local_addr");
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(MetaServerServer::new(service))
            .serve_with_incoming(incoming)
            .await
            .expect("metaserver serve");
    });

    (format!("http://{}", addr), store)
}

/// Poll the MetaServer store until at least `expected` entries are registered, or timeout.
async fn wait_for_metaserver_entries(store: &BlockHashStore, expected: u64, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        store.run_pending_tasks().await;
        let count = store.entry_count();
        if count >= expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} metaserver entries (got {count})"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// P2P smoke test: start an in-process MetaServer, save blocks with P2P
/// backing enabled, verify hashes are registered, then complete roundtrip.
#[tokio::test]
async fn p2p_smoke_roundtrip() {
    const NUM_BLOCKS: usize = 4;
    const BLOCK_SIZE: usize = 1024;

    let (metaserver_addr, metaserver_store) = start_test_metaserver().await;

    let harness = RoundtripHarness::new(
        HarnessConfig::new("test-p2p-smoke", "test-ns-p2p", NUM_BLOCKS, BLOCK_SIZE)
            .with_hash_salt(55)
            .with_storage_config(StorageConfig {
                enable_lfu_admission: false,
                hint_value_size_bytes: None,
                max_prefetch_blocks: 100,
                baking_store_config: Some(BakingStoreConfig {
                    p2p_coordinator_addr: metaserver_addr,
                    p2p_node_addr: "127.0.0.1:50055".to_string(),
                    node_id: "test-node".to_string(),
                }),
                ssd_cache_config: None,
                enable_numa_affinity: false,
                // Uninitialized engine: insert path doesn't use RDMA,
                // but the P2P store requires a non-None engine to be created.
                transfer_engine: Some(Arc::new(MooncakeTransferEngine::new())),
            }),
    );

    // Save all blocks: GPU → pinned memory → cache + P2P hash registration
    harness.save_all().await;
    harness.assert_cache_eventually_all().await;

    // Verify block hashes were registered with the MetaServer
    wait_for_metaserver_entries(&metaserver_store, NUM_BLOCKS as u64, CACHE_WAIT_TIMEOUT).await;

    // Normal roundtrip: zero GPU → pin → load → verify
    harness.zero_gpu_and_assert();
    harness.expect_query_prefetch_done_all();
    harness.load_all_and_wait().await.expect("batch_load");
    harness.assert_gpu_matches_host();
}

/// Test the QueryBlockHashes RPC: insert via gRPC then query via gRPC.
#[tokio::test]
async fn metaserver_grpc_query_returns_inserted_hashes() {
    let (addr, _store) = start_test_metaserver().await;
    let mut client = MetaServerClient::connect(addr).await.expect("connect");

    let namespace = "test-ns";
    let node = "127.0.0.1:50055";
    let hashes = vec![vec![1u8, 2, 3], vec![4u8, 5, 6], vec![7u8, 8, 9]];

    // Insert
    let insert_resp = client
        .insert_block_hashes(InsertBlockHashesRequest {
            namespace: namespace.to_string(),
            node: node.to_string(),
            block_hashes: hashes.clone(),
            domain_addresses: vec![],
        })
        .await
        .expect("insert rpc")
        .into_inner();
    assert!(insert_resp.status.unwrap().ok);
    assert_eq!(insert_resp.inserted_count, 3);

    // Query all: expect all found
    let query_resp = client
        .query_block_hashes(QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: hashes.clone(),
        })
        .await
        .expect("query rpc")
        .into_inner();
    assert!(query_resp.status.unwrap().ok);
    assert_eq!(query_resp.total_queried, 3);
    assert_eq!(query_resp.found_count, 3);
    assert_eq!(query_resp.existing_hashes.len(), 3);
    assert_eq!(query_resp.node_blocks.len(), 1);
    assert_eq!(query_resp.node_blocks[0].node, node);

    // Query with a mix: one unknown hash
    let mixed = vec![hashes[0].clone(), vec![0xde, 0xad]];
    let query_resp = client
        .query_block_hashes(QueryBlockHashesRequest {
            namespace: namespace.to_string(),
            block_hashes: mixed,
        })
        .await
        .expect("query rpc mixed")
        .into_inner();
    assert_eq!(query_resp.total_queried, 2);
    assert_eq!(query_resp.found_count, 1);

    // Query wrong namespace: expect zero hits
    let query_resp = client
        .query_block_hashes(QueryBlockHashesRequest {
            namespace: "other-ns".to_string(),
            block_hashes: hashes,
        })
        .await
        .expect("query rpc wrong ns")
        .into_inner();
    assert_eq!(query_resp.found_count, 0);
}

/// Empty block_hashes should return an error status for both Insert and Query.
#[tokio::test]
async fn metaserver_grpc_empty_hashes_returns_error() {
    let (addr, _store) = start_test_metaserver().await;
    let mut client = MetaServerClient::connect(addr).await.expect("connect");

    let insert_resp = client
        .insert_block_hashes(InsertBlockHashesRequest {
            namespace: "ns".to_string(),
            node: "node".to_string(),
            block_hashes: vec![],
            domain_addresses: vec![],
        })
        .await
        .expect("insert rpc")
        .into_inner();
    assert!(!insert_resp.status.unwrap().ok);
    assert_eq!(insert_resp.inserted_count, 0);

    let query_resp = client
        .query_block_hashes(QueryBlockHashesRequest {
            namespace: "ns".to_string(),
            block_hashes: vec![],
        })
        .await
        .expect("query rpc")
        .into_inner();
    assert!(!query_resp.status.unwrap().ok);
    assert_eq!(query_resp.found_count, 0);
}

/// Health RPC should always return ok.
#[tokio::test]
async fn metaserver_grpc_health_returns_ok() {
    let (addr, _store) = start_test_metaserver().await;
    let mut client = MetaServerClient::connect(addr).await.expect("connect");

    let resp = client
        .health(HealthRequest {})
        .await
        .expect("health rpc")
        .into_inner();
    assert!(resp.status.unwrap().ok);
}
