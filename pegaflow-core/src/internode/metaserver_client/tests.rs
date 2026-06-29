use super::*;
use pegaflow_proto::proto::engine::meta_server_server::{MetaServer, MetaServerServer};
use pegaflow_proto::proto::engine::{
    HeartbeatNodeResponse, InsertBlockHashesResponse, NodePrefixResult, QueryPrefixBlocksRequest,
    QueryPrefixBlocksResponse, RemoveBlockHashesResponse, ResponseStatus, UnregisterNodeResponse,
};
use std::collections::BTreeMap;
#[cfg(feature = "rdma")]
use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use tokio::net::TcpListener;
use tokio::sync::{Notify, oneshot};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status, async_trait};

type RequestLog = Mutex<Vec<(String, Vec<Vec<u8>>)>>;
type RequestSet = BTreeMap<String, Vec<Vec<u8>>>;
type TimeoutLog = Mutex<Vec<&'static str>>;

#[derive(Default)]
struct FakeMetaServerState {
    heartbeat_count: AtomicUsize,
    insert_count: AtomicUsize,
    remove_count: AtomicUsize,
    query_count: AtomicUsize,
    unregister_count: AtomicUsize,
    fail_insert_with_stale_session: AtomicUsize,
    fail_query_unavailable: AtomicUsize,
    insert_requests: RequestLog,
    remove_requests: RequestLog,
    timeout_methods: TimeoutLog,
    heartbeat_notify: Notify,
    insert_notify: Notify,
    remove_notify: Notify,
    query_notify: Notify,
    unregister_notify: Notify,
}

#[derive(Clone)]
struct FakeMetaServer {
    state: Arc<FakeMetaServerState>,
}

fn record_timeout<T>(state: &FakeMetaServerState, method: &'static str, request: &Request<T>) {
    if request.metadata().get("grpc-timeout").is_some() {
        state.timeout_methods.lock().unwrap().push(method);
    }
}

#[async_trait]
impl MetaServer for FakeMetaServer {
    async fn heartbeat_node(
        &self,
        request: Request<HeartbeatNodeRequest>,
    ) -> Result<Response<HeartbeatNodeResponse>, Status> {
        record_timeout(&self.state, "heartbeat_node", &request);
        self.state.heartbeat_count.fetch_add(1, Ordering::SeqCst);
        self.state.heartbeat_notify.notify_waiters();
        Ok(Response::new(HeartbeatNodeResponse {
            stale_after_secs: 2,
        }))
    }

    async fn unregister_node(
        &self,
        request: Request<UnregisterNodeRequest>,
    ) -> Result<Response<UnregisterNodeResponse>, Status> {
        record_timeout(&self.state, "unregister_node", &request);
        self.state.unregister_count.fetch_add(1, Ordering::SeqCst);
        self.state.unregister_notify.notify_waiters();
        Ok(Response::new(UnregisterNodeResponse { removed_owners: 0 }))
    }

    async fn insert_block_hashes(
        &self,
        request: Request<InsertBlockHashesRequest>,
    ) -> Result<Response<InsertBlockHashesResponse>, Status> {
        record_timeout(&self.state, "insert_block_hashes", &request);
        let request = request.into_inner();
        self.state.insert_count.fetch_add(1, Ordering::SeqCst);
        if self
            .state
            .fail_insert_with_stale_session
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(Status::failed_precondition("stale node session"));
        }
        let inserted_count = request.block_hashes.len() as u64;
        self.state
            .insert_requests
            .lock()
            .unwrap()
            .push((request.namespace, request.block_hashes));
        self.state.insert_notify.notify_waiters();
        Ok(Response::new(InsertBlockHashesResponse {
            status: Some(ResponseStatus {
                ok: true,
                message: String::new(),
            }),
            inserted_count,
        }))
    }

    async fn remove_block_hashes(
        &self,
        request: Request<RemoveBlockHashesRequest>,
    ) -> Result<Response<RemoveBlockHashesResponse>, Status> {
        record_timeout(&self.state, "remove_block_hashes", &request);
        let request = request.into_inner();
        self.state.remove_count.fetch_add(1, Ordering::SeqCst);
        let removed_count = request.block_hashes.len() as u64;
        self.state
            .remove_requests
            .lock()
            .unwrap()
            .push((request.namespace, request.block_hashes));
        self.state.remove_notify.notify_waiters();
        Ok(Response::new(RemoveBlockHashesResponse {
            status: Some(ResponseStatus {
                ok: true,
                message: String::new(),
            }),
            removed_count,
        }))
    }

    async fn query_prefix_blocks(
        &self,
        request: Request<QueryPrefixBlocksRequest>,
    ) -> Result<Response<QueryPrefixBlocksResponse>, Status> {
        record_timeout(&self.state, "query_prefix_blocks", &request);
        self.state.query_count.fetch_add(1, Ordering::SeqCst);
        self.state.query_notify.notify_waiters();
        if self
            .state
            .fail_query_unavailable
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(Status::unavailable("synthetic connection failure"));
        }
        Ok(Response::new(QueryPrefixBlocksResponse {
            nodes: vec![NodePrefixResult {
                node: "node-b:50055".to_string(),
                prefix_len: 2,
            }],
        }))
    }
}

async fn start_fake_metaserver() -> (String, Arc<FakeMetaServerState>, oneshot::Sender<()>) {
    let state = Arc::new(FakeMetaServerState::default());
    let service = FakeMetaServer {
        state: Arc::clone(&state),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let incoming = TcpListenerStream::new(listener);
    tokio::spawn(async move {
        let service = MetaServerServer::new(service)
            .max_decoding_message_size(MAX_GRPC_MESSAGE_SIZE)
            .max_encoding_message_size(MAX_GRPC_MESSAGE_SIZE);
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await
            .unwrap();
    });
    (format!("http://{addr}"), state, shutdown_tx)
}

fn start_client(addr: String) -> MetaServerClient {
    MetaServerClient::new(MetaServerClientConfig::new(
        addr,
        "node-a:50055".to_string(),
    ))
    .expect("valid fake MetaServer endpoint")
}

fn collect_requests(requests: &RequestLog) -> RequestSet {
    let mut grouped = BTreeMap::new();
    for (namespace, hashes) in requests.lock().unwrap().iter() {
        let namespace_hashes: &mut Vec<Vec<u8>> = grouped.entry(namespace.clone()).or_default();
        namespace_hashes.extend(hashes.iter().cloned());
    }
    for hashes in grouped.values_mut() {
        hashes.sort();
    }
    grouped
}

fn expected_requests(entries: &[(&str, Vec<u8>)]) -> RequestSet {
    let mut grouped: RequestSet = BTreeMap::new();
    for (namespace, hash) in entries {
        grouped
            .entry((*namespace).to_string())
            .or_default()
            .push(hash.clone());
    }
    for hashes in grouped.values_mut() {
        hashes.sort();
    }
    grouped
}

async fn wait_for_count(notify: &Notify, count: &AtomicUsize, expected: usize) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    loop {
        if count.load(Ordering::SeqCst) >= expected {
            return;
        }
        tokio::select! {
            _ = notify.notified() => {}
            _ = tokio::time::sleep_until(deadline) => {
                panic!("timed out waiting for count {expected}");
            }
        }
    }
}

#[tokio::test]
async fn heartbeat_loop_sends_initial_heartbeat_and_unregisters_on_shutdown() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    let client = start_client(addr);

    wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
    client.shutdown().await;
    wait_for_count(&service.unregister_notify, &service.unregister_count, 1).await;

    let _ = shutdown_tx.send(());
}

#[test]
fn invalid_metaserver_endpoint_is_returned_as_error() {
    let result = MetaServerClient::new(MetaServerClientConfig::new(
        "not a uri".to_string(),
        "node-a:50055".to_string(),
    ));

    assert!(result.is_err());
}

#[cfg(feature = "rdma")]
#[test]
fn retryable_query_status_matches_tonic_transport_and_timeout_shapes() {
    assert!(is_retryable_query_status(&Status::unavailable(
        "transport error"
    )));
    assert!(is_retryable_query_status(&Status::deadline_exceeded(
        "deadline exceeded"
    )));
    assert!(is_retryable_query_status(&Status::cancelled(
        "Timeout expired"
    )));
    assert!(is_retryable_query_status(&Status::unknown(
        "Service was not ready: transport error"
    )));
    assert!(is_retryable_query_status(&Status::unknown(
        "transport error: Connection reset by peer"
    )));

    assert!(!is_retryable_query_status(&Status::cancelled(
        "client cancelled"
    )));
    assert!(!is_retryable_query_status(&Status::invalid_argument(
        "bad request"
    )));
}

#[tokio::test]
async fn stale_session_write_error_triggers_new_heartbeat() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    service
        .fail_insert_with_stale_session
        .store(1, Ordering::SeqCst);
    let client = start_client(addr);

    wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
    client.try_register_namespace("ns".to_string(), vec![vec![1]]);
    wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 2).await;

    client.shutdown().await;
    let _ = shutdown_tx.send(());
}

#[cfg(feature = "rdma")]
#[tokio::test]
async fn metaserver_rpcs_carry_deadline_metadata() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    let client = start_client(addr);

    wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;
    client.try_register_namespace("ns".to_string(), vec![vec![1]]);
    client.try_unregister(vec![("ns".to_string(), vec![2])]);
    let nodes = client
        .query_prefix("ns", &[vec![1], vec![2]])
        .await
        .unwrap();
    assert_eq!(nodes.len(), 1);

    wait_for_count(&service.insert_notify, &service.insert_count, 1).await;
    wait_for_count(&service.remove_notify, &service.remove_count, 1).await;

    client.shutdown().await;
    wait_for_count(&service.unregister_notify, &service.unregister_count, 1).await;

    let methods: BTreeSet<&'static str> = service
        .timeout_methods
        .lock()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(
        methods,
        BTreeSet::from([
            "heartbeat_node",
            "insert_block_hashes",
            "remove_block_hashes",
            "query_prefix_blocks",
            "unregister_node",
        ])
    );

    let _ = shutdown_tx.send(());
}

#[cfg(feature = "rdma")]
#[tokio::test]
async fn query_prefix_refreshes_channel_and_retries_once_on_unavailable() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    service.fail_query_unavailable.store(1, Ordering::SeqCst);
    let client = start_client(addr);

    let nodes = client
        .query_prefix("ns", &[vec![1], vec![2]])
        .await
        .unwrap();

    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node, "node-b:50055");
    assert_eq!(nodes[0].prefix_len, 2);
    assert_eq!(service.query_count.load(Ordering::SeqCst), 2);

    client.shutdown().await;
    let _ = shutdown_tx.send(());
}

#[test]
fn unsent_after_failure_counts_only_unsent_tail() {
    let ns = |name: &str, n: usize| (name.to_string(), vec![vec![0u8]; n]);
    let namespaces = vec![ns("a", 100), ns("b", 50), ns("c", 30)];

    // First chunk of "b" failed (nothing of b sent yet): all of b + c.
    assert_eq!(unsent_after_failure(&namespaces, 1, 0), 50 + 30);
    // 40 hashes of "b" already landed before the failing chunk: b's tail + c.
    assert_eq!(unsent_after_failure(&namespaces, 1, 40), 10 + 30);
    // Failure in the last namespace: only its unsent tail, no later namespaces.
    assert_eq!(unsent_after_failure(&namespaces, 2, 20), 10);
    // Offset past the namespace length saturates to zero unsent.
    assert_eq!(unsent_after_failure(&namespaces, 2, 999), 0);
}

#[tokio::test]
async fn large_namespace_removal_splits_into_bounded_rpcs() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    let client = start_client(addr);
    wait_for_count(&service.heartbeat_notify, &service.heartbeat_count, 1).await;

    // One namespace coalesced past the per-RPC cap. Use sha256-sized (32B)
    // hashes so the total payload is large enough to prove the registration
    // loop still emits bounded RPCs instead of one burst-sized request.
    // Chunking holds each request to MAX_HASHES_PER_RPC * 32B = 512 KiB.
    let sha256_hash = |i: u32| {
        let mut h = vec![0u8; 32];
        h[..4].copy_from_slice(&i.to_le_bytes());
        h
    };
    let total = MAX_HASHES_PER_RPC * 10 + 1;
    let expected_chunks = total.div_ceil(MAX_HASHES_PER_RPC);
    let entries: Vec<(String, Vec<u8>)> = (0..total as u32)
        .map(|i| ("ns".to_string(), sha256_hash(i)))
        .collect();
    client.try_unregister(entries);

    wait_for_count(
        &service.remove_notify,
        &service.remove_count,
        expected_chunks,
    )
    .await;

    let logged = service.remove_requests.lock().unwrap().clone();
    assert_eq!(
        logged.len(),
        expected_chunks,
        "removal split into wrong RPC count"
    );
    let mut sent: Vec<Vec<u8>> = Vec::with_capacity(total);
    for (namespace, hashes) in &logged {
        assert_eq!(namespace, "ns");
        assert!(
            hashes.len() <= MAX_HASHES_PER_RPC,
            "chunk of {} hashes exceeds cap {MAX_HASHES_PER_RPC}",
            hashes.len()
        );
        sent.extend(hashes.iter().cloned());
    }
    sent.sort();
    let mut expected: Vec<Vec<u8>> = (0..total as u32).map(sha256_hash).collect();
    expected.sort();
    assert_eq!(sent, expected, "chunking lost or duplicated hashes");

    client.shutdown().await;
    let _ = shutdown_tx.send(());
}

#[tokio::test]
async fn mixed_insert_remove_drain_preserves_last_write_per_namespace() {
    let (addr, service, shutdown_tx) = start_fake_metaserver().await;
    let (tx, rx) = mpsc::channel(16);

    let insert_then_remove = vec![0xa0];
    let remove_then_insert = vec![0xb0];
    let remove_only = vec![0xc0];
    let insert_only = vec![0xd0];

    tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
        "ns-first".to_string(),
        vec![insert_then_remove.clone()],
    )))
    .unwrap();
    tx.try_send(MetaServerCommand::Remove(BlockHashBatch::from_entries(
        vec![("ns-first".to_string(), insert_then_remove.clone())],
    )))
    .unwrap();
    tx.try_send(MetaServerCommand::Remove(BlockHashBatch::from_entries(
        vec![
            ("ns-second".to_string(), remove_then_insert.clone()),
            ("ns-remove".to_string(), remove_only.clone()),
        ],
    )))
    .unwrap();
    tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
        "ns-second".to_string(),
        vec![remove_then_insert.clone()],
    )))
    .unwrap();
    tx.try_send(MetaServerCommand::Insert(BlockHashBatch::single_namespace(
        "ns-insert".to_string(),
        vec![insert_only.clone()],
    )))
    .unwrap();

    let endpoint = metaserver_endpoint(addr.clone()).expect("valid fake MetaServer endpoint");
    let loop_task = tokio::spawn(registration_loop(
        rx,
        addr,
        endpoint,
        "node-a:50055".to_string(),
    ));

    wait_for_count(&service.insert_notify, &service.insert_count, 2).await;
    wait_for_count(&service.remove_notify, &service.remove_count, 2).await;

    assert_eq!(
        collect_requests(&service.insert_requests),
        expected_requests(&[
            ("ns-second", remove_then_insert),
            ("ns-insert", insert_only),
        ])
    );
    assert_eq!(
        collect_requests(&service.remove_requests),
        expected_requests(&[("ns-first", insert_then_remove), ("ns-remove", remove_only)])
    );

    let (done_tx, done_rx) = oneshot::channel();
    tx.send(MetaServerCommand::Shutdown(done_tx)).await.unwrap();
    done_rx.await.unwrap();
    loop_task.await.unwrap();
    let _ = shutdown_tx.send(());
}
