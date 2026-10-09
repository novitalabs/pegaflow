// RAII release of a remote transfer-lock session acquired via
// QueryBlocksForTransfer. See `rdma_fetch` for the fetch flow that owns it.

use log::warn;
use pegaflow_proto::proto::engine::ReleaseTransferLockRequest;
use pegaflow_proto::proto::engine::engine_client::EngineClient;
use tonic::transport::Channel;

/// Releases a transfer session exactly once, on whichever exit runs first:
/// `release()` on the coded completion paths, or `Drop` when the fetch task
/// panics or its future is dropped mid-await. A session that is never
/// released pins the remote blocks until the holder's GC expires it.
pub(super) struct TransferLockGuard {
    client: EngineClient<Channel>,
    session_id: String,
    remote_addr: String,
    req_id: String,
    // Captured at construction (always inside the runtime) so Drop can spawn
    // even from a panic unwind; spawning on a shut-down runtime is a no-op.
    handle: tokio::runtime::Handle,
}

impl TransferLockGuard {
    pub(super) fn new(
        client: EngineClient<Channel>,
        session_id: String,
        remote_addr: &str,
        req_id: &str,
    ) -> Self {
        Self {
            client,
            session_id,
            remote_addr: remote_addr.to_string(),
            req_id: req_id.to_string(),
            handle: tokio::runtime::Handle::current(),
        }
    }

    /// Release on a completed fetch (success or handled error). Fire-and-forget.
    /// `transferred` tells the holder every block arrived, so it may demote
    /// its source copies.
    pub(super) fn release(mut self, transferred: bool) {
        self.spawn_release(transferred);
    }

    fn spawn_release(&mut self, transferred: bool) {
        let session_id = std::mem::take(&mut self.session_id);
        if session_id.is_empty() {
            return;
        }
        let mut client = self.client.clone();
        self.handle.spawn(async move {
            let req = ReleaseTransferLockRequest {
                transfer_session_id: session_id.clone(),
                transferred,
            };
            if let Err(e) = client.release_transfer_lock(req).await {
                warn!("ReleaseTransferLock failed for session {session_id}: {e}");
            }
        });
    }
}

impl Drop for TransferLockGuard {
    fn drop(&mut self) {
        if self.session_id.is_empty() {
            return;
        }
        // Only panic/cancellation reaches here — the coded paths call release().
        warn!(
            "RDMA fetch aborted without releasing transfer lock; releasing via drop guard: session={} remote={} req_id={}",
            self.session_id, self.remote_addr, self.req_id
        );
        self.spawn_release(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use parking_lot::Mutex;

    use pegaflow_proto::proto::engine::engine_server::{Engine, EngineServer};
    use pegaflow_proto::proto::engine::{
        HealthRequest, HealthResponse, LoadRequest, LoadResponse, QueryBlocksForTransferRequest,
        QueryBlocksForTransferResponse, QueryRequest, QueryResponse, RdmaHandshakeRequest,
        RdmaHandshakeResponse, RegisterContextRequest, RegisterContextResponse, ReleaseRequest,
        ReleaseResponse, ReleaseTransferLockResponse, SaveRequest, SaveResponse, SessionEvent,
        SessionRequest, ShutdownRequest, ShutdownResponse, UnregisterRequest, UnregisterResponse,
    };
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Endpoint;
    use tonic::{Request, Response, Status};

    /// Stub engine that records the `transferred` flag of each ReleaseTransferLock.
    struct ReleaseCounter(Arc<Mutex<Vec<bool>>>);

    #[tonic::async_trait]
    impl Engine for ReleaseCounter {
        async fn release_transfer_lock(
            &self,
            request: Request<ReleaseTransferLockRequest>,
        ) -> Result<Response<ReleaseTransferLockResponse>, Status> {
            self.0.lock().push(request.into_inner().transferred);
            Ok(Response::new(ReleaseTransferLockResponse {
                status: None,
                released_blocks: 0,
            }))
        }

        async fn query_blocks_for_transfer(
            &self,
            _request: Request<QueryBlocksForTransferRequest>,
        ) -> Result<Response<QueryBlocksForTransferResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn rdma_handshake(
            &self,
            _request: Request<RdmaHandshakeRequest>,
        ) -> Result<Response<RdmaHandshakeResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn health(
            &self,
            _request: Request<HealthRequest>,
        ) -> Result<Response<HealthResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn register_context_batch(
            &self,
            _request: Request<RegisterContextRequest>,
        ) -> Result<Response<RegisterContextResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn save(
            &self,
            _request: Request<SaveRequest>,
        ) -> Result<Response<SaveResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn load(
            &self,
            _request: Request<LoadRequest>,
        ) -> Result<Response<LoadResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn query_prefetch(
            &self,
            _request: Request<QueryRequest>,
        ) -> Result<Response<QueryResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn release(
            &self,
            _request: Request<ReleaseRequest>,
        ) -> Result<Response<ReleaseResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn unregister_context(
            &self,
            _request: Request<UnregisterRequest>,
        ) -> Result<Response<UnregisterResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        async fn shutdown(
            &self,
            _request: Request<ShutdownRequest>,
        ) -> Result<Response<ShutdownResponse>, Status> {
            Err(Status::unimplemented("stub"))
        }
        type SessionStream = futures::stream::Empty<Result<SessionEvent, Status>>;
        async fn session(
            &self,
            _request: Request<SessionRequest>,
        ) -> Result<Response<Self::SessionStream>, Status> {
            Err(Status::unimplemented("stub"))
        }
    }

    /// Serve a ReleaseCounter on an ephemeral loopback port; return a
    /// connected client and the shared counter.
    async fn start_counter_server() -> (EngineClient<Channel>, Arc<Mutex<Vec<bool>>>) {
        let counter = Arc::new(Mutex::new(Vec::new()));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let service = EngineServer::new(ReleaseCounter(Arc::clone(&counter)));
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(TcpListenerStream::new(listener)),
        );
        let channel = Endpoint::from_shared(format!("http://{addr}"))
            .expect("endpoint")
            .connect_lazy();
        (EngineClient::new(channel), counter)
    }

    async fn wait_for_count(counter: &Mutex<Vec<bool>>, expected: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while counter.lock().len() < expected {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("release RPC should arrive");
    }

    fn guard(client: &EngineClient<Channel>, session: &str) -> TransferLockGuard {
        TransferLockGuard::new(client.clone(), session.to_string(), "remote", "req")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn releases_exactly_once_on_every_exit_path() {
        let (client, counter) = start_counter_server().await;

        // Completed transfer on the coded path.
        guard(&client, "transferred").release(true);
        wait_for_count(&counter, 1).await;

        // Handled transfer error on the coded path.
        guard(&client, "failed").release(false);
        wait_for_count(&counter, 2).await;

        // Drop without release (future cancelled) still releases.
        drop(guard(&client, "dropped"));
        wait_for_count(&counter, 3).await;

        // Panic unwinding through the guard still releases.
        let g = guard(&client, "panicked");
        let task = tokio::spawn(async move {
            let _g = g;
            panic!("simulated fetch panic");
        });
        assert!(task.await.is_err());
        wait_for_count(&counter, 4).await;

        // No double release: settle window after all paths. Only a completed
        // transfer may demote the holder's source copies.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(*counter.lock(), vec![true, false, false, false]);

        // Empty session (holder returned none) never sends an RPC.
        guard(&client, "").release(true);
        drop(guard(&client, ""));
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(counter.lock().len(), 4);
    }
}
