//! Regression guard for the "everything hangs after a few `curl
//! .../instances/cleanup`" incident, and for the structural fix that put the
//! CUDA tensor registry behind a dedicated thread ([`RegistryHandle`]).
//!
//! Root cause: registry mutations (`register` / `cleanup` / session-close
//! cleanup) take the Python GIL and run a blocking `torch.cuda.empty_cache()`
//! that performs a CUDA device sync. When the GPU is wedged that sync never
//! returns. Originally those calls ran *inline on async worker threads* (HTTP
//! `cleanup_handler` and several gRPC handlers). A single stuck call pinned a
//! tokio worker; a few of them pinned every worker on the shared runtime, and
//! the whole server went dark — including pure-Rust endpoints like `/health`
//! and `/metrics` that touch neither the GIL nor the registry. They did not
//! hang on the GIL; they hung because no worker was left to schedule them
//! (executor starvation).
//!
//! The fix moves the registry onto one dedicated `cuda-registry` thread; every
//! handler (HTTP and gRPC) only `.await`s a reply. A wedged CUDA call now pins
//! that single thread, never an async worker.
//!
//! This test reproduces the real mechanism without a wedged GPU: it holds the
//! process-wide GIL from a side thread, then drives the gRPC `register_context`
//! hot path. The registry actor blocks inside `materialize_tensor`'s
//! `Python::attach` — exactly where a stuck `empty_cache` would block — wedging
//! its thread. We then assert that the unrelated endpoints, on the SAME
//! 2-worker runtime, stay responsive: HTTP `/health` + `/metrics` AND a gRPC
//! `health` RPC. Pre-fix this starved all of them; post-fix it must not.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use pegaflow_core::{PegaEngine, StorageConfig};
use pegaflow_server::http_server::start_http_server;
use pegaflow_server::proto::engine::engine_client::EngineClient;
use pegaflow_server::proto::engine::engine_server::EngineServer;
use pegaflow_server::proto::engine::{HealthRequest, RegisterContextRequest};
use pegaflow_server::{CudaTensorRegistry, GrpcEngineService, RegistryHandle};
use prometheus::Registry;
use pyo3::Python;
use tokio::sync::Notify;
use tonic::transport::{Channel, Server};

fn ephemeral_addr() -> SocketAddr {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind ephemeral port");
    listener.local_addr().expect("ephemeral local addr")
}

/// Build the engine and return it together with the CUDA context it was
/// allocated against. The caller must keep the context alive for as long as the
/// engine is used: the pinned-memory pool (cudaHostAlloc) needs a live CUDA
/// context current on the allocating thread.
fn build_engine() -> (Arc<PegaEngine>, Arc<CudaContext>) {
    let ctx = CudaContext::new(0).expect("CUDA init on device 0");
    ctx.bind_to_thread().expect("bind CUDA context to thread");
    let engine = Arc::new(
        PegaEngine::new_with_config(
            16 << 20,
            false,
            StorageConfig {
                enable_lfu_admission: false,
                ..StorageConfig::default()
            },
        )
        .expect("test engine should start"),
    );
    (engine, ctx)
}

/// Issue one HTTP/1.1 request on a fresh connection and read the full response.
/// Returns `Err` if the server does not answer within `timeout` — that timeout
/// IS the hang we are trying to observe.
fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    timeout: Duration,
) -> std::io::Result<String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let req = format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes())?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    Ok(buf)
}

fn wait_until_serving(addr: SocketAddr) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if http_request(addr, "GET", "/health", Duration::from_millis(500))
            .map(|resp| resp.contains("200"))
            .unwrap_or(false)
        {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!("server never became ready at {addr}");
}

struct TestCluster {
    http_addr: SocketAddr,
    client: EngineClient<Channel>,
    rt: tokio::runtime::Runtime,
    // Kept alive so the engine's pinned pool stays valid for the whole test.
    _ctx: Arc<CudaContext>,
}

impl TestCluster {
    /// Assert an HTTP GET answers within `timeout` and the response contains
    /// `needle`. The `.expect` failure message IS the starvation signal.
    fn assert_http(&self, path: &str, needle: &str, timeout: Duration, ctx: &str) {
        let resp = http_request(self.http_addr, "GET", path, timeout)
            .unwrap_or_else(|e| panic!("{path} hung/failed ({ctx}): {e}"));
        assert!(
            resp.contains(needle),
            "unexpected {path} response ({ctx}): {resp:?}"
        );
    }

    /// Assert a gRPC `health` RPC answers within `timeout`. A timeout means the
    /// gRPC server was starved by the wedged registry — the bug we guard.
    fn assert_grpc_health(&self, timeout: Duration, ctx: &str) {
        let mut client = self.client.clone();
        let outcome = self.rt.block_on(async move {
            tokio::time::timeout(timeout, client.health(HealthRequest {})).await
        });
        let resp = outcome
            .unwrap_or_else(|_| {
                panic!("gRPC health timed out ({ctx}): registry wedge starved gRPC")
            })
            .unwrap_or_else(|e| panic!("gRPC health RPC failed ({ctx}): {e}"));
        assert!(
            resp.into_inner().status.is_some(),
            "gRPC health returned no status ({ctx})"
        );
    }

    /// Assert a `register_context` RPC does NOT complete within `within` — i.e.
    /// the registry actor really is wedged on the GIL. Without this, the
    /// responsiveness checks could false-pass: if the GIL hold silently failed,
    /// the register RPCs would just complete and never test starvation at all.
    fn assert_registry_wedged(&self, within: Duration) {
        let mut client = self.client.clone();
        let outcome = self.rt.block_on(async move {
            tokio::time::timeout(
                within,
                client.register_context_batch(wedge_register_request(99)),
            )
            .await
        });
        assert!(
            outcome.is_err(),
            "register_context completed despite the GIL wedge — the wedge did not take \
             effect, so this test proves nothing about starvation"
        );
    }
}

/// A `register_context` request whose only purpose is to make the registry
/// actor enter `materialize_tensor` (and thus `Python::attach`). It passes
/// validation but its `wrapper_bytes` are never parsed — the actor blocks on
/// the GIL before it touches them.
fn wedge_register_request(i: usize) -> RegisterContextRequest {
    RegisterContextRequest {
        instance_id: format!("wedge-{i}"),
        namespace: "wedge".to_string(),
        tp_rank: 0,
        tp_size: 1,
        world_size: 1,
        device_id: 0,
        num_layers: 1,
        layer_names: vec!["k0".to_string()],
        wrapper_bytes: vec![vec![0u8; 4]],
        num_blocks: vec![1],
        bytes_per_block: vec![1],
        kv_stride_bytes: vec![1],
        segments: vec![1],
        pp_rank: 0,
    }
}

/// Start a gRPC + HTTP server pair that share ONE registry actor, on a runtime
/// with `worker_threads` async workers, plus a connected gRPC client.
fn start_cluster(worker_threads: usize) -> TestCluster {
    let http_addr = ephemeral_addr();
    let grpc_addr = ephemeral_addr();
    let (engine, ctx) = build_engine();
    let registry = RegistryHandle::spawn(CudaTensorRegistry::empty());
    let shutdown = Arc::new(Notify::new());
    let hll_tracker = Arc::new(std::sync::Mutex::new(
        pegaflow_common::hll::MultiWindowHllTracker::new(
            vec![("24h".into(), Duration::from_secs(86400))],
            14,
        ),
    ));
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .enable_all()
        .build()
        .expect("build server runtime");

    let service = GrpcEngineService::new(
        Arc::clone(&engine),
        registry.clone(),
        Arc::clone(&shutdown),
        hll_tracker,
    );
    rt.spawn(async move {
        Server::builder()
            .add_service(EngineServer::new(service))
            .serve(grpc_addr)
            .await
            .expect("gRPC serve");
    });

    rt.block_on(async {
        start_http_server(
            http_addr,
            Arc::clone(&engine),
            registry,
            true,
            Some(Registry::new()),
            Arc::clone(&shutdown),
        )
        .await
        .expect("start http server");
    });
    wait_until_serving(http_addr);

    let endpoint = format!("http://{grpc_addr}");
    let client = rt.block_on(async {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match EngineClient::connect(endpoint.clone()).await {
                Ok(c) => break c,
                Err(e) => {
                    assert!(Instant::now() < deadline, "gRPC client connect failed: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    });

    TestCluster {
        http_addr,
        client,
        rt,
        _ctx: ctx,
    }
}

/// Regression guard: a wedged registry actor must not starve the async runtime.
///
/// On a deliberately under-provisioned 2-worker runtime we wedge the single
/// `cuda-registry` thread via the real mechanism (held GIL inside
/// `materialize_tensor`) and pile up 4 stuck `register_context` RPCs. Pre-fix,
/// those registry calls ran inline on the async workers and this starved
/// `/health`, `/metrics`, and every gRPC RPC. Post-fix, handlers only `.await`
/// the registry thread, so all unrelated endpoints must keep answering quickly.
#[test]
fn wedged_registry_actor_does_not_starve_async_endpoints() {
    let cluster = start_cluster(2);

    // Baseline: everything answers instantly when nothing is wedged.
    cluster.assert_http("/health", "200", Duration::from_secs(2), "baseline");
    cluster.assert_http("/metrics", "200", Duration::from_secs(2), "baseline");
    cluster.assert_grpc_health(Duration::from_secs(2), "baseline");

    // Hold the process-wide GIL forever from a side thread. This is the real
    // thing that makes a registry op block: any later `Python::attach` waits
    // here. (No torch needed — the actor blocks acquiring the GIL, before it
    // would import torch.)
    thread::spawn(|| {
        Python::attach(|_py| {
            loop {
                thread::sleep(Duration::from_secs(3600));
            }
        });
    });
    thread::sleep(Duration::from_millis(300)); // ensure the GIL is actually held

    // Drive the gRPC register hot path. Each RPC reaches the registry actor,
    // which blocks in `Python::attach`; the RPCs hang (their replies never
    // come) but only ever `.await` — no async worker is pinned.
    for i in 0..4 {
        let mut client = cluster.client.clone();
        cluster.rt.spawn(async move {
            let _ = client
                .register_context_batch(wedge_register_request(i))
                .await;
        });
    }
    thread::sleep(Duration::from_millis(500)); // let the actor pick up + block

    // Confirm the wedge is real before trusting the responsiveness checks: a
    // register RPC must NOT complete while the actor is GIL-blocked.
    cluster.assert_registry_wedged(Duration::from_millis(800));

    // The invariant under test: unrelated endpoints stay responsive despite the
    // wedged registry actor and stuck register RPCs, on only 2 async workers.
    let t = Instant::now();
    cluster.assert_http("/health", "200", Duration::from_secs(2), "wedged");
    cluster.assert_http("/metrics", "200", Duration::from_secs(2), "wedged");
    cluster.assert_grpc_health(Duration::from_secs(2), "wedged");
    eprintln!(
        "[fixed / 2 workers] /health + /metrics + gRPC health stayed responsive in {:?} despite a wedged registry actor and 4 stuck register RPCs",
        t.elapsed()
    );

    // The GIL-holder thread keeps the actor wedged; drop the runtime without
    // joining the stuck register tasks.
    cluster.rt.shutdown_background();
}
