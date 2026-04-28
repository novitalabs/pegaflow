//! PegaFlow P/D Disaggregation Router
//!
//! Simple router that coordinates prefill (P) and decode (D) nodes.
//! Flow:
//! 1. Receive request
//! 2. Send to P node (max_tokens=1)
//! 3. Forward to D node (P response means KV is ready)

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
};
use clap::Parser;
use log::{error, info};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_stream::StreamExt;

const PREFILL_MAX_TOKENS: u64 = 1;

#[derive(Clone)]
struct RouterState {
    prefill_clients: Arc<Vec<Client>>,
    decode_clients: Arc<Vec<Client>>,
    prefill_urls: Arc<Vec<String>>,
    decode_urls: Arc<Vec<String>>,
    p_index: Arc<AtomicUsize>,
    d_index: Arc<AtomicUsize>,
    pd_push: bool,
    d_pegaflow_addrs: Arc<Vec<String>>,
    decode_instance_ids: Arc<Vec<String>>,
    // Track in-flight requests per node
    p_inflight: Arc<Vec<AtomicUsize>>,
    d_inflight: Arc<Vec<AtomicUsize>>,
}

impl RouterState {
    fn new(
        prefill_endpoints: Vec<String>,
        decode_endpoints: Vec<String>,
        d_pegaflow_addrs: Vec<String>,
        decode_instance_ids: Vec<String>,
        pd_push: bool,
    ) -> Self {
        let prefill_clients = prefill_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build prefill client")
            })
            .collect();

        let decode_clients = decode_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build decode client")
            })
            .collect();

        let p_inflight: Vec<AtomicUsize> = prefill_endpoints
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();
        let d_inflight: Vec<AtomicUsize> = decode_endpoints
            .iter()
            .map(|_| AtomicUsize::new(0))
            .collect();

        Self {
            prefill_clients: Arc::new(prefill_clients),
            decode_clients: Arc::new(decode_clients),
            prefill_urls: Arc::new(prefill_endpoints),
            decode_urls: Arc::new(decode_endpoints),
            p_index: Arc::new(AtomicUsize::new(0)),
            d_index: Arc::new(AtomicUsize::new(0)),
            pd_push,
            d_pegaflow_addrs: Arc::new(d_pegaflow_addrs),
            decode_instance_ids: Arc::new(decode_instance_ids),
            p_inflight: Arc::new(p_inflight),
            d_inflight: Arc::new(d_inflight),
        }
    }

    fn get_next_p(&self) -> (Client, String, usize) {
        let idx = self.p_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.prefill_clients.len();
        self.p_inflight[idx].fetch_add(1, Ordering::Relaxed);
        (
            self.prefill_clients[idx].clone(),
            self.prefill_urls[idx].clone(),
            idx,
        )
    }

    fn get_next_d(&self) -> (Client, String, usize) {
        let idx = self.d_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.decode_clients.len();
        self.d_inflight[idx].fetch_add(1, Ordering::Relaxed);
        (
            self.decode_clients[idx].clone(),
            self.decode_urls[idx].clone(),
            idx,
        )
    }

    fn finish_p(&self, idx: usize) {
        self.p_inflight[idx].fetch_sub(1, Ordering::Relaxed);
    }

    fn finish_d(&self, idx: usize) {
        self.d_inflight[idx].fetch_sub(1, Ordering::Relaxed);
    }

    fn d_pegaflow_addr(&self, idx: usize) -> Option<&str> {
        self.d_pegaflow_addrs.get(idx).map(String::as_str)
    }

    fn decode_instance_id(&self, idx: usize) -> Option<&str> {
        self.decode_instance_ids.get(idx).map(String::as_str)
    }

    fn get_inflight_summary(&self) -> String {
        let p_counts: Vec<usize> = self
            .p_inflight
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let d_counts: Vec<usize> = self
            .d_inflight
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        format!("P={:?} D={:?}", p_counts, d_counts)
    }
}

fn inject_pd_params(
    body: &mut Value,
    role: &str,
    pd_request_id: &str,
    d_pegaflow_addr: &str,
    dst_instance_id: &str,
) {
    let mut params = body
        .get("kv_transfer_params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut pegaflow = serde_json::Map::new();
    pegaflow.insert("request_id".to_string(), json!(pd_request_id));
    match role {
        "source" => {
            pegaflow.insert("type".to_string(), json!("prefill_push"));
            pegaflow.insert("decode_endpoint".to_string(), json!(d_pegaflow_addr));
            pegaflow.insert("decode_instance_id".to_string(), json!(dst_instance_id));
        }
        "target" => {
            pegaflow.insert("type".to_string(), json!("decode_load"));
        }
        _ => {}
    }
    params.insert("pegaflow".to_string(), Value::Object(pegaflow));
    body["kv_transfer_params"] = Value::Object(params);
}

async fn run_prefill_request(
    state: RouterState,
    p_client: Client,
    p_idx: usize,
    p_url: String,
    req_id: String,
    p_body: Value,
) {
    let start = Instant::now();
    let result = async {
        let response = p_client
            .post(&p_url)
            .header("X-Request-Id", &req_id)
            .json(&p_body)
            .send()
            .await
            .map_err(|e| format!("send failed: {e}"))?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .map_err(|e| format!("response parse failed: {e}"))?;
        if !status.is_success() {
            return Err(format!("status={status} body={body:?}"));
        }
        Ok::<_, String>(())
    }
    .await;

    state.finish_p(p_idx);
    match result {
        Ok(()) => info!(
            "prefill done: req={} P[{}] latency={}ms inflight=[{}]",
            req_id,
            p_idx,
            start.elapsed().as_millis(),
            state.get_inflight_summary()
        ),
        Err(err) => error!(
            "prefill failed: req={} P[{}] latency={}ms error={} inflight=[{}]",
            req_id,
            p_idx,
            start.elapsed().as_millis(),
            err,
            state.get_inflight_summary()
        ),
    }
}

async fn handle_completion(
    State(state): State<RouterState>,
    _headers: HeaderMap,
    Json(body): Json<Value>,
    api_path: &str,
) -> Response {
    let arrive_time = Instant::now();

    // Use existing request_id or generate new one
    let req_id = body
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    info!(
        "request arrived: req={} inflight=[{}]",
        req_id,
        state.get_inflight_summary()
    );

    // Save original values to restore for D request
    let org_max_tokens = body.get("max_tokens").cloned();
    let org_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let stream_options = body.get("stream_options").cloned();

    let (p_client, p_url, p_idx) = state.get_next_p();
    let p_url = format!("{}{}", p_url, api_path);
    let (d_client, d_url, d_idx) = state.get_next_d();
    let d_url = format!("{}{}", d_url, api_path);

    let d_pegaflow_addr = state.d_pegaflow_addr(d_idx).map(str::to_string);
    let dst_instance_id = state.decode_instance_id(d_idx).map(str::to_string);

    if state.pd_push && (d_pegaflow_addr.is_none() || dst_instance_id.is_none()) {
        state.finish_p(p_idx);
        state.finish_d(d_idx);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "P/D push requires --decode-pegaflow and --decode-instance for each decode endpoint"
            })),
        )
            .into_response();
    }

    // Prepare P request (short generation, non-streaming). It runs concurrently
    // with the D request; D waits in get_num_new_matched_tokens() until IMM.
    let mut p_body = body.clone();
    p_body["max_tokens"] = json!(PREFILL_MAX_TOKENS);
    p_body["stream"] = json!(false);
    p_body["request_id"] = json!(req_id.clone());
    if state.pd_push {
        inject_pd_params(
            &mut p_body,
            "source",
            &req_id,
            d_pegaflow_addr.as_deref().unwrap_or_default(),
            dst_instance_id.as_deref().unwrap_or_default(),
        );
    }

    // Remove stream_options since stream=false
    p_body
        .as_object_mut()
        .map(|obj| obj.remove("stream_options"));

    // Ensure min_tokens <= max_tokens to avoid 400 from P node
    if let Some(min_tokens) = p_body.get("min_tokens").and_then(|v| v.as_i64()) {
        p_body["min_tokens"] = json!(min_tokens.min(PREFILL_MAX_TOKENS as i64));
    } else {
        p_body["min_tokens"] = json!(0);
    }

    tokio::spawn(run_prefill_request(
        state.clone(),
        p_client,
        p_idx,
        p_url,
        req_id.clone(),
        p_body,
    ));

    // Prepare D request (restore original settings)
    let mut d_body = body;
    if let Some(max_tokens) = org_max_tokens {
        d_body["max_tokens"] = max_tokens;
    }
    d_body["stream"] = json!(org_stream);
    d_body["request_id"] = json!(req_id.clone());
    if let Some(stream_opts) = stream_options {
        d_body["stream_options"] = stream_opts;
    }
    if state.pd_push {
        inject_pd_params(
            &mut d_body,
            "target",
            &req_id,
            d_pegaflow_addr.as_deref().unwrap_or_default(),
            dst_instance_id.as_deref().unwrap_or_default(),
        );
    }

    if org_stream {
        // Streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let stream = d_response.bytes_stream();
        let state_clone = state.clone();
        let stream = tokio_stream::wrappers::ReceiverStream::new({
            let (tx, rx) = tokio::sync::mpsc::channel(100);
            let req_id = req_id.clone();
            tokio::spawn(async move {
                let mut stream = Box::pin(stream);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok::<_, std::io::Error>(bytes)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Stream error: req={} error={}", req_id, e);
                            break;
                        }
                    }
                }
                // D node finished (stream ended)
                state_clone.finish_d(d_idx);
                let total = arrive_time.elapsed().as_millis();
                info!(
                    "done (stream): req={} D[{}] total={}ms inflight=[{}]",
                    req_id,
                    d_idx,
                    total,
                    state_clone.get_inflight_summary()
                );
            });
            rx
        });

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        // Non-streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let d_status = d_response.status();
        let d_result: Value = match d_response.json().await {
            Ok(v) => v,
            Err(e) => {
                state.finish_d(d_idx);
                error!("D response parse failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode response error: {}", e)})),
                )
                    .into_response();
            }
        };

        // D node finished
        state.finish_d(d_idx);
        let total = arrive_time.elapsed().as_millis();
        info!(
            "done: req={} D[{}] total={}ms inflight=[{}]",
            req_id,
            d_idx,
            total,
            state.get_inflight_summary()
        );

        (d_status, Json(d_result)).into_response()
    }
}

async fn chat_completions(
    state: State<RouterState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> Response {
    handle_completion(state, headers, body, "/v1/chat/completions").await
}

async fn completions(state: State<RouterState>, headers: HeaderMap, body: Json<Value>) -> Response {
    handle_completion(state, headers, body, "/v1/completions").await
}

#[derive(Parser)]
#[command(name = "pegaflow-router")]
#[command(version)]
#[command(about = "PegaFlow P/D Disaggregation Router")]
struct Args {
    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Prefill endpoints
    #[arg(long, required = true, num_args = 1..)]
    prefill: Vec<String>,

    /// Decode endpoints
    #[arg(long, required = true, num_args = 1..)]
    decode: Vec<String>,

    /// Enable PegaFlow CPU-staging P/D push metadata injection.
    #[arg(long, default_value_t = false)]
    pd_push: bool,

    /// D-side PegaFlow gRPC endpoints, aligned with --decode.
    #[arg(long, num_args = 1..)]
    decode_pegaflow: Vec<String>,

    /// D-side PegaFlow/vLLM instance IDs, aligned with --decode.
    #[arg(long, num_args = 1..)]
    decode_instance: Vec<String>,
}

#[tokio::main]
async fn main() {
    pegaflow_common::logging::init_stderr("info");
    info!("Starting pegaflow-router v{}", env!("CARGO_PKG_VERSION"));

    let args = Args::parse();

    if args.pd_push {
        assert!(
            args.decode_pegaflow.len() == args.decode.len(),
            "--decode-pegaflow count ({}) must match --decode count ({})",
            args.decode_pegaflow.len(),
            args.decode.len()
        );
        assert!(
            args.decode_instance.len() == args.decode.len(),
            "--decode-instance count ({}) must match --decode count ({})",
            args.decode_instance.len(),
            args.decode.len()
        );
    }

    let state = RouterState::new(
        args.prefill.clone(),
        args.decode.clone(),
        args.decode_pegaflow.clone(),
        args.decode_instance.clone(),
        args.pd_push,
    );

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting on {}", addr);
    info!("Prefill nodes: {:?}", args.prefill);
    info!("Decode nodes: {:?}", args.decode);
    info!(
        "P/D push: enabled={} decode_pegaflow={:?} decode_instance={:?}",
        args.pd_push, args.decode_pegaflow, args.decode_instance
    );

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
