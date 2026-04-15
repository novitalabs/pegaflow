use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{Router, routing::get};
use log::info;
use prometheus::{Registry, TextEncoder};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

#[derive(Clone)]
struct AppState {
    prometheus_registry: Registry,
}

async fn health_handler() -> &'static str {
    "ok"
}

async fn metrics_handler(State(state): State<AppState>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = state.prometheus_registry.gather();
    (
        StatusCode::OK,
        encoder
            .encode_to_string(&metric_families)
            .unwrap_or_else(|e| format!("# Error encoding metrics: {e}")),
    )
}

pub async fn start_http_server(
    addr: std::net::SocketAddr,
    prometheus_registry: Registry,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        prometheus_registry,
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    info!("Starting HTTP server on {} (/health, /metrics)", addr);

    let handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await
            .ok();
    });

    Ok(handle)
}
