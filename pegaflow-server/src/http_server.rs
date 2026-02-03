use axum::{Router, routing::get};
use log::info;
use prometheus::{Registry, TextEncoder};
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

/// Handler for health check endpoint
async fn health_handler() -> &'static str {
    "ok"
}

/// Handler for Prometheus /metrics endpoint
async fn metrics_handler(axum::extract::State(registry): axum::extract::State<Registry>) -> String {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    encoder
        .encode_to_string(&metric_families)
        .unwrap_or_else(|e| format!("# Error encoding metrics: {e}"))
}

/// Start HTTP server for health check and optional Prometheus metrics
pub async fn start_http_server(
    addr: std::net::SocketAddr,
    enable_prometheus: bool,
    prometheus_registry: Option<Registry>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let handle = if enable_prometheus {
        let registry = prometheus_registry
            .expect("prometheus_registry must be Some when enable_prometheus is true");
        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/metrics", get(metrics_handler))
            .with_state(registry);
        info!("Starting HTTP server on {} (/health and /metrics)", addr);

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown.notified().await;
                })
                .await
                .ok();
        })
    } else {
        let app = Router::new().route("/health", get(health_handler));
        info!("Starting HTTP server on {} (/health only)", addr);

        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    shutdown.notified().await;
                })
                .await
                .ok();
        })
    };

    Ok(handle)
}
