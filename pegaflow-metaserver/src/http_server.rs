use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::{
    Json, Router,
    routing::{get, post},
};
use log::{info, warn};
use prometheus::{Registry, TextEncoder};
use serde::Serialize;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Notify;

use crate::store::{BlockHashStore, MANUAL_CLEANUP_AGE_SECS};

#[derive(Clone)]
struct AppState {
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
}

#[derive(Debug, Serialize)]
struct CleanupResponse {
    removed_owners: usize,
    removed_keys: usize,
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

async fn cleanup_expired_blocks_handler(
    State(state): State<AppState>,
) -> Result<Json<CleanupResponse>, StatusCode> {
    let store = Arc::clone(&state.store);
    let stats = match tokio::task::spawn_blocking(move || {
        store.remove_owners_older_than(std::time::Duration::from_secs(MANUAL_CLEANUP_AGE_SECS))
    })
    .await
    {
        Ok(stats) => stats,
        Err(err) => {
            warn!("manual cleanup worker failed: {err}");
            return Err(StatusCode::INTERNAL_SERVER_ERROR);
        }
    };
    Ok(Json(CleanupResponse {
        removed_owners: stats.removed_owners,
        removed_keys: stats.removed_keys,
    }))
}

fn public_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

fn admin_app(state: AppState) -> Router {
    Router::new()
        .route(
            "/admin/cleanup-expired-blocks",
            post(cleanup_expired_blocks_handler),
        )
        .with_state(state)
}

pub async fn start_http_server(
    addr: std::net::SocketAddr,
    admin_addr: std::net::SocketAddr,
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    if !admin_addr.ip().is_loopback() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("admin HTTP address must be loopback, got {admin_addr}"),
        ));
    }
    let listener = TcpListener::bind(addr).await?;
    let admin_listener = TcpListener::bind(admin_addr).await?;

    let state = AppState {
        prometheus_registry: prometheus_registry.clone(),
        store: Arc::clone(&store),
    };
    let admin_state = AppState {
        prometheus_registry,
        store,
    };

    info!(
        "Starting HTTP server on {} (/health, /metrics); admin cleanup on {} (localhost only)",
        addr, admin_addr
    );

    let handle = tokio::spawn(async move {
        let public_shutdown = Arc::clone(&shutdown);
        let public = axum::serve(listener, public_app(state)).with_graceful_shutdown(async move {
            public_shutdown.notified().await;
        });
        let admin = axum::serve(admin_listener, admin_app(admin_state)).with_graceful_shutdown(
            async move {
                shutdown.notified().await;
            },
        );
        let (public_result, admin_result) = tokio::join!(public, admin);
        if let Err(err) = public_result {
            warn!("HTTP server stopped with error: {err}");
        }
        if let Err(err) = admin_result {
            warn!("Admin HTTP server stopped with error: {err}");
        }
    });

    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn cleanup_route_removes_old_owners_and_preserves_fresh_replicas() {
        let store = Arc::new(crate::store::tests::manual_cleanup_fixture());
        let app = admin_app(AppState {
            prometheus_registry: Registry::new(),
            store: Arc::clone(&store),
        });
        for expected in [
            br#"{"removed_owners":2,"removed_keys":1}"#.as_slice(),
            br#"{"removed_owners":0,"removed_keys":0}"#.as_slice(),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/admin/cleanup-expired-blocks")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap()
                    .as_ref(),
                expected,
            );
        }
        assert!(store.query_prefix("ns", &[vec![1]]).is_empty());
        assert_eq!(
            store.query_prefix("ns", &[vec![2]])[0].nodes[0].as_ref(),
            "b"
        );
        assert_eq!(
            store.query_prefix("ns", &[vec![3]])[0].nodes[0].as_ref(),
            "a"
        );
        assert_eq!(store.owner_count(), 2);
        assert_eq!(store.entry_count(), 2);
        assert_eq!(store.node_counts(), (2, 0));
        assert_eq!(store.redundancy_snapshot().keys_1, 2);
    }

    #[tokio::test]
    async fn public_route_does_not_expose_admin_cleanup() {
        let response = public_app(AppState {
            prometheus_registry: Registry::new(),
            store: Arc::new(BlockHashStore::new()),
        })
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/admin/cleanup-expired-blocks")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
