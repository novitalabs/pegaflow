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

async fn run_cleanup(
    store: Arc<BlockHashStore>,
    cleanup: impl FnOnce(&BlockHashStore) -> crate::store::SweepStats + Send + 'static,
) -> Result<Json<CleanupResponse>, StatusCode> {
    match tokio::task::spawn_blocking(move || cleanup(&store)).await {
        Ok(stats) => Ok(Json(CleanupResponse {
            removed_owners: stats.removed_owners,
            removed_keys: stats.removed_keys,
        })),
        Err(err) => {
            warn!("manual cleanup worker failed: {err}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
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
    run_cleanup(Arc::clone(&state.store), |store| {
        store.remove_owners_older_than(std::time::Duration::from_secs(MANUAL_CLEANUP_AGE_SECS))
    })
    .await
}

/// POST /admin/clear-all-blocks
///
/// Drops every block ownership record regardless of age. Node sessions and the
/// KV data held by the nodes themselves are untouched, so cleared blocks stay
/// resident locally but become invisible to cross-node prefix queries.
async fn clear_all_blocks_handler(
    State(state): State<AppState>,
) -> Result<Json<CleanupResponse>, StatusCode> {
    let response = run_cleanup(Arc::clone(&state.store), |store| store.clear_blocks()).await?;
    warn!(
        "Manual clear-all-blocks: removed keys={} owners={}",
        response.removed_keys, response.removed_owners
    );
    Ok(response)
}

fn public_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route(
            "/admin/cleanup-expired-blocks",
            post(cleanup_expired_blocks_handler),
        )
        .route("/admin/clear-all-blocks", post(clear_all_blocks_handler))
        .with_state(state)
}

pub async fn start_http_server(
    addr: std::net::SocketAddr,
    prometheus_registry: Registry,
    store: Arc<BlockHashStore>,
    shutdown: Arc<Notify>,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let listener = TcpListener::bind(addr).await?;

    let state = AppState {
        prometheus_registry,
        store,
    };

    info!(
        "Starting HTTP server on {} (/health, /metrics, /admin/cleanup-expired-blocks, /admin/clear-all-blocks)",
        addr
    );

    let handle = tokio::spawn(async move {
        let result = axum::serve(listener, public_app(state))
            .with_graceful_shutdown(async move {
                shutdown.notified().await;
            })
            .await;
        if let Err(err) = result {
            warn!("HTTP server stopped with error: {err}");
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
        let app = public_app(AppState {
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
    async fn clear_all_blocks_route_removes_fresh_owners_too() {
        let store = Arc::new(crate::store::tests::manual_cleanup_fixture());
        let app = public_app(AppState {
            prometheus_registry: Registry::new(),
            store: Arc::clone(&store),
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/admin/clear-all-blocks")
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
            br#"{"removed_owners":4,"removed_keys":3}"#.as_slice(),
        );
        assert_eq!(store.entry_count(), 0);
        assert_eq!(store.owner_count(), 0);
    }

    #[tokio::test]
    async fn cleanup_route_is_available_on_public_listener() {
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

        assert_eq!(response.status(), StatusCode::OK);
    }
}
