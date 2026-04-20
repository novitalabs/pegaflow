use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::{KeyValue, global};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Instant;
use tonic::Status;

use crate::store::BlockHashStore;

// ---------------------------------------------------------------------------
// Store gauges
// ---------------------------------------------------------------------------

struct StoreGaugeHandles {
    _entries: ObservableGauge<u64>,
    _active_nodes: ObservableGauge<u64>,
}

static STORE_GAUGES: OnceLock<StoreGaugeHandles> = OnceLock::new();

/// Register observable gauges backed by the given store.
pub fn register_store_gauges(store: &Arc<BlockHashStore>) {
    let s1 = Arc::clone(store);
    let s2 = Arc::clone(store);
    STORE_GAUGES.get_or_init(|| {
        let meter = global::meter("pegaflow_metaserver");
        let entries = meter
            .u64_observable_gauge("pegaflow_metaserver_store_entries")
            .with_description("Number of unique block keys in the store")
            .with_callback(move |observer| {
                observer.observe(s1.entry_count(), &[]);
            })
            .build();
        let active_nodes = meter
            .u64_observable_gauge("pegaflow_metaserver_active_nodes")
            .with_description("Number of currently tracked nodes (healthy + suspect)")
            .with_callback(move |observer| {
                observer.observe(s2.node_count() as u64, &[]);
            })
            .build();
        StoreGaugeHandles {
            _entries: entries,
            _active_nodes: active_nodes,
        }
    });
}

// ---------------------------------------------------------------------------
// TTL sweep counter
// ---------------------------------------------------------------------------

struct SweepMetrics {
    removed: Counter<u64>,
}

static SWEEP_METRICS: LazyLock<SweepMetrics> = LazyLock::new(|| {
    let meter = global::meter("pegaflow_metaserver");
    SweepMetrics {
        removed: meter
            .u64_counter("pegaflow_metaserver_ttl_sweep_removed")
            .with_description("Total block keys removed by TTL sweep")
            .build(),
    }
});

pub fn record_ttl_sweep(removed: u64) {
    SWEEP_METRICS.removed.add(removed, &[]);
}

// ---------------------------------------------------------------------------
// Purge sweep counter (block keys removed due to dead-node hard-delete)
// ---------------------------------------------------------------------------

static PURGE_SWEEP_REMOVED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    let meter = global::meter("pegaflow_metaserver");
    meter
        .u64_counter("pegaflow_metaserver_purge_sweep_removed")
        .with_description("Total block keys removed by dead-node purge sweep")
        .build()
});

pub fn record_purge_sweep(removed: u64) {
    if removed > 0 {
        PURGE_SWEEP_REMOVED.add(removed, &[]);
    }
}

// ---------------------------------------------------------------------------
// Node purge counter
// ---------------------------------------------------------------------------

static NODE_PURGED: LazyLock<Counter<u64>> = LazyLock::new(|| {
    let meter = global::meter("pegaflow_metaserver");
    meter
        .u64_counter("pegaflow_metaserver_node_purged_total")
        .with_description("Total nodes purged (timeout or bye)")
        .build()
});

pub fn record_node_purge(count: u64) {
    if count > 0 {
        NODE_PURGED.add(count, &[]);
    }
}

// ---------------------------------------------------------------------------
// RPC metrics
// ---------------------------------------------------------------------------

struct RpcMetrics {
    request_count: Counter<u64>,
    request_duration: Histogram<f64>,
}

impl RpcMetrics {
    fn new() -> Self {
        let meter = global::meter("pegaflow_metaserver_rpc");
        let request_count = meter
            .u64_counter("pegaflow_metaserver_rpc_requests")
            .with_description("Total RPC requests handled by pegaflow metaserver")
            .build();
        let request_duration = meter
            .f64_histogram("pegaflow_metaserver_rpc_duration")
            .with_description("RPC latency in seconds")
            .with_unit("s")
            .with_boundaries(
                [
                    0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
                ]
                .into(),
            )
            .build();
        Self {
            request_count,
            request_duration,
        }
    }

    fn record(&self, method: &'static str, status: &str, duration: f64) {
        let labels = [
            KeyValue::new("method", method.to_string()),
            KeyValue::new("status", status.to_string()),
        ];
        self.request_count.add(1, &labels);
        self.request_duration.record(duration, &labels);
    }
}

static RPC_METRICS: LazyLock<RpcMetrics> = LazyLock::new(RpcMetrics::new);

pub fn record_rpc_result<T>(method: &'static str, result: &Result<T, Status>, start: Instant) {
    let status = match result {
        Ok(_) => "ok".to_string(),
        Err(status) => status.code().to_string(),
    };
    let duration = start.elapsed().as_secs_f64();
    RPC_METRICS.record(method, &status, duration);
}
