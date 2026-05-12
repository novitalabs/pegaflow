use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::{KeyValue, global};
use pegaflow_common::hll::MultiWindowHllTracker;
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Instant;
use tonic::Status;

struct HllGaugeHandles {
    _cardinality: ObservableGauge<f64>,
    _total_requests: ObservableGauge<u64>,
}

static HLL_GAUGES: OnceLock<HllGaugeHandles> = OnceLock::new();

/// Register HLL observable gauges backed by the multi-window tracker.
///
/// Emits one time series per configured window, labeled with `window=<label>`.
/// Hit rate is intentionally not exported — derive it in Prometheus via
/// `1 - pegaflow_hll_cardinality / pegaflow_hll_total_requests`.
pub fn register_hll_gauges(tracker: &Arc<Mutex<MultiWindowHllTracker>>) {
    let t_card = Arc::clone(tracker);
    let t_total = Arc::clone(tracker);

    HLL_GAUGES.get_or_init(|| {
        let meter = global::meter("pegaflow-core");

        let cardinality = meter
            .f64_observable_gauge("pegaflow_hll_cardinality")
            .with_description("Estimated distinct blocks seen in the sliding window")
            .with_callback(move |observer| {
                if let Ok(mut t) = t_card.lock() {
                    for (label, m) in t.metrics() {
                        observer.observe(m.cardinality, &[KeyValue::new("window", label)]);
                    }
                }
            })
            .build();

        let total_requests = meter
            .u64_observable_gauge("pegaflow_hll_total_requests")
            .with_description("Total block requests in the sliding window")
            .with_callback(move |observer| {
                if let Ok(mut t) = t_total.lock() {
                    for (label, m) in t.metrics() {
                        observer.observe(m.total_requests, &[KeyValue::new("window", label)]);
                    }
                }
            })
            .build();

        HllGaugeHandles {
            _cardinality: cardinality,
            _total_requests: total_requests,
        }
    });
}

struct RpcMetrics {
    request_count: Counter<u64>,
    request_duration: Histogram<f64>,
}

impl RpcMetrics {
    fn new() -> Self {
        let meter = global::meter("pegaflow_server_rpc");
        let request_count = meter
            .u64_counter("pegaflow_rpc_requests")
            .with_description("Total RPC requests handled by pegaflow server")
            .build();
        let request_duration = meter
            .f64_histogram("pegaflow_rpc_duration")
            .with_description("RPC latency in seconds")
            .with_unit("s")
            // Buckets tuned for single-node/IPC workloads, covering sub-ms to ~seconds tail
            .with_boundaries(
                [
                    0.0005, // 0.5ms
                    0.001,  // 1ms
                    0.002,  // 2ms
                    0.005,  // 5ms
                    0.01,   // 10ms
                    0.02,   // 20ms
                    0.05,   // 50ms
                    0.1,    // 100ms
                    0.2,    // 200ms
                    0.5,    // 500ms
                    1.0,    // 1s
                    2.0,    // 2s
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
