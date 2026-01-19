use opentelemetry::metrics::{Counter, Histogram};
use opentelemetry::{global, KeyValue};
use std::sync::LazyLock;
use std::time::Instant;
use tonic::Status;

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
