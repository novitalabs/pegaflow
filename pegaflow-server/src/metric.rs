use opentelemetry::metrics::{Counter, Histogram, ObservableGauge};
use opentelemetry::{KeyValue, global};
use pegaflow_common::hll::HllTracker;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tonic::Status;

static HLL_TRACKER: OnceLock<Mutex<HllTracker>> = OnceLock::new();

/// Initialize the global HLL tracker. Must be called once at startup.
pub fn init_hll_tracker(slot_secs: u64, window_secs: u64, bucket_bits: u8) {
    HLL_TRACKER.get_or_init(|| {
        Mutex::new(HllTracker::new(
            Duration::from_secs(slot_secs),
            Duration::from_secs(window_secs),
            bucket_bits,
        ))
    });
}

/// Access the global HLL tracker.
pub fn hll_tracker() -> &'static Mutex<HllTracker> {
    HLL_TRACKER
        .get()
        .expect("HLL tracker not initialized; call init_hll_tracker() first")
}

struct HllGaugeHandles {
    _hit_rate: ObservableGauge<f64>,
    _cardinality: ObservableGauge<f64>,
    _total_requests: ObservableGauge<u64>,
}

static HLL_GAUGES: OnceLock<HllGaugeHandles> = OnceLock::new();

/// Register HLL observable gauges. Must be called after `init_hll_tracker`.
pub fn register_hll_gauges() {
    HLL_GAUGES.get_or_init(|| {
        let meter = global::meter("pegaflow-core");

        let hit_rate = meter
            .f64_observable_gauge("pegaflow_hll_estimated_hit_rate")
            .with_description("Estimated cache hit rate assuming infinite cache [0.0, 1.0]")
            .with_callback(|observer| {
                if let Ok(mut t) = hll_tracker().lock() {
                    observer.observe(t.metric().estimated_hit_rate, &[]);
                }
            })
            .build();

        let cardinality = meter
            .f64_observable_gauge("pegaflow_hll_cardinality")
            .with_description("Estimated distinct blocks seen in the sliding window")
            .with_callback(|observer| {
                if let Ok(mut t) = hll_tracker().lock() {
                    observer.observe(t.metric().cardinality, &[]);
                }
            })
            .build();

        let total_requests = meter
            .u64_observable_gauge("pegaflow_hll_total_requests")
            .with_description("Total block requests in the sliding window")
            .with_callback(|observer| {
                if let Ok(mut t) = hll_tracker().lock() {
                    observer.observe(t.metric().total_requests, &[]);
                }
            })
            .build();

        HllGaugeHandles {
            _hit_rate: hit_rate,
            _cardinality: cardinality,
            _total_requests: total_requests,
        }
    });
}

/// Record block hashes into the HLL tracker for hit rate estimation.
pub fn record_hll_hashes(block_hashes: &[Vec<u8>]) {
    if let Ok(mut tracker) = hll_tracker().lock() {
        for hash in block_hashes {
            if let Ok(h) = <&[u8; 32]>::try_from(hash.as_slice()) {
                tracker.record(h);
            }
        }
    }
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
