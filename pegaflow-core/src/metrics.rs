use opentelemetry::{
    global,
    metrics::{Counter, Histogram, Meter, UpDownCounter},
};
use std::sync::OnceLock;

pub(crate) struct CoreMetrics {
    pub pool_used_bytes: UpDownCounter<i64>,
    pub pool_capacity_bytes: UpDownCounter<i64>,
    pub pool_alloc_failures: Counter<u64>,

    pub inflight_blocks: UpDownCounter<i64>,
    pub inflight_gc_cleaned: Counter<u64>,

    pub cache_block_hits: Counter<u64>,
    pub cache_block_misses: Counter<u64>,
    pub cache_block_insertions: Counter<u64>,
    pub cache_block_admission_rejections: Counter<u64>,
    pub cache_block_evictions: Counter<u64>,

    pub save_bytes: Counter<u64>,
    pub save_duration_seconds: Histogram<f64>,

    pub load_bytes: Counter<u64>,
    pub load_duration_seconds: Histogram<f64>,
    pub load_failures: Counter<u64>,

    // SSD cache metrics
    pub ssd_write_bytes: Counter<u64>,
    pub ssd_write_duration_seconds: Histogram<f64>,
    pub ssd_write_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_write_queue_pending: UpDownCounter<i64>,
    pub ssd_write_queue_full: Counter<u64>,

    pub ssd_prefetch_success: Counter<u64>,
    pub ssd_prefetch_failures: Counter<u64>,
    pub ssd_prefetch_duration_seconds: Histogram<f64>,
    pub ssd_prefetch_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_prefetch_inflight: UpDownCounter<i64>,
    pub ssd_prefetch_queue_full: Counter<u64>,
}

fn init_meter() -> Meter {
    global::meter("pegaflow-core")
}

/// Custom histogram boundaries for SSD throughput in bytes/s (1 to 40 GB/s, step 1 GB/s)
fn ssd_throughput_boundaries() -> Vec<f64> {
    // 1.0e9, 2.0e9, 3.0e9, ..., 40.0e9 (40 buckets in bytes/s)
    (1..=40).map(|i| i as f64 * 1.0e9).collect()
}

/// Custom histogram boundaries for duration in seconds (1ms to 5s)
/// Optimized for local IPC workloads where most operations are 10-100ms
fn duration_seconds_boundaries() -> Vec<f64> {
    vec![
        0.001, // 1ms
        0.005, // 5ms
        0.01,  // 10ms
        0.025, // 25ms
        0.05,  // 50ms
        0.1,   // 100ms
        0.2,   // 200ms
        0.3,   // 300ms
        0.4,   // 400ms
        0.5,   // 500ms
        1.0,   // 1s
        2.0,   // 2s
        5.0,   // 5s
    ]
}

pub(crate) fn core_metrics() -> &'static CoreMetrics {
    static METRICS: OnceLock<CoreMetrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let meter = init_meter();

        CoreMetrics {
            pool_used_bytes: meter
                .i64_up_down_counter("pegaflow_pool_used_bytes")
                .with_unit("bytes")
                .with_description("Current pinned pool usage in bytes")
                .build(),
            pool_capacity_bytes: meter
                .i64_up_down_counter("pegaflow_pool_capacity_bytes")
                .with_unit("bytes")
                .with_description("Total pinned pool capacity in bytes")
                .build(),
            pool_alloc_failures: meter
                .u64_counter("pegaflow_pool_alloc_failures")
                .with_description("Pinned pool allocation failures after eviction retries")
                .build(),

            inflight_blocks: meter
                .i64_up_down_counter("pegaflow_inflight_blocks")
                .with_description("Current inflight (partial) blocks awaiting all slots")
                .build(),
            inflight_gc_cleaned: meter
                .u64_counter("pegaflow_inflight_gc_cleaned")
                .with_description("Stale inflight blocks cleaned by background GC")
                .build(),

            cache_block_hits: meter
                .u64_counter("pegaflow_cache_block_hits")
                .with_description("Complete blocks found in cache (cache hit)")
                .build(),
            cache_block_misses: meter
                .u64_counter("pegaflow_cache_block_misses")
                .with_description("Complete blocks not found in cache (cache miss)")
                .build(),
            cache_block_insertions: meter
                .u64_counter("pegaflow_cache_block_insertions")
                .with_description("New blocks inserted into cache")
                .build(),
            cache_block_admission_rejections: meter
                .u64_counter("pegaflow_cache_block_admission_rejections")
                .with_description("Blocks rejected by cache admission policy")
                .build(),
            cache_block_evictions: meter
                .u64_counter("pegaflow_cache_block_evictions")
                .with_description("Blocks evicted from cache due to memory pressure")
                .build(),

            save_bytes: meter
                .u64_counter("pegaflow_save_bytes")
                .with_unit("bytes")
                .with_description("Total bytes saved from GPU to CPU storage")
                .build(),
            save_duration_seconds: meter
                .f64_histogram("pegaflow_save_duration")
                .with_unit("s")
                .with_description("Save operation latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),

            load_bytes: meter
                .u64_counter("pegaflow_load_bytes")
                .with_unit("bytes")
                .with_description("Total bytes loaded from CPU storage to GPU")
                .build(),
            load_duration_seconds: meter
                .f64_histogram("pegaflow_load_duration")
                .with_unit("s")
                .with_description("Load operation latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            load_failures: meter
                .u64_counter("pegaflow_load_failures")
                .with_description("Load operation failures (e.g., transfer errors)")
                .build(),

            ssd_write_bytes: meter
                .u64_counter("pegaflow_ssd_write_bytes")
                .with_unit("bytes")
                .with_description("Bytes written to SSD cache")
                .build(),
            ssd_write_duration_seconds: meter
                .f64_histogram("pegaflow_ssd_write_duration")
                .with_unit("s")
                .with_description("SSD block write latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            ssd_write_throughput_bytes_per_second: meter
                .f64_histogram("pegaflow_ssd_write_throughput")
                .with_unit("bytes/s")
                .with_description("SSD write throughput per batch in bytes/s")
                .with_boundaries(ssd_throughput_boundaries())
                .build(),
            ssd_write_queue_pending: meter
                .i64_up_down_counter("pegaflow_ssd_write_queue_pending")
                .with_description("Current pending blocks in SSD write queue")
                .build(),
            ssd_write_queue_full: meter
                .u64_counter("pegaflow_ssd_write_queue_full")
                .with_description("Write requests dropped due to full queue")
                .build(),

            ssd_prefetch_success: meter
                .u64_counter("pegaflow_ssd_prefetch_success")
                .with_description("Blocks successfully prefetched from SSD cache")
                .build(),
            ssd_prefetch_failures: meter
                .u64_counter("pegaflow_ssd_prefetch_failures")
                .with_description("SSD prefetch failures (short read, rebuild error, stale)")
                .build(),
            ssd_prefetch_duration_seconds: meter
                .f64_histogram("pegaflow_ssd_prefetch_duration")
                .with_unit("s")
                .with_description("SSD block prefetch latency in seconds")
                .with_boundaries(duration_seconds_boundaries())
                .build(),
            ssd_prefetch_throughput_bytes_per_second: meter
                .f64_histogram("pegaflow_ssd_prefetch_throughput")
                .with_unit("bytes/s")
                .with_description("SSD prefetch throughput per batch in bytes/s")
                .with_boundaries(ssd_throughput_boundaries())
                .build(),
            ssd_prefetch_inflight: meter
                .i64_up_down_counter("pegaflow_ssd_prefetch_inflight")
                .with_description("Current in-flight SSD prefetch operations")
                .build(),
            ssd_prefetch_queue_full: meter
                .u64_counter("pegaflow_ssd_prefetch_queue_full")
                .with_description("Prefetch requests dropped due to full queue")
                .build(),
        }
    })
}
