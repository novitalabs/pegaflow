use opentelemetry::{
    KeyValue, global,
    metrics::{Counter, Histogram, Meter, ObservableGauge, UpDownCounter},
};
use std::sync::{Arc, LazyLock, OnceLock};

use crate::backing::RdmaTransport;

// ---------------------------------------------------------------------------
// Tier-attribution label sets for `cache_tier_block_requests`.
//
// Stored as `LazyLock<[KeyValue; 1]>` so the hot path passes a `&[KeyValue]`
// slice without rebuilding the attribute on every counter add.
// ---------------------------------------------------------------------------

static TIER_RAM: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "ram")]);
static TIER_RDMA: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "rdma")]);
static TIER_SSD: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "ssd")]);
static TIER_MISS: LazyLock<[KeyValue; 1]> = LazyLock::new(|| [KeyValue::new("tier", "miss")]);

pub(crate) struct CoreMetrics {
    // Pinned pool (allocator-level)
    pub pool_capacity_bytes: UpDownCounter<i64>,
    pub pool_used_bytes: UpDownCounter<i64>,
    pub pool_alloc_failures: Counter<u64>,

    // Inflight (write path safety/health)
    pub inflight_bytes: UpDownCounter<i64>,
    pub inflight_gc_cleaned: Counter<u64>,

    // Cache (sealed blocks in memory)
    pub cache_resident_bytes: UpDownCounter<i64>,
    pub cache_block_hits: Counter<u64>,
    pub cache_block_misses: Counter<u64>,
    /// Per-decision block attribution for `query_prefetch`. Labelled by `tier`
    /// (`ram` | `rdma` | `ssd` | `miss`). Each `query_prefetch` decision adds
    /// at most four times (one per non-zero tier) and the sum across tiers
    /// equals the request's `block_hashes.len()`.
    pub cache_tier_block_requests: Counter<u64>,
    pub cache_block_insertions: Counter<u64>,
    pub cache_block_admission_rejections: Counter<u64>,
    pub cache_block_evictions: Counter<u64>,
    pub cache_block_evictions_still_referenced: Counter<u64>,
    pub cache_eviction_reclaimed_bytes: Counter<u64>,

    // GPU <-> CPU transfer
    pub save_bytes: Counter<u64>,
    pub save_duration_seconds: Histogram<f64>,

    pub load_bytes: Counter<u64>,
    pub load_duration_seconds: Histogram<f64>,
    pub load_failures: Counter<u64>,

    // SSD cache
    pub ssd_write_bytes: Counter<u64>,
    pub ssd_write_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_write_queue_pending: UpDownCounter<i64>,
    pub ssd_write_queue_full: Counter<u64>,
    pub ssd_write_inflight: UpDownCounter<i64>,

    pub ssd_prefetch_bytes: Counter<u64>,
    pub ssd_prefetch_success: Counter<u64>,
    pub ssd_prefetch_failures: Counter<u64>,
    pub ssd_prefetch_throughput_bytes_per_second: Histogram<f64>,
    pub ssd_prefetch_inflight: UpDownCounter<i64>,
    pub ssd_prefetch_queue_full: Counter<u64>,
    pub ssd_prefetch_backpressure_blocks: Counter<u64>,

    // MetaServer registration
    pub metaserver_registration_blocks: Counter<u64>,
    pub metaserver_registration_failures: Counter<u64>,
    pub metaserver_registration_queue_full: Counter<u64>,

    // MetaServer removal
    pub metaserver_removal_blocks: Counter<u64>,
    pub metaserver_removal_failures: Counter<u64>,
    pub metaserver_removal_queue_full: Counter<u64>,

    // Cross-node transfer lock (serving side)
    pub transfer_lock_active: UpDownCounter<i64>,
    pub transfer_lock_timeouts_total: Counter<u64>,

    // RDMA remote fetch (client side)
    pub rdma_fetch_total: Counter<u64>,
    pub rdma_fetch_duration_seconds: Histogram<f64>,
    pub rdma_fetch_bytes: Counter<u64>,
}

fn init_meter() -> Meter {
    global::meter("pegaflow-core")
}

/// Custom histogram boundaries for SSD throughput in bytes/s (1 to 40 GB/s, step 1 GB/s)
fn ssd_throughput_boundaries() -> Vec<f64> {
    // 1.0e9, 2.0e9, 3.0e9, ..., 40.0e9 (40 buckets in bytes/s)
    (1..=40).map(|i| i as f64 * 1.0e9).collect()
}

/// Histogram boundaries for RDMA remote fetch (gRPC + handshake + RDMA READ).
fn rdma_fetch_duration_boundaries() -> Vec<f64> {
    vec![
        0.01, // 10ms
        0.02, // 20ms
        0.05, // 50ms
        0.1,  // 100ms
        0.2,  // 200ms
        0.5,  // 500ms
        1.0,  // 1s
        2.0,  // 2s
    ]
}

/// Tail-focused transfer duration boundaries in seconds.
///
/// Load/save debugging cares more about sustained tail regressions than small
/// sub-10ms jitter, so buckets stay coarse at the low end and distinguish
/// transfer stalls up to one minute.
fn duration_seconds_boundaries() -> Vec<f64> {
    vec![
        0.01,  // 10ms
        0.025, // 25ms
        0.05,  // 50ms
        0.1,   // 100ms
        0.25,  // 250ms
        0.5,   // 500ms
        1.0,   // 1s
        1.5,   // 1.5s
        2.0,   // 2s
        3.0,   // 3s
        5.0,   // 5s
        7.5,   // 7.5s
        10.0,  // 10s
        15.0,  // 15s
        30.0,  // 30s
        60.0,  // 60s
    ]
}

struct RdmaGaugeHandles {
    _qps: ObservableGauge<u64>,
}

static RDMA_GAUGES: OnceLock<RdmaGaugeHandles> = OnceLock::new();

/// Register RDMA observable gauges backed by the given transport.
/// Must be called after [`RdmaTransport`] is created; safe to call multiple times (no-op after first).
pub(crate) fn register_rdma_gauges(transport: &Arc<RdmaTransport>) {
    let t = Arc::clone(transport);
    RDMA_GAUGES.get_or_init(|| {
        let meter = init_meter();
        let qps = meter
            .u64_observable_gauge("pegaflow_rdma_qps")
            .with_description("Active RC queue pairs across all RDMA NICs")
            .with_callback(move |observer| {
                observer.observe(t.engine().num_qps() as u64, &[]);
            })
            .build();
        RdmaGaugeHandles { _qps: qps }
    });
}

pub(crate) fn record_cache_tier_block_requests(ram: usize, rdma: usize, ssd: usize, miss: usize) {
    let metrics = core_metrics();
    if ram > 0 {
        metrics
            .cache_tier_block_requests
            .add(ram as u64, &*TIER_RAM);
    }
    if rdma > 0 {
        metrics
            .cache_tier_block_requests
            .add(rdma as u64, &*TIER_RDMA);
    }
    if ssd > 0 {
        metrics
            .cache_tier_block_requests
            .add(ssd as u64, &*TIER_SSD);
    }
    if miss > 0 {
        metrics
            .cache_tier_block_requests
            .add(miss as u64, &*TIER_MISS);
    }
}

pub(crate) fn core_metrics() -> &'static CoreMetrics {
    static METRICS: OnceLock<CoreMetrics> = OnceLock::new();
    METRICS.get_or_init(|| {
        let meter = init_meter();

        CoreMetrics {
            // Pool
            pool_capacity_bytes: meter
                .i64_up_down_counter("pegaflow_pool_capacity_bytes")
                .with_unit("bytes")
                .with_description("Total pinned pool capacity in bytes")
                .build(),
            pool_used_bytes: meter
                .i64_up_down_counter("pegaflow_pool_used_bytes")
                .with_unit("bytes")
                .with_description("Current pinned pool usage in bytes")
                .build(),
            pool_alloc_failures: meter
                .u64_counter("pegaflow_pool_alloc_failures")
                .with_description("Pinned pool allocation failures after eviction retries")
                .build(),

            // Inflight
            inflight_bytes: meter
                .i64_up_down_counter("pegaflow_inflight_bytes")
                .with_unit("bytes")
                .with_description("Current bytes in inflight blocks (memory allocated but not yet sealed)")
                .build(),
            inflight_gc_cleaned: meter
                .u64_counter("pegaflow_inflight_gc_cleaned")
                .with_description("Stale inflight blocks cleaned by background GC")
                .build(),

            // Cache
            cache_resident_bytes: meter
                .i64_up_down_counter("pegaflow_cache_resident_bytes")
                .with_unit("bytes")
                .with_description("Current sealed block bytes resident in cache (sum of footprints)")
                .build(),
            cache_block_hits: meter
                .u64_counter("pegaflow_cache_block_hits")
                .with_description("Complete blocks found in cache (cache hit)")
                .build(),
            cache_block_misses: meter
                .u64_counter("pegaflow_cache_block_misses")
                .with_description("Complete blocks not found in cache (cache miss)")
                .build(),
            cache_tier_block_requests: meter
                .u64_counter("pegaflow_cache_tier_block_requests")
                .with_description(
                    "Per-decision query_prefetch block attribution by storage tier \
                     (tier=ram|rdma|ssd|miss). The sum across tiers equals the \
                     request's block count for that decision. This is decision \
                     attribution, not service attribution; backing failures must be \
                     inspected via pegaflow_rdma_fetch_total{status=\"error\"} \
                     and pegaflow_ssd_prefetch_failures_total.",
                )
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
            cache_block_evictions_still_referenced: meter
                .u64_counter("pegaflow_cache_block_evictions_still_referenced")
                .with_description("Evicted cache blocks that still had external references (eviction did not immediately reclaim memory)")
                .build(),
            cache_eviction_reclaimed_bytes: meter
                .u64_counter("pegaflow_cache_eviction_reclaimed_bytes")
                .with_unit("bytes")
                .with_description("Estimated bytes actually reclaimed in pinned allocator after cache eviction")
                .build(),

            // Transfer
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

            // SSD
            ssd_write_bytes: meter
                .u64_counter("pegaflow_ssd_write_bytes")
                .with_unit("bytes")
                .with_description("Bytes written to SSD cache")
                .build(),
            ssd_write_throughput_bytes_per_second: meter
                .f64_histogram("pegaflow_ssd_write_throughput")
                .with_unit("bytes/s")
                .with_description("SSD write throughput per block in bytes/s")
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
            ssd_write_inflight: meter
                .i64_up_down_counter("pegaflow_ssd_write_inflight")
                .with_description("Current in-flight SSD write operations")
                .build(),

            ssd_prefetch_bytes: meter
                .u64_counter("pegaflow_ssd_prefetch_bytes")
                .with_unit("bytes")
                .with_description("Bytes prefetched from SSD cache")
                .build(),
            ssd_prefetch_success: meter
                .u64_counter("pegaflow_ssd_prefetch_success")
                .with_description("Blocks successfully prefetched from SSD cache")
                .build(),
            ssd_prefetch_failures: meter
                .u64_counter("pegaflow_ssd_prefetch_failures")
                .with_description("SSD prefetch failures (short read, rebuild error, stale)")
                .build(),
            ssd_prefetch_throughput_bytes_per_second: meter
                .f64_histogram("pegaflow_ssd_prefetch_throughput")
                .with_unit("bytes/s")
                .with_description("SSD prefetch throughput per block in bytes/s")
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
            ssd_prefetch_backpressure_blocks: meter
                .u64_counter("pegaflow_ssd_prefetch_backpressure_blocks")
                .with_description("Blocks treated as missing due to max prefetch backpressure")
                .build(),

            // MetaServer registration
            metaserver_registration_blocks: meter
                .u64_counter("pegaflow_metaserver_registration_blocks")
                .with_description("Block hashes sent to MetaServer for registration")
                .build(),
            metaserver_registration_failures: meter
                .u64_counter("pegaflow_metaserver_registration_failures")
                .with_description("MetaServer registration RPC failures")
                .build(),
            metaserver_registration_queue_full: meter
                .u64_counter("pegaflow_metaserver_registration_queue_full")
                .with_description("Block hashes dropped due to full registration queue")
                .build(),

            // MetaServer removal
            metaserver_removal_blocks: meter
                .u64_counter("pegaflow_metaserver_removal_blocks")
                .with_description("Block hashes sent to MetaServer for removal")
                .build(),
            metaserver_removal_failures: meter
                .u64_counter("pegaflow_metaserver_removal_failures")
                .with_description("MetaServer removal RPC failures")
                .build(),
            metaserver_removal_queue_full: meter
                .u64_counter("pegaflow_metaserver_removal_queue_full")
                .with_description("Block hashes dropped due to full removal queue")
                .build(),

            // Transfer lock
            transfer_lock_active: meter
                .i64_up_down_counter("pegaflow_transfer_lock_active")
                .with_description("Currently locked blocks for cross-node RDMA transfer")
                .build(),
            transfer_lock_timeouts_total: meter
                .u64_counter("pegaflow_transfer_lock_timeouts_total")
                .with_description("Transfer lock sessions expired by timeout (potential issue)")
                .build(),

            // RDMA remote fetch (client side)
            rdma_fetch_total: meter
                .u64_counter("pegaflow_rdma_fetch_total")
                .with_description("RDMA remote fetch attempts (status=ok|error)")
                .build(),
            rdma_fetch_duration_seconds: meter
                .f64_histogram("pegaflow_rdma_fetch_duration")
                .with_unit("s")
                .with_description("End-to-end RDMA fetch latency (gRPC + handshake + RDMA READ)")
                .with_boundaries(rdma_fetch_duration_boundaries())
                .build(),
            rdma_fetch_bytes: meter
                .u64_counter("pegaflow_rdma_fetch_bytes")
                .with_unit("bytes")
                .with_description("Total bytes fetched via RDMA from remote nodes")
                .build(),
        }
    })
}
