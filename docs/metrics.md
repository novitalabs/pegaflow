# PegaFlow Metrics Guide

This guide explains how to collect, export, and visualize metrics from PegaFlow.

## Overview

PegaFlow supports two methods for exposing metrics:

### Method 1: Direct Prometheus (Recommended)

```
PegaFlow Server → Prometheus → Grafana
   (/metrics)      (scrape)    (visualize)
```

- Simpler deployment (2 components)
- PegaFlow exposes `/metrics` endpoint directly
- Use `examples/metric-prometheus/`

### Method 2: OTLP via OpenTelemetry Collector (Deprecated)

> **DEPRECATED**: This method is deprecated now, but we will keep it.
> Please use Method 1 (Direct Prometheus) instead.

```
PegaFlow Server → OpenTelemetry Collector → Prometheus → Grafana
   (OTLP/gRPC)         (HTTP scrape)       (HTTP queries)
```

- More flexible (supports multiple backends)
- Useful if you already have OTel infrastructure
- Use `examples/metric/`

## Available Metrics

PegaFlow exposes the following metrics for monitoring KV cache operations:

### Pool Metrics (Pinned Memory)
- **pegaflow_pool_used_bytes** (Gauge)
  - Current pinned memory pool usage in bytes
  - Use case: Monitor memory pressure

- **pegaflow_pool_capacity_bytes** (Gauge)
  - Total pinned memory pool capacity in bytes
  - Use case: Derive pool utilization

- **pegaflow_pool_largest_free_bytes** (Gauge)
  - Largest contiguous free region in pinned pool (fragmentation signal)
  - Use case: Distinguish true exhaustion vs fragmentation (largest_free << free_bytes)

- **pegaflow_pool_alloc_failures_total** (Counter)
  - Total allocation failures after eviction retries
  - Use case: Detect memory exhaustion issues

### Cache Metrics (Block-level)
- **pegaflow_cache_block_hits_total** (Counter)
  - Legacy resident-cache hit counter
  - Use case: Backward-compatible dashboards only; do not use for tier
    contribution analysis

- **pegaflow_cache_block_misses_total** (Counter)
  - Legacy resident-cache miss counter
  - Use case: Backward-compatible dashboards only; do not use for tier
    contribution analysis

- **pegaflow_cache_tier_block_requests_total** (Counter)
  - Per-decision `query_prefetch` block attribution by `tier`
    (`ram`, `rdma`, `ssd`, or `miss`)
  - Use case: Calculate overall hit ratio and each cache tier's contribution
    from one consistent denominator
  - Invariant: for each attributed decision,
    `ram + rdma + ssd + miss == block_hashes.len()`
  - Semantics: this is decision attribution. `tier="rdma"` and `tier="ssd"`
    mean the block was selected to be satisfied by that backing tier; they do
    not guarantee the later backing operation succeeded.

- **pegaflow_cache_block_insertions_total** (Counter)
  - New blocks inserted into cache
  - Use case: Track cache growth

- **pegaflow_cache_block_evictions_total** (Counter)
  - Blocks evicted from cache due to memory pressure
  - Use case: Monitor eviction frequency, tune pool size

- **pegaflow_cache_block_evictions_by_class_total** (Counter)
  - Blocks evicted from cache due to memory pressure, labelled by replacement class (`reclaimable` or `retained`)
  - Use case: Verify remote-fetched replicas are reclaimed before locally produced blocks

- **pegaflow_cache_block_evictions_still_referenced_total** (Counter)
  - Evicted blocks that still had external references (eviction did not immediately reclaim pinned memory)
  - Use case: Explain "evictions spike but pool_used_bytes doesn't drop"

- **pegaflow_cache_eviction_reclaimed_bytes_total** (Counter)
  - Estimated bytes actually reclaimed in pinned allocator after cache eviction
  - Use case: Measure effectiveness of eviction under real reference patterns

- **pegaflow_cache_resident_blocks** (Gauge)
  - Current number of sealed blocks resident in cache, labelled by replacement class (`reclaimable` or `retained`)
  - Use case: Track cache size and source-based replacement pressure in blocks

- **pegaflow_cache_resident_bytes** (Gauge)
  - Current sealed block bytes resident in cache (sum of footprints)
  - Use case: Attribute pinned pool usage to cache residency

- **pegaflow_cache_residence_duration_seconds** (Histogram)
  - RAM resident block lifetime from its first successful cache insertion to
    removal, measured in seconds
  - Labels: `reason` (`pressure` or `cleanup`)
  - `pressure`: allocator pressure removed the block through LRU reclaim
  - `cleanup`: the memory-cache cleanup endpoint removed the block
  - Buckets: `1s`, `5s`, `10s`, `30s`, `1m`, `2m`, `5m`, `10m`, `30m`,
    `1h`, `2h`, `6h`, `12h`, `24h`, and `+Inf`
  - Use case: Track typical and tail cache residence time and visualize the
    eviction-age distribution
  - Scope: each RAM residence is a separate lifetime. A block reinserted after
    eviction starts a new lifetime; cache hits, duplicate inserts, and
    replacement-class changes do not reset the original insertion time.
  - The lifetime ends when the block leaves the resident cache, even if an
    outstanding `Arc` keeps its pinned memory allocated. Correlate
    `pegaflow_cache_block_evictions_still_referenced_total` and
    `pegaflow_cache_eviction_reclaimed_bytes_total` to diagnose delayed memory
    reclamation.
  - SSD ring-cache overwrite and blocks still resident at server shutdown are
    not observed.

- **pegaflow_pinned_for_load_entries** (Gauge)
  - Current number of pinned_for_load entries (instance_id, block_key)
  - Use case: Diagnose load-path pins keeping evicted blocks alive

- **pegaflow_pinned_for_load_refs** (Gauge)
  - Current outstanding pinned_for_load consumer refcount (sum of per-entry counts)
  - Use case: Detect stuck consumers / missing release on load path

- **pegaflow_pinned_for_load_unique_blocks** (Gauge)
  - Current number of unique blocks referenced by pinned_for_load
  - Use case: Understand how many distinct blocks are being kept alive by pins

- **pegaflow_pinned_for_load_unique_bytes** (Gauge)
  - Current bytes referenced by pinned_for_load (unique blocks; sum of footprints)
  - Use case: Attribute pinned pool usage to load-path pins

### HLL Reuse Metrics
- **pegaflow_hll_cardinality** (Gauge)
  - Estimated distinct `(namespace, block hash)` objects classified as misses in a configured sliding window
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Derive approximate prefix reuse over longer windows without
    storing every block hash

- **pegaflow_hll_total_requests** (Gauge)
  - Total queried blocks in the same configured sliding window, including ready blocks and duplicates
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Denominator for HLL-based reference reuse rate

- **pegaflow_hll_estimated_hit_rate** (Gauge)
  - Server-computed miss-based infinite-cache reuse reference from the same
    HLL snapshot as the two gauges above
  - Labels: `window`
  - Value is clamped to `[0, 1]`

The MetaServer exports the active-node register union. Do not sum per-server
cardinality gauges: the same object observed on multiple nodes must count only
once at cluster scope.

- **pegaflow_metaserver_hll_cardinality** (Gauge)
  - Estimated distinct cache objects in the active-node HLL register union
  - Labels: `window`
- **pegaflow_metaserver_hll_total_requests** (Gauge)
  - Sum of observations in the latest reports from active node sessions
  - Labels: `window`
- **pegaflow_metaserver_hll_estimated_hit_rate** (Gauge)
  - Miss-based infinite-cache reuse reference from the same aggregate snapshot
  - Labels: `window`
- **pegaflow_metaserver_hll_active_nodes** (Gauge)
  - Active node sessions included in the union
  - Labels: `window`
- **pegaflow_metaserver_hll_snapshot_age** (Gauge, unit `s`)
  - Age of the oldest active-node report in the aggregate
  - Labels: `window`

Missing or damaged HLL reports are best-effort observability failures: the
MetaServer keeps the node live for metadata, excludes that report from the
cluster union, and the server increments
`pegaflow_metaserver_hll_report_failures` when it cannot build a report.
All active reports must use the same window label, duration, and `bucket_bits`;
an incompatible report is excluded from the union while the node remains live.

The existing cardinality and total metrics are retained. Existing PromQL
continues to work, but new dashboards should prefer the direct gauge because
it applies the same cardinality clamp as the tracker:

```promql
1 - (
  pegaflow_hll_cardinality{window="1h"}
  /
  clamp_min(pegaflow_hll_total_requests{window="1h"}, 1)
)
```

```promql
pegaflow_hll_estimated_hit_rate{window="1h"}
```

This is a metrics semantic update, not an `/metrics` protocol breaking change:
metric names, types, existing labels, and the HTTP endpoint are unchanged;
the new gauge is additive. The default HLL size changes from 16,384 registers
(`bucket_bits=14`, about 0.8% standard error) to 65,536 registers
(`bucket_bits=16`, about 0.4%). Three default windows use about 192 KiB of
register storage; sliding slots make the live tracker a few MiB per server.
The setting remains configurable with `--metric-hll-bucket-bits`.

### Save Metrics (GPU → CPU)
- **pegaflow_save_bytes_total** (Counter)
  - Total bytes saved from GPU to CPU storage
  - Use case: Monitor save throughput

- **pegaflow_save_duration_seconds** (Histogram)
  - Save operation latency distribution
  - Use case: Track save performance (p50, p99)

### Load Metrics (CPU → GPU)
- **pegaflow_load_bytes_total** (Counter)
  - Total bytes loaded from CPU storage to GPU
  - Use case: Monitor load throughput

- **pegaflow_load_duration_seconds** (Histogram)
  - Load operation latency distribution
  - Use case: Track load performance (p50, p99)

- **pegaflow_load_failures_total** (Counter)
  - Load operation failures (e.g., transfer errors)
  - Use case: Detect data transfer issues

### SSD Cache Metrics
- **pegaflow_ssd_write_bytes_total** (Counter) - Bytes written to SSD cache
- **pegaflow_ssd_write_duration_seconds** (Histogram) - SSD write latency
- **pegaflow_ssd_prefetch_success_total** (Counter) - Successful SSD prefetches
- **pegaflow_ssd_prefetch_failures_total** (Counter) - Failed SSD prefetches
- **pegaflow_ssd_prefetch_duration_seconds** (Histogram) - SSD prefetch latency

### Tier Attribution Semantics

`pegaflow_cache_tier_block_requests_total{tier}` is the canonical metric for
explaining how each cache tier contributes to prefix-query hit ratio. It is
emitted once for each `query_prefetch` decision and uses exactly one label:
`tier`.

Tier values:

- `ram`: blocks already present in the resident RAM cache at the decision point
- `rdma`: blocks selected to be satisfied by RDMA remote fetch
- `ssd`: blocks selected to be satisfied by SSD prefetch
- `miss`: blocks no tier selected for that decision, including SSD prefetch
  backpressure and residual blocks after RDMA partial availability

This metric intentionally records decisions, not completed service outcomes.
For backing failure correlation, use:

- `pegaflow_rdma_fetch_total{status="error"}` for RDMA fetch failures
- `pegaflow_ssd_prefetch_failures_total` for SSD prefetch failures

The legacy `pegaflow_cache_block_hits_total` and
`pegaflow_cache_block_misses_total` counters are retained for compatibility.
Their `Loading { hit, loading }` path only increments hits by `hit`; `loading`
does not enter the legacy denominator. New dashboards should use
`pegaflow_cache_tier_block_requests_total{tier}` instead of mixing legacy and
tier counters.

### RPC Metrics
- **pegaflow_rpc_requests_total** (Counter) - Total RPC requests by method and status
- **pegaflow_rpc_duration_seconds** (Histogram) - RPC latency distribution

## Configuration

### PegaFlow Server Parameters

**Metrics Parameters:**

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`)
  - Always enabled for health check at `/health`
  - Use `--enable-prometheus` to also expose `/metrics` endpoint

- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
  - When enabled, metrics are available at `http://<http-addr>/metrics`
  - Health check is always available at `http://<http-addr>/health`

- `--metrics-otel-endpoint`: OTLP gRPC endpoint for metrics export (optional)
  - Example: `http://127.0.0.1:4321`
  - Leave unset to disable OTLP export

- `--metrics-period-secs`: Metric export interval in seconds (default: `5`)
  - Only used when `--metrics-otel-endpoint` is set

- `--metric-hll-windows`: Comma-separated HLL sliding windows for estimated
  prefix reuse (default: `15m,1h,1d`)
  - Supported units: `s`, `m`, `h`, `d`
  - Each configured duration becomes a canonical `window` label. For example,
    the default config exports `window="15m"`, `window="1h"`, and `window="1d"`.
  - Empty entries such as `15m,,1h` and duplicate durations such as `1h,60m`
    are rejected at startup.

- `--metric-hll-bucket-bits`: HLL bucket index bits (default: `16`)
  - `2^16 = 65,536` registers per window and about 0.4% standard error.
  - Higher values use more memory and lower estimation error; `18` remains
    the supported maximum.

**Example: Prometheus Metrics**
```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

**Example: OTLP Export Only**
```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --enable-prometheus=false \
  --metrics-otel-endpoint http://127.0.0.1:4321
```

**Example: Health Check Only (No Metrics)**
```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --enable-prometheus=false
```

### Environment Variables

- `RUST_LOG`: Control logging verbosity (e.g., `info,pegaflow_core=debug`)

## Quick Start: Direct Prometheus (Recommended)

The `examples/metric-prometheus/` directory provides a simple monitoring stack.

### 1. Start PegaFlow Server

```bash
# From repository root
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --http-addr 0.0.0.0:9091 \
  --enable-prometheus
```

### 2. Start the Monitoring Stack

```bash
cd examples/metric-prometheus

docker compose up -d
# To stop: docker compose down
```

This starts two services:
- **Prometheus** (port: 9090) - Scrapes metrics from PegaFlow
- **Grafana** (port: 3000) - Visualizes metrics

### 3. Access Grafana Dashboard

1. Open browser: http://localhost:3000
2. Login: `admin` / `admin`
3. Navigate to **Dashboards** → **PegaFlow Metrics**

### 4. Test Metrics Endpoint

```bash
curl http://localhost:9091/metrics
```

## Quick Start: OTLP Method

The `examples/metric/` directory provides a full OTel-based monitoring stack using the OTLP exporter.

### 1. Start the Monitoring Stack

```bash
cd examples/metric

docker compose up -d
```

This starts three services:
- **OpenTelemetry Collector** (ports: 4320, 4321, 8889)
- **Prometheus** (port: 9090)
- **Grafana** (port: 3000)

### 2. Start PegaFlow Server

```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --metrics-otel-endpoint http://127.0.0.1:4321
```

### 3. Access Grafana Dashboard

Same as above: http://localhost:3000

## Architecture Details

### Direct Prometheus Architecture (Recommended)

```
┌─────────────────┐
│ PegaFlow Server │
│   :50055 gRPC   │
│   :9091 /metrics│
└────────┬────────┘
         │ Prometheus scrape
         ▼
┌─────────────────┐
│   Prometheus    │
│     :9090       │
└────────┬────────┘
         │ PromQL queries
         ▼
┌─────────────────┐
│    Grafana      │
│     :3000       │
└─────────────────┘
```

### OTLP Architecture (Deprecated)

```
┌─────────────────┐
│ PegaFlow Server │
│   :50055 gRPC   │
└────────┬────────┘
         │ OTLP/gRPC (4321)
         ▼
┌─────────────────┐
│ OTel Collector  │
│     :8889       │
└────────┬────────┘
         │ Prometheus scrape
         ▼
┌─────────────────┐
│   Prometheus    │
│     :9090       │
└────────┬────────┘
         │ PromQL queries
         ▼
┌─────────────────┐
│    Grafana      │
│     :3000       │
└─────────────────┘
```

### Port Reference

| Service            | Port  | Protocol | Purpose                              |
|--------------------|-------|----------|--------------------------------------|
| PegaFlow Server    | 50055 | gRPC     | Engine service                       |
| PegaFlow Server    | 9091  | HTTP     | Prometheus metrics endpoint          |
| PegaFlow MetaServer | 50056 | gRPC     | Cross-node metadata and HLL reports  |
| PegaFlow MetaServer | 9092  | HTTP     | Prometheus metrics endpoint          |
| OTel Collector     | 4321  | gRPC     | OTLP gRPC receiver (deprecated)      |
| OTel Collector     | 8889  | HTTP     | Prometheus exporter (deprecated)     |
| Prometheus         | 9090  | HTTP     | Query API & Web UI                   |
| Grafana            | 3000  | HTTP     | Dashboard UI                         |

## PromQL Query Examples

```promql
# Overall cache hit ratio from decision attribution (last 5 minutes)
sum(rate(pegaflow_cache_tier_block_requests_total{tier!="miss"}[5m])) /
sum(rate(pegaflow_cache_tier_block_requests_total[5m]))

# RAM contribution to total requested blocks
sum(rate(pegaflow_cache_tier_block_requests_total{tier="ram"}[5m])) /
sum(rate(pegaflow_cache_tier_block_requests_total[5m]))

# RDMA contribution to total requested blocks
sum(rate(pegaflow_cache_tier_block_requests_total{tier="rdma"}[5m])) /
sum(rate(pegaflow_cache_tier_block_requests_total[5m]))

# SSD contribution to total requested blocks
sum(rate(pegaflow_cache_tier_block_requests_total{tier="ssd"}[5m])) /
sum(rate(pegaflow_cache_tier_block_requests_total[5m]))

# Miss ratio from the same denominator
sum(rate(pegaflow_cache_tier_block_requests_total{tier="miss"}[5m])) /
sum(rate(pegaflow_cache_tier_block_requests_total[5m]))

# Average save latency (p50)
histogram_quantile(0.5, rate(pegaflow_save_duration_seconds_bucket[5m]))

# Average load latency (p99)
histogram_quantile(0.99, rate(pegaflow_load_duration_seconds_bucket[5m]))

# Save throughput (MB/s)
rate(pegaflow_save_bytes_total[1m]) / 1e6

# Pool memory utilization
pegaflow_pool_used_bytes / pegaflow_pool_capacity_bytes

# RAM cache residence-time quantiles for pressure evictions
histogram_quantile(
  0.50,
  sum by (le) (
    rate(pegaflow_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.95,
  sum by (le) (
    rate(pegaflow_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

histogram_quantile(
  0.99,
  sum by (le) (
    rate(pegaflow_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
  )
)

# RAM cache residence-time buckets for a Grafana heatmap
sum by (le) (
  rate(pegaflow_cache_residence_duration_seconds_bucket{reason="pressure"}[5m])
)

# HLL estimated hit rate for the 1h window (preferred)
pegaflow_hll_estimated_hit_rate{window="1h"}

# Cluster HLL estimated hit rate for the 1h window
pegaflow_metaserver_hll_estimated_hit_rate{window="1h"}

# Backward-compatible derivation from the retained gauges
1 - (
  pegaflow_hll_cardinality{window="1h"}
  /
  clamp_min(pegaflow_hll_total_requests{window="1h"}, 1)
)

# HLL estimated hit rate for every configured window
1 - (
  pegaflow_hll_cardinality
  /
  clamp_min(pegaflow_hll_total_requests, 1)
)
```

## Troubleshooting

### Metrics not appearing (Direct Prometheus)

1. Check PegaFlow is exposing metrics:
   ```bash
   curl http://localhost:9091/metrics
   ```

2. Check Prometheus targets:
   - Open http://localhost:9090/targets
   - Verify `pegaflow` target is UP

3. If Docker cannot reach host, ensure `extra_hosts` is configured:
   ```yaml
   extra_hosts:
     - "host.docker.internal:host-gateway"
   ```

### Metrics not appearing (OTLP) - Deprecated

1. Check OTel Collector is receiving data:
   ```bash
   docker-compose logs otel-collector | grep pegaflow
   ```

2. Check Prometheus is scraping OTel Collector:
   - Open http://localhost:9090/targets
   - Verify `otel-collector` target is UP

## Best Practices

1. **Monitor tier-attributed hit ratio**: Aim for >80% hit rate in production
   using `pegaflow_cache_tier_block_requests_total`
   - Low hit rate → consider increasing `--pool-size`

2. **Watch eviction rate**: High evictions indicate memory pressure
   - Use `rate(pegaflow_cache_block_evictions_total[5m])`

3. **Track allocation failures**: Any failures indicate critical issues
   - Alert on `pegaflow_pool_alloc_failures_total > 0`

4. **Analyze latency distributions**: Use histogram quantiles
   - p50: Typical case performance
   - p99: Worst-case user experience

## References

- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
