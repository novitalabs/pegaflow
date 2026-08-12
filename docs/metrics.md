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
  - Per-server estimated distinct `(namespace, block hash)` objects classified
    as misses in a configured sliding window
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Derive approximate prefix reuse over longer windows without
    storing every block hash

- **pegaflow_hll_total_requests** (Gauge)
  - Total queried blocks in the same configured sliding window, including
    already-ready blocks and duplicates
  - Labels: `window` (`15m`, `1h`, `1d` by default)
  - Use case: Denominator for HLL-based estimated hit-rate PromQL

An individual PegaFlow server does not export a separate HLL hit-rate gauge.
Use PromQL so the ratio is computed from values in the same scrape:

```promql
1 - (
  pegaflow_hll_cardinality{window="1h"}
  /
  clamp_min(pegaflow_hll_total_requests{window="1h"}, 1)
)
```

The MetaServer exports the active-node register union. Do not sum the
per-server cardinality gauges: the same object observed on multiple nodes must
be distinct only once at cluster scope.

- **pegaflow_metaserver_hll_cardinality** (Gauge)
  - Estimated distinct cache objects in the active-node register union
  - Labels: `window`
- **pegaflow_metaserver_hll_total_requests** (Gauge)
  - Sum of observations in the latest reports from all active node sessions
  - Labels: `window`
- **pegaflow_metaserver_hll_estimated_hit_rate** (Gauge)
  - Miss-based infinite-cache theoretical reuse reference derived from the
    union cardinality of final query misses
    and total observations in one aggregate snapshot
  - Labels: `window`
- **pegaflow_metaserver_hll_active_nodes** (Gauge)
  - Number of active node sessions included in the cluster union
  - Labels: `window`
- **pegaflow_metaserver_hll_snapshot_age_seconds** (Gauge)
  - Age of the oldest active-node report in the aggregate
  - Labels: `window`

Every node heartbeat carries all configured windows, including all-zero
registers before traffic. A missing, damaged, incomplete, or schema-incompatible
report rejects the whole heartbeat. Node-local sliding windows are not aligned
to a global epoch, so the cluster value is a near-time union whose skew is
bounded by heartbeat/report freshness rather than a strict common boundary.

The miss-based HLL reuse metric is a theoretical infinite-cache reference.
With exact distinct-miss counting it is an upper bound on reuse for the same
observation stream. Raw HLL cardinality is still an estimate and can be high
or low; use a cardinality lower confidence bound when a statistical upper
bound is required. HLL windows and actual counters must use the same interval
and active node set. With the default `bucket_bits=14`, the HLL standard error
is approximately 0.8%.
P2P is already represented by the `rdma` tier in actual attribution; do not add
the RDMA ratio to the HLL reference.

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
  prefix reuse (default: `15m,1h,24h`)
  - Supported units: `s`, `m`, `h`, `d`
  - Each configured duration becomes a canonical `window` label. For example,
    the default config exports `window="15m"`, `window="1h"`, and `window="1d"`.
  - Empty entries such as `15m,,1h` and duplicate durations such as `1h,60m`
    are rejected at startup.

- `--metric-hll-bucket-bits`: HLL bucket index bits (default: `14`)
  - Higher values use more memory and lower estimation error.

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
| PegaFlow MetaServer| 50056 | gRPC     | Cross-node metadata and HLL reports  |
| PegaFlow MetaServer| 9092  | HTTP     | Prometheus metrics endpoint          |
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

# Cluster HLL miss-based theoretical reuse reference for the 15m window
pegaflow_metaserver_hll_estimated_hit_rate{job="pegaflow-metaserver",window="15m"}

# Actual tier-attributed ratio over the same 15m interval
sum(increase(pegaflow_cache_tier_block_requests_total{job="pegaflow",tier!="miss"}[15m]))
/
clamp_min(sum(increase(pegaflow_cache_tier_block_requests_total{job="pegaflow"}[15m])), 1)

# RAM / RDMA / SSD / miss decomposition with the same denominator
sum by (tier) (increase(pegaflow_cache_tier_block_requests_total{job="pegaflow"}[15m]))
/
clamp_min(sum(increase(pegaflow_cache_tier_block_requests_total{job="pegaflow"}[15m])), 1)

# Cluster HLL participation and freshness
pegaflow_metaserver_hll_active_nodes{job="pegaflow-metaserver",window="15m"}
pegaflow_metaserver_hll_snapshot_age_seconds{job="pegaflow-metaserver",window="15m"}

# Per-server HLL estimated reuse for every configured window
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
