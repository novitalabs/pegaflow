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

- **pegaflow_pool_alloc_failures_total** (Counter)
  - Total allocation failures after eviction retries
  - Use case: Detect memory exhaustion issues

### Cache Metrics (Block-level)
- **pegaflow_cache_block_hits_total** (Counter)
  - Number of complete blocks found in cache
  - Use case: Calculate cache hit ratio

- **pegaflow_cache_block_misses_total** (Counter)
  - Number of complete blocks not found in cache
  - Use case: Calculate cache miss ratio

- **pegaflow_cache_block_insertions_total** (Counter)
  - New blocks inserted into cache
  - Use case: Track cache growth

- **pegaflow_cache_block_evictions_total** (Counter)
  - Blocks evicted from cache due to memory pressure
  - Use case: Monitor eviction frequency, tune pool size

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

### RPC Metrics
- **pegaflow_rpc_requests_total** (Counter) - Total RPC requests by method and status
- **pegaflow_rpc_duration_seconds** (Histogram) - RPC latency distribution

## Configuration

### PegaFlow Server Parameters

**Metrics Parameters:**

- `--metrics-addr`: Address for Prometheus HTTP endpoint (e.g., `0.0.0.0:9091`)
  - When set, exposes `/metrics` endpoint at the specified address
  - Leave unset to disable direct Prometheus metrics

- `--metrics-otel-endpoint` **(DEPRECATED)**: OTLP gRPC endpoint for metrics export
  - Example: `http://127.0.0.1:4321`
  - Leave unset to disable OTLP export
  - **Note**: This option is deprecated. Use `--metrics-addr` instead.

- `--metrics-period-secs` **(DEPRECATED)**: Metric export interval in seconds (default: `5`)
  - Only used when `--metrics-otel-endpoint` is set
  - **Note**: This option is deprecated. Use `--metrics-addr` instead.

**Example: Direct Prometheus (Recommended)**
```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --metrics-addr 0.0.0.0:9091
```

**Example: OTLP (Deprecated)**
```bash
# DEPRECATED: Use --metrics-addr instead
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --metrics-otel-endpoint http://127.0.0.1:4321
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
  --metrics-addr 0.0.0.0:9091
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

## Quick Start: OTLP Method (Deprecated)

> **DEPRECATED**: This method is deprecated. Please use the Direct Prometheus method above.

The `examples/metric/` directory provides a full OTel-based monitoring stack.

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
# DEPRECATED: Use --metrics-addr instead
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
| OTel Collector     | 4321  | gRPC     | OTLP gRPC receiver (deprecated)      |
| OTel Collector     | 8889  | HTTP     | Prometheus exporter (deprecated)     |
| Prometheus         | 9090  | HTTP     | Query API & Web UI                   |
| Grafana            | 3000  | HTTP     | Dashboard UI                         |

## PromQL Query Examples

```promql
# Cache hit ratio (last 5 minutes)
rate(pegaflow_cache_block_hits_total[5m]) /
(rate(pegaflow_cache_block_hits_total[5m]) + rate(pegaflow_cache_block_misses_total[5m]))

# Average save latency (p50)
histogram_quantile(0.5, rate(pegaflow_save_duration_seconds_bucket[5m]))

# Average load latency (p99)
histogram_quantile(0.99, rate(pegaflow_load_duration_seconds_bucket[5m]))

# Save throughput (MB/s)
rate(pegaflow_save_bytes_total[1m]) / 1e6

# Pool memory utilization
pegaflow_pool_used_bytes / pegaflow_pool_capacity_bytes
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

1. **Monitor cache hit ratio**: Aim for >80% hit rate in production
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
