# PegaFlow Metrics Guide

This guide explains how to collect, export, and visualize metrics from PegaFlow.

## Overview

PegaFlow exports metrics using the **OpenTelemetry Protocol (OTLP)** over gRPC. The metrics pipeline follows this flow:

```
PegaFlow Server → OpenTelemetry Collector → Prometheus → Grafana
   (OTLP/gRPC)         (HTTP scrape)       (HTTP queries)
```

## Available Metrics

PegaFlow exposes 11 core metrics for monitoring KV cache operations:

### Pool Metrics (Pinned Memory)
- **pegaflow_pool_used_bytes** (UpDownCounter)
  - Current pinned memory pool usage in bytes
  - Type: Gauge-like (tracks allocation deltas)
  - Use case: Monitor memory pressure

- **pegaflow_pool_capacity_bytes** (UpDownCounter)
  - Total pinned memory pool capacity in bytes (emitted once per pool lifecycle)
  - Type: Gauge-like (set on pool creation, cleared on drop)
  - Use case: Derive pool utilization without manual dashboard constants

- **pegaflow_pool_alloc_failures_total** (Counter)
  - Total allocation failures after eviction retries
  - Type: Counter
  - Use case: Detect memory exhaustion issues

### Cache Metrics (Block-level)
- **pegaflow_cache_block_hits_total** (Counter)
  - Number of complete blocks found in cache
  - Type: Counter
  - Use case: Calculate cache hit ratio

- **pegaflow_cache_block_misses_total** (Counter)
  - Number of complete blocks not found in cache
  - Type: Counter
  - Use case: Calculate cache miss ratio

- **pegaflow_cache_block_insertions_total** (Counter)
  - New blocks inserted into cache
  - Type: Counter
  - Use case: Track cache growth

- **pegaflow_cache_block_evictions_total** (Counter)
  - Blocks evicted from cache due to memory pressure
  - Type: Counter
  - Use case: Monitor eviction frequency, tune pool size

### Save Metrics (GPU → CPU)
- **pegaflow_save_bytes_total** (Counter)
  - Total bytes saved from GPU to CPU storage
  - Type: Counter
  - Unit: bytes
  - Use case: Monitor save throughput

- **pegaflow_save_duration_ms** (Histogram)
  - Save operation latency distribution
  - Type: Histogram
  - Unit: milliseconds
  - Use case: Track save performance (p50, p99)

### Load Metrics (CPU → GPU)
- **pegaflow_load_bytes_total** (Counter)
  - Total bytes loaded from CPU storage to GPU
  - Type: Counter
  - Unit: bytes
  - Use case: Monitor load throughput

- **pegaflow_load_duration_ms** (Histogram)
  - Load operation latency distribution
  - Type: Histogram
  - Unit: milliseconds
  - Use case: Track load performance (p50, p99)

- **pegaflow_load_failures_total** (Counter)
  - Load operation failures (e.g., transfer errors)
  - Type: Counter
  - Use case: Detect data transfer issues

## Configuration

### PegaFlow Server Parameters

The PegaFlow server accepts the following metrics-related command-line arguments:

```bash
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb \
  --metrics-otel-endpoint http://127.0.0.1:4321 \
  --metrics-period-secs 5
```

**Metrics Parameters:**
- `--metrics-otel-endpoint`: OTLP gRPC endpoint for metrics export
  - Default: `http://127.0.0.1:4321`
  - Set to empty string to disable metrics export
  - Protocol: OTLP over gRPC

- `--metrics-period-secs`: Metric export interval in seconds
  - Default: `5` seconds
  - Controls how frequently metrics are pushed to the collector

### Environment Variables

PegaFlow uses the OpenTelemetry SDK, which respects standard OTel environment variables:

- `OTEL_EXPORTER_OTLP_ENDPOINT`: Override default OTLP endpoint
- `OTEL_METRIC_EXPORT_INTERVAL`: Override export interval (milliseconds)
- `RUST_LOG`: Control logging verbosity (e.g., `info,pegaflow_core=debug`)

## Quick Start with Docker Compose

The `examples/metric/` directory provides a ready-to-use monitoring stack.

### 1. Start the Monitoring Stack

```bash
cd examples/metric

docker compose up -d
# If the above command doesn't work
# docker-compose up -d

# To stop the stack:
# docker compose down
```

This starts three services:
- **OpenTelemetry Collector** (ports: 4320, 4321, 8889)
- **Prometheus** (port: 9090)
- **Grafana** (port: 3000)

### 2. Start PegaFlow Server

```bash
# From repository root
cargo run -r -p pegaflow-server -- \
  --addr 0.0.0.0:50055 \
  --device 0 \
  --pool-size 30gb
```

The server will automatically export metrics to `http://127.0.0.1:4321` (the OTel Collector).

### 3. Access Grafana Dashboard

1. Open browser: http://localhost:3000
2. Login credentials:
   - Username: `admin`
   - Password: `admin`
3. Navigate to **Dashboards** → **PegaFlow Metrics**

> **Note**: If Grafana asks for a data source, use: `http://prometheus:9090`

The pre-configured dashboard includes:
- Cache hit/miss rate (per second)
- Total cache hits/misses
- Pool usage and failures
- Save/load throughput and latency

### 4. Query Metrics Directly

**Prometheus UI**: http://localhost:9090

Example PromQL queries:

```promql
# Cache hit ratio (last 5 minutes)
rate(pegaflow_cache_block_hits_total[5m]) /
(rate(pegaflow_cache_block_hits_total[5m]) + rate(pegaflow_cache_block_misses_total[5m]))

# Average save latency (p50)
histogram_quantile(0.5, rate(pegaflow_save_duration_ms_bucket[5m]))

# Average load latency (p99)
histogram_quantile(0.99, rate(pegaflow_load_duration_ms_bucket[5m]))

# Save throughput (MB/s)
rate(pegaflow_save_bytes_total[1m]) / 1e6

# Pool memory usage
pegaflow_pool_used_bytes
```

## Architecture Details

### Component Diagram

```
┌─────────────────┐
│ PegaFlow Server │
│  (Rust + OTLP)  │
└────────┬────────┘
         │ OTLP/gRPC (4321)
         ▼
┌─────────────────┐
│ OTel Collector  │
│  - Receives OTLP│
│  - Exposes :8889│
└────────┬────────┘
         │ Prometheus scrape (8889)
         ▼
┌─────────────────┐
│   Prometheus    │
│  - Stores TSDB  │
│  - Query API    │
└────────┬────────┘
         │ PromQL queries (9090)
         ▼
┌─────────────────┐
│    Grafana      │
│  - Dashboards   │
│  - Alerts       │
└─────────────────┘
```

### Port Reference

| Service            | Port | Protocol          | Purpose                     |
|--------------------|------|-------------------|-----------------------------|
| OTel Collector     | 4320 | HTTP              | OTLP HTTP receiver          |
| OTel Collector     | 4321 | gRPC              | OTLP gRPC receiver (default)|
| OTel Collector     | 8889 | HTTP              | Prometheus exporter         |
| Prometheus         | 9090 | HTTP              | Query API & Web UI          |
| Grafana            | 3000 | HTTP              | Dashboard UI                |
| PegaFlow Server    | 50055| gRPC              | Engine service              |

## Customization

### Modify OTel Collector Configuration

Edit `otel-collector-config.yaml` to:
- Add additional exporters (e.g., StatsD, InfluxDB)
- Configure sampling or filtering
- Add custom processors

Example: Add debug logging
```yaml
exporters:
  debug:
    verbosity: detailed
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug, prometheus]  # Added debug
```

### Modify Prometheus Scrape Interval

Edit `prometheus.yml`:
```yaml
global:
  scrape_interval: 10s  # Changed from 5s
```

### Add Custom Grafana Dashboards

1. Place JSON dashboard files in `grafana/dashboards/`
2. Update `grafana/provisioning/dashboards/dashboards.yaml` if needed
3. Restart Grafana: `docker-compose restart grafana`

## Troubleshooting

### Metrics not appearing in Grafana

1. Check PegaFlow server is exporting:
   ```bash
   # Should see periodic "Exporting metrics" logs
   RUST_LOG=opentelemetry_sdk=debug cargo run -r -p pegaflow-server
   ```

2. Verify OTel Collector is receiving data:
   ```bash
   docker-compose logs otel-collector | grep pegaflow
   ```

3. Check Prometheus is scraping:
   - Open http://localhost:9090/targets
   - Verify `otel-collector` target is UP

4. Test Prometheus query:
   - Open http://localhost:9090/graph
   - Query: `pegaflow_pool_used_bytes`

### High cardinality warnings

If you see warnings about high cardinality:
- Avoid adding labels with many unique values (e.g., block hashes)
- Use aggregation in Prometheus queries
- Consider adjusting retention policies

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

5. **Correlate throughput with latency**:
   - High throughput + high latency → bandwidth bottleneck
   - Low throughput + low latency → underutilization

## References

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Dashboard Guide](https://grafana.com/docs/grafana/latest/dashboards/)
