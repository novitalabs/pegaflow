# Server Configuration

## PegaFlow Server

```bash
pegaflow-server
```

### Options

- `--addr`: Bind address (default: `127.0.0.1:50055`)
- `--devices`: CUDA device IDs to initialize, comma-separated (default: auto-detect all available GPUs, e.g., `--devices 0,1,2,3`)
- `--pool-size`: Pinned memory pool size (default: `30gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--hint-value-size`: Hint for typical value size to tune cache and allocator (optional, supports: `kb`, `mb`, `gb`, `tb`)
- `--use-hugepages`: Use huge pages for pinned memory (default: `false`, requires pre-configured `/proc/sys/vm/nr_hugepages`)
- `--enable-lfu-admission`: Enable TinyLFU cache admission policy (default: plain LRU)
- `--disable-numa-affinity`: Disable NUMA-aware memory allocation (default: enabled)
- `--blockwise-alloc`: Allocate each block separately instead of contiguous batch allocation. Reduces memory fragmentation when blocks are freed in different order (default: `false`)
- `--log-level`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)

### HTTP & Metrics

- `--http-addr`: HTTP server address for health check and Prometheus metrics (default: `0.0.0.0:9091`, always enabled)
- `--enable-prometheus`: Enable Prometheus `/metrics` endpoint (default: `true`)
- `--metrics-otel-endpoint`: OTLP metrics export endpoint (optional, leave unset to disable)
- `--metrics-period-secs`: Metrics export period in seconds (default: `10`, only used with OTLP)

### SSD Cache

- `--ssd-cache-path`: Enable SSD cache by providing cache file path (optional)
- `--ssd-cache-capacity`: SSD cache capacity (default: `512gb`, supports: `kb`, `mb`, `gb`, `tb`)
- `--ssd-write-queue-depth`: SSD write queue depth, max pending write batches (default: `8`)
- `--ssd-prefetch-queue-depth`: SSD prefetch queue depth, max pending prefetch batches (default: `2`)
- `--ssd-write-inflight`: SSD write inflight, max concurrent block writes (default: `2`)
- `--ssd-prefetch-inflight`: SSD prefetch inflight, max concurrent block reads (default: `16`)
- `--max-prefetch-blocks`: Max blocks allowed in prefetching state, backpressure for SSD prefetch (default: `800`)

### Cross-Node (Multi-Node Setup)

- `--nics`: RDMA NIC names for inter-node transfer (e.g., `--nics mlx5_0 mlx5_1`). When set, pinned memory is registered for RDMA access on these NICs. Required for P2P KV cache sharing.
- `--metaserver-addr`: MetaServer gRPC address for cross-node block hash registry (e.g., `http://10.0.0.100:50056`). When set, saved block hashes are inserted to the metaserver for cross-node discovery. Requires `--addr` to be a routable IP (not `0.0.0.0` or `127.0.0.1`).
- `--transfer-lock-timeout-secs`: Transfer lock timeout in seconds (default: `120`). Blocks held for cross-node RDMA transfer are locked for at most this duration before being force-released (crash recovery).
- `--metaserver-queue-depth`: MetaServer registration queue depth, max pending registration batches

## MetaServer

For multi-node setups, start a MetaServer to coordinate block hashes across nodes. Each pegaflow-server registers its block hashes with the MetaServer, enabling cross-node KV cache discovery.

```bash
pegaflow-metaserver
```

Then point each pegaflow-server to the MetaServer:

```bash
pegaflow-server --metaserver-addr http://<metaserver-host>:50056
```

### Options

- `--addr`: Bind address (default: `127.0.0.1:50056`)
- `--log-level`: Log level: `trace`, `debug`, `info`, `warn`, `error` (default: `info`)
- `--max-capacity-mb`: Maximum cache capacity in MB (default: `512`)
- `--ttl-minutes`: Cache entry TTL in minutes (default: `120`)
