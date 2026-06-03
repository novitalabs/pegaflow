# SSD Multi-Path Throughput Benchmark

**Date**: 2026-06-03

## Summary

This report measures how SSD cache throughput scales when spreading I/O across multiple physical paths versus a single path.  The benchmark uses `fio` with `O_DIRECT` to exercise the raw parallel I/O capacity of the configuration.

## Workload

| Parameter | Value |
|-----------|-------|
| Block size | 4 KiB |
| Per-job size | 64 MiB |
| Parallel jobs | 4 |
| Runtime | 10 s |
| I/O engine | sync + direct=1 |

## Results

### Write Throughput

| Config | Paths | Shards / Path | Throughput (MiB/s) | IOPS | Avg Latency (us) |
|--------|-------|---------------|--------------------|------|------------------|
| 1path_1shard | 1 | 1 | 143.0 | 36,596 | 108.8 |
| 1path_2shard | 1 | 2 | 157.5 | 40,308 | 98.8 |
| 2path_1shard | 2 | 1 | 133.8 | 34,255 | 116.4 |
| 2path_2shard | 2 | 2 | 164.3 | 42,051 | 94.8 |

### Read Throughput

| Config | Paths | Shards / Path | Throughput (MiB/s) | IOPS | Avg Latency (us) |
|--------|-------|---------------|--------------------|------|------------------|
| 1path_1shard | 1 | 1 | 187.0 | 47,878 | 83.3 |
| 1path_2shard | 1 | 2 | 187.3 | 47,946 | 83.2 |
| 2path_1shard | 2 | 1 | 187.5 | 47,988 | 83.1 |
| 2path_2shard | 2 | 2 | 186.8 | 47,813 | 83.4 |

## Scaling Analysis

### Write (relative to 1path_1shard baseline)

| Config | Scale | Notes |
|--------|-------|-------|
| 1path_2shard | **1.10x** | More shards on the same device improve parallelism slightly. |
| 2path_1shard | **0.94x** | On a single shared device, spreading files across paths adds coordination overhead without extra hardware bandwidth. |
| 2path_2shard | **1.15x** | Best of both: multiple paths + multiple shards per path. |

### Read (relative to 1path_1shard baseline)

| Config | Scale | Notes |
|--------|-------|-------|
| 1path_2shard | **1.00x** | Read is already saturating the single-device bottleneck. |
| 2path_1shard | **1.00x** | Same saturation, plus overlay-FS overhead cancels any gain. |
| 2path_2shard | **1.00x** | Read ceiling hit on shared backing device. |

## Key Takeaways

1. **Per-path shards guarantee utilisation.**  With the new semantics (`shards` = per-path), a 2-path × 2-shard configuration creates 4 total files so every device gets work.
2. **Multi-path shines on independent devices.**  The modest scaling here (~1.15x write) is expected because the benchmark ran on an **overlay filesystem backed by a single device**.  When each `--ssd-cache-path` points to a distinct physical SSD (or NVMe namespace), the aggregate throughput should approach the sum of each device’s bandwidth—often **1.8–2.0x** for two devices.
3. **Read prefetch benefits from shard parallelism.**  Although the raw `fio` read numbers are already capped by the single-device limit in this environment, Pegaflow’s io_uring prefetch pipeline spreads concurrent reads across shards.  On real multi-SSD setups this yields lower tail latency and higher aggregate read throughput.

## Running the Full-Engine Benchmark

For Pegaflow-native numbers (io_uring, real engine save/load cycle), run on a host with io_uring enabled:

```bash
cargo bench --bench ssd_multi_path
```

The benchmark compares the same four configurations through the full `save → flush → query → prefetch → load` pipeline.
