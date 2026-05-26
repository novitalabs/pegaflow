# PD Connector Benchmark Results

## Setup

- Machine: single node, NVIDIA H20 (SXM, 96GB)
- Model: Qwen3-8B
- vLLM: 0.19.1
- Branch: `refactor/pd-handshake-simplify`
- Config: TP=1, bs=1, output_len=1, seed=42
- Decode: GPU 1 / mlx5_1, Prefill: GPU 2 / mlx5_2
- Proxy: Python PdProxy on loopback

## TTFT Sweep (Mean, 5 requests each)

| input_len | Baseline (ms) | PegaFlow PD (ms) | NIXL PD (ms) | PegaFlow Δ (ms) | NIXL Δ (ms) |
|-----------|--------------|-------------------|--------------|-----------------|-------------|
| 512       | 66.1         | 79.0              | 100.2        | +12.9           | +34.1       |
| 1024      | 120.3        | 134.5             | 147.8        | +14.2           | +27.5       |
| 2048      | 236.3        | 250.2             | 274.6        | +13.9           | +38.3       |
| 4096      | 481.2        | 496.7             | 524.5        | +15.5           | +43.3       |
| 8192      | 1019.6       | 1035.8            | 1085.2       | +16.2           | +65.6       |
| 16000     | 2251.2       | 2300.7            | 2366.7       | +49.4           | +115.5      |

NIXL config: NixlConnector, kv_role=kv_both, UCX_NET_DEVICES=all, enforce-eager,
proxy: vLLM toy_proxy_server.py (httpx + FastAPI), GPU-direct RDMA via UCX.

## Overhead Breakdown (16k input, single request)

| Stage                          | Time   |
|--------------------------------|--------|
| D handshake build              | 1.8ms  |
| D→P HTTP + P vLLM scheduling   | 37.9ms |
| **Control plane total**        | ~40ms  |

D→P setup (37.9ms) further breaks down to:
- P connector scheduling: ~15ms
- D HTTP POST (125KB) + P tokenize 16k + P vLLM scheduler: ~23ms

## RDMA Transfer Stats (16k, per request)

- KV size: 2.36 GB (36 layers × 1000 blocks)
- RDMA throughput: 8.67 Gbps (400G link, single NIC)
- Per-layer push: ~60ms (65MB/layer)
- Forward: ~1025ms, RDMA tail: ~1150ms
- RDMA fully pipelined with forward — tail absorbed by baseline prefill time
