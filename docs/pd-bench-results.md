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

## H20 Kimi K2.5 Two-Node MLA Debug Run (2026-05-28)

Standalone experiment note: [h20-kimi-pd-mla-debug.md](h20-kimi-pd-mla-debug.md).

Note: this c4 run is a diagnostic queueing run. H20 PD MLA acceptance pressure
tests should use `--max-concurrency 1`, sweep only input length, and compare
direct baseline vs P/D proxy with the same fixed non-connector vLLM serving flags
(`--load-format dummy --max-num-batched-tokens 32768`).
Use `scripts/run_pd_h20_kimi.sh start-baseline` for the direct baseline and
`scripts/run_h20_kimi_ttft_sweep.sh` for the paired c1 sweep.

### Setup

- Branch/commit: `docs/pd-mla-design`, `874e0c6 fix: improve pd rdma push throughput`
- Prefill/proxy node: `h20-99`
- Decode node: `h20-100`
- Model: `/data/models/Kimi-K2.5`
- TP: 8 on P and D
- vLLM flags: `--load-format dummy`, `--max-num-batched-tokens 32768`,
  no explicit `--block-size`, no explicit `--max-model-len`
- Proxy: `http://127.0.0.1:18100/v1/completions`
- NIC rank map:
  - ranks 0,1: `mlx5_1`
  - ranks 2,3: `mlx5_2`
  - ranks 4,5: `mlx5_3`
  - ranks 6,7: `mlx5_4`

### Code Changes Tested

- D-side prefill HTTP dispatch is parallelized with 8 sender threads.
- P-side RDMA layer push is parallelized with 4 sender threads per worker.
- Native RDMA write window reservation uses CAS before submit, so concurrent Python
  push threads cannot overrun the global write window.
- Hot per-layer RDMA write logs are DEBUG; per-request summary logs remain INFO.
- P-side final logs include per-request queue/event/native timing aggregates.

### Benchmark Command

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:18100 \
  --endpoint /v1/completions \
  --model /data/models/Kimi-K2.5 \
  --trust-remote-code \
  --dataset-name random \
  --random-range-ratio 0.1 \
  --random-input-len 16384 \
  --random-output-len 1 \
  --request-rate inf \
  --max-concurrency 4 \
  --num-prompts 50 \
  --save-result \
  --result-dir pd_h20_logs/bench \
  --result-filename proxy-16k-c4-nicdelta-50.json
```

### Serving Result

| run | success | duration_s | req/s | total_tok/s | mean_TTFT_ms | p99_TTFT_ms |
|-----|---------|------------|-------|-------------|--------------|-------------|
| proxy-16k-c4-nicdelta-50 | 50/50 | 114.23 | 0.438 | 7113.63 | 8911.54 | 12200.33 |

Previous reference points from the same debug session. The `batch32768` runs are
diagnostic unless the direct baseline is also restarted with the same fixed 32k
vLLM serving flags.

| run | success | duration_s | req/s | total_tok/s | mean_TTFT_ms | p99_TTFT_ms |
|-----|---------|------------|-------|-------------|--------------|-------------|
| d-baseline-16k | 50/50 | 135.45 | 0.369 | 5999.04 | 2701.34 | 3513.22 |
| proxy-16k-c1-prefill-parallel-batch32768 | 50/50 | 130.52 | 0.383 | 6225.64 | 2609.82 | 2883.63 |
| proxy-16k-c4-prefill-parallel-batch32768-50 | 50/50 | 113.71 | 0.440 | 7145.87 | 8869.98 | 12085.89 |
| proxy-16k-c4-windowfix-batch32768 | 20/20 | 46.62 | 0.429 | 7080.97 | 8726.38 | 11874.28 |

### NIC Counter Result

Counters are from `port_xmit_data` / `port_rcv_data`, converted with 4 bytes per
counter unit. The sampling window was 179.6s and includes idle tail after the
114.23s benchmark. Each NIC moved 115.86GB in the active direction. If all bytes
are attributed to the benchmark window, that is about 8.1Gbps average per NIC.

Prefill node `h20-99` transmit:

| NIC | bytes sent | avg over sample | peak 1s |
|-----|------------|-----------------|---------|
| mlx5_1 | 115.86GB | 5.16Gbps | 19.81Gbps |
| mlx5_2 | 115.86GB | 5.16Gbps | 19.50Gbps |
| mlx5_3 | 115.86GB | 5.16Gbps | 19.81Gbps |
| mlx5_4 | 115.86GB | 5.16Gbps | 19.81Gbps |

Decode node `h20-100` receive:

| NIC | bytes received | avg over sample | peak 1s |
|-----|----------------|-----------------|---------|
| mlx5_1 | 115.86GB | 5.16Gbps | 20.14Gbps |
| mlx5_2 | 115.86GB | 5.16Gbps | 19.96Gbps |
| mlx5_3 | 115.86GB | 5.16Gbps | 20.46Gbps |
| mlx5_4 | 115.86GB | 5.16Gbps | 20.46Gbps |

### Debug Conclusion

- Correctness and request success passed: 50/50 requests completed.
- Four NICs are used evenly. The rank map is effective; this is not a single-NIC
  routing issue.
- Bandwidth saturation did not pass. Per-NIC peak is only about 20Gbps and
  active-window average is about 8.1Gbps, far below the expected H20 RDMA link
  capacity.
- P logs show each rank pushes about 1.1GB per 16k request, but
  `wait_writes_ms` is usually sub-ms to tens of ms while `wait_sender_ms` and
  `schedule_to_imm_ms` are often about 1-2s. Native RDMA submit/complete is not
  the long pole in this run.
- D logs show large `queue_wait_ms` for concurrent requests, often several
  seconds. Some requests later observe IMM almost immediately because the P-side
  push already completed before the D waiter reached them.
- The next performance target should be reducing P-side layer scheduling/event
  wait and D-side RDMA done waiter queueing. Increasing NIC count alone will not
  raise bandwidth until the upper pipeline can feed RDMA continuously.

### Cleanup

After this run, services were stopped on both nodes:

- `h20-99`: proxy stopped, prefill stopped, decode stopped.
- `h20-100`: proxy stopped, prefill stopped, decode stopped.

Logs/results remain on the machines:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.log`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-p.csv`
- `h20-100:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-d.csv`
