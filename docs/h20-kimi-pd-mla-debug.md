# H20 Kimi K2.5 PD MLA Debug Experiment

## Summary

This experiment debugged two-node PD MLA transfer performance for Kimi K2.5 on
H20. The run passed request correctness but did not saturate RDMA bandwidth.

- Date: 2026-05-28
- Branch: `docs/pd-mla-design`
- Code commit under test: `874e0c6 fix: improve pd rdma push throughput`
- Documentation commit: `0851fd0 docs: record h20 pd mla benchmark results`
- Prefill/proxy node: `h20-99`
- Decode node: `h20-100`
- Model: `/data/models/Kimi-K2.5`
- Latest fixed c1 P/D sweep result: 50/50 requests completed at every tested
  input length. 16k mean TTFT is 2429.02ms; 30k mean TTFT is 4848.75ms.
- RDMA result: all 4 NICs were used evenly, but serving peak bandwidth was
  only about 12-14Gbps per NIC
- Benchmark discipline after this run: use `--max-concurrency 1` for acceptance
  pressure tests, and compare only runs started with the same non-connector vLLM
  flags. The c4 run below is kept only as a queueing diagnostic.

The main conclusion is that the 4-NIC rank map is working, but the upper P/D
pipeline is not feeding RDMA continuously enough to fill the links.

## Setup

The P/D services in this historical run were started with:

- TP=8 on both prefill and decode
- `--load-format dummy`
- `--max-num-batched-tokens 32768`
- no explicit `--block-size`
- no explicit `--max-model-len`
- `--no-enable-prefix-caching`

Rank-to-NIC map:

| ranks | NIC |
|-------|-----|
| 0, 1 | `mlx5_1` |
| 2, 3 | `mlx5_2` |
| 4, 5 | `mlx5_3` |
| 6, 7 | `mlx5_4` |

Service endpoints:

| role | node | endpoint |
|------|------|----------|
| proxy | h20-99 | `http://127.0.0.1:18100` |
| prefill | h20-99 | `http://10.96.191.99:18101` |
| decode | h20-100 | `http://10.96.191.100:18102` |

The services were stopped after the run.

## Fixed Startup Contract For The Next Sweep

The final comparison must keep the vLLM serving shape fixed. Baseline and P/D
proxy runs should use the same non-connector flags:

```bash
vllm serve /data/models/Kimi-K2.5 \
  --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --load-format dummy \
  --max-num-batched-tokens 32768 \
  --trust-remote-code \
  --no-enable-prefix-caching
```

The direct baseline starts this command without `--kv-transfer-config`. The P/D
run starts the same command on P and D with the PdConnector
`--kv-transfer-config`, plus the proxy. Do not compare a default-baseline run
against a P/D run with a different scheduler/batching shape.

The launch script now has a dedicated direct-baseline role for this:

```bash
ssh h20-100 'cd /root/develop/xingming/pegaflow && scripts/run_pd_h20_kimi.sh stop && scripts/run_pd_h20_kimi.sh start-baseline'
```

For P/D proxy runs:

```bash
ssh h20-100 'cd /root/develop/xingming/pegaflow && scripts/run_pd_h20_kimi.sh stop'
cd /root/develop/xingming/pegaflow
scripts/run_pd_h20_kimi.sh stop
scripts/run_pd_h20_kimi.sh start-cluster
```

## Code Changes Tested

- D-to-P prefill HTTP dispatch uses 8 `AsyncPrefillSender` workers.
- P-side RDMA layer push now uses one FIFO sender thread per worker. vLLM
  computes layers in order; extra Python senders only wait on later CUDA events
  and add queueing noise.
- P-side layer push now uses scheduler-provided block ids instead of reading
  `slot_mapping` back to CPU.
- D-side rank0 prefill dispatch caches compact per-rank layer templates after
  peer layout gather and only fills shared `block_ids` per request.
- D-side local RDMA wait registration now opens a minimal one-layer/one-block
  request while still sending the full per-rank handshake set to P.
- Native RDMA write window reservation was changed to reserve with CAS before
  submit, avoiding write-window overrun under concurrent Python push threads.
- Hot per-layer RDMA write logs were moved to DEBUG.
- Per-request P-side final logs now include queue/event/native timing aggregates.

## Benchmark Command Actually Run

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

This was a diagnostic c4 run to expose queueing. It should not be used as the
acceptance benchmark shape.

## Acceptance Sweep Command

Future H20 PD MLA pressure tests should use concurrency 1 and only sweep input
length. The helper script keeps the benchmark shape fixed:

```bash
LENGTHS="1024 4096 8192 16384 30000" scripts/run_h20_kimi_ttft_sweep.sh baseline
LENGTHS="1024 4096 8192 16384 30000" scripts/run_h20_kimi_ttft_sweep.sh proxy
```

Both modes use:

- `--max-concurrency 1`
- `--request-rate inf`
- `--random-range-ratio 0.0`
- `--random-output-len 1`
- `--num-prompts 50`
- `--temperature 0`

Proxy mode also samples `mlx5_1` through `mlx5_4` on both nodes and writes NIC
summary files next to the benchmark JSON files.

Build the final table from the generated JSON and NIC summaries:

```bash
uv run --no-project python scripts/summarize_h20_kimi_ttft_sweep.py \
  pd_h20_logs/bench/ttft-sweep
```

## Fixed 32k C1 Sweep Progress

The direct baseline leg was run on 2026-05-29 with the fixed serving contract:

- baseline node: `h20-100`
- benchmark client: `h20-99`
- vLLM flags: `--load-format dummy`, `--max-num-batched-tokens 32768`,
  no explicit `--block-size`, no explicit `--max-model-len`
- vLLM resolved max model length: `262144`
- benchmark: `--max-concurrency 1`, `--request-rate inf`,
  `--random-range-ratio 0.0`, `--random-output-len 1`, `--num-prompts 50`

The P/D proxy leg was rerun for all input lengths after the single FIFO sender,
compact handshake, compact JSON, scheduler-block push, D handshake template
cache, and minimal D wait-handshake changes.

| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | baseline_success | proxy_success | baseline_req_s | proxy_req_s | proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |
|-----------|-----------------------|-----------------------|----------|-----------|-----------------------|-------------------|------------------|---------------|----------------|-------------|-----------------------------|------------------------------|-------|
| 1024 | 158.67 | 199.19 | 40.51 | 25.53% | 235.17 | 448.19 | 50/50 | 50/50 | 6.30 | 5.01 | 3.08 | 6.37 | proxy label `kimi-proxy-fixed32k-waitmini` |
| 4096 | 553.42 | 600.52 | 47.10 | 8.51% | 559.53 | 782.72 | 50/50 | 50/50 | 1.81 | 1.66 | 5.79 | 8.35 | proxy label `kimi-proxy-fixed32k-waitmini` |
| 8192 | 1111.23 | 1183.15 | 71.92 | 6.47% | 1120.30 | 1431.78 | 50/50 | 50/50 | 0.90 | 0.84 | 6.48 | 8.80 | proxy label `kimi-proxy-fixed32k-waitmini` |
| 16384 | 2334.77 | 2429.02 | 94.25 | 4.04% | 2346.75 | 2809.90 | 50/50 | 50/50 | 0.43 | 0.41 | 6.71 | 13.62 | proxy label `kimi-proxy-fixed32k-waitmini` |
| 30000 | 4728.88 | 4848.75 | 119.87 | 2.53% | 4738.49 | 4894.93 | 50/50 | 50/50 | 0.21 | 0.21 | 6.38 | 10.56 | proxy label `kimi-proxy-fixed32k-waitmini` |

Artifacts:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-baseline-fixed32k-in1024-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-baseline-fixed32k-in4096-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-baseline-fixed32k-in8192-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-baseline-fixed32k-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-baseline-fixed32k-in30000-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-compacths2-singlefifo-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-compacths2-singlefifo-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-compacths2-singlefifo-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-jsoncompact-singlefifo-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-jsoncompact-singlefifo-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-jsoncompact-singlefifo-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-trace-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-trace-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-trace-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in1024-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in4096-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in8192-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in30000-out1-c1-n50-seed20260528.json`

## Rejected Early-Prefill Experiment

An experimental proxy-side early-prefill path was tested on 2026-05-29 and then
reverted. The design started P prefill from the proxy immediately, let D
allocate KV blocks, and sent a small handshake update to P once D had the RDMA
handshakes. It added a delayed-handshake state machine on P but did not improve
the 16k/c1 acceptance result in a meaningful way.

The first 16k run exposed a request-body bug: P-early was forced to
`stream=false` while still receiving `stream_options`, so P returned HTTP 400
and D RDMA waits timed out after 30s. After removing `stream_options` from the
P-early body, the 16k/c1 run completed. Mean TTFT changed by only -1.09ms versus
the stable waitmini run, while p99 was worse.

| run | success | mean_TTFT_ms | p50_TTFT_ms | p99_TTFT_ms | req_s | avg_RDMA_Gbps_per_NIC | peak_RDMA_Gbps_per_NIC | verdict |
|-----|---------|--------------|-------------|-------------|-------|------------------------|-------------------------|---------|
| `kimi-proxy-fixed32k-waitmini` | 50/50 | 2429.02 | 2380.71 | 2809.90 | 0.41 | 6.71 | 12.14 | kept |
| `kimi-proxy-fixed32k-earlyp2` | 50/50 | 2427.93 | 2374.32 | 3040.55 | 0.41 | 6.71 | 18.52 | reverted |

Artifacts:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-earlyp2-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-earlyp2-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-earlyp2-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`

## RDMA-only Integration Test

Before another connector change, the RDMA path was isolated with
`scripts/pd_rdma_two_node_it.py`. The test uses the same 8-rank Kimi 16k shape
as the P/D run: 61 layers, 1024 blocks, 18432 bytes per block, and the same
rank-to-NIC map.

The current IT command fails the run when aggregate bandwidth is below
1000Gbps, when any active NIC direction is below 300Gbps, when any active NIC
direction moves less than 0.98x of the expected bytes, or when D-side sampled
payload bytes do not match the P-side rank/offset deterministic pattern. D
resets sampled destination ranges before every iteration and verifies them
immediately after that iteration completes, so a later skipped transfer cannot be
hidden by a previous successful transfer.

```bash
python scripts/pd_rdma_two_node_it.py decode \
  --iterations 4 \
  --min-bandwidth-gbps 1000 \
  --min-nic-gbps 300 \
  --min-nic-byte-ratio 0.98 \
  --verify-sample-bytes 1048576

python scripts/pd_rdma_two_node_it.py prefill \
  --decode-host 10.96.191.100 \
  --iterations 4 \
  --min-bandwidth-gbps 1000 \
  --min-nic-gbps 300 \
  --min-nic-byte-ratio 0.98 \
  --verify-sample-bytes 1048576
```

Verified result on h20 on 2026-05-29:

| side | bytes | elapsed_ms | aggregate_Gbps | per-NIC active direction |
|------|-------|------------|----------------|--------------------------|
| P push | 36.84GB | 238.63 | 1235.16 | 313.16Gbps xmit on each of `mlx5_1..4` |
| D receive | 36.84GB | 239.19 | 1232.26 | 312.43Gbps recv on each of `mlx5_1..4` |

Each NIC moved 9.34GB in the active direction versus 9.21GB expected. D sampled
25.17MB per iteration, 100.66MB total across 4 iterations, and passed
deterministic payload verification after every iteration.

Artifacts:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/rdma-it-kimi16k-itercheck-iters4-prefill.json`
- `h20-100:/root/develop/xingming/pegaflow/pd_h20_logs/bench/rdma-it-kimi16k-itercheck-iters4-decode.json`

This rules out RDMA engine bandwidth as the current bottleneck. The low NIC
average seen in vLLM pressure runs is caused by upper-layer submission cadence
and prefill overlap, not by a one-NIC routing issue or slow native RDMA writes.

## Serving Result

| run | success | duration_s | req/s | total_tok/s | mean_TTFT_ms | p99_TTFT_ms |
|-----|---------|------------|-------|-------------|--------------|-------------|
| `proxy-16k-c4-nicdelta-50` | 50/50 | 114.23 | 0.438 | 7113.63 | 8911.54 | 12200.33 |

Reference runs from the same debug session. The `batch32768` runs are useful
diagnostic evidence, but they are not a valid baseline-aligned sweep unless the
direct baseline is restarted with the same fixed 32k vLLM serving flags.

| run | success | duration_s | req/s | total_tok/s | mean_TTFT_ms | p99_TTFT_ms |
|-----|---------|------------|-------|-------------|--------------|-------------|
| `d-baseline-16k` | 50/50 | 135.45 | 0.369 | 5999.04 | 2701.34 | 3513.22 |
| `proxy-16k-c1-prefill-parallel-batch32768` | 50/50 | 130.52 | 0.383 | 6225.64 | 2609.82 | 2883.63 |
| `kimi-proxy-fixed32k-singlefifo-in16384-out1-c1-n50-seed20260528` | 50/50 | 130.68 | 0.383 | 6269.62 | 2613.22 | 2772.61 |
| `kimi-proxy-fixed32k-compacths-singlefifo-in16384-out1-c1-n50-seed20260528` | 50/50 | 127.20 | 0.393 | 6439.95 | 2543.45 | 3119.31 |
| `kimi-proxy-fixed32k-compacths2-singlefifo-in16384-out1-c1-n50-seed20260528` | 50/50 | 126.48 | 0.395 | 6477.43 | 2529.09 | 2917.96 |
| `kimi-proxy-fixed32k-jsoncompact-singlefifo-in16384-out1-c1-n50-seed20260528` | 50/50 | 125.94 | 0.397 | 6505.33 | 2518.21 | 3016.47 |
| `kimi-proxy-fixed32k-schedblocks-in16384-out1-c1-n50-seed20260528` | 50/50 | 124.32 | 0.402 | 6589.97 | 2485.86 | 2986.99 |
| `kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528` | 50/50 | 122.47 | 0.408 | 6689.38 | 2448.93 | 2897.89 |
| `kimi-proxy-fixed32k-trace-in16384-out1-c1-n50-seed20260528` | 50/50 | 122.05 | 0.410 | 6712.22 | 2440.53 | 2845.69 |
| `kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528` | 50/50 | 121.48 | 0.412 | 6744.16 | 2429.02 | 2809.90 |
| `proxy-16k-c4-prefill-parallel-batch32768-50` | 50/50 | 113.71 | 0.440 | 7145.87 | 8869.98 | 12085.89 |
| `proxy-16k-c4-windowfix-batch32768` | 20/20 | 46.62 | 0.429 | 7080.97 | 8726.38 | 11874.28 |

## Final Table

| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | baseline_success | proxy_success | baseline_req_s | proxy_req_s | proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |
|-----------|-----------------------|-----------------------|----------|-----------|-----------------------|-------------------|------------------|---------------|----------------|-------------|-----------------------------|------------------------------|-------|
| 1024 | 158.67 | 199.19 | 40.51 | 25.53% | 235.17 | 448.19 | 50/50 | 50/50 | 6.30 | 5.01 | 3.08 | 6.37 | minimal D wait handshake |
| 4096 | 553.42 | 600.52 | 47.10 | 8.51% | 559.53 | 782.72 | 50/50 | 50/50 | 1.81 | 1.66 | 5.79 | 8.35 | minimal D wait handshake |
| 8192 | 1111.23 | 1183.15 | 71.92 | 6.47% | 1120.30 | 1431.78 | 50/50 | 50/50 | 0.90 | 0.84 | 6.48 | 8.80 | minimal D wait handshake |
| 16384 | 2334.77 | 2429.02 | 94.25 | 4.04% | 2346.75 | 2809.90 | 50/50 | 50/50 | 0.43 | 0.41 | 6.71 | 13.62 | minimal D wait handshake |
| 30000 | 4728.88 | 4848.75 | 119.87 | 2.53% | 4738.49 | 4894.93 | 50/50 | 50/50 | 0.21 | 0.21 | 6.38 | 10.56 | minimal D wait handshake |

## NIC Counter Result

The NIC counters use `port_xmit_data` and `port_rcv_data`, converted with 4 bytes
per counter unit. The sampling window was 179.6s and includes idle tail after the
114.23s benchmark. Each NIC moved 115.86GB in the active direction. If all bytes
are attributed to the benchmark window, the active-window average is about
8.1Gbps per NIC.

Prefill node `h20-99` transmit:

| NIC | bytes sent | avg over sample | peak 1s |
|-----|------------|-----------------|---------|
| `mlx5_1` | 115.86GB | 5.16Gbps | 19.81Gbps |
| `mlx5_2` | 115.86GB | 5.16Gbps | 19.50Gbps |
| `mlx5_3` | 115.86GB | 5.16Gbps | 19.81Gbps |
| `mlx5_4` | 115.86GB | 5.16Gbps | 19.81Gbps |

Decode node `h20-100` receive:

| NIC | bytes received | avg over sample | peak 1s |
|-----|----------------|-----------------|---------|
| `mlx5_1` | 115.86GB | 5.16Gbps | 20.14Gbps |
| `mlx5_2` | 115.86GB | 5.16Gbps | 19.96Gbps |
| `mlx5_3` | 115.86GB | 5.16Gbps | 20.46Gbps |
| `mlx5_4` | 115.86GB | 5.16Gbps | 20.46Gbps |

Latest fixed 16k/c1 `waitmini` run: each P NIC transmitted 116.76GB over a
139.3s monitor window, averaging 6.71Gbps per NIC with max 1s peak 12.14Gbps.
Each D NIC received 116.76GB over a 139.4s window, averaging 6.70Gbps per NIC
with max 1s peak 13.62Gbps.

Re-reading the fixed c1 sweep CSVs with a `>1Gbps` active-sample filter shows
the low serving bandwidth is not just idle tail dilution:

| input_len | P xmit active mean | P xmit active p95 | P xmit max 1s | D recv active mean | D recv active p95 | D recv max 1s |
|-----------|--------------------|-------------------|---------------|--------------------|-------------------|---------------|
| 1024 | 6.03Gbps | 6.36Gbps | 6.37Gbps | 7.18Gbps | 8.18Gbps | 8.80Gbps |
| 4096 | 7.70Gbps | 8.35Gbps | 8.35Gbps | 7.24Gbps | 8.12Gbps | 8.80Gbps |
| 8192 | 7.79Gbps | 8.80Gbps | 8.80Gbps | 7.17Gbps | 8.02Gbps | 10.00Gbps |
| 16384 | 7.71Gbps | 8.50Gbps | 12.14Gbps | 7.83Gbps | 14.27Gbps | 18.53Gbps |
| 30000 | 7.04Gbps | 7.78Gbps | 10.00Gbps | 7.04Gbps | 7.79Gbps | 10.56Gbps |

For 16k/c1, this is the expected ceiling of the current layer-wise serving
shape. Each rank pushes 1,151,336,448 bytes. At the verified RDMA-only rate of
about 313Gbps/NIC, that payload would take 29.4ms if submitted as a ready batch.
In serving, the same bytes become ready over the prefill layer window, about
1.18s, which gives 7.83Gbps/NIC. The gap is about 40x, so c1 serving cannot
saturate the NICs unless the system has many more independent prefills in flight,
deliberately batches ready KV at the cost of overlap, or changes the P/D contract
around D-side completion.

## Log Evidence

P-side final logs show about 1.1GB pushed per rank for a 16k request. However,
native write completion is not the long section:

- `wait_writes_ms`: usually sub-ms to tens of ms
- `wait_sender_ms`: often about 1-2s
- `schedule_to_imm_ms`: often about 1-2s
- per-request `tail_gbps`: often about 8-9Gbps, sometimes lower under c4 queueing
- future P-side logs include `ready_window_gbps`, `link_gbps`, and
  `ready_link_util_pct` to show how much of the selected NIC link is fed by
  layer-ready KV

D-side wait logs show large queueing under concurrency:

- `queue_wait_ms`: often several seconds
- Some requests later see near-zero `wait_ms`, because the P-side IMM already
  arrived before the D waiter reached that request.

For the fixed 16k/c1 `handshakecache` run:

- P-side `wait_writes_ms`: p50 0.83ms, p95 0.89ms.
- P-side `push_native_avg_ms`: p50 0.03ms, p95 0.04ms per layer push.
- P-side `wait_sender_ms`: p50 1036.67ms. This is dominated by waiting for
  CUDA layer events and the prefill tail, not RDMA completion.
- D rank0 prefill dispatch: p50 0.07ms total, p95 0.14ms.
- D rank0 handshake build inside that dispatch: p50 0.04ms, p95 0.10ms.
- Proxy accept to D rank0 dispatch: p50 108.94ms, p95 125.41ms.
- Proxy accept to D RDMA done: p50 2401.43ms, p95 2427.77ms.
- D RDMA done to first proxy chunk: p50 23.63ms, p95 36.30ms.
- D-to-P HTTP payload after compact handshake and compact JSON: p50 280KB,
  p95 295KB. Before the compact handshake this was about 3.76MB.

For the fixed 16k/c1 `trace` run:

- `proxy_to_matched_ms`: p50 95.79ms, p95 104.66ms.
- `matched_to_wait_ms`: p50 0.28ms, p95 0.37ms.
- Rank0 `matched_to_worker_ms`: p50 9.04ms, p95 10.04ms.
- Rank0 `open_request_ms`: p50 7.34ms, p95 8.35ms.
- `proxy_to_dispatch_ms`: p50 104.96ms, p95 113.77ms.
- `matched_to_dispatch_ms`: p50 9.16ms, p95 10.13ms.
- D rank0 dispatch body itself remains small: p50 0.07ms, p95 0.12ms.

For the fixed 16k/c1 `waitmini` run:

- `proxy_to_matched_ms`: p50 94.86ms, p95 102.48ms.
- `matched_to_wait_ms`: p50 0.29ms, p95 0.37ms.
- Rank0 `open_request_ms`: p50 0.27ms, p95 0.42ms.
- `proxy_to_dispatch_ms`: p50 96.83ms, p95 104.99ms.
- `matched_to_dispatch_ms`: p50 1.99ms, p95 2.75ms.
- Proxy stream header latency: p50 91.34ms, p95 98.55ms.
- Proxy TTFT: p50 2409.07ms, p95 2444.66ms.
- P-side `wait_sender_ms`: p50 1037.43ms.
- P-side `wait_writes_ms`: p50 0.84ms.
- P-side `push_native_avg_ms`: p50 0.03ms per layer push.

After scheduler-block push, cached D handshake templates, and minimal D wait
registration, D-side handshake/open registration is no longer material in the
TTFT delta. The waitmini run shows that most fixed overhead is already present
before D's scheduler-side `get_num_new_matched_tokens` callback runs. The main
remaining target is therefore D's vLLM HTTP/API to EngineCore ingress before
scheduler matching; one known later component is that vLLM decrements a full
external KV hit by one token, so D recomputes the last prompt token to produce
sampling logits.

The last-token recompute is a vLLM scheduler invariant, not a value that the
PdConnector can change by reporting more matched tokens. In local vLLM
`v1/core/sched/scheduler.py`, `_update_waiting_for_remote_kv()` caches the
external blocks and then changes a full prompt hit from `num_tokens` to
`num_tokens - 1` before the request is promoted back to `WAITING`. Removing that
without a vLLM-side replacement path would leave the request with no token to
schedule, so the connector-only options are exhausted here. Avoiding this cost
requires changing the product contract: either D must receive logits/hidden
state from P, or the proxy/API path must use P's first generated token and hand
the continuation to D.

The low serving RDMA average should also be read together with this overlap
model. Native RDMA can move the same Kimi 16k shape at 1.23Tbps when the test
submits a contiguous transfer. In the real serving path, layer-wise push sends
KV as each CUDA layer event becomes ready, so the NICs are idle between layer
readiness events. This is good for TTFT overlap but means the whole-request NIC
average will not resemble the RDMA-only peak unless multiple independent
prefills are in flight or the design intentionally batches the tail transfer.

## TCP and HTTP Microbench

To check whether the remaining fixed 16k/c1 delta is caused by slow TCP, a
temporary Python echo server was run on each H20 node on 2026-05-29. The payload
sizes cover the real proxy-to-D body size (~170KB) and D-to-P body size (~272KB).
The HTTP client used `urllib.request`, matching the proxy and D-to-P sender style
more closely than a pooled async client.

| direction | payload | TCP p50_ms | TCP p95_ms | HTTP p50_ms | HTTP p95_ms |
|-----------|---------|------------|------------|-------------|-------------|
| h20-99 -> h20-100 | 170KB | 0.27 | 0.37 | 0.71 | 1.06 |
| h20-99 -> h20-100 | 272KB | 0.22 | 0.35 | 0.81 | 1.28 |
| h20-100 -> h20-99 | 170KB | 0.19 | 0.28 | 0.77 | 1.12 |
| h20-100 -> h20-99 | 272KB | 0.24 | 0.32 | 1.05 | 1.84 |

The measured TCP/HTTP cost is not large enough to explain the real P/D timing:
in the latest fixed 16k/c1 run, proxy accept to D scheduler matching was p50
94.86ms and p95 102.48ms. The remaining delta is therefore application-layer
serialization before P starts: proxy sends only D first, D reaches scheduler
allocation, then D dispatches the P prefill request. Since the P-side prefill
HTTP latency is already approximately the direct baseline TTFT, this serialized
pre-P-start section currently shows up almost directly as P/D TTFT overhead.

## Local vLLM/Proxy Microbench

Local checks on `/data/models/Kimi-K2.6` with the same random 16k benchmark
shape show why D ingress is a plausible fixed-cost source:

| local check | p50_ms | p95_ms | notes |
|-------------|--------|--------|-------|
| Kimi tokenizer encode 16k random prompt | 55.81 | 61.05 | `transformers.AutoTokenizer`, `add_special_tokens=True` |
| proxy two-hop text prompt | 2.52 | 2.94 | fake D server, local loopback |
| proxy two-hop token-list prompt before shallow-copy change | 11.36 | 11.97 | dominated by list `deepcopy` |
| proxy two-hop token-list prompt after shallow-copy change | 5.64 | 6.30 | top-level request copy only |

The benchmark client sends string prompts for the OpenAI completions path, so
this proxy microbench does not by itself explain the H20 P/D overhead. It does
confirm that tokenized request bodies avoid tokenizer work at the cost of JSON
list parsing/copying, and that proxy forwarding is not a tens-of-ms component
for string prompts.

## Conclusion

Correctness passed, and all 4 NICs were used evenly. The c4 diagnostic result
did not pass the bandwidth target: peak was only about 20Gbps per NIC, far below
the expected H20 RDMA link capacity. The RDMA-only integration test reaches
about 1.23Tbps aggregate, so native RDMA bandwidth is available when the upper
pipeline feeds it continuously.

The evidence points away from a single-NIC routing issue. The next performance
work should focus on the remaining upper-layer latency:

- decide whether TTFT should use P's first generated token or transfer
  logits/hidden state so D does not recompute the last prompt token;
- reduce proxy-to-D header latency and D/proxy request setup before rank0
  dispatch;
- keep P-side layer-wise push as one FIFO sender unless a future design can
  prove readiness-aware scheduling without waiting on later CUDA events.

## Artifact Paths

Remote result files:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.log`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-p.csv`
- `h20-100:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-d.csv`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-handshakecache-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528-h20-99-nic-summary.txt`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/ttft-sweep/kimi-proxy-fixed32k-waitmini-in16384-out1-c1-n50-seed20260528-h20-100-nic-summary.txt`

Service cleanup was verified after the experiment:

- `h20-99`: proxy stopped, prefill stopped, decode stopped.
- `h20-100`: proxy stopped, prefill stopped, decode stopped.
