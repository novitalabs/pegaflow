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
- Result: 50/50 requests completed, 7113.63 total tok/s
- RDMA result: all 4 NICs were used evenly, but peak bandwidth was only about
  20Gbps per NIC
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

The P/D proxy leg has only been rerun for 16k after the single FIFO sender,
compact handshake, compact JSON, scheduler-block push, and D handshake template
cache changes. Other input lengths are still pending.

| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | baseline_success | proxy_success | baseline_req_s | proxy_req_s | proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |
|-----------|-----------------------|-----------------------|----------|-----------|-----------------------|-------------------|------------------|---------------|----------------|-------------|-----------------------------|------------------------------|-------|
| 1024 | 158.67 | TBD | TBD | TBD | 235.17 | TBD | 50/50 | TBD | 6.30 | TBD | TBD | TBD | missing proxy |
| 4096 | 553.42 | TBD | TBD | TBD | 559.53 | TBD | 50/50 | TBD | 1.81 | TBD | TBD | TBD | missing proxy |
| 8192 | 1111.23 | TBD | TBD | TBD | 1120.30 | TBD | 50/50 | TBD | 0.90 | TBD | TBD | TBD | missing proxy |
| 16384 | 2334.77 | 2440.53 | 105.76 | 4.53% | 2346.75 | 2845.69 | 50/50 | 50/50 | 0.43 | 0.41 | 6.68 | 12.14 | proxy label `kimi-proxy-fixed32k-trace` |
| 30000 | 4728.88 | TBD | TBD | TBD | 4738.49 | TBD | 50/50 | TBD | 0.21 | TBD | TBD | TBD | missing proxy |

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
| `proxy-16k-c4-prefill-parallel-batch32768-50` | 50/50 | 113.71 | 0.440 | 7145.87 | 8869.98 | 12085.89 |
| `proxy-16k-c4-windowfix-batch32768` | 20/20 | 46.62 | 0.429 | 7080.97 | 8726.38 | 11874.28 |

## Final Table Shape

The final experiment table should be keyed by input length:

| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | baseline_success | proxy_success | baseline_req_s | proxy_req_s | proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |
|-----------|-----------------------|-----------------------|----------|-----------|-----------------------|-------------------|------------------|---------------|----------------|-------------|-----------------------------|------------------------------|-------|
| 1024 | 158.67 | TBD | TBD | TBD | 235.17 | TBD | 50/50 | TBD | 6.30 | TBD | TBD | TBD | missing proxy |
| 4096 | 553.42 | TBD | TBD | TBD | 559.53 | TBD | 50/50 | TBD | 1.81 | TBD | TBD | TBD | missing proxy |
| 8192 | 1111.23 | TBD | TBD | TBD | 1120.30 | TBD | 50/50 | TBD | 0.90 | TBD | TBD | TBD | missing proxy |
| 16384 | 2334.77 | 2440.53 | 105.76 | 4.53% | 2346.75 | 2845.69 | 50/50 | 50/50 | 0.43 | 0.41 | 6.68 | 12.14 | scheduler trace instrumentation |
| 30000 | 4728.88 | TBD | TBD | TBD | 4738.49 | TBD | 50/50 | TBD | 0.21 | TBD | TBD | TBD | missing proxy |

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

Latest fixed 16k/c1 `trace` run: each P NIC transmitted 116.27GB over a
139.3s monitor window, averaging 6.68Gbps per NIC with max 1s peak 12.14Gbps.
Each D NIC received 116.76GB over a 140.2s window, averaging 6.66Gbps per NIC
with max 1s peak 9.06Gbps.

## Log Evidence

P-side final logs show about 1.1GB pushed per rank for a 16k request. However,
native write completion is not the long section:

- `wait_writes_ms`: usually sub-ms to tens of ms
- `wait_sender_ms`: often about 1-2s
- `schedule_to_imm_ms`: often about 1-2s
- per-request `tail_gbps`: often about 8-9Gbps, sometimes lower under c4 queueing

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

After scheduler-block push and cached D handshake templates, D-side handshake
construction is no longer material in the TTFT delta. The trace run shows that
most fixed overhead is already present before D's scheduler-side
`get_num_new_matched_tokens` callback runs. Scheduler wait construction is
sub-ms, and scheduler-to-worker dispatch adds about 9ms, mostly RDMA
`open_request`. The main remaining target is therefore D's vLLM HTTP/API to
EngineCore ingress before scheduler matching; one known later component is that
vLLM decrements a full external KV hit by one token, so D recomputes the last
prompt token to produce sampling logits.

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
in the latest fixed 16k/c1 run, proxy accept to D rank0 dispatch was p50
108.94ms and p95 125.41ms. The remaining delta is therefore application-layer
serialization before P starts: proxy sends only D first, D reaches scheduler
allocation, then D dispatches the P prefill request. Since the P-side prefill
HTTP latency is already approximately the direct baseline TTFT, this serialized
pre-P-start section currently shows up almost directly as P/D TTFT overhead.

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

Service cleanup was verified after the experiment:

- `h20-99`: proxy stopped, prefill stopped, decode stopped.
- `h20-100`: proxy stopped, prefill stopped, decode stopped.
