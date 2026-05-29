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

- D-side prefill HTTP dispatch was parallelized with 8 sender threads.
- P-side RDMA layer push was parallelized with 4 sender threads per worker.
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
| `proxy-16k-c4-prefill-parallel-batch32768-50` | 50/50 | 113.71 | 0.440 | 7145.87 | 8869.98 | 12085.89 |
| `proxy-16k-c4-windowfix-batch32768` | 20/20 | 46.62 | 0.429 | 7080.97 | 8726.38 | 11874.28 |

## Final Table Shape

The final experiment table should be keyed by input length:

| input_len | baseline_mean_TTFT_ms | proxy_PD_mean_TTFT_ms | delta_ms | delta_pct | baseline_p99_TTFT_ms | proxy_p99_TTFT_ms | proxy_avg_RDMA_Gbps_per_NIC | proxy_peak_RDMA_Gbps_per_NIC | notes |
|-----------|-----------------------|-----------------------|----------|-----------|-----------------------|-------------------|-----------------------------|------------------------------|-------|
| 1024 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed 32k, c1 |
| 4096 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed 32k, c1 |
| 8192 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed 32k, c1 |
| 16384 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed 32k, c1 |
| 30000 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed 32k, c1 |

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

## Conclusion

Correctness passed, and all 4 NICs were used evenly. The result did not pass the
bandwidth target: peak was only about 20Gbps per NIC, far below the expected H20
RDMA link capacity.

The evidence points away from a single-NIC routing issue. The next performance
work should focus on making the upper pipeline feed RDMA continuously:

- reduce P-side layer scheduling and CUDA event wait gaps;
- reduce P-side per-request push queue buildup;
- reduce D-side RDMA done waiter queueing;
- then rerun NIC counter sampling to verify sustained per-NIC bandwidth.

## Artifact Paths

Remote result files:

- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.json`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/proxy-16k-c4-nicdelta-50.log`
- `h20-99:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-p.csv`
- `h20-100:/root/develop/xingming/pegaflow/pd_h20_logs/bench/nic-proxy-c4-50-d.csv`

Service cleanup was verified after the experiment:

- `h20-99`: proxy stopped, prefill stopped, decode stopped.
- `h20-100`: proxy stopped, prefill stopped, decode stopped.
