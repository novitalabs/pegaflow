# PegaFlow P/D Disaggregation Design

> **⚠️ Experimental** — This feature is functional but not recommended for production deployments.

## Overview

Prefill/Decode disaggregation separates the prefill (P) and decode (D) phases to different vLLM instances, improving resource utilization.

```
┌────────┐         ┌────────┐         ┌────────┐
│ Router │ ──1──→  │   P    │         │   D    │
│        │ ←─2───  │        │         │        │
│        │         │ async  │         │        │
│        │         │ save   │         │        │
│        │ ←─3───  │ done!  │         │        │
│        │ ──4──────────────────────→ │        │
└────────┘         └────────┘         └────────┘
            ↓               ↓
         PegaEngine (shared CPU storage)
```

## Flow

1. Router sends request to P node (max_tokens=1)
2. P returns first token immediately (non-blocking)
3. P's save worker completes async KV write, callbacks Router
4. Router receives callback, forwards request to D node
5. D node's `get_num_new_matched_tokens()` queries PegaEngine, finds KV exists
6. D loads KV via `start_load_kv()` and continues decode

## Key Design Decisions

### Why Callback Instead of Blocking

`wait_for_save()` blocking would hurt throughput. Callback allows P to continue processing other requests while KV is being saved.

### Why Router Doesn't Need block_hashes

D node receives the same prompt, computes the same block_hashes (via vLLM's internal logic), and queries PegaEngine directly. No need to pass block_hashes through Router.

### Multi-P Multi-D Support

As long as all P/D instances:
- Connect to the same PegaEngine
- Use the same TP size
- Use the same block_size

Router only needs to do load balancing.

## Implementation

### Environment Variables

```bash
# P node
PEGAFLOW_ROUTER_ENDPOINT=http://router:8080

# D node (no special config needed)
```

### Connector Changes (planned, not yet implemented)

The async callback path (`_notify_router` → `/kv_ready`) is not yet implemented in the connector.
The current Router uses a synchronous flow: it waits for P's HTTP response before forwarding to D.

### Router Implementation

The Rust router lives at `pegaflow-server/src/bin/pegaflow-router.rs`. It is a standalone binary (not part of the default build) that can be run with:

```bash
cargo run --release --bin pegaflow-router -- \
    --prefill http://p-node:8000 \
    --decode http://d-node:8001
```

See `examples/run_vllm_pd_with_pega.py` for a complete multi-GPU launch script.

## Benchmark Results (H800, Qwen3-8B, 5K input tokens)

| Configuration  | TTFT mean (ms) | TPOT mean (ms) | TPOT p99 (ms) | ITL p99 (ms) |
| -------------- | -------------- | -------------- | ------------- | ------------ |
| P/D (1P+1D)    | 573.78         | 15.68          | 15.89         | 21.71        |
| Baseline (DP2) | 438.24         | 22.67          | 24.32         | 142.70       |

The P/D setup trades higher TTFT for **significantly more stable decode latency** — TPOT p99 drops from 24.32ms to 15.89ms, and ITL p99 improves dramatically from 142.70ms to 21.71ms.

## Limitations

- Router uses synchronous P→D handoff (no async KV-ready callback yet)
- No built-in timeout/retry for P or D node failures
- No Prometheus metrics for P/D latency breakdown
