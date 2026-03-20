# PegaFlow P/D Disaggregation Design

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

### Connector Changes (connector.py)

```python
class PegaKVConnector:
    def __init__(self, ...):
        self._router_endpoint = os.environ.get("PEGAFLOW_ROUTER_ENDPOINT")

    def _decrement_layer_counter(self, request_ids: list[str]) -> None:
        with self._save_completion_lock:
            for req_id in request_ids:
                if req_id in self._req_pending_layers:
                    self._req_pending_layers[req_id] -= 1

                    if self._req_pending_layers[req_id] == 0:
                        self._completed_saves.add(req_id)
                        del self._req_pending_layers[req_id]

                        # Callback to router
                        if self._router_endpoint:
                            self._notify_router(req_id)

    def _notify_router(self, req_id: str):
        url = f"{self._router_endpoint}/kv_ready"
        try:
            requests.post(url, json={"request_id": req_id}, timeout=1.0)
        except Exception as e:
            logger.warning(f"Failed to notify router: {e}")
```

### Router Implementation

```python
class PegaPDRouter:
    def __init__(self, prefill_endpoints: list[str], decode_endpoints: list[str]):
        self._pending_requests: dict[str, asyncio.Event] = {}

    async def handle_chat_completion(self, request: dict):
        req_id = str(uuid.uuid4())
        done_event = asyncio.Event()
        self._pending_requests[req_id] = done_event

        # 1. Send to P (max_tokens=1)
        p_request = {**request, "max_tokens": 1}
        p_response = await self.send_to_p(p_request)

        # 2. Wait for callback
        await done_event.wait()
        del self._pending_requests[req_id]

        # 3. Send to D
        return await self.send_to_d(request)

    # Callback endpoint: POST /kv_ready
    async def on_kv_ready(self, req_id: str):
        if req_id in self._pending_requests:
            self._pending_requests[req_id].set()
```

## Benchmark Results (H800, Qwen3-8B, 5K input tokens)

| Configuration  | TTFT mean (ms) | TPOT mean (ms) | TPOT p99 (ms) | ITL p99 (ms) |
| -------------- | -------------- | -------------- | ------------- | ------------ |
| P/D (1P+1D)    | 573.78         | 15.68          | 15.89         | 21.71        |
| Baseline (DP2) | 438.24         | 22.67          | 24.32         | 142.70       |

The P/D setup trades higher TTFT for **significantly more stable decode latency** — TPOT p99 drops from 24.32ms to 15.89ms, and ITL p99 improves dramatically from 142.70ms to 21.71ms.

## TODO

- [ ] Implement `_notify_router()` in connector
- [ ] Implement Router service
- [ ] Handle timeout/error cases
- [ ] Add metrics for P/D latency breakdown
