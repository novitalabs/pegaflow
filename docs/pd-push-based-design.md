# PegaFlow P/D CPU-Staging Push Design

This document focuses only on push-based P/D transfer paths. The goal is to
understand how PegaFlow can own P/D RDMA traffic instead of letting P/D transfer
compete independently with ordinary KV cache traffic.

## Decision

The first implementation will use **CPU-staging push**:

```text
P RDMA WRITEs KV into PegaFlow-managed CPU/pinned memory on the D side.
D then installs the staged KV into vLLM GPU KV cache with an H2D copy.
```

Direct GPU push is intentionally deferred. It remains a future fast path, but it
should not be the default path until GPU memory lifetime, RDMA-to-GPU ordering,
and D-side KV block pinning are validated under production load.

## Problem Statement

In P/D disaggregation, P produces prompt KV and D consumes it before decode. A
pull-based design lets D fetch KV after P publishes it. A push-based design
instead prepares receive resources on D first, then lets P write KV into those
resources.

The push-based design has two materially different receive targets, but only one
is selected for the initial path:

- **CPU staging target, selected**: P writes into PegaFlow-managed CPU/pinned
  memory near D, then D copies H2D into vLLM KV cache blocks.
- **Direct GPU target, future**: P writes into D vLLM KV cache blocks.

These are not small variants. They have different lifetime, scheduling, memory
pressure, and TTFT trade-offs.

## Common Push-Based Shape

```text
Router selects D
D/PegaFlow-D prepares CPU/pinned receive resources
Router or PegaFlow control plane gives P a receive handle
P/PegaFlow-P pushes KV to D-side staging target by RDMA WRITE
D/PegaFlow-D observes completion, ideally via write-with-imm
D connector installs staged KV into vLLM GPU KV blocks
D connector reports finished_recving
D vLLM resumes decode
P connector reports finished_sending and releases P-side blocks
```

The router should carry a short-lived transfer handle, not long-lived RDMA
capabilities. PegaFlow-P and PegaFlow-D can exchange real address/rkey/QP details
through PegaFlow-controlled RPC.

## First-Cut Request Behavior

The initial request path is:

1. Router receives the user request and selects one `(P, D)` pair.
2. Router sends `PreparePdReceive` to the D-side PegaFlow endpoint.
   The endpoint is logically part of the D vLLM deployment. Since the external
   endpoint is vLLM, this likely requires a small background Rust server or
   embedded PegaFlow control server started with the vLLM process.
3. D-side PegaFlow allocates registered CPU/pinned staging memory for the
   expected KV blocks and returns a short-lived receive descriptor.
4. Router embeds that descriptor into the requests sent to both P and D, for
   example under `kv_transfer_params.pegaflow_pd`.
5. D-side vLLM may receive the request before P has pushed KV. In that state the
   D connector polls PegaFlow-D, sees the staging lease is not ready, and returns
   `None` from `get_num_new_matched_tokens()`. vLLM keeps the request waiting and
   does not allocate D GPU KV blocks yet.
6. P-side connector sees the receive descriptor. After P computes prompt KV, it
   asks PegaFlow-P to push the relevant KV blocks to D-side staging memory with
   RDMA WRITE with immediate.
7. P continues its normal async save path. The P/D push traffic and P's own
   offloading/save traffic must both be admitted by PegaFlow's RDMA scheduler so
   they do not collide on NIC/PCIe bandwidth.
8. D-side PegaFlow observes the immediate value on the receive CQ and marks the
   staging lease ready.
9. On a later scheduler poll, D connector sees the staging lease is ready and
   returns `(N, True)` from `get_num_new_matched_tokens()`. Only then does vLLM
   allocate D GPU KV blocks and enter `WAITING_FOR_REMOTE_KVS`.
10. D worker `start_load_kv()` copies the staged KV into the newly allocated D
   vLLM GPU KV blocks and then reports `finished_recving`.
11. P-side connector reports `finished_sending` after PegaFlow marks the push
   transaction completed, failed, or expired.

```d2
direction: right

user: User
router: Router

p_vllm: "P vLLM"
p_pf: "P PegaFlow"
d_vllm: "D vLLM"
d_pf: "D PegaFlow"

user -> router: request
router -> router: select (P, D)

router -> d_pf: "PreparePdReceive(prompt/KV shape)"
d_pf -> d_pf: "alloc CPU pinned staging\nregister RDMA memory"
d_pf -> router: "receive descriptor\n(handle, slots, ptr/size/numa, imm)"

router -> p_vllm: "prefill request\n+ kv_transfer_params.pegaflow_pd"
router -> d_vllm: "decode request\n+ same receive handle"
d_vllm -> d_pf: "poll staging state"
d_pf -> d_vllm: "pending\n(no GPU KV allocation)"

p_vllm -> p_pf: "publish source KV blocks\nwith receive descriptor"

p_pf -> d_pf: "RDMA WRITE with imm\nP KV -> D CPU staging"
p_pf -> p_pf: "normal async save/offload\nthrough same RDMA scheduler"

d_pf -> d_pf: "CQE imm received\nmark staging ready"
d_vllm -> d_pf: "poll staging state"
d_pf -> d_vllm: "ready"
d_vllm -> d_vllm: "allocate GPU KV blocks\nenter async load"
d_vllm -> d_pf: "acquire staged KV"
d_pf -> d_vllm: "H2D install into D KV blocks"
d_vllm -> router: "decode output"

p_pf -> p_vllm: finished_sending
d_pf -> d_vllm: finished_recving
```

The receive descriptor format should follow the same spirit as existing P2P
RDMA metadata: per-slot pointer, length, NUMA/NIC affinity, and enough identity
to validate the request. In the current transfer engine, rkeys are exchanged in
the RDMA handshake's registered-memory snapshot rather than carried per slot.
Because this descriptor is a write capability, it must be scoped to one
transfer, short-lived, and revocable. A later version can replace raw pointer
exposure in the router payload with an opaque handle that PegaFlow-P resolves
directly from PegaFlow-D.

## vLLM Interface Fit

D still needs to enter vLLM's async external-KV path:

1. D request arrives with `kv_transfer_params`.
2. D connector `get_num_new_matched_tokens()` polls PegaFlow-D for the staging
   lease state.
3. If the lease is not ready, the connector returns `(None, False)`. vLLM skips
   the request for this scheduler step and does not allocate D GPU KV blocks.
4. After RDMA WRITE with immediate completes and PegaFlow-D marks the lease
   ready, the connector returns `(N, True)`.
5. vLLM allocates destination KV blocks and marks the request
   `WAITING_FOR_REMOTE_KVS`.
6. D connector `update_state_after_alloc()` sees the destination block IDs.
7. D worker `start_load_kv()` installs staged KV into the destination blocks
   with H2D.
8. D worker `get_finished()` returns `finished_recving` when the blocks are safe
   for decode.

P uses the delayed-release path:

1. P connector `request_finished...()` returns `True` so vLLM keeps source KV
   blocks alive.
2. P connector returns `kv_transfer_params` or a PegaFlow publish handle.
3. PegaFlow-P pushes the source KV into the D receive target.
4. P worker returns `finished_sending` only after the D-side transaction has
   completed, failed, or timed out.

## Selected Path: Push Into PegaFlow CPU Staging Near D

```text
PegaFlow-D allocates registered CPU/pinned receive buffers
PegaFlow-P RDMA WRITEs KV into those buffers
D connector polls until staging is ready
D vLLM allocates KV blocks for the external tokens
D connector copies staged KV into D GPU KV blocks
D reports finished_recving
```

### Why This Is First

- PegaFlow owns the receive resource, so admission, timeout, quota, and release
  do not depend on vLLM GPU block lifetime.
- D decode GPU memory is not pinned while the network transfer is still in
  flight.
- It aligns with the main PegaFlow objective: P/D RDMA and ordinary KV-cache
  RDMA should enter the same managed traffic plane.
- It gives a natural place for heterogeneous TP, block size, KV layout, or
  future conversion/compression.
- Failure handling is cleaner: PegaFlow can discard a staging lease without
  corrupting or stranding vLLM GPU KV blocks.

### Costs

- Requires an extra D-side H2D copy before decode.
- Consumes D-side PegaFlow CPU/pinned pool capacity.
- Adds one more local bandwidth domain to schedule: NIC->CPU followed by
  CPU->GPU.
- TTFT can be worse when H2D is on the critical path and D GPU memory would have
  been safe to reserve directly.

### When It Should Be Used

This is the default for the initial P/D implementation. It should be used for:

- long prompts,
- heterogeneous or not-yet-validated P/D layouts,
- production traffic where RDMA traffic isolation matters,
- cases where D decode concurrency is more important than best-case TTFT.

## Deferred Fast Path: Push Directly Into D vLLM GPU KV

```text
D vLLM allocates KV blocks
PegaFlow-D maps block IDs to GPU KV addresses
PegaFlow-D exports a receive handle for those exact GPU pages
PegaFlow-P RDMA WRITEs KV into D GPU memory
D observes completion and reports finished_recving
```

### Benefits

- Lowest copy count: no D-side CPU staging and no extra H2D.
- Best theoretical TTFT once the push starts, because data lands where decode
  will read it.
- Avoids consuming D-side CPU pool for large prompt KV.
- Matches the ideal "P produces, D receives" direction.

### Costs

- Pins D vLLM GPU KV blocks for the whole transfer window. Long prompts can
  occupy decode GPU memory for seconds before D can run decode.
- D scheduler must allocate KV blocks before P can push. This couples router,
  D scheduler admission, and P transfer start.
- GPU memory registration/rkey lifetime must track vLLM KV cache lifetime
  exactly.
- Local validation showed same-process `cudaMalloc` GPU memory can be
  `ibv_reg_mr`-registered, but a CUDA IPC imported GPU pointer fails
  registration with `EFAULT`; direct GPU push therefore needs RDMA registration
  in the vLLM allocation-owning process, not a sidecar that only imports CUDA
  IPC memory.
- Correctness depends on RDMA-to-GPU memory ordering. A CQE or write-with-imm
  must imply the target KV is visible to later GPU kernels, or D must insert the
  right synchronization.
- Failed or slow P pushes strand D KV blocks until timeout/recovery.
- Heterogeneous TP/block-size/layout conversion is harder because the target is
  already D's final physical layout.

### Conditions Before Enabling

- Homogeneous or explicitly compatible P/D layouts.
- Strict admission that bounds the GPU block pinning window.
- Validated GPUDirect RDMA registration and memory-ordering behavior.
- Clear CUDA synchronization before D reports `finished_recving`.
- Recovery path for slow or failed P pushes that does not strand D KV blocks.

## Trade-Off Summary

| Dimension | CPU Staging Push | Direct GPU Push |
|---|---|---|
| Status | Initial/default path | Future fast path |
| Copy count | RDMA to CPU + H2D | RDMA directly to final KV |
| D GPU memory pressure | Low until install | High during transfer |
| TTFT best case | Worse by H2D cost | Better |
| Failure isolation | Easier | Harder |
| Heterogeneous support | Easier | Harder |
| RDMA scheduling ownership | Strongest | Good, but tied to vLLM blocks |
| Implementation risk | Lower | Higher |

## Admission Policy Implication

CPU staging push reserves PegaFlow pool memory first, not vLLM GPU KV memory.
Initial admission should therefore focus on:

- estimated bytes and transfer time,
- D-side CPU/pinned staging pool capacity,
- NIC and PCIe topology,
- current PegaFlow RDMA queue occupancy,
- D-side H2D install budget,
- request priority and timeout.

GPU direct push, when added later, should require a stricter second policy that
also accounts for D GPU KV free blocks, decode queue depth, and expected GPU
pinning time.

## Completion Semantics

RDMA WRITE with immediate is attractive for push-based transfer because it can
signal D-side completion without a separate notification RPC. PegaFlow should
still define completion at the transaction level:

- network write completed,
- all expected blocks received,
- data is visible to the next consumer,
- target lease can transition to `ready`.

For CPU staging, this is mostly a CPU memory visibility problem. For direct GPU
push, PegaFlow must validate the required GPU synchronization before reporting
`finished_recving` to vLLM.

## Initial Transaction

The initial CPU-staging transaction is:

```text
PreparePdReceive(target=cpu_staging)
Push(handle, source_blocks)
QueryPrefetch(handle)  # D polls; ready only after write-with-imm
Install(handle)        # H2D into D vLLM KV blocks
TTL/GC or internal consume release
```

GPU direct can later reuse the same high-level transaction with
`target=gpu_kv`, where `Install(handle)` becomes a synchronization/no-op step.

## Implementation Notes

- `PreparePdReceive` should allocate D-side PegaFlow staging buffers and return a
  short-lived receive handle.
- The router should carry the receive descriptor initially, but the long-term
  target is to carry only an opaque handle.
- PegaFlow-P should resolve the handle with PegaFlow-D and perform RDMA WRITE.
- D connector should return `(N, True)` only after the transaction is accepted
  and the D-side staging lease is ready. It should return `(None, False)` while
  waiting for P's RDMA WRITE with immediate, so vLLM does not allocate D GPU KV
  blocks early.
- D worker `start_load_kv()` should copy already-staged KV into allocated vLLM
  KV blocks and only then report `finished_recving`.
- P worker should report `finished_sending` after PegaFlow marks the transaction
  completed, failed, or expired.

## Engineering Roadmap

This roadmap is based on the current code structure:

- `python/pegaflow/connector/` already has the scheduler/worker split needed for
  vLLM lifecycle integration.
- `pegaflow-proto/proto/engine.proto` has save/load/query and P2P transfer RPCs,
  but no P/D receive transaction RPCs yet.
- `pegaflow-transfer` supports RDMA READ and WRITE, but does not expose
  RDMA WRITE with immediate yet. The receive CQ is effectively unused today.
- `pegaflow-core` already has a pinned allocator and GPU H2D/D2H worker pool.
  CPU-staging receive buffers should be allocated from this existing pinned
  pool so they are covered by RDMA memory registration.
- Existing `Load` requires a prior pin reservation in `ReadCache`. Therefore a
  ready P/D receive must either insert-and-pin staged blocks or otherwise expose
  them through the same consume path before D calls `Load`.

### Phase 1: P/D Control Plane RPCs

Goal: add real receive transaction state to PegaFlow server.

The initial design should add only two new RPCs and reuse the existing
`QueryPrefetch` polling path.

New RPCs:

```text
PreparePdReceive(instance_id, namespace, req_id, block_hashes, shape/bytes)
  -> handle, slots, imm, expires_at

PushPdBlocks(source_instance_id, namespace, req_id, dst_instance_id, handle,
             receive_descriptor, block_hashes, imm)
  -> ok, accepted_blocks
```

Existing RPC to extend:

```text
QueryPrefetch(instance_id, req_id, block_hashes, pd_receive_handle?)
  -> PREFETCH_LOADING | PREFETCH_DONE
```

Proposed wire shape:

```proto
message PreparePdReceiveRequest {
  string instance_id = 1;      // D-side vLLM/PegaFlow instance.
  string request_id = 2;       // Router/vLLM request id.
  string namespace = 3;        // Model/cache namespace.
  repeated bytes block_hashes = 4;
  KvShape shape = 5;           // dtype, layers, block size, heads, head dim.
  uint64 expire_after_ms = 6;
}

message PreparePdReceiveResponse {
  bool ok = 1;
  string handle = 2;
  repeated PdReceiveSlot slots = 3;
  uint32 imm_data = 4;
  uint64 expires_at_ms = 5;
  string error = 6;
}

message PdReceiveSlot {
  bytes block_hash = 1;
  uint64 k_ptr = 2;
  uint64 k_size = 3;
  uint64 v_ptr = 4;
  uint64 v_size = 5;
  int32 numa_node = 6;
}

message PushPdBlocksRequest {
  string source_instance_id = 1;  // P-side instance.
  string request_id = 2;
  string namespace = 3;
  string dst_instance_id = 4;
  string pd_receive_handle = 5;
  repeated PdReceiveSlot dst_slots = 6;
  repeated bytes block_hashes = 7;
  uint32 imm_data = 8;
}

message PushPdBlocksResponse {
  bool ok = 1;
  uint32 accepted_blocks = 2;
  string error = 3;
}
```

`PdReceiveSlot` intentionally does not carry `rkey` in the first version. P2P
already synchronizes registered-memory metadata through the RDMA handshake, so
the transfer layer should resolve `ptr -> rkey` from that snapshot instead of
duplicating registration lifetime in the business-level RPC.

`PreparePdReceive` is called by the router against D-side PegaFlow after it
selects a `(P, D)` pair. It creates a D-side receive lease, allocates registered
CPU staging memory, and returns a descriptor that the router places in the P and
D requests. It must not allocate D vLLM GPU KV blocks.

`PushPdBlocks` is a P-local submission RPC. It tells PegaFlow-P to accept a push
job and, once the source KV blocks are available from the normal async save
path, RDMA WRITE them into D's receive slots. The response means accepted, not
completed; D readiness is driven only by RDMA WRITE with immediate.

`QueryPrefetch` is already the scheduler-side polling mechanism for external KV
availability. P/D CPU staging should use the same contract:

- `PREPARED` / `WRITING` receive lease -> `PREFETCH_LOADING`
- `READY` receive lease -> pin staged blocks and return `PREFETCH_DONE(hit=N)`
- `FAILED` / `EXPIRED` receive lease -> return zero/partial hit or an error,
  depending on the final failure policy

When `QueryPrefetch` observes a ready P/D receive, it must provide the same
safety as the current cache path: the blocks D will load must be pinned for that
D instance before `Load` consumes them. There are two viable implementations:

- insert staged blocks into `ReadCache` and pin them in `QueryPrefetch`, or
- add a dedicated staged-block consume path used by `Load`.

The first option reuses the current `Load` implementation and should be the
initial target.

`CompletePdReceive` is intentionally not part of the first-cut API. RDMA WRITE
with immediate is the only D-side ready signal; adding an explicit completion
RPC would create two competing completion sources. Hot-path `ReleasePdReceive`
is also not part of the first-cut API. The D-side receive lease manager should
free staging memory through `CONSUMED`, TTL/GC, or failure transitions. A later
non-hot-path cancel RPC can be added if abandoned staging pressure becomes a
real problem.

### Phase 2: D-Side Staging Allocator

Goal: allocate and track D-side CPU staging leases.

- Add a receive lease manager in `pegaflow-core` or `pegaflow-server`.
- Allocate staging buffers from `StorageEngine::allocate(...)`, preserving NUMA
  affinity for the selected D GPU/NIC.
- Represent staged KV as the same `RawBlock`/`SealedBlock` shape used by the
  existing cache path once the transfer is complete.
- Add TTL/GC so abandoned P/D transfers release pinned memory.
- Track states: `prepared`, `writing`, `ready`, `consumed`, `failed`, `expired`.

### Phase 3: RDMA WRITE Path

Goal: make PegaFlow-P write P's saved CPU KV into D staging.

Product behavior should use RDMA WRITE with immediate. Engineering-wise this
requires extending `pegaflow-transfer`:

- add a transfer op or flag for immediate data,
- configure receive WQ/CQ for imm completions,
- post receive WQEs,
- expose a D-side completion stream or callback that marks the lease ready.

P-side source should initially come from PegaFlow's normal async save output:
P saves GPU KV into its local PegaFlow pinned cache, then PegaFlow-P locks those
source blocks and writes them to D. Later we can optimize by piggybacking on the
fresh save allocations before insertion.

### Phase 4: D-Side Install

Goal: reuse the existing `Load` path for staged KV.

- After `QueryPrefetch` reports ready and pins staged blocks, D connector returns
  `(N, True)`.
- vLLM allocates D GPU KV blocks and calls `update_state_after_alloc()`.
- D worker `start_load_kv()` calls existing `EngineRpcClient.load(...)` with the
  staged block hashes and D block IDs.
- Existing GPU load worker performs H2D and signals `PyLoadState`.
- Worker reports `finished_recving` through the existing path.

### Phase 5: Router Integration

Goal: complete 1P1D request flow.

- Router selects `(P, D)`.
- Router calls D-side `PreparePdReceive`.
- Router sends the receive descriptor to both:
  - P request, so P can push,
  - D request, so D can poll without allocating GPU KV.
- D request may be sent before P finishes; pending polling should be cheap and
  should not reserve D GPU KV.
- P request release is delayed until normal save and P/D push complete.

### Phase 6: Unified Traffic Scheduling

Goal: prevent P/D push from colliding with P's normal async save/offload and
other PegaFlow RDMA traffic.

- Introduce a transfer scheduler above `TransferEngine::batch_transfer_async`.
- Classify traffic:
  - P/D push,
  - remote cache/P2P fetch,
  - background save/offload-related traffic.
- Add per-NIC concurrency and byte budgets.
- Give P/D push priority without starving cache traffic.
- Export metrics for queue time, RDMA time, H2D time, P held-block duration, and
  staging pool pressure.

### Phase 7: Compatibility and Optimizations

Deferred until the basic CPU-staging path is correct:

- delta transfer when D already owns a prefix,
- heterogeneous TP/block-size/layout conversion,
- cross-layer blocks,
- MLA/HMA matrix,
- direct GPU push fast path.

## Open Questions

- **Delta transfer**: D may already have a prefix of the requested KV in its
  local cache or via PegaFlow remote-cache/P2P discovery. In that case P only
  needs to push the missing delta blocks. The initial implementation should not
  optimize this path; it can conservatively push the full external KV span.
  Later, D-side PegaFlow could include already-present block hashes in
  `PreparePdReceive`, so PegaFlow-P only writes missing blocks. This likely fits
  better once P2P/cache discovery can cover the shared-prefix case.
