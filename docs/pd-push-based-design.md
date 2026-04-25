# PegaFlow P/D CPU-Staging Push Design

This document records the selected push-based P/D path and the vLLM connector
entry points it must fit. It intentionally stays at the design level; transport
details live in `rdma-write-with-imm.md`, and receive-manager internals live in
`pd-receive-manager.md`.

## Decision

The initial P/D implementation uses **CPU-staging push**:

```text
P writes prompt KV into PegaFlow-managed CPU/pinned memory near D.
D later copies the staged KV into vLLM GPU KV cache blocks.
```

Direct GPU push is deferred. We validated that same-process `cudaMalloc` memory
can be RDMA registered, but CUDA IPC imported GPU pointers fail `ibv_reg_mr`
with `EFAULT`. A direct GPU path therefore needs registration in the vLLM
allocation-owning process plus explicit RDMA-to-GPU ordering validation.

The most important design point is:

```text
D scheduler connector get_num_new_matched_tokens() is the root of the
CPU-staging receive state machine.
```

It is the only vLLM hook that can keep a request pending **before** D GPU KV
blocks are allocated.

## vLLM Connector Contract

Relevant scheduler-side sequence:

1. `get_num_new_matched_tokens(request, num_local_cached_tokens)` is called
   before D allocates GPU KV blocks.
2. Returning `(None, False)` skips the request for this scheduler step. D GPU KV
   blocks are not allocated.
3. Returning `(N, True)` tells vLLM that `N` external tokens are available and
   should be loaded asynchronously.
4. vLLM then allocates D GPU KV blocks for those external tokens and moves the
   request to `WAITING_FOR_REMOTE_KVS`.
5. `update_state_after_alloc(request, blocks, N)` is called after GPU block
   allocation; this is the first hook that sees destination block IDs.
6. `build_connector_meta()` sends per-step load metadata to workers.
7. D worker `start_load_kv()` performs CPU staging -> GPU KV H2D.
8. D worker `get_finished()` reports `finished_recving`; scheduler then resumes
   the request.

The base connector contract says `get_num_new_matched_tokens()` may be called
multiple times and should be side-effect free. For PegaFlow, this should be
implemented as an idempotent, non-blocking state machine:

```text
ensure_receive_prepare_started(pd_request_id)
read_receive_state(pd_request_id)
return based on state
```

The method may kick off async prepare once, but repeated calls must not allocate
duplicate staging memory or block the scheduler thread.

## D-Side State Machine

```text
Absent
  -> PreparingCpuStaging
  -> WaitingForPWriteImm
  -> CpuStagedReady
  -> GpuAllocated
  -> H2DLoading
  -> Done
```

Behavior in `get_num_new_matched_tokens()`:

- `Absent`: start an idempotent D PegaFlow receive prepare; return
  `(None, False)`.
- `PreparingCpuStaging` / `WaitingForPWriteImm`: return `(None, False)`.
- `CpuStagedReady`: return `(N, True)`.
- `Failed` / `Expired`: return zero/partial hit or fail according to the final
  failure policy.

`GpuAllocated` begins only after vLLM has accepted `(N, True)`. `H2DLoading`
belongs to the D worker-side connector, not the scheduler hook.

## End-to-End Flow

```d2
direction: right

router: Router
p_vllm: "P vLLM"
p_worker: "P vLLM worker\nPegaFlow in-process runtime"
d_sched: "D vLLM scheduler connector"
d_worker: "D vLLM worker connector"
d_pf: "D PegaFlow"

router -> d_sched: "decode request\npd_request_id + rendezvous"
router -> p_vllm: "prefill request\nsame pd_request_id"

d_sched -> d_sched: "get_num_new_matched_tokens()"
d_sched -> d_pf: "PreparePdReceive\nidempotent, async"
d_sched -> d_sched: "return (None, False)\nwhile not ready"

p_vllm -> p_worker: "source KV ready\nmetadata-driven save path"
p_worker -> d_pf: "GetPdReceiveDescriptor\npd_request_id + receive_rank"
d_pf -> p_worker: "CPU staging descriptor\nslabs + layouts + imm"
p_worker -> d_pf: "RDMA WRITE KV\nthen WRITE_WITH_IMM"

d_pf -> d_pf: "IMM complete\nall expected slices ready"
d_sched -> d_pf: "poll/query receive state"
d_pf -> d_sched: "CpuStagedReady"
d_sched -> d_sched: "return (N, True)"

d_sched -> d_sched: "vLLM allocates D GPU KV blocks"
d_sched -> d_worker: "connector metadata\nstaging handle + D block ids"
d_worker -> d_pf: "Load staged KV"
d_pf -> d_worker: "H2D into vLLM KV blocks"
d_worker -> d_sched: "finished_recving"
```

## Router Responsibility

The router must stay out of tokenizer and block-manager semantics. It should
only choose the `(P, D)` pair and pass a short-lived rendezvous identity:

```text
pd_request_id
d_pegaflow_addr
p_pegaflow_addr
d_instance_id
p_instance_id
role hints
```

The router should not pass raw pointers, rkeys, token counts, block IDs, or
receive descriptors. It should also not pass vLLM CUDA device IDs, TP ranks, or
layer names. Those are owned by the P/D vLLM processes and their registered
PegaFlow instance topology.

## D Prepare Responsibility

D scheduler connector has the request after tokenization and local prefix-cache
lookup. It can compute the external KV span conservatively:

```text
computed_blocks = num_computed_tokens / vLLM_block_size
num_blocks = len(request.block_hashes) - computed_blocks
external_tokens = num_blocks * vLLM_block_size
```

Only complete block hashes are eligible for P/D receive. Partial prompt tails
are recomputed by D; otherwise D can wait for a block P will never save.

It calls local D PegaFlow `PreparePdReceive` with:

- `pd_request_id`;
- D instance / model identity;
- complete block count and block hashes for the staged span;
- optional expected IMM count override;
- TTL and admission metadata.

D PegaFlow uses registered KV layout and instance topology to expand the single
request-level prepare into receive ranks. In the first supported deployment,
homogeneous TP4 P/D maps P worker TP rank `r` to D receive rank `r`, for
`r = 0..3`. D PegaFlow does not tokenize and does not allocate vLLM GPU KV
blocks.

## P Push Responsibility

P-side data movement happens inside the P vLLM worker process. There is no
hot-path RPC from the P connector to a local PegaFlow server for this transfer:
the worker process already owns the source CUDA tensors, so the in-process
PegaFlow runtime registers the source GPU memory and owns RDMA WRITE plus
WRITE-with-immediate to D CPU staging.

The P worker connector initializes a PyO3 `KvEgressRuntime` when outbound P/D
push is enabled. The runtime owns:

- a `pegaflow-transfer::TransferEngine`;
- RDMA connection state to D PegaFlow peers;
- P-side GPU memory registration for vLLM KV tensors;
- descriptor polling against D PegaFlow;
- the async push task lifecycle.

P scheduler/worker metadata should describe intent, not raw transport state:

```text
pd_request_id
d_pegaflow_addr
dst_instance_id
source block_ids
source block_hashes
optional handle
```

It should not carry D raw pointers, rkeys, CUDA device IDs, TP ranks, layer
names, or per-NIC transfer details. P workers use their own
`effective_tp_rank` as the D `receive_rank` when fetching descriptors.

The first P implementation can conservatively report `finished_sending` only
after RDMA WRITE, final IMM send, and any normal local save work from the same
save task complete. The target behavior is stricter lifetime splitting:

```text
RDMA + IMM done:
  D receive manager can observe readiness

local D2H save done:
  PegaFlow local/offload copy has a stable CPU copy

both done:
  P GPU KV blocks can be released/reused by vLLM
```

The save worker therefore treats one vLLM `wait_for_save()` as a fan-out point:
run P/D egress first, then run the normal local save path.

```d2
direction: right

p_worker: "P worker connector"
runtime: "KvEgressRuntime\nin vLLM worker process"
d_pf: "D PegaFlow"
local_save: "normal PegaFlow save\nD2H/offload"

p_worker -> runtime: "enqueue PdPushIntent\nsource block ids + rendezvous"
runtime -> d_pf: "poll GetPdReceiveDescriptor"
d_pf -> runtime: "PENDING or READY descriptor"
runtime -> d_pf: "RDMA WRITE\nP GPU KV -> D CPU staging"
runtime -> d_pf: "WRITE_WITH_IMM"
p_worker -> local_save: "then normal save\nsame block ids"
```

The P/D push path and local offload save share scheduling and completion
lifetime, but not the first data movement. A single source GPU block batch may
have multiple sinks:

```text
source P GPU KV blocks
  -> RDMA WRITE sink to D CPU staging
  -> local D2H/offload save sink
```

This avoids an extra P-side CPU staging bounce for P/D transfer and keeps the
normal local save behavior independent.

## TP4 Receive Mapping

The first version is explicitly TP4 P/D, not a TP1 fallback:

```text
P instance TP4:
  worker tp_rank 0 -> D receive_rank 0
  worker tp_rank 1 -> D receive_rank 1
  worker tp_rank 2 -> D receive_rank 2
  worker tp_rank 3 -> D receive_rank 3
```

D scheduler calls `PreparePdReceive(instance_id, pd_request_id, num_blocks, ...)`
once. D PegaFlow snapshots registered D workers and allocates staging for every
registered receive rank. P workers call
`GetPdReceiveDescriptor(dst_instance_id, pd_request_id, receive_rank=<local tp>)`
to get only their rank's descriptor.

Default readiness accounts for WRITE_WITH_IMM fanout:

```text
expected_imm_count = receive_rank_count * D_local_rdma_nic_count
```

This matches the current transfer primitive, which sends one immediate signal
per connected NIC/QP. The value can still be overridden by the caller for later
TP mismatch or grouped-rank policies.

## CPU Staging Layout

The current implementation allocates one NUMA-aware staging slab per D receive
rank per request. This avoids per-layer allocation and keeps the first TP4 path
simple enough to validate:

```text
TP4:
  receive_rank 0 -> one CPU slab on rank 0 preferred NUMA
  receive_rank 1 -> one CPU slab on rank 1 preferred NUMA
  receive_rank 2 -> one CPU slab on rank 2 preferred NUMA
  receive_rank 3 -> one CPU slab on rank 3 preferred NUMA
```

The descriptor is coarse:

- slab base, size, NUMA node;
- receive rank, D device, D TP rank, slab index, NUMA node;
- per-layer offset and stride within the rank slice;
- `imm_data` or equivalent opaque completion token.

P computes addresses from `base + layer_offset + block * stride`.
We should avoid `layer * block` descriptor growth.

The descriptor shape can later support coalescing several receive ranks into one
NUMA slab without changing router/proxy request parameters.

Phase-one implementation may allocate one slab per rank if it is simpler, but
the API should remain compatible with per-NUMA slabs.

RDMA registration creates one important constraint: descriptor pointers must be
inside memory that is already visible in the RDMA handshake snapshot. Returning
a freshly allocated per-request pointer is not sufficient unless the peers
rehshake or dynamically exchange MR metadata. The preferred D-side allocation
model is therefore:

```text
D startup / registration:
  allocate and RDMA-register per-NUMA P/D staging arenas

PreparePdReceive:
  sub-allocate request slices inside those arenas
  return descriptor offsets/pointers inside registered arenas
```

## TP And Fan-In

Readiness is per receive lease and means all expected P writes for the D slice
have completed.

- `P_TP == D_TP`: each D rank waits for one P rank.
- `P_TP > D_TP`: one D rank may wait for multiple P rank slices.
- `P_TP < D_TP`: one P rank may write multiple D rank slices.

The transport layer treats immediate data as opaque. The P/D receive manager
maps it to lease state and expected contributor counts.

## Minimal Control Plane

Initial RPC/control operations:

```text
PreparePdReceive        D scheduler connector -> local D PegaFlow
GetPdReceiveDescriptor  P in-process runtime -> D PegaFlow
GetPdReceiveDescriptor  D scheduler connector -> local D PegaFlow (data_ready poll)
LoadPdReceive           D worker connector -> local D PegaFlow
```

No hot-path `CompletePdReceive` is needed; RDMA WRITE with immediate is the
completion signal. No hot-path `ReleasePdReceive` is needed; D-side TTL/GC and
consume transitions own cleanup.

## Validation

`examples/run_pd_push_tp4.py` launches a single-host TP4 P/D smoke stack:

```text
GPUs 0-3: P vLLM TP4 + PegaFlow server
GPUs 4-7: D vLLM TP4 + PegaFlow server
NICs: mlx5_1..mlx5_4
```

The validation prompt is intentionally long enough to produce complete
`request.block_hashes`; short prompts may correctly bypass P/D receive and let D
compute the whole prompt locally.

## Why CPU Staging First

- D GPU KV blocks are not pinned while P is still writing.
- PegaFlow owns admission, TTL, NUMA placement, and RDMA scheduling.
- P/D push traffic and normal KV/offload traffic can share one PegaFlow traffic
  manager.
- Heterogeneous TP/layout/block-size support is easier before data enters final
  D GPU KV layout.
- Failure cleanup can discard CPU staging without corrupting vLLM KV blocks.

Costs:

- one extra D-side H2D copy;
- D CPU/pinned pool pressure;
- extra local bandwidth scheduling for NIC->CPU and CPU->GPU.

## Open Questions

- Delta transfer: if D already has a prefix, P should eventually push only
  missing blocks. Defer until P2P/cache discovery can cover it cleanly.
- Failure policy: recompute, fail, or retry from another P.
- Exact per-NUMA slab allocator and descriptor schema.
- Direct GPU push admission and synchronization requirements.

## Roadmap

1. Current validation path: TP4 P/D, CPU staging receive, P GPU RDMA WRITE into
   D CPU staging, final WRITE-with-IMM, D `LoadPdReceive` H2D.
2. Add TP fan-in accounting, TTL cleanup hardening, and metrics.
3. Add unified RDMA admission for P/D push, P save/offload, and remote cache
   traffic.
4. Revisit delta transfer and direct GPU push.
