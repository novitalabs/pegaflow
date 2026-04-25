# P/D Receive Manager Design

This document defines the layer above `pegaflow-transfer` for CPU-staging P/D
push. The transfer layer now provides RDMA WRITE-with-immediate as an opaque
transport signal. The receive manager owns the P/D meaning of that signal:
leases, request lifecycle, fan-in, TTL, and vLLM-visible state.

## Scope

The receive manager is a D-side component used by the D connector/control
server. It is responsible for:

- allocating D-side CPU/pinned staging memory for one P/D receive lease;
- generating an opaque `imm_data: u32` token for that lease;
- storing a short-lived receive descriptor for PegaFlow-P to fetch by
  rendezvous reference;
- consuming `TransferEngine::take_imm_receiver()` events;
- demuxing `ImmCompletion { imm_data, ... }` into lease state;
- exposing prefetch-like state to the D vLLM connector;
- releasing expired or completed staging memory.

It is not part of the transport layer. `pegaflow-transfer` must not parse or
reserve `imm_data` values.

## Layering

```text
P/D control plane + vLLM connector
        |
        v
P/D receive manager
        |
        v
pegaflow-transfer
  - normal RDMA WRITE
  - final WRITE_WITH_IMM
  - opaque ImmCompletion queue
```

The receive manager can use transfer events like this:

```rust
let mut imm_rx = transfer
    .take_imm_receiver()
    .expect("one P/D manager owns the IMM stream");

while let Ok(completion) = imm_rx.recv().await {
    receive_manager.on_imm(completion);
}
```

`completion.imm_data` is an opaque payload. The manager chooses how to map it to
a lease.

## Lease State

A receive lease is the D-side state for one P/D push.

```rust
struct PdReceiveLease {
    handle: PdReceiveHandle,
    request_id: String,
    imm_data: u32,
    state: PdReceiveState,
    expected_imm_count: usize,
    observed_imm_count: usize,
    ranks: Vec<PdReceiveRankDesc>,
    slabs: Vec<CpuStagingSlab>,
    layers: Vec<PdReceiveLayerLayout>,
    created_at: Instant,
    expires_at: Instant,
    block_plan: PdBlockPlan,
}

enum PdReceiveState {
    Prepared,
    Writing,
    Ready,
    Loading,
    Done,
    Failed,
    Expired,
}
```

Phase one defaults to:

```text
expected_imm_count = receive_rank_count * D_local_rdma_nic_count
```

because the current `write_imm_async(remote_addr, imm_data)` primitive sends one
immediate signal per connected NIC/QP. Future TP mismatch/fan-in may override
this value explicitly.

## `imm_data` Allocation

D generates `imm_data` when preparing the lease and exposes it through the
descriptor fetched by PegaFlow-P. The router only carries a rendezvous
reference, such as `{d_pegaflow_addr, dst_instance_id, request_id}`.

Recommended first allocator:

```text
imm_data = (generation << slot_bits) | slot_index
```

Why:

- active leases need unique `imm_data` values;
- slot reuse must not let a late IMM complete a new lease;
- the transfer layer can stay opaque because only the receive manager knows the
  encoding;
- TTL cleanup can retire `(slot, generation)` safely.

The manager keeps at least these maps:

```rust
leases_by_handle: HashMap<PdReceiveHandle, LeaseId>
leases_by_imm: HashMap<u32, LeaseId>
```

If source identity is needed for stronger fan-in dedupe, the manager can also
key observed contributors by `(local_qpn, imm_data)` initially, and later by an
explicit source peer/session id once the control plane carries it.

## Prepare Flow

D-side prepare is initiated from the D scheduler connector's
`get_num_new_matched_tokens()` state machine. This is before vLLM allocates D
GPU KV blocks. The prepare call must be idempotent because vLLM may ask the
connector about the same request multiple times.

The scheduler connector computes a conservative external KV span from
tokenized request state, local prefix-cache hits, and vLLM block size, then asks
D PegaFlow to allocate CPU staging. Exact destination GPU block IDs are not
known until `update_state_after_alloc()`.

```text
PreparePdReceive(instance_id, request_id, external_tokens/num_blocks,
                 expected_imm_count?):
  resolve instance_id to registered D topology
  expand every registered D TP rank into a receive rank
  validate layer registrations for each receive rank
  compute staging bytes from KVCacheRegistration and requested block count
  choose NUMA from each GpuContext::preferred_numa()
  allocate lease slot + generation
  allocate one CPU/pinned staging slab per receive rank
  build per-rank/per-layer offset/stride layout inside the slab
  register staging memory with pegaflow-transfer if needed
  imm_data = encode(slot, generation)
  expected_imm_count = override or receive_rank_count * local_nic_count
  state = Prepared
  return handle
```

Response shape:

```rust
struct PreparePdReceiveResponse {
    handle: PdReceiveHandle,
    imm_data: u32,
    expires_at_ms: u64,
}
```

The response intentionally does not need to return slots to the router. Slots
are kept in D PegaFlow and fetched by PegaFlow-P through
`GetPdReceiveDescriptor`.

The descriptor is intentionally coarse. Phase one allocates one slab per receive
rank, which is enough for homogeneous TP4 P/D and avoids per-layer allocation.
The descriptor should remain compatible with later NUMA-level slab coalescing.
It returns:

- slab base pointer, byte length, and NUMA node;
- receive-rank metadata: receive rank, D device, D TP rank, slab index, NUMA;
- one layout entry per layer: slab index, layer offset, block stride, segment
  count, segment size, padded segment stride, number of blocks, and receive
  rank;
- `imm_data` for the final WRITE-with-immediate signal.

P computes destination addresses locally:

```text
dst = slab.base_ptr
    + layer.layer_offset
    + block_index * layer.block_stride
    + segment_index * layer.padded_segment_stride
len = layer.segment_size
```

This avoids `layer * block` slot metadata growth while still letting P issue
normal RDMA WRITEs.

Important boundary:

- Router does not send prompt token counts, block IDs, or descriptors to
  PegaFlow-D.
- PegaFlow-D does not run tokenizer logic.
- D scheduler connector owns the external-token/block-count decision because it
  is the first component that sees tokenized request state and local cache hits.
- PegaFlow-D uses registered KV layout and topology metadata only to convert the
  receive plan to NUMA-aware CPU staging bytes.

## Descriptor Fetch Flow

PegaFlow-P fetches the descriptor from D PegaFlow using the rendezvous reference
that router placed in the P request.

```text
GetPdReceiveDescriptor(dst_instance_id, request_id, receive_rank, handle?):
  if lease missing or not prepared yet: return PENDING
  if lease expired/failed: return EXPIRED/FAILED
  if receive_rank >= 0: return descriptor for only that rank
  if receive_rank < 0: return request-level descriptor/status
  return descriptor(handle, ranks, slabs, layer_layouts, imm_data, expires_at, data_ready)
```

This makes D prepare and P prefill naturally race:

- if P finishes prefill first, it waits for the descriptor;
- if D prepare finishes first, descriptor sits in D PegaFlow until P fetches it;
- router does not need an HTTP callback endpoint and never carries raw RDMA
  write capabilities.

## P Push Flow

P receives a rendezvous reference from router, fetches its local TP rank's
descriptor from D PegaFlow, and treats `imm_data` as an opaque token.

```text
P receives rendezvous_ref(d_pegaflow_addr, dst_instance_id, request_id)
P computes prompt KV
P fetches descriptor(receive_rank = local effective_tp_rank) from D PegaFlow
P computes destination addresses from slab + layer layout
P schedules normal RDMA WRITEs into D CPU staging
P schedules final write_imm_async(D, imm_data)
P reports finished_sending after PegaFlow accepts/completes the push policy
```

The P-side send completion for `write_imm_async` means the signal WR completed
locally. It does not mean D has loaded GPU KV blocks.

For homogeneous TP4 mapping, this means one D scheduler prepare creates four
receive-rank descriptors. P rank `r` writes only descriptor `receive_rank = r`.
A later TP-mismatch design can let multiple P ranks contribute to one D receive
rank without putting rank/device details into router JSON.

## D Readiness Flow

D connector sees the request before the push may have finished. It should poll
or query receive-manager state through the same shape as existing prefetch.

```text
Prepared/Writing -> PREFETCH_LOADING / (None, False)
Ready            -> PREFETCH_DONE / (N, True)
Loading          -> external KV load in progress
Done             -> finished_recving
Failed/Expired   -> request failure path
```

Important behavior:

- D GPU KV blocks are not allocated while the lease is Prepared/Writing.
- After enough IMM completions arrive, the lease becomes Ready.
- `GetPdReceiveDescriptor` remains usable before data readiness so P can fetch
  write destinations; D scheduler gates on `data_ready == true`.
- Only then does D connector allow vLLM to allocate KV blocks and enter async
  load/H2D.

## IMM Event Handling

```rust
fn on_imm(completion: ImmCompletion) {
    let Some(lease_id) = leases_by_imm.get(&completion.imm_data) else {
        metrics.unexpected_imm += 1;
        return;
    };

    let lease = leases.get_mut(lease_id);
    if lease.state == Prepared {
        lease.state = Writing;
    }

    if observe_contributor(lease, completion) {
        lease.observed_imm_count += 1;
    }

    if lease.observed_imm_count == lease.expected_imm_count {
        lease.state = Ready;
        wake_prefetch_waiter(lease.handle);
    }
}
```

For phase one, `observe_contributor` can simply count completions. For fan-in,
it should reject duplicate contributors.

## TTL And Cleanup

The manager owns cleanup. We do not need a `ReleasePdReceive` RPC in the first
phase.

TTL policy:

- Prepared/Writing leases expire if IMM does not arrive before `expires_at`.
- Ready leases expire if D never starts loading.
- Done leases can be released immediately after `finished_recving`.
- Failed/Expired leases release CPU staging slabs and retire the generation.

Late IMM behavior:

- if `imm_data` is unknown, count/log `unexpected_imm`;
- if generation does not match the active slot, count/log `late_imm`;
- never let a late IMM complete a reused lease.

## Metrics

Minimum metrics:

- active leases by state;
- CPU staging bytes allocated/free;
- prepare latency;
- time from prepare to first IMM;
- time from prepare to Ready;
- H2D load latency;
- expired leases;
- unexpected/late IMM completions;
- duplicate fan-in contributors;
- IMM receiver lag or dropped receiver events if applicable.

## Implementation Roadmap

1. Add D-side `PdReceiveManager` with lease allocation, `imm_data` allocator,
   and TTL cleanup.
2. Wire manager to `TransferEngine::take_imm_receiver()` in the server runtime.
3. Expose descriptor/data-ready query methods matching the existing prefetch
   connector states.
4. Add `PreparePdReceive` for D connector and `GetPdReceiveDescriptor` for
   PegaFlow-P rendezvous.
5. Add P-side push method that writes slab/layout destinations and calls
   `write_imm_async`.
6. Add fan-in source identity and duplicate detection for TP mismatch.
7. Add production metrics and stress tests.
