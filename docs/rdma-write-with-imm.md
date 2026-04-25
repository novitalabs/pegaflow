# RDMA WRITE With Immediate

This note records the PegaFlow transfer-layer WRITE-with-immediate path used by
P/D CPU staging. The control-plane RPCs remain separate; this layer only exposes
transport send completions and receiver-side immediate-completion events.

## Why It Matters

P/D push wants this sequence:

```text
PegaFlow-D allocates registered CPU staging
PegaFlow-P RDMA WRITEs KV bytes into that staging
PegaFlow-P sends immediate data with the final write
PegaFlow-D observes the imm CQE and marks the receive lease READY
D connector later enters async load and H2D copies staged KV
```

The immediate value is the only hot-path completion signal. We should not add a
`CompletePdReceive` RPC unless WRITE-with-imm proves unusable.

## Implementation Summary

`pegaflow-transfer` now supports a minimal WRITE-with-immediate signal path:

- `TransferEngine::write_imm_async(remote_addr, imm_data)` posts one
  zero-SGE `RDMA_WRITE_WITH_IMM` signal per connected NIC/QP and returns
  send-side completion receivers.
- `TransferEngine::take_imm_receiver()` returns the single consumer of
  receiver-side `ImmCompletion` events.
- Handshake metadata includes an internal per-NIC signal MR. Higher layers do
  not pass signal pointers/rkeys.
- RC sessions now retain and poll `recv_cq`.
- RC sessions post zero-SGE receive WQEs after QP connection and repost after
  each immediate completion.
- The recv CQ poller publishes `ImmCompletion { nic_idx, local_qpn, imm_data }`
  without interpreting `imm_data`.

## Sideway Support

PegaFlow already uses `sideway = 0.4.1` through `pegaflow-transfer`, and sideway
does expose the primitives needed for WRITE-with-imm:

- `sideway::ibverbs::queue_pair::SendOperationFlags` includes
  `WriteWithImmediate`.
- `WorkRequestOperationType` includes `WriteWithImmediate`.
- `WorkRequestHandle::setup_write_imm(rkey, remote_addr, imm_data)` is
  available for generic QPs.
- The basic QP path maps it to `IBV_WR_RDMA_WRITE_WITH_IMM`.
- The extended QP path maps it to `ibv_wr_rdma_write_imm(...)`.
- `WorkCompletionOperationType` includes `ReceiveWithImmediate`
  (`IBV_WC_RECV_RDMA_WITH_IMM`).
- `GenericWorkCompletion::imm_data()` can read the immediate value from either
  basic or extended CQ.
- `QueuePair::start_post_recv()` and `PostRecvGuard` are available for posting
  receive WQEs.

The implementation uses these sideway APIs directly. It does not add raw
`rdma-mummy-sys` verbs calls.

## PPLX Garden Reference

`pplx-garden` has a useful production-shaped reference in `fabric-lib`:

- Public API:
  - `SingleTransferRequest { imm_data: Option<u32>, ... }`
  - `ImmTransferRequest { imm_data, dst_mr, ... }`
  - `TransferCompletionEntry::ImmData(u32)`
  - `TransferCompletionEntry::ImmCountReached(u32)`
  - `TransferEngine::add_imm_callback(...)`
  - `TransferEngine::set_imm_count_expected(...)`
  - `TransferEngine::get_imm_counter(...)`
- Verbs implementation:
  - `verbs_rdma_op::opcode_imm(...)` selects
    `IBV_WR_RDMA_WRITE_WITH_IMM` when `imm_data` is present.
  - `SingleWriteOpIter::new_imm(...)` creates a zero-SGE
    `IBV_WR_RDMA_WRITE_WITH_IMM`; this is a pure ready signal.
  - `VerbsDomain` creates a dedicated RMA SRQ and posts zero-byte receive WQEs
    with `post_imm_recv()`.
  - `handle_cqe(...)` handles `IBV_WC_RECV_RDMA_WITH_IMM`, immediately reposts
    another imm recv, extracts `wc.imm_data`, and routes it through an
    `ImmCountMap`.
- Test reference:
  - `tests/fabric_lib/test_transfer_engine.py::test_single_write_cpu_tensor`
    sends a CPU RDMA write with `imm_data=555`, waits for the remote imm
    callback, and verifies the destination CPU buffer contents.

The important design lesson is that immediate data should be a first-class
completion stream:

```text
WRITE completion on P -> source buffers can be released
IMM completion on D   -> receive lease can become READY
```

PPLX also separates two immediate modes that PegaFlow likely wants:

- `submit_write(... imm_data=Some(x))`: data movement and ready signal in the
  same WR.
- `submit_imm(x, dst_mr)`: a zero-SGE WRITE-with-imm used as a barrier/signal
  after one or more normal WRITEs.

For P/D CPU staging, PegaFlow uses the second form: write all KV segments with
normal RDMA WRITEs, then post final zero-SGE WRITE-with-imm signal(s). With one
active QP this is one IMM CQE per P/D lease; with multiple active QPs or future
fan-in, the P/D manager owns the expected-count policy.

## PegaFlow Transfer API Shape

The public transfer API should not expose a per-request signal pointer. The
transfer layer should own any per-peer/per-NIC signal memory needed to make
WRITE-with-imm legal. P/D control plane should only carry KV slots and
`imm_data`.

Sender-side API:

```rust
impl TransferEngine {
    pub fn write_imm_async(
        &self,
        remote_addr: &str,
        imm_data: u32,
    ) -> Result<Vec<mea::oneshot::Receiver<Result<usize>>>>;
}
```

Semantics:

- `remote_addr` selects the connected peer.
- `imm_data` is an opaque 32-bit immediate payload owned by the higher layer.
- The operation sends one final WRITE-with-imm ready signal per active NIC/QP.
- The returned receivers are send-side completions. They tell P when the signal
  WR has completed locally; they are not the D-side ready event.
- The remote address/rkey for the signal write is internal to the transfer
  handshake or connection state.

Receiver-side API:

```rust
#[derive(Clone, Copy, Debug)]
pub struct ImmCompletion {
    pub nic_idx: usize,
    pub local_qpn: u32,
    pub imm_data: u32,
}

pub type ImmCompletionReceiver =
    mea::mpsc::UnboundedReceiver<ImmCompletion>;

impl TransferEngine {
    pub fn take_imm_receiver(&self) -> Option<ImmCompletionReceiver>;
}
```

Semantics:

- The transfer layer reports only transport events: an opaque `imm_data`
  payload arrived on a given NIC/QP.
- `take_imm_receiver()` is single-consumer. The P/D receive manager owns that
  receiver and is responsible for demuxing `imm_data` into receive leases.
- The internal poller never invokes user code while holding transport state. It
  only publishes `ImmCompletion` into the receiver queue.
- It does not know whether the immediate corresponds to a P/D lease, barrier,
  counter, fan-in contributor, or future transfer class.
- It does not generate, parse, reserve, or deduplicate `imm_data`.

P/D waiting state belongs above the transfer layer:

```text
PreparePdReceive -> lease state PREPARED
first matching ImmCompletion -> optional lease state WRITING
ImmCompletion { imm_data } -> P/D receive manager looks up lease/fan-in state
expected imm count reached -> P/D receive manager marks lease READY
GetPdReceiveDescriptor(handle) while PREPARED/WRITING -> state READY, data_ready false
GetPdReceiveDescriptor(handle) after expected IMM -> state READY, data_ready true
```

This keeps the transport generic and lets P/D own TTL, failure policy, retries,
and request lifecycle.

## Verbs Semantics

For RC QPs, RDMA WRITE with immediate is still a one-sided data movement, but
the immediate notification is delivered to the remote receive CQ. That implies:

- D must have a receive CQ and must poll it.
- D must post receive WQEs before P sends WRITE-with-imm.
- The receive WQE does not carry the KV payload; the payload lands in the remote
  address from the RDMA WRITE.
- The CQE must expose `imm_data`.
- The CQE should let D map `imm_data -> receive lease handle`.
- P still needs send-side completion to know when it can release local source
  blocks.

The current API intentionally keeps normal data writes separate from the final
signal write. Higher layers schedule data movement through
`batch_transfer_async(TransferOp::Write, ...)`, then call `write_imm_async(...)`
when they want to publish readiness.

Receiver side needs:

- receive CQ creation and polling thread/task;
- enough posted receive WQEs per QP;
- single-consumer completion receiver for opaque `imm_data`;
- error handling for CQE failure status;
- reposting policy after each receive completion;
- metrics for recv CQ lag, posted recv depth, imm completions, and failures.

## Integration Test

The hardware-gated test lives in:

```text
pegaflow-transfer/tests/write_imm.rs
```

It is ignored by default and must be run with explicit RNIC selection:

```bash
PEGAFLOW_IT_NICS=mlx5_1 \
cargo test -p pegaflow-transfer --test write_imm -- --ignored --nocapture
```

The test is written from the P/D caller's perspective:

- creates a local P engine and D engine;
- exchanges in-memory handshake metadata;
- D takes the transfer-layer `ImmCompletionReceiver`;
- a tiny test `PdReceiveManager` allocates opaque `imm_data` values and keeps a
  lease table above the transfer layer;
- P writes CPU data with normal RDMA WRITE and sends the final
  WRITE-with-imm signal;
- D manager demuxes `ImmCompletion` by `imm_data`, marks the lease ready, and
  the test verifies destination CPU bytes.

It covers:

- data WRITE plus final IMM readiness;
- multiple in-flight P/D leases with different opaque `imm_data` values.

Validation on h20 with `mlx5_1` passed:

```text
test result: ok. 2 passed; 0 failed
```

Observed P/D caller-path timing from one run:

```text
setup+handshake: 12.7-13.6 ms
64 KiB data WRITE send completion: 109 us
WRITE_WITH_IMM send completion: 73 us
D manager ready wait after P IMM send completion: 75 us
data WRITE start -> D manager ready: 263 us
multi-inflight IMM send completions: 93-120 us
multi-inflight ready waits: 65-68 us
```

These numbers include the current Rust worker thread, oneshot, MPSC, and test
manager polling overhead. They are smoke-test timings, not a pure verbs
microbenchmark.

## Ordering Question

The key correctness question is whether D can treat the imm CQE as "all prior
WRITE bytes for this lease are visible in CPU memory".

For CPU staging, the desired rule is:

```text
P posts all data WRITEs
P posts final WRITE_WITH_IMM
D observes imm CQE
D may read CPU staging and start H2D
```

The validation should write a deterministic pattern over all bytes and verify
from D after the imm CQE. If there is any doubt, PegaFlow should enforce one QP
ordering for all descriptors of a lease or add a send-side ordering barrier
before the final immediate.

## P/D Integration Boundary

Once WRITE-with-imm is validated, P/D should use it like this:

```text
PreparePdReceive:
  allocate D CPU staging and return handle + imm_data

P worker in-process egress:
  fetch rank descriptor from D PegaFlow
  P schedules normal RDMA WRITEs for KV bytes
  P schedules final WRITE_WITH_IMM carrying imm_data

D recv CQ completion:
  publish opaque imm_data to P/D receive manager
  P/D receive manager updates lease/fan-in state
  mark lease READY

GetPdReceiveDescriptor(pd_receive_handle):
  PREPARED/WRITING -> data_ready false
  READY -> data_ready true
```

`CompletePdReceive` remains unnecessary if this path works.

## Open Validation Items

- Confirm immediate byte order (`htonl`/`ntohl`) used by rdma-mummy or raw
  verbs wrapper.
- Decide whether P/D immediate data is a compact integer lease id or an index
  into a D-side table keyed by `(qp, imm_data)`. The transfer layer treats it as
  opaque either way.
- Confirm CQ polling thread placement and NUMA affinity.
- Confirm interaction with existing P2P RDMA READ traffic sharing the same QPs.
- Add production metrics for recv CQ lag, posted recv depth, IMM completions,
  unexpected receiver drop, and repost failures.
