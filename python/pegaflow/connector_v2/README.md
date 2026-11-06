# connector_v2

vLLM KV connector driven by `pegaflow.semantics`. Load it with

```
--kv-transfer-config '{"kv_connector": "PegaKVConnectorV2",
  "kv_connector_module_path": "pegaflow.connector_v2", "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {"model_profile": "default"}}'
```

Use `kv_load_failure_policy=recompute`. Under vLLM's default `fail`, any failed
load (for example an expired lease) errors the request. The scheduler logs a
warning at init when it sees `fail`.

Endpoint: `pegaflow_host` / `pegaflow_port` in the extra config, then
`PEGAFLOW_HOST` / `PEGAFLOW_PORT`, default `127.0.0.1:50055`.

## Layering

| Layer | Knows | Must not know |
|-------|-------|---------------|
| `connector.py` | vLLM connector API | models, policy |
| `core/` | semantics + server contract (query, lease, load, save) | vLLM imports, spec classes |
| `adapters/` | one vLLM `KVCacheSpec` class each | requests, server |
| `policy.py` + `models/*.toml` | choices inside the semantic range | GPU layout, server |

- **Hits.** Each transferable group is queried. Storage group 0 is a full
  group and uses the server's prefix query, which can prefetch. The other
  groups use membership queries (local only for now). A model without a full
  group is `Unsupported`.
  - vLLM re-queries a waiting request every scheduling pass, so queries
    that cannot beat vLLM's local hit are skipped. The bound is
    `semantics.resolve` with every unqueried group's blocks counted as
    present, checked before any query and again after group 0. If it is
    not past the local hit, the remaining groups are not queried and
    create no lease (group 0's lease is released). A full-only layout
    skips only when no full block lies past the local hit.
  - A hit vLLM did not allocate (for example, its KV pool was short) is
    kept with its leases. vLLM asks again for the same waiting head every
    pass; if its local hit is unchanged and the hit is younger than
    `Scheduler.HIT_REUSE_MAX_S` (2 s), the same hit is offered again
    without a server query or `policy.accept` (neither input changed).
    That cap is a bound, not a tuned value: it is far below the server's
    lease TTL (600 s) and bounds how long a longer hit, saved meanwhile by
    another request, goes unseen. Otherwise the held leases are released
    and the request is looked up again (or waits, see below).
  - `semantics.resolve` gives the longest legal `p`. `policy.accept` may lower it.
  - Only the blocks in `semantics.needed(g, p)` that are past vLLM's local hit get loaded.
  - Leases that cover none of those blocks are released immediately. Unwanted
    blocks inside a kept lease get `None` targets.
  - If vLLM allocates a shorter hit than planned, and that hit is still a legal
    resume point, the load is re-planned. Otherwise the load is reported failed.
  - Each request gets its own `load` RPC, so one bad lease fails one request.
  - vLLM prefix-caches an async load's blocks only once the load lands. A
    request whose blocks another request is still loading asynchronously
    (with prefix caching on) is not looked up: `get_num_new_matched_tokens`
    returns `None` until those loads finish or fail (`finished_recving` /
    `failed_recving` in `update_connector_output`, or the loader finishes),
    so the shared blocks become vLLM's local hit and only the tail is
    loaded. It waits only if the in-flight blocks give a resume point past
    the local hit (`semantics.resolve`), holds no lease meanwhile, and gives
    up after `loading_wait_ms` (0 disables the wait). Counted as `deferred`
    in `hit stats`.
  - Short hits (at most `sync_load_tokens`, default 256) are loaded
    synchronously: vLLM schedules the request at once and `start_load_kv`
    waits for the copy before the forward (~1 ms). The wait has no deadline:
    until the server signals, its copy may still land in the blocks, so the
    load is neither failed nor done before that. Longer hits load
    asynchronously between steps, so one big copy does not stall the batch.
    A failed sync load is reported as `invalid_block_ids` (the blocks holding
    the hit's tokens), which vLLM recomputes. Sync loads are only used when
    vLLM can recover them: one KV cache group and no block zeroing
    (`needs_kv_cache_zeroing`); otherwise every load is asynchronous.
    The server copies a device's loads one at a time in arrival order, so
    the workers send a step's sync loads before its async ones.
    The server keeps the first copy of a key, so nothing computed on a load
    that may have failed is saved:
    - a request whose blocks include a sync-loaded block saves nothing until
      vLLM has processed that step's output (`max_concurrent_batches` plans
      later); the held range is then offered again;
    - once vLLM reports a failed sync load, the scheduler plans no save for
      the rest of the process (logged as an error). Under the recompute
      policy vLLM evicts nothing: the failed blocks and every block computed
      on them stay in its prefix cache under their correct hashes, so any
      later request (a sharer's next turn, a hit on a preempted loader's
      freed blocks) may extend bad KV. From then on every load is also
      asynchronous, so no further failed load enters vLLM's prefix cache.

    Residual risk: this protects the store, not vLLM's own outputs. Until the
    rewound holder recomputes, requests in this process that hit the failed
    blocks in vLLM's prefix cache (a sharer vLLM did not rewind, a hit on a
    preempted or aborted loader's freed blocks) can produce wrong tokens.
    vLLM 0.32 has no connector hook to evict them.

    An async load needs neither: vLLM schedules the request only after it
    processed the failure and reset its computed tokens.
- **Saves.** Each step, `semantics.saveable(g, before, after)` gives the blocks
  that became storable. `policy.choose_saves` filters them. The saves are
  asynchronous: `wait_for_save` records a CUDA event on the compute stream and
  queues the job; a saver thread waits on the event, then issues the save RPC.
  The step thread never waits on the GPU. Block lifetime:
  - `request_finished` always returns `False`: vLLM frees the blocks at
    once, as with its own `OffloadingConnector`. Each block of the request's
    chosen saves (since its last preemption) is fenced on the plan step of
    its last save. The scheduler sees every block a step writes before the
    forward: new requests' tables past their prefix hit, blocks appended to
    running requests, resumed requests' tables, zeroed and copy-on-write
    target blocks, and load targets. If one is fenced, the plan's
    `fence_step` asks every worker to wait in `handle_preemptions` until the
    saves of plans up to that step are done. A save is done once its RPC
    returned (the server replies after the GPU->CPU copy completed) or
    failed; with one FIFO saver per worker that covers every earlier job.
    A fence clears once every worker reports that step saved
    (`PegaWorkerMetadata.saved_through`, min over ranks), or once a planned
    fence or flush covers it. A block the request shares through the prefix
    cache stays fenced whoever frees it. A fence on a batched (held) job
    sends the batch first (see `save_batch_steps`);
  - a step whose plan has `flush` (vLLM preempted a request and freed its
    blocks, or the layout has window groups whose blocks are freed while the
    request runs) waits for all pending saves in `handle_preemptions`. That
    is before the step's forward writes any block, but vLLM's V2 model runner
    zeroes new blocks and applies copy-on-write copies before it. So layouts
    where vLLM may zero (`needs_kv_cache_zeroing`) or copy (`hash_block_size`
    smaller than a group's `block_size`) blocks, or that have a checkpoint
    group, pin their saves instead: the scheduler takes one reference per
    saved block in vLLM's `BlockPool` (`touch`) when it plans the save and
    drops it (`free_blocks`) once every worker reports that step's saves
    done (`PegaWorkerMetadata.saved_through`, min over ranks). A pinned block
    is never handed out again, so vLLM cannot zero, copy into or reuse it
    while it is read. Such layouts need no flush and no fence, and their
    saves are not batched (pins stay short). `has_pending_push_work` keeps
    vLLM stepping until every planned save is reported done (pins released,
    held batches sent). Without a bound block
    pool these layouts save nothing (fail closed). A pin raises the block's
    `ref_cnt`; vLLM 0.32 reads `ref_cnt == 1` (`is_block_writable`) only for
    speculative Mamba blocks, so check this before enabling spec decode on a
    checkpoint layout;
  - a save the server reports failed still releases its blocks (a failed
    save is a future miss). The save handler reports engine errors as a
    gRPC status, so the RPC raises. Only the binding's `ValueError` proves
    no copy runs. The binding raises it for InvalidArgument, or before any
    RPC is sent; tonic's client never makes InvalidArgument, and the
    handler returns it before it copies or after its copy synchronized. So
    it counts as a failed save.
  - any other raise quiesces the saver (fail closed, then bounded
    recovery). For `Worker.SAVE_QUIESCE_S` (10 s) no save RPC is sent, new
    saves are dropped, and the raised job's fences and pins (and those of
    every later job) hold. Then the job counts as done, the jobs queued
    behind it (held batches released meanwhile included) are dropped
    without an RPC, and saving resumes. Each consecutive raise doubles the
    window, up to `SAVE_QUIESCE_MAX_S` (60 s); a save RPC that stores its
    blocks resets it. Shutdown ends a window early. Each raise is logged
    and counted as `raised` in `save stats`.
    During a window, earlier steps' fences clear as usual. A step waits
    for the rest of the window if it writes a block fenced on the raised
    step or later (also blocks whose save was dropped: the scheduler does
    not know), and every flush does (preemption, window-group layouts).
    Pinned layouts never wait, but their pins of the window's planned
    saves stay in vLLM's block pool until it ends. This covers server
    errors such as an exhausted pinned pool (Internal) or an unknown
    instance (FailedPrecondition). The binding raises those as the same
    class as transport errors (`PegaFlowError`), and the client also
    reports its own decode and HTTP/2 errors as Internal. The window is a
    heuristic, not a proof: no reply bounds a copy the server may still
    run. It is over 3x the longest save job seen in the gate logs (2.98 s
    for compute wait plus RPC; p99 0.63 s over 3853 `save stats` windows).
    A saver thread that dies stops saving for good: fences and holds are
    released, later saves are dropped, and no other copy runs.
    Server limitations:
    - The server seals a job's blocks only after its copy synchronized. But
      the job whose RPC raised may still be copying, and vLLM may reuse its
      blocks once the window ends. A GPU copy that fails part-way returns
      Internal with earlier copies still queued. A call the client drops (no
      reply) is cancelled on the server only when the connection notices.
      Closing that needs a server that answers only once no copy of the job
      runs.
    - An exhausted pinned pool fails in allocation, before any copy of the
      request is issued, so it needs no window. The server reports it as
      Internal, like a part-way copy failure, so the connector cannot tell
      them apart. A distinct reply for errors raised before the copy (for
      example `SaveResponse.status.ok = false`) would make it a plain failed
      save.
    - A save RPC that hangs blocks `flush` and fences for good (logged every
      10 s): the client has no per-call deadline, and the server may still
      be copying.
- **Checkpoint groups** (Mamba / GDN state, vLLM `mamba_cache_mode="align"`).
  vLLM's block table for such a group is null except the running state
  block, so positions do not say which block holds which state. Saves come
  from `SchedulerOutput.kv_connector_block_state.boundary_state_offloads`
  (`(group, block id, boundary tokens)`): `saveable` still decides which
  boundary a step computed, and only an offer for exactly that boundary is
  saved. A boundary vLLM did not offer (it retains only what
  `--prefix-cache-retention-interval` keeps) is not saved. An offered block
  holds its state only until vLLM frees or reuses it, so layouts with a
  checkpoint group pin it (see Saves) until the copy is done. Loads need nothing
  special: the needed block `[p - bt, p)` is always past the local hit, and
  vLLM allocates a fresh state block for an external hit.
- **Hit stats.** The scheduler logs `hit stats` (cumulative) at most every
  30 s while lookups run, whenever the last live request finishes, and at
  shutdown (vLLM may kill the engine core without calling it): lookups, and per set of storage groups missing at the
  longest resume point the full group backs, how often and how many tokens
  were lost, counted once per request and resume point (vLLM re-queries a
  waiting request every scheduling pass) (`missing=` with no group: the policy lowered `p`). This is the
  group-coherence signal for server eviction. Load counters:
  - `deferred`: requests that waited for another request's async load;
  - `sync_planned`: sync loads planned (a sync load is a short hit loaded
    before the forward).

  The same line carries `lookup_cost`: `calls` / `us` / `max_us`, the wall
  time of every server lookup vLLM's scheduler thread waited for (re-queries
  of a waiting request included), `rpcs`, its `query_prefetch` calls, and
  `reused`, re-asks answered from a held hit without a lookup (see Hits).
- **Namespace.** It is a hash of:
  - the adapters' declarations: group kinds, block sizes, per-rank segments
    and shards, transfer backend, page_first, and `LayerDecl.layout` when an
    adapter declares it (e.g. the Mamba state shapes and dtypes);
  - vLLM's `kv_cache_layout` (the page byte order, e.g. LBNHC `[N, H, C]` vs
    LBHNC `[H, N, C]`);
  - the parallel shape;
  - the model: path, revision, quantization;
  - the dtypes and vLLM's prefix hash algorithm.

  Declarations carry sizes, not byte order. So an adapter must reject any byte
  order that neither its declaration nor `kv_cache_layout` names. For example,
  `full_attention` rejects kernel-split HNC pages.
- **Save cost.** Every 500 `wait_for_save` calls (and at shutdown) each worker
  logs `save stats`:
  - steps, and steps with a save;
  - blocks and bytes;
  - on the step thread: `submit_ms`, preemption `flushes` / `flush_ms`, and
    `fences` / `fence_ms` / `max_fence_ms` (steps that waited for a save
    before writing a freed block, and that wait);
  - on the saver thread: `wait_ms` (compute event), `rpc_ms`, `max_job_ms`,
    `max_pending` jobs;
  - failed saves, and `raised` (save RPCs that raised with a copy possibly
    running; each starts a quiesce window); `rpcs` counts save RPCs sent,
    `pressure_rpcs` batches released early on KV pressure;
  - loads, on the step thread: `sync_loads` (sync loads whose RPC the server
    accepted), `sync_first_steps` (steps where a sync load was moved ahead
    of an async one), `sync_wait_ms` / `max_sync_wait_ms` (waiting for the
    sync copies, per step), and `load_steps` / `load_issue_ms` /
    `max_load_issue_ms` (all of `start_load_kv`: every load RPC plus the
    sync wait, over steps with a load plan);
  - async loads, in `get_finished`: `async_done` (load RPCs seen ready,
    landed or failed), `async_first_poll` (of those, ready at the first
    poll, the issuing step's), `async_polls` / `max_async_polls` (polls
    until ready, the landing one included), and `async_ready_ms` /
    `max_async_ready_ms` (from the start of the issuing step's
    `start_load_kv` to the poll that saw it ready). vLLM promotes a request the step after the poll that reports
    it, so 1 poll is its minimum round trip.
- **Init checks.** These run on both roles:
  - unknown spec classes are `Unsupported`;
  - every TP rank's declarations must pass `semantics.seal_plan` (V0).

  At registration, every declared segment must equal `LayerView.bytes_per_block`,
  the size the server copies per segment.

## Adding a model

1. Start vLLM with this connector. An unknown spec class fails at init with
   `Unsupported("no adapter for <Class> (group i) ...")` on both roles.
2. Add `adapters/<name>.py` exposing `ADAPTER` with `group`, `layer` and `view`
   (see the `Adapter` protocol in `adapters/__init__.py`).
3. Add one line to `_REGISTRY`: `"<Class>": "<name>"`.

Registered adapters:

| Spec class | Module | Declares | Onboarded by |
|---|---|---|---|
| `FullAttentionSpec` | `full_attention` | `Group.full(block_size)`; one segment = the whole page (K/V interleaved per cell); `Heads` shard, replicas beyond `max_tp_shards` (checked against `kv_tp_replicas`); dense layer-compact pages only, no kernel-split HNC pages; rejects window, chunked, padded, packed, quantized and `non_causal` specs | Qwen3-4B |
| `MambaSpec` | `mamba` | `Group.checkpoint(block_size)`; one segment = the whole (padded) state page; state shapes/dtypes in `LayerDecl.layout`; `Heads` shard (or `Replicated` for `tp_replicated`); `align` mode only, rejects speculative/prefill-checkpoint blocks, per-token states, DCP | Qwen3.5-4B (in progress) |

The shared rules live in the semantics crate, so an adapter only declares
facts: the dependency kind, the tokens per block, the bytes per segment, and
the shard. It never decides what to load or save.

## Model profiles (the RSI surface)

`policy.py` is the place to tune behaviour. Knobs are read from
`models/default.toml`, then from `models/<profile>.toml` if that file exists.
The profile is `kv_connector_extra_config["model_profile"]`, else vLLM's
`model_config.architecture`.

- Unknown knobs are an error. Profile names must match `[A-Za-z0-9_.-]+`.
- The policy can only narrow what semantics allows:
  - it sees immutable copies (`frozenset`s, a tuple of `SaveItem`s);
  - `accept` is capped at the legal `p`, and the result is re-checked against
    the server answers;
  - `choose_saves` may only return offered items. Anything else raises.

  So a bad policy costs hit rate, never correctness.

## Tests

`python/tests/test_connector_v2_core.py` is part of the default (source-only)
unit gate. It uses a stub adapter, a fake server, and a pure-Python semantics
stub that is injected through the `semantics=` argument of `Scheduler` and
`build_layout`. Cases parametrized `native` re-run against the real
`pegaflow.semantics`, and `test_stub_matches_native` cross-checks the stub.
Both skip without the native extension.
