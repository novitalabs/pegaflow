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
  - `semantics.resolve` gives the longest legal `p`. `policy.accept` may lower it.
  - Only the blocks in `semantics.needed(g, p)` that are past vLLM's local hit get loaded.
  - Leases that cover none of those blocks are released immediately. Unwanted
    blocks inside a kept lease get `None` targets.
  - If vLLM allocates a shorter hit than planned, and that hit is still a legal
    resume point, the load is re-planned. Otherwise the load is reported failed.
  - Each request gets its own `load` RPC, so one bad lease fails one request.
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
  - a finished request with saves returns `True` from `request_finished`;
    vLLM keeps its blocks until every worker reports it in `finished_sending`,
    i.e. until every save job submitted before its finish is done (stored or
    failed);
  - a step whose plan has `flush` (vLLM preempted a request and freed its
    blocks, or the layout has window groups whose blocks are freed while the
    request runs) waits for all pending saves in `handle_preemptions`. That
    is before the step's forward writes any block, but vLLM's V2 model runner
    zeroes new blocks and applies copy-on-write copies before it. So layouts
    where vLLM may zero (`needs_kv_cache_zeroing`) or copy (`hash_block_size`
    smaller than a group's `block_size`) blocks save synchronously instead:
    `wait_for_save` returns once the step's saves are done;
  - a save that fails, raises, or whose saver thread died still releases its
    blocks (a failed save is a future miss). A save RPC that hangs blocks
    `flush` and held requests for good (logged every 10 s): the client has
    no per-call deadline, and the server may still be copying.
- **Namespace.** It is a hash of:
  - the adapters' declarations: group kinds, block sizes, per-rank segments
    and shards, transfer backend, page_first;
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
  - on the step thread: `submit_ms` and preemption `flushes` / `flush_ms`;
  - on the saver thread: `wait_ms` (compute event), `rpc_ms`, `max_job_ms`,
    `max_pending` jobs;
  - failed saves.
- **Init checks.** These run on both roles:
  - unknown spec classes are `Unsupported`;
  - so are checkpoint groups, until the core handles vLLM's
    `boundary_state_offloads` (P5);
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
| `FullAttentionSpec` | `full_attention` | `Group.full(block_size)`; one segment = the whole page (K/V interleaved per cell); `Heads` shard, replicas beyond `max_tp_shards` (checked against `kv_tp_replicas`); dense layer-compact pages only, no kernel-split HNC pages; rejects window, chunked, padded, packed, quantized and `non_causal` specs | Qwen3-4B (`docs/rsi/models/Qwen3-4B/ITERATIONS.md`) |

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
