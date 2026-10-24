# Review: P3 it1, Qwen3-4B onboarding of connector_v2 (2026-10-23)

Reviewer: independent agent, fresh context, adversarial ("can a wrong layout load wrong bytes, or a lazy
connector pass the gate?").
Scope: commit 1b31d7a (`git diff 57e49cf..1b31d7a`): `python/pegaflow/connector_v2/adapters/full_attention.py`,
the registry entry, `python/tests/test_connector_v2_core.py` (+152 lines), README adapter row,
`docs/rsi/models/Qwen3-4B/ITERATIONS.md`, `.claude/skills/pegaflow-new-model/SKILL.md`, PROGRESS.md.
Context read: connector_v2 core (`layout.py`, `scheduler.py`, `worker.py`, `connector.py`), `pegaflow-semantics`,
`python/src/lib.rs`, server register/save/load paths (`pegaflow-server/src/registry.rs`, `service.rs`
register_context_batch/save, `pegaflow-core/src/layout.rs`, `pegaflow-core/src/lib.rs` strided registration
and slot-count check, `offload.rs` batch save), `rsi/gates/run_gate.py`.
vLLM cross-checks were made against the v0.32.0 sources the onboarding agent fetched (`/tmp/vllm032/`:
`kv_cache_interface.py`, `kv_cache_layout.py`, `multi_connector.py`, `sched/scheduler.py`,
`kv_connector_model_runner_mixin.py`, `worker/utils.py`). Claims that rely on runtime behaviour not visible in those files are
marked **[unverified-0.32]**.

## Verdict: APPROVE-WITH-FIXES

The Qwen3-4B layout declaration is correct:
- Geometry. View `[4745, 8, 16, 256]`, stride `[32768, 256, 2048, 1]`, bf16:
  - cells = 8*16*256*2 = 65536 = page, ratio 1, `_dense` holds, `stride[0]*2 == 65536`.
  - The server's `KVCacheLayout` (segments=1, block_stride = 65536) addresses block b at `data_ptr + b*65536`,
    which is exactly vLLM's `create_kv_cache_views` offset for layer l: `l*layer_stride + b*block_stride`.
  - `registry.materialize_tensor` sizes each layer from the view start to the end of the shared allocation,
    so every layer passes the bounds check. Layer 35 ends exactly at the buffer end
    (4745*65536 == layer_stride == 310968320).
- Kernel-block split. The reconstruction (`ratio = block_size // N`, `shape[0] // ratio`) matches vLLM's own
  `create_kv_cache_views` rule: split is only legal when `block_stride == dense page`, and then
  `block_stride //= ratio`.
- Rejections. Block-outer layouts (BLNHC, BLHNC, BHLNC) and LHBNC are rejected with a clear message.
- TP. TP2 is accepted as `Heads(r, 2)`, which is right because the page shrinks with the local head count.
  `max_tp_shards` replicas are handled.
- Gate integrity and isolation pass (see below).

One must-fix: the namespace does not encode the page byte order, so two layouts the adapter accepts share
stored blocks and the README promises the opposite. The gate cannot see this (fresh server per arm). Fix it
before the adapter is reused by P4-P6 or deployed against a long-lived server; it does not block the it1 GPU
slice run.

## Must-fix

**M1. Same namespace for byte-incompatible page layouts. Silent wrong KV across instances or restarts.**
- Where: `python/pegaflow/connector_v2/adapters/full_attention.py:9-12,73-87`,
  `python/pegaflow/connector_v2/core/layout.py:130-137,148-163`,
  `python/pegaflow/connector_v2/connector.py:70-77`, `python/pegaflow/connector_v2/README.md:42-46`.
- The view check accepts both LBNHC (`[N][H][C]` pages) and LBHNC (`[H][N][C]` pages). The test proves both
  are accepted: `test_connector_v2_core.py:871-872`.
- Both declare the same `LayerDecl((page,), Heads(...))`. The namespace doc is built from:
  - groups: spec class, kind, block_tokens, window, per-rank segments + shard repr;
  - parallel;
  - transfer;
  - identity.

  Nothing in it reflects `kv_cache_config.kv_cache_layout` or the view strides, so both layouts produce the
  same namespace.
- The same applies inside LBHNC with a kernel split (ratio > 1). The page is stored as `ratio x [H][N_k][C]`,
  while the unsplit page is `[H][N][C]`: different bytes, same declaration. For LBNHC the split is
  byte-identical, so it is safe.
- Server-side checks do not catch it. Registration only checks bounds, and the load path only checks
  `group_total_slots` ("namespace is shared by incompatible KV layouts"), which is equal here.
- Scenario:
  - vLLM A runs Qwen3-4B with the default LBNHC and saves.
  - Against the same pegaflow-server, vLLM B runs the same model with `VLLM_KV_CACHE_LAYOUT=HND` (alias
    LBHNC), or with an attention backend that resolves to HND. The same can happen when vLLM A is restarted
    with a different backend.
  - B gets external hits and loads pages with heads and tokens transposed. Outputs are garbage, and no error
    is raised anywhere.
  - The gate starts a fresh server per arm and workload, so G2 digest cannot observe this.
- The README states "Any layout change gives a new namespace automatically", which is false today.
- Fix (any one; the first is the cheapest and covers future adapters):
  1. Add `kv_cache_config.kv_cache_layout` (a 0.32 `KVCacheConfig` field, present on both roles) to the
     identity dict or to the namespace doc.
  2. Put a byte-order tag in the adapter's declaration, e.g. `LayerDecl` field or fingerprint entry
     `page_order = "NHC" | "HNC" | f"{ratio}xHNC"`. Note that today `view()` runs only on the worker, while
     the namespace is computed on both roles from `layer()`, so the tag must come from information the
     scheduler also has (`kv_cache_layout`, kernel block size), not from the worker-side strides.
  3. Reject LBHNC (and any split with H > 1 outside N) until it is needed, and drop it from the test.

  Add one table case asserting LBNHC and LBHNC give different namespaces (or that LBHNC is rejected).

## Should-fix

**S1. `non_causal` is not fail-closed.** `python/pegaflow/connector_v2/adapters/full_attention.py:92-100`.
- `_DEFAULT_FIELDS` blocks sliding_window, chunking, padding, head slots, state content and quant mode, but not
  `FullAttentionSpec.non_causal` (vLLM 0.32 `kv_cache_interface.py:593`, prefix-LM bidirectional attention).
- vLLM's docstring says it "does not affect the KV cache layout", which is true for bytes. But in a
  bidirectional span, a block's K/V depends on tokens after the block, so a prefix-keyed reuse can be wrong
  even though the bytes are well formed. vLLM itself adjusts prefix caching for it.
- Scenario: gemma-4-E4B-it (P6) image/prefix-LM layers land in a `FullAttentionSpec` group with
  `non_causal=True`. The adapter accepts it and serves a block saved under a different suffix; G2 may still
  pass if the gate's prompts never diverge after a bidirectional span. **[unverified-0.32]** whether Gemma4
  sets it in 0.32.
- Fix: add `"non_causal": False` to `_DEFAULT_FIELDS` and one row to
  `test_full_attention_rejects_unvalidated_specs`.

**S2. Synchronous save with a full stream sync on almost every decode step is the main G4 risk.**
`python/pegaflow/connector_v2/core/worker.py:156-171` (the sync is at `:164`),
`python/pegaflow/connector_v2/connector.py:122`.
- `wait_for_save` calls `torch.cuda.current_stream().synchronize()` and then a blocking save RPC. The server
  does the GPU->CPU copy inside the RPC; only the index insert is async.
- With 32 concurrent replay conversations, some request crosses a 16-token boundary on nearly every decode
  step. So nearly every step pays a host sync, 36 per-layer copy descriptors and an RPC round trip.
- This defeats vLLM's async-scheduling overlap **[unverified-0.32: async scheduling default and where
  `wait_for_save` sits relative to sampling]**.
- This is already deferred as P2 S5. For the it1 slice, have the run record save ms/step (wall time around
  `Worker.save`) and steps with saves / total steps, so G4 failures can be attributed rather than guessed.
- Baseline warm TTFT p50 is about 2.5 s at concurrency 32, so loads probably still win on TTFT. TPOT/
  throughput regressions are the likelier symptom if G4 includes them.

## Nits

- n1 Spurious warning under MultiConnector. `python/pegaflow/connector_v2/connector.py:96-100`.
  - 0.32 MultiConnector builds each child's `KVTransferConfig` from the child dict only, so the child sees
    the default `"fail"`. The scheduler reads the OUTER config (`sched/scheduler.py:183-184`), and
    `run_gate.py:188` sets `recompute` there. Recompute is therefore effective.
  - This is neither a gate bug nor a connector behaviour bug, just a misleading log line on every gate run.
  - Fix: reword it to "child config says fail; effective policy is the top-level config", or skip it when
    `vllm_config.kv_transfer_config` is not the connector's own config. Do not ask for a gate change: gates
    are human-owned and correct here.
- n2 `max_tp_shards=None` ignores `KVCacheConfig.kv_tp_replicas`. `full_attention.py:49`.
  - When TP > num KV heads and vLLM leaves `max_tp_shards` unset, replicas are declared as distinct `Heads`
    shards. That wastes storage and dedup, but does not corrupt data. It matches vLLM's own "None = not
    replicated" reading, so it is only a nit.
  - Consider cross-checking `tp_size // shards == kv_cache_config.kv_tp_replicas` and raising `Unsupported`
    on disagreement.
- n3 `test_full_attention_registers_one_page_segment` (`python/tests/test_connector_v2_core.py:945-963`)
  mostly re-asserts what the `qwen3-4b-LBNHC` view case and the `tp1` layout case already pin (segment ==
  page, num_blocks, direct backend).
  - Per the CLAUDE.md "merge tests that protect the same contract" rule, fold its only unique assertion
    (36 names reach `register_context_batch` with `segments == 1`) into the `tp1` case.
- n4 The `tp2`/`tp4-replicas` fixtures (`test_connector_v2_core.py:902-908`) keep `page_size_bytes=65536`
  while each rank holds fewer heads. This is harmless because the layout test never builds a view, but it
  reads as if a TP2 page equals the TP1 page. Set the page to the per-rank value to keep the fixture honest.
- n5 The test imports the private `_tp_ranks` (`test_connector_v2_core.py:25`). Assert on `layout`/fake RPC
  output instead, or make the helper public in `core`.
- n6 `_check_spec` compares `kv_quant_mode` (IntEnum) against `0` with `!=` (`full_attention.py:115-120`).
  This is correct today. A comment saying the enum's default value is 0 would stop a future reader from
  "fixing" it to an `is` check.

## Contract with vLLM 0.32 and the gate

- MultiConnector[under_test, RsiProbe] (`rsi/gates/run_gate.py:182-189`):
  - Matched tokens. The first child with a non-zero match wins, and the probe always returns 0, so the
    connector's hit is used. The others get `update_state_after_alloc(..., 0)`, which the core handles by
    releasing.
  - Transfer results are unioned.
  - HMA. Both children declare `SupportsHMA`, so HMA stays on.
  - OK.
- `kv_load_failure_policy=recompute` only on the outer config: effective. See n1, which is not a gate bug.
  Failed loads are reported via `failed_recving`. `_handle_failed_recving` acts on WAITING_FOR_REMOTE_KVS and
  resets `num_computed_tokens` under recompute. OK.
- `PEGAFLOW_PORT` (`run_gate.py:215`): `endpoint_of` reads it when extra is empty. Empty extra config: OK.
  `instance_id` falls back to `engine_id`, which is unique per run.
- Block ids for saves come from `scheduler_output.kv_connector_block_state.get_block_ids`.
  - 0.32 builds `req_ids` from `num_scheduled_tokens` plus boundary offloads, so every request saved in a
    step has ids.
  - A None is retried by keeping `_saved_upto` (`scheduler.py:339-344`). OK.
- Hit cap: lookups are trimmed to `(num_tokens-1)//16` blocks, so the hit never covers the last token. This
  is consistent with vLLM's own last-token rule and with the oracle.

## Gate integrity

`git diff 57e49cf..1b31d7a --stat -- rsi docs/rsi/GATES.md docs/rsi/models/Qwen3-4B/gate.json
docs/rsi/models/Qwen3-4B/spec.json` is empty:
- No gate code, gate doc, gate config or committed spec dump was changed.
- No existing test was removed or loosened; the test diff is additions only.
- PROGRESS.md only appends log lines and updates the P3 status row.

Pass.

## Isolation

The old connector was read only to check for copying.
- The block-size and kernel-split arithmetic is conceptually similar to
  `python/pegaflow/connector/worker.py` registration inference. But the arithmetic is forced by vLLM's public
  `create_kv_cache_views` contract, which itself calls the quantity `ratio`.
- Names, structure and error texts differ. The new code adds things with no analogue in the old code:
  - the `_dense` stride proof;
  - the `_DEFAULT_FIELDS` fail-closed set;
  - `max_tp_shards` / `has_layer_views` handling.
- A grep of the old connector for `max_tp_shards`, `kv_tp_replicas`, `page_size_padded`, `num_head_slots`,
  `has_layer_views` and `_dense`-like helpers finds nothing.
- `TRANSFER_BACKENDS` predates this commit (P2) and mirrors the `python/src/lib.rs` contract.
- ITERATIONS.md cites only allowed sources (vLLM 0.32, `pegaflow-core/src/layout.rs`,
  `pegaflow-server/src/registry.rs`, proto).

No evidence of copying.

## Predicted GPU gate failure modes (it1 slice, then official)

- **G2 digest / outputs.** Expected pass if registration succeeds. Most likely failures, in order:
  1. Registration rejected by the server: per-layer bounds, or `device_id` mapping through
     `CUDA_VISIBLE_DEVICES` under `--gpu 1` (`connector.py:243-260`). The server would see GPU 1 while the
     worker's local ordinal is 0. The mapping looks right; confirm in the log.
  2. A slot-count mismatch at load if the server's per-instance slot math disagrees with 36 one-segment
     layers ("namespace is shared by incompatible KV layouts").
  3. CUDA IPC of 36 views of one storage: each wrapper re-opens the same allocation. This should be fine,
     but if it fails it shows up at registration, not as wrong bytes.

  If digests mismatch while registration succeeded, suspect M1-style byte order first: check that the
  probe's `spec-pp0-tp0.json` stride matches `[32768, 256, 2048, 1]`.
- **G3 E_ext >= 0.95.** Expected around 1.0 on warm passes. Risks:
  1. Visibility lag. The save inserts fire-and-forget (`send_raw_insert`), while the oracle makes prompt
     blocks visible from the next step. Concurrent same-prefix requests in a gsm8k cold wave can miss. This
     affects the cold pass most; warm should be unaffected.
  2. `query_prefetch` errors are swallowed as 0 hits (`scheduler.py:113-118`). Grep the log for
     "query failed".
  3. Lease consumer count = `world_size`, which is fine at TP1.
  4. Evictions: `G3.no_eviction` with a 160 GB pool and a few GB of Qwen3-4B KV per pass should be safe.
- **G4 TTFT.** The async load adds one scheduler iteration (WAITING_FOR_REMOTE_KVS), and every new request does
  a synchronous `query_prefetch` gRPC on the scheduler thread. Both are small next to a 2.5 s baseline
  warm p50. The real risk is S2 (sync saves) slowing steady-state decode and therefore queueing. Collect
  save ms/step in the slice run.

## Resolution (it1b, onboarding agent, 2026-10-23)

Uncommitted on top of 1b31d7a. Local default gate (source-only, stub semantics):
`test_connector_v2_core.py` 69 passed, 43 skipped (native cases need the extension). ruff format/check is
clean. No GPU run: .3 is still "No route to host".

| Finding | Fix |
|---|---|
| M1 namespace vs page byte order | `core/layout.py` adds `kv_cache_config.kv_cache_layout` to the namespace doc. vLLM 0.32 sets it on every `KVCacheConfig` (`engine/core.py:330`), and the scheduler config is a deepcopy of it, so both roles hash the same value. The one byte-order difference the layout name misses is the kernel split: an HNC page split in `ratio` is stored as `ratio x [H, N/ratio, C]`. `full_attention.view` now rejects a kernel-split page unless `stride(N) == H * stride(H)` (NHC, where the split is byte-identical). Tests: `test_namespace_covers_page_byte_order` (LBNHC != LBHNC, same layout gives the same namespace) and view case `LBHNC-split`. The README namespace text now lists `kv_cache_layout` and states the adapter's duty to reject any byte order the namespace does not name. |
| S1 `non_causal` | Added to `_DEFAULT_FIELDS` (fail closed), with a comment on why; new reject case `non_causal`. |
| S2 save cost unobservable | `core/worker.py` `SaveStats`: steps, save_steps, blocks, bytes (from `bytes_per_block * segments` per layer), sync ms, RPC ms, max save-step ms, failed. It logs `[PegaKVConnectorV2] save stats ...` at INFO every 500 `wait_for_save` calls and at `shutdown()`, through the `pegaflow.connector_v2.core.worker` logger. vLLM 0.32 configures only the `vllm` logger (propagate=False), so INFO from `pegaflow.connector_v2.*` would hit Python's last-resort handler and be dropped. `connector.py` now attaches a stream handler to the package logger when no root handler exists, so the line reaches `vllm.log`. The save path itself is unchanged. Covered in `test_worker_save_by_storage_group`. |
| n1 misleading recompute warning | Downgraded to INFO and reworded: the child's own config says `fail`, and the effective policy is the top-level `kv_transfer_config`'s. A comment explains why MultiConnector children always see the default. No gate change. |
| n2 `kv_tp_replicas` ignored | `Parallel` gains `kv_tp_replicas` (from `KVCacheConfig`, both roles; `_with_worker_ranks` now uses `dataclasses.replace`). `full_attention._shard` raises `Unsupported` when the declared replicas per shard are not a multiple of it. vLLM's value is the gcd across layers, so "multiple" is the consistent check. New reject case `max_tp_shards=None` with `kv_tp_replicas=2`. |
| n3 redundant 36-layer test | Folded into `test_full_attention_layout`. Every case (tp1/tp2/tp4-replicas) now builds 36 layers and asserts rank 0's `register_context_batch` payload (segments == 1, bytes == kv_stride == per-rank page, num_blocks, direct backend). |
| n4 TP fixtures with the TP1 page | tp2/tp4 use `HALF_SPEC` (4 kv heads, 32768 B page) and a matching `HALF_KV` view. |
| n5 private `_tp_ranks` import | Removed; the test builds rank `Parallel`s with `dataclasses.replace`. |
| n6 `kv_quant_mode` comparison | Comment: IntEnum, `NONE == 0`, compare by value. |

Not done (by design): no save redesign (S5 stays deferred, now with data from S2), no gate edits.
