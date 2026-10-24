# Qwen3-4B onboarding iterations (connector_v2)

Spec: `spec.json` (vLLM 0.32.0, TP1, 4090). One transferable group:
`FullAttentionSpec`, 36 layers, block_size 16, bf16, 8 kv heads x 128,
`page_size_bytes == real_page_size_bytes == 65536`, no padding/quant/window,
`kv_cache_layout = LBNHC`, one `KVCacheTensor` with `block_stride = 65536`
(= page) and `layer_stride = num_blocks * page`, hash_block_size 16.

## it1 (2026-10-23): first adapter

Written in isolation. Sources used: the spec dump, the vLLM v0.32.0 source
(fetched from GitHub), `pegaflow.semantics`, the proto/PyO3 client and the
generic `pegaflow-core/src/layout.rs` / `pegaflow-server/src/registry.rs`.

### Design

- **Group**: `Group.full(block_size)`. Qwen3 attends every past token, so
  resuming at `p` needs blocks `[0, p/16)`. Semantics already covers this, so
  no Kind change was needed.
- **Segments**: one segment per block = the whole 65536 B page. The probe's
  layer view is `[B, H=8, N=16, C=256]` with strides `[32768, 256, 2048, 1]`
  (elements), so the page is `[N][H][K128|V128]`. K and V are interleaved per
  (token, head), and two K/V segments would split each block in the wrong
  places. One page is also the unit the attention kernel reads, so a stored
  block round-trips byte for byte.
- **Server addressing**: the gRPC register path takes no block stride, so the
  server addresses block `b` at `data_ptr + b * bytes_per_block` (contiguous,
  one segment). The adapter therefore registers `bytes_per_block =
  kv_stride_bytes = page`. `view()` proves the geometry instead of assuming
  it:
  - the per-block `[H, N, C]` dims tile a dense range;
  - `N` divides `block_size` (kernel-block splitting);
  - the kernel-block cells times the ratio equal the page;
  - dim 0 strides exactly one kernel block.

  Anything else (block-outer layouts such as BLNHC, padded pages, holes) is
  `Unsupported` at init. Silently wrong addressing is not possible.
- **Shard**: `Heads(tp_rank // replicas, shards)`, where shards =
  `min(tp, max_tp_shards or tp)`. TP1 gives `Heads(0, 1)`. Ranks beyond
  `max_tp_shards` hold identical bytes, so `seal_plan` dedupes them. DCP > 1
  is `Unsupported` until a DCP model is onboarded (Tokens shard semantics are
  not pinned down yet).
- **Fail closed** on spec fields this adapter has not been validated on:
  - `sliding_window`, `attention_chunk_size`;
  - `page_size_padded`, `num_head_slots`, `state_content_bytes`;
  - `tokens_per_state != 1`, `kv_quant_mode != 0`;
  - `has_layer_views == False`.

  Each of these needs its own declaration when a model brings it. For
  example, a FullAttentionSpec carrying `sliding_window` (hybrid allocator
  off) is probably still Full for storage, but this is unverified.
- **Policy**: no `models/Qwen3ForCausalLM.toml`. The defaults (min_hit_tokens
  0, loading_wait_ms 1000) are what this model needs, and a profile file
  would only repeat them.
- **Core**: checked against vLLM 0.32 and found compatible, so there are no
  core changes:
  - `kv_connector_block_state.get_block_ids`;
  - `get_transfer_results` / `failed_recving` with recompute;
  - MultiConnector routing: the first child with a hit wins, and the others
    get `update_state_after_alloc(.., 0)`, which releases their leases;
  - `get_loaded_kv_cache_group_ids`;
  - an empty extra config;
  - `PEGAFLOW_PORT` via `endpoint_of`.

### Changes

- `python/pegaflow/connector_v2/adapters/full_attention.py` (new), registered
  as `"FullAttentionSpec": "full_attention"` in `adapters/__init__.py`.
- `python/tests/test_connector_v2_core.py`: adapter section, all cases run on
  both the stub and the native semantics:
  - **view geometry**: Qwen3 LBNHC, LBHNC, kernel-split accepted; BLNHC,
    holes, wrong page rejected;
  - **layout + seal_plan**: TP1, TP2, TP4 with `max_tp_shards=2` replicas;
  - **fail-closed fields**: each unvalidated spec field is rejected;
  - **Qwen3-4B registration**: the 36-layer worker registration as the server
    sees it.
- `connector_v2/README.md`: registered-adapter table.

### Evidence

- Local (Mac, source-only, stub semantics): `test_connector_v2_core.py`
  66 passed, 41 skipped (native cases need the extension). ruff format/check
  clean.
- GPU validation: NOT run. 10.121.196.3 was unreachable from the jump host
  ("No route to host") at both the start and the end of this iteration.
- Verdict table: none yet.

### Next on .3 (when reachable)

1. Sync, write `.rsi-rev`, and rerun `test_connector_v2_core.py` with the
   shared venv. This covers the native cases.
2. Slice gate:
   `run_gate --model ~/kexi/models/Qwen3-4B --tag qwen3-4b-p3-it1-slice --arms plain,baseline,pegaflow.connector_v2:PegaKVConnectorV2 --gsm8k-limit 64 --replay-convs 8 --replay-turns 6 --gpu 1`.
3. Only if every slice verdict passes, run the full official run with the
   same arms and no limits.

### Open risks

- **No end-to-end evidence yet.** Registration size checks are proven only
  against the server's layout rules, read from source; the CUDA IPC path has
  not been exercised. The first gate run may expose:
  - a `size_bytes` / bounds issue;
  - the server's handling of `segments=1` with `kv_stride == bytes_per_block`.
- **G4**: saves are synchronous in `wait_for_save` (deferred S5). The replay
  warm TTFT should still beat baseline because loads skip prefill, but cold
  passes pay the copy inline. Async saves are the likely it2 topic if G4
  fails.
- **Spurious warning under MultiConnector**: vLLM builds each child's
  `KVTransferConfig` from the child dict only, so the child sees
  `kv_load_failure_policy="fail"` (the default) even when the parent says
  `recompute`. The scheduler uses the parent's value, so behaviour is
  correct, but the connector logs a spurious "use recompute" warning. If the
  gate wants the child to see it, `run_gate` should also put
  `kv_load_failure_policy` into the child dict (gate-owned, not changed
  here).
- **Hash trimming**: the scheduler queries at most `(num_tokens-1)//16`
  blocks. This is correct for Full (the last token is always recomputed),
  but a prompt of exactly `16k + 1` tokens gets its full `k` blocks, while
  one of `16k` tokens loses its last block by design.

### Gate concerns (not edited; for the gate owner)

- See the MultiConnector child-config note above. G3 is unaffected.

## it1b (2026-10-23): review fixes

Review: `docs/rsi/reviews/P3-qwen3-4b-it1.md` (APPROVE-WITH-FIXES). Its
Resolution table has the per-finding detail.

### Changes

- **M1:**
  - the namespace now includes vLLM's `kv_cache_layout` (`core/layout.py`);
  - `full_attention.view` rejects kernel-split HNC pages, the one byte order
    the layout name does not capture;
  - README namespace text corrected.
- **S1:** `non_causal` specs fail closed.
- **S2:** `Worker.save_stats`, logged as `save stats` every 500 steps and at
  shutdown. This is the it2 input for the async-save decision. `connector.py`
  routes `pegaflow.connector_v2.*` INFO to stderr: vLLM only configures its own
  `vllm` logger, so core INFO lines were silently dropped before.
- **n1:** the child-config recompute warning is now an accurate INFO line.
- **n2:** `Parallel.kv_tp_replicas`, cross-checked by the adapter.
- **n3:** registration assertions merged into `test_full_attention_layout`.
- **n4:** per-rank pages in the TP fixtures.
- **n5:** the test no longer imports `_tp_ranks`.
- **n6:** comment on the `kv_quant_mode` comparison.

### Evidence

- Local stub run: 69 passed, 43 skipped. ruff clean.
- .3 is still unreachable, so there is no GPU verdict yet.

### What to read from the first slice run

- **Save cost:** `grep "save stats" <arm>/<workload>/vllm.log`. A high
  `avg_save_step_ms` relative to the decode step time, or `save_steps`
  approaching `steps` on replay, means async saves (S5) are it2.
- **Load-failure policy:** the line "own kv_load_failure_policy=fail; the
  effective policy is the top-level ..." is expected under the gate's
  MultiConnector and is not an error.

### Open risks (delta)

- The `kv_cache_layout` namespace input relies on vLLM 0.32 setting it on both
  roles (verified in source: `engine/core.py` sets it on every config before
  the scheduler copy). If a future vLLM leaves it None on one role, the
  scheduler and worker namespaces diverge and registration or queries mismatch
  loudly (no hits), not silently.

### it1b slice gate result (.44, 2026-10-23)

Run `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it1-slice` (64 gsm8k, 8 convs x 6
turns; not official). All verdicts pass.

| Check | Value |
|---|---|
| G0.spec | 0 diffs |
| G2 digest external (gsm8k / replay) | 1492/1492, 617/617 matched; 0 mismatched |
| G2 gsm8k | 0.8594 baseline = cold = warm; outputs identical |
| G3 gsm8k warm | H 0.978, oracle 0.992, E_ext 1.0, excess 0 |
| G3 replay warm | H 0.991, oracle 0.998, E_ext 1.0, excess 0 |
| G3 no_eviction / load errors | pass / 0 |
| G4 replay warm TTFT p50 | 0.101-0.117 s vs baseline 0.209 s |
| replay cold TTFT p50 | 0.266 s vs baseline 0.211 s |

Save stats (gsm8k cold window): 151 of 500 steps save, ~21.5 ms avg (max 34 ms)
per save step, sync 3175 ms vs rpc 77 ms, 0 failed. The synchronous D2H wait
is the whole cost; it explains the +26% cold TTFT. Candidate for it2: S5
async saves, judged by the official run's cold numbers.

Next: official run `qwen3-4b-p3-it1` on GPU1 (in flight). If it passes, mark
it1 accepted and start it2 (async save) or move on to P4.

### it1b under the hard G4 (2026-10-23)

G4 became a hard gate (GATES.md). Re-judging the slice above: warm replay
passes both checks; `G4.replay_cold.ttft_p50` (+26%), `G4.replay_cold.req_per_s`
(-8%), `G4.gsm8k_cold.seconds` (+4.1%) and `G4.gsm8k_warm.seconds` (+3.2%) fail
the 2% overhead bound. it1b is not accepted.

### it1 official run under the hard G4 (2026-10-23)

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it1` (official, GPU1, connector code
of 41fb158). Judged at run time with the old G4: every verdict passed (G0, all
G2 digests 0 mismatch, gsm8k accuracy 0.858 = baseline, G3 oracle hit 0.992 /
0.998, no eviction). Re-judged offline with `_perf_verdicts`:

| check | v2 | baseline | verdict |
|---|---|---|---|
| replay warm TTFT p50 (warm2/3) | 0.515 / 0.524 s | 2.534 / 2.543 s | pass (4.9x) |
| replay warm req/s | 9.60 / 9.55 | 3.44 / 3.43 | pass (2.8x) |
| replay cold TTFT p50 | 1.333 s | 2.797 s | pass |
| replay cold req/s | 6.15 | 3.29 | pass |
| gsm8k cold seconds | 175.4 | 162.0 | FAIL (+8.3%) |
| gsm8k warm seconds | 174.0 | 162.0 | FAIL (+7.4%) |

At 32x12 the replay working set exceeds vLLM's GPU cache, so even the cold
pass reloads earlier turns from PegaFlow and beats baseline; the 8x6 slice was
too small to show this. The remaining failure is per-step save overhead on the
short-prefix gsm8k workload, which is it2's target. it1 is not accepted.

Hit numbers (official): gsm8k warm oracle hit 0.992, replay warm oracle hit
0.998, `E_ext` 1.0 on both, excess within one block per request, no eviction,
digests 0 mismatched (gsm8k 6773 external blocks, replay 18286).

Save stats from the official vLLM logs (`save stats` lines, 500-step windows):

| workload | window | save steps | sync ms | rpc ms | avg ms / save step |
|---|---|---|---|---|---|
| gsm8k | 19.0 s | 429/500 | 10474 | 293 | 25.1 |
| gsm8k | 19.3 s | 419/500 | 9929 | 287 | 24.4 |
| gsm8k | 17.7 s | 314/500 | 8719 | 212 | 28.4 |
| replay | 13.2 s | 281/500 | 8534 | 199 | 31.1 |
| replay | 19.6 s | 325/500 | 8818 | 217 | 27.8 |

The device sync is ~50% of wall time in each window and the RPC ~1.5%, so the
synchronous wait in `Worker.save` explains the gsm8k overhead; it2 removes it.
Reference: the black-box `old` arm (p0-final) is within +0.4% of baseline on
gsm8k cold, so the 2% bound is reachable.

## it2 (2026-10-23): asynchronous saves, synchronous short loads

Written in isolation (vLLM 0.32 source on the remote venv, `connector_v2`,
`pegaflow.pyi`, the generic save/load RPC path in `pegaflow-server/src/service.rs`
`save` and `pegaflow-core/src/gpu_worker.rs`). Two measured changes, each
fixing one per-step overhead the gate showed.

### it2a: asynchronous saves (commit 0bb44bf)

Hypothesis: `Worker.save` blocks the step on `torch.cuda.current_stream().synchronize()`
(~10 s of every ~19 s gsm8k window in the official it1 run), so every step
that saves pays its forward twice.

Change:
- `wait_for_save` records a CUDA event on the compute stream and queues the
  job. A saver thread waits on the event (`Event.synchronize` releases the GIL;
  checked on .44) and then issues the save RPC. The server copies with its own
  stream and only returns after the copy, so "RPC returned" means the bytes
  are in PegaFlow. The event covers the step's forward, so the saver never
  reads KV that is not yet written.
- Block lifetime (vLLM 0.32):
  - `request_finished` returns True for a request that had saves since its
    blocks were last freed. vLLM keeps its blocks (`_free_request` with delay)
    until every worker reports it in `finished_sending`. The worker reports it
    once every job submitted before vLLM told it about the finish is done,
    whether the job stored or failed. The plan's `finished` list carries the
    ids; `start_load_kv` runs before `wait_for_save` of the same step, so the
    fence is "jobs submitted so far". A request with no saves returns False
    (no extra step of block hold).
  - Preemption frees blocks without `request_finished`. The scheduler sees
    `preempted_req_ids` in `build_connector_meta` and sets `plan.flush`; the
    worker waits for every pending job in `handle_preemptions`, which vLLM
    calls before the step's forward. (Corrected in it2c: the V1 model runner
    calls it before zeroing and copy-on-write too, the V2 runner after them;
    safe for Qwen3-4B, which has neither. Layouts that may zero or copy now
    save synchronously.)
  - Window groups free blocks that slide out while the request runs, so any
    layout with a window group flushes before every step (not Qwen3-4B;
    keeps the core safe for P4-P6 until a cheaper rule is measured).
  - Shutdown flushes (30 s bound) before stopping the thread.
- `save stats` now splits step-thread cost (`submit_ms`, `flushes`,
  `flush_ms`) from saver-thread cost (`wait_ms`, `rpc_ms`, `max_job_ms`,
  `max_pending`).

Slice1 (`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2-slice1`, rev 0bb44bf,
full gsm8k 1319 + replay 8x6, GPU2): every verdict passes except
`G4.gsm8k_warm.seconds` 167.5 vs 162.7 s (+2.9%). gsm8k cold 164.4 vs 162.6 s
(+1.1%, was +8.3% in the official it1 run). Replay cold TTFT 0.211 vs 0.210 s
(was +26% in the it1b slice).

Save stats, gsm8k, 500-step windows (official it1 vs slice1):

| | window | save steps | step thread | saver wait | rpc |
|---|---|---|---|---|---|
| it1 (sync) | 19.5 s | 446 | sync 10382 ms | - | 749 ms |
| it2a | 18.2 s | 450 | submit 14 ms, 0 flushes | 10301 ms | 750 ms |

The saver's wait is now off the step thread; `max_pending` stayed at 3 jobs,
so the saver keeps up. 0 failed saves.

### it2b: synchronous short loads (commit 2ff447e)

Why warm gsm8k still failed: the remaining cost was not saves (warm stores
2 new blocks) but the async load round trip. The probe trace shows a warm
gsm8k request (it loads ~5 blocks, p50 64 tokens) waits 7.6 steps on average
(max 15) between admission and its first computed step: one step to plan the
load, the load completes off-step, `finished_recving` arrives with the next
step's output, then vLLM schedules it. vLLM's warm TTFT sum was 282 s vs
179 s for baseline, and vLLM ran 4596 iterations vs 4505. A 5-block load
takes ~1 ms on the server (`pegaflow_load_duration_seconds` 1.39 s / 1319).

Change: hits of at most `sync_load_tokens` (new policy knob, default 256) are
returned as synchronous (`load_kv_async=False`). vLLM schedules the request
in the same step and calls `start_load_kv` before the forward
(`has_sync_kv_loads`); the worker issues the load and spins on the load state
before returning. Failure handling: vLLM already counts the tokens as
computed, so a failed sync load (RPC error, bad state, 30 s timeout, failed
plan) reports the GPU blocks holding the hit's tokens through
`get_block_ids_with_load_errors`, and vLLM rewinds and recomputes them
(`kv_load_failure_policy=recompute`). vLLM can only map invalid blocks back
to requests with one KV cache group, and worker-side block zeroing is not
ordered against the server's copy, so the core enables sync loads only for
layouts with one group and `needs_kv_cache_zeroing == False` (Qwen3-4B);
every other layout keeps async loads. Knob evidence (policy rules): the
gsm8k warm check fails without it (slice1) and passes with it (slice2);
replay warm hits are longer than 256 tokens for most requests (p50 720) and
stay async, so a big copy does not stall a whole batch.

### Slice2 result (all verdicts pass)

Run `~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2-slice2`, rev 2ff447e, GPU2,
full gsm8k (1319) + replay 8 convs x 6 turns, 3 replay passes. Not official
(replay size). Every verdict passes (`pass: true`).

| Check | it2 | baseline | verdict |
|---|---|---|---|
| G0.spec | 0 diffs | | pass |
| G2 digest gsm8k (self / vs baseline) | external 7033/7033 matched, local 169788/169788, 0 mismatched | | pass |
| G2 digest replay (self) | external 617/617, local 3195/3195, 0 mismatched | | pass |
| G2 gsm8k accuracy | 0.8582 cold = warm | 0.8582 | pass (0 pp) |
| G2 gsm8k outputs | 0 cold / 0 warm differ | | pass |
| G3 gsm8k warm | E_ext 1.0, excess 0, 0 load errors | | pass |
| G3 replay warm | E_ext 1.0, excess 0, 0 load errors | | pass |
| G3 no_eviction | 0 in every pass | | pass |
| G4 replay warm TTFT p50 (warm2/3) | 0.0910 / 0.0904 s | 0.2096 / 0.2102 s | pass (2.3x) |
| G4 replay warm req/s (warm2/3) | 6.853 / 6.850 | 5.513 / 5.506 | pass (+24%) |
| G4 replay cold TTFT p50 | 0.2080 s | 0.2095 s | pass (-0.7%) |
| G4 replay cold req/s | 5.073 | 5.084 | pass (-0.2%) |
| G4 gsm8k cold seconds | 164.6 | 163.1 | pass (+0.9%) |
| G4 gsm8k warm seconds | 163.5 | 163.1 | pass (+0.3%) |

Hit rates (prompt tokens; local = vLLM GPU cache, external = loaded by the
connector, total = local + external, H* = oracle hit rate, E_ext from G3):

| arm | pass | local | external | total | H* | E_ext |
|---|---|---|---|---|---|---|
| baseline | gsm8k cold | 0.933 | 0 | 0.933 | 0.933 | - |
| baseline | gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0.0 |
| baseline | replay cold | 0.791 | 0 | 0.791 | 0.791 | - |
| baseline | replay warm/2/3 | 0.791 | 0 | 0.791 | 0.998 | 0.0 |
| v2 | gsm8k cold | 0.933 | 0 | 0.933 | 0.933 | - |
| v2 | gsm8k warm | 0.916 | 0.077 | 0.992 | 0.992 | 1.0 |
| v2 | replay cold | 0.791 | 0 | 0.791 | 0.782 | - |
| v2 | replay warm/2/3 | 0.730 | 0.268 | 0.998 | 0.998 | 1.0 |

External hit rate H (of non-local prompt tokens): gsm8k warm 0.911, replay
warm 0.991, unchanged from it1b. At-capacity numbers (H*_C, H_lru, E_C) are
not produced by this gate version (unlimited pool, `G3.no_eviction`).

vLLM metrics, gsm8k (sum over 1319 requests):

| | TTFT sum | iterations |
|---|---|---|
| baseline warm | 178.9 s | 4505 |
| it2a warm (async loads) | 282.3 s | 4596 |
| it2b warm (sync short loads) | 175.1 s | 4499 |

Save stats (slice2, 500-step windows): gsm8k ~18.3 s windows, 440-451 save
steps, submit 15 ms, 0 flushes, saver wait ~10 s, rpc ~0.8 s, max_pending 3,
0 failed; replay ~14 s windows, 151 save steps, submit ~5 ms, wait ~3 s, rpc
~0.1 s, 0 failed. No `sync load failed` lines.

### Tests

`test_connector_v2_core.py`: the worker fake now injects a compute mark;
new contract tests:
- `test_async_save_holds_blocks_until_copied[saved|save-failed]`: `save`
  returns before the compute mark, a finished request with saves is held and
  reported in `finished_sending` only after its job is done (stored or
  failed), a request without saves is not held.
- `test_save_flush_before_block_reuse`: flush on preemption and always with a
  window group; a preempted request's saves are covered by the flush.
- `test_sync_loads`: short hit is sync and reports nothing on success; a
  failed sync load reports exactly the hit's GPU blocks as invalid; long
  hits and layouts without sync support stay async.
Remote (.44, shared venv): 156 passed (`test_connector_v2_core.py` +
`test_semantics_bindings.py`); ruff format/check clean.

### Open risks

- **Slice, not official.** Replay is 8x6; the official 32x12 run (which also
  exercises GPU-cache pressure and preemption) is the orchestrator's next
  step. `flushes=0` here: the preemption flush path is unit-tested only.
- **Margins.** gsm8k cold +0.9% and warm +0.3% vs the 2% bound; run-to-run
  noise between baselines is ~0.5 s (162.6 vs 163.1 s). The remaining ~1%
  on cold is most likely the saver's D2H copies sharing the GPU with the
  forward (not profiled; inter-token latency sum is +1.0% vs baseline).
- **Held blocks.** A finished request's blocks stay held for one to a few
  steps after finish (until its last save lands). Under heavy memory pressure
  this can cause extra preemption; not seen in the slice. The old arm (black
  box) is the reference to compare in the official run.
- **Sync load stall.** A sync load spins the step thread up to 30 s if the
  server never signals; the policy bound keeps normal loads ~1 ms. Long hits
  remain async.
- **Window groups flush every step** (P4-P6 safety rule, unmeasured cost):
  revisit with data when the first window model is onboarded.
- **Held-out replay slice** (policy rule): not run; `sync_load_tokens` was
  chosen from gsm8k load sizes (p90 96 tokens), not tuned to the replay trace.


### it2 official run (2026-10-23)

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2` (official, GPU1, rev 2ff447e,
32 x 12 replay, full gsm8k). Every required verdict passes (G0, G2 x6, G3 x3,
G4 x6). Exit 0.

| check | v2 | baseline | verdict |
|---|---|---|---|
| replay warm TTFT p50 (warm2/3) | 0.541 / 0.552 s | 2.560 / 2.538 s | pass (4.7x) |
| replay warm req/s | 10.02 / 10.05 | 3.43 / 3.44 | pass (2.9x) |
| replay cold TTFT p50 | 1.242 s | 2.785 s | pass |
| replay cold req/s | 6.80 | 3.30 | pass |
| gsm8k cold seconds | 163.5 | 162.0 | pass (+0.9%) |
| gsm8k warm seconds | 162.4 | 162.4 | pass (+0.0%) |

Hit rates (fractions of prompt tokens):

| arm / pass | local | external | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|
| baseline gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0 | 0 |
| v2 gsm8k warm | 0.922 | 0.070 | 0.992 | 0.992 | 1.0 | 0 |
| baseline replay warm2 | 0.377 | 0 | 0.377 | 0.998 | 0 | 0 |
| v2 replay warm / warm2 / warm3 | 0.35 | 0.65 | 0.998 | 0.998 | 1.0 | 0 |

Digests: gsm8k 6448 external blocks, replay 18531, 0 mismatched. gsm8k
accuracy 0.8582 = baseline. Save stats (replay, 500-step window): step-thread
submit 12 ms, saver wait 9.5 s off the step thread, rpc 277 ms, max 2 pending,
0 failed. Preemption flushes: 0 (no preemption in this run either; the
preemption wait stays unit-test-only evidence). Capacity (G5) not judged: the
G5 gate is still under review fixes; at the pre-fix slice v2 scored E_C 0.40,
limited by server reclaim/eviction (`feat/rsi-server-reclaim`).

Status: accepted for G0-G4 once the it2 code review passes; G5 pending.

## it2c (2026-10-23): it2 review fixes

Review `docs/rsi/reviews/P3-qwen3-4b-it2.md` (APPROVE-WITH-FIXES). Commits
50e186b, 056eb4b (connector_v2 + tests), this docs commit. Mostly failure
paths, but two changes run on the measured Qwen3-4B path (corrected in it2d):
the M1 hold delays the saves of every request that sync-loaded by
`max_concurrent_batches` plans, and the S2 sync-load wait now sleeps
20-200 us per poll instead of `sleep(0)`. Still async saves (`saves=async`)
and sync short loads.

### Changes

- **M1, failed sync load feeds saves.** vLLM schedules step N+1 before it
  processes step N's output (`max_concurrent_batches`, 2 with async
  scheduling), so a failed sync load of step N is computed on in N and N+1
  before vLLM rewinds the request. The scheduler records every sync-loaded
  GPU block as unconfirmed for `confirm_steps = max_concurrent_batches`
  plans; any request whose block table holds one (the loader, or a local
  prefix hit on its blocks) saves nothing in those plans and keeps
  `saved_upto` at `before`, so the range is offered again afterwards, from
  vLLM's rewound `num_computed_tokens` if the load failed. vLLM rewinds only
  the first request holding a shared invalid block
  (`_update_requests_with_invalid_blocks`); the others keep KV computed on
  the bad bytes. So `update_connector_output` forwards `invalid_block_ids`
  to the scheduler, which checks every request alive at the error once at
  its next schedule and stops saving for any whose computed tokens still
  include a failed block (logged `computed on a failed load`). Async load
  failures need neither: vLLM schedules a `WAITING_FOR_REMOTE_KVS` request
  only after it processed `failed_recving` and reset its computed tokens
  (`_handle_failed_recving`, `_update_waiting_for_remote_kv`), and async
  loads never share blocks.
- **S1, V2 runner order.** `core.may_zero_or_copy_blocks(kv_cache_config)`:
  `needs_kv_cache_zeroing`, or any group with `block_size != hash_block_size`
  (partial prefix hits copy-on-write), or an unknown field. Those layouts run
  the worker with `sync_saves`: `wait_for_save` returns only after the
  step's save job finished, before the next step's zeroing or CoW on either
  model runner. Qwen3-4B: no zeroing, hash block == block, so async.
- **S2, sync load deadline.** Removed `SYNC_LOAD_TIMEOUT_S`. The wait polls
  with a sleep doubling from 20 us to 200 us (no busy spin), logs every 10 s,
  and reports invalid only on a final error state.
- **S4, saver death.** Each job completes in `finally` (`_completed`
  advances even on `BaseException`); if the saver thread exits, `poll`
  releases every held request, `flush` returns, and later saves are dropped
  and counted as failed. A hung save RPC is not released: the PyO3 client has
  no per-call deadline (`EngineRpcClient.call` blocks on the tonic future,
  only `connect_timeout`), and a client-side timeout would not stop the
  server's copy into the blocks, so `flush` keeps waiting and logs every
  10 s. Adding a server-side save deadline is a server/client change, not
  done here.
- **Nits.** `policy.py` and the README name no model class; `.rsi-rev` is
  written with the synced commit (056eb4b for the slice).

### Tests

`test_connector_v2_core.py` (131) + `test_semantics_bindings.py` on .44:
166 passed; ruff check/format clean. New or extended contracts:
- `test_sync_load_outcome_gates_saves[sync-ok|sync-failed|sync-failed-shared-later]`:
  no save touching a sync-loaded block in the load step or the step vLLM
  schedules before seeing it; after success the held range is saved; after
  failure the loader re-saves from the rewound point and a request vLLM did
  not rewind never saves; unrelated requests save at once.
- `test_sync_loads[sync-slow-ok]`: a load that stays pending while the
  clock passes any deadline is neither failed nor done before it finishes.
- `test_async_save_holds_blocks_until_copied[raises|saver-dies]`: blocks are
  released, flush returns, and a dead saver drops later saves.
- `test_layouts_that_zero_or_copy_save_synchronously[plain|zeroing|cow|unknown]`.

Full default Python gate: 518 passed, 1 failed
(`test_e2e_version_preflight`: installed pegaflow version vs checkout on
.44; not touched by this change).

### Slice (all verdicts pass)

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2c-slice1`, rev 056eb4b, GPU1,
full gsm8k + replay 8 x 6, same size as it2 slice2. `pass: true` for both
arms.

| Check | it2c | baseline | it2 slice2 (v2 / baseline) | verdict |
|---|---|---|---|---|
| G0.spec | 0 diffs | | 0 diffs | pass |
| G2 digests gsm8k (self, vs baseline) | external 6578/6578, local 170243, 0 mismatched | | 7033/7033 | pass |
| G2 digest replay | external 617/617, local 3195, 0 mismatched | | 617/617 | pass |
| G2 gsm8k accuracy / outputs | 0.8582, 0 differ | 0.8582 | 0.8582 | pass |
| G3 gsm8k / replay warm | E_ext 1.0 / 1.0, excess 0, 0 load errors | | 1.0 / 1.0 | pass |
| G3 no_eviction | 0 | | 0 | pass |
| G4 replay warm TTFT p50 (warm2/3) | 0.0939 / 0.0958 s | 0.2095 / 0.2103 s | 0.0910 / 0.0904 vs 0.2096 / 0.2102 | pass |
| G4 replay warm req/s | 6.865 / 6.788 | 5.519 / 5.521 | 6.853 / 6.850 vs 5.513 / 5.506 | pass |
| G4 replay cold TTFT p50 | 0.2085 s | 0.2108 s | 0.2080 vs 0.2095 | pass |
| G4 replay cold req/s | 5.086 | 5.085 | 5.073 vs 5.084 | pass |
| G4 gsm8k cold seconds | 163.8 | 175.6 | 164.6 vs 163.1 | pass |
| G4 gsm8k warm seconds | 162.3 (+0.2%) | 162.0 | 163.5 vs 163.1 (+0.3%) | pass |

The baseline's gsm8k cold pass (175.6 s) is an outlier: its plain arm ran
138.7 s and its warm pass 162.0 s, and the first pass overlapped the
held-out run's startup on GPU0. So the cold ratio (-6.7%) is not evidence
of a speedup; v2's own cold time moved 164.6 -> 163.8 s (-0.5%) vs slice2,
and v2 cold against slice2's baseline would be +0.4%.

Hit rates (fractions of prompt tokens):

| arm / pass | local | external | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|
| baseline gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0 | 0 |
| v2 gsm8k warm | 0.921 | 0.072 | 0.992 | 0.992 | 1.0 | 0 |
| baseline replay warm/2/3 | 0.791 | 0 | 0.791 | 0.998 | 0 | 0 |
| v2 replay warm/2/3 | 0.730 | 0.268 | 0.998 | 0.998 | 1.0 | 0 |

Save stats: gsm8k 18.1 s windows, 450 save steps, submit 15.8 ms, 0 flushes,
saver wait 10.2 s, rpc 0.77 s, max_pending 3, 0 failed; replay 14.1 s,
163 save steps, submit 5.4 ms, 0 failed. No `sync load failed`,
`save failed`, `computed on a failed load` or `flush waiting` lines.

### Held-out replay slice (S3)

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2c-heldout`, GPU0, replay only,
8 x 6 from `~/kexi/datasets/r2egym-heldout-from33.jsonl` (lines 33.. of
`r2egym-256.jsonl`, i.e. conversations the official run's 32 never use;
`run_gate` has no skip flag). Rev 50e186b: the summary records the
`.rsi-rev` of its start; 056eb4b only changed the failure-tracking path
(`_check_failed_loads`), which never ran (0 load errors). G0 and the gsm8k
checks are void (no gsm8k workload), so the run's overall `pass` is false;
every replay verdict passes.

| Check | v2 | baseline | verdict |
|---|---|---|---|
| G2 digest replay | external 729/729, local 4129, 0 mismatched | | pass |
| G3 replay warm | E_ext 1.0, excess 0, 0 load errors | | pass |
| G3 no_eviction | 0 | | pass |
| G4 replay warm TTFT p50 (warm2/3) | 0.0896 / 0.0894 s | 0.2362 / 0.2372 s | pass (2.6x) |
| G4 replay warm req/s | 7.867 / 7.926 | 6.117 / 6.103 | pass (+29%) |
| G4 replay cold TTFT p50 | 0.2238 s | 0.2549 s | pass |
| G4 replay cold req/s | 5.614 | 5.636 | pass (-0.4%) |

Hits: baseline warm local 0.785 = total (H* 0.998); v2 warm local 0.729,
external 0.269, total 0.998 = H*, E_ext 1.0.

External load sizes per warm request (`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2c-load-stats.json`,
from the probe's `first_computed - local_hit`, script
`~/kexi/rsi-tools/load_stats.py`); sync = at most `sync_load_tokens` (256):

| run / workload | warm reqs | sync | async | p50 | p90 | max |
|---|---|---|---|---|---|---|
| it2 slice2 replay | 117 | 27 | 90 | 720 | 2000 | 3504 |
| it2 official replay (32 x 12) | 792 | 69 | 723 | 3152 | 5792 | 6688 |
| it2c held-out replay | 138 | 42 | 96 | 864 | 1744 | 3936 |
| it2 slice2 gsm8k | 1319 | 1294 | 25 | 64 | 96 | 1152 |

The "replay hits p50 ~720 tokens, mostly async" claim holds on the slice
and the held-out trace (70% async), and more so on the official trace
(91% async); gsm8k is 98% sync.

### Open risks

- M1's hold delays the first save of a request that sync-loaded by one step
  (two plans) on every short hit; no measurable G4 cost in the slice. The
  failure path itself is unit-tested only (0 load errors in every run).
- A hung save RPC still blocks `flush` and keeps held requests (logged);
  fixing it needs a server-side save deadline.
- `sync_saves` layouts (zeroing or CoW) pay the full save wait on the step
  thread, i.e. it1's cost; when such a model is onboarded, measure it and
  find a cheaper ordering point.

## it2d (2026-10-23): fail closed after a failed sync load

it2c re-review (APPROVE-WITH-FIXES), M1': under vLLM's recompute policy
`_handle_invalid_blocks` evicts nothing, and `_update_requests_with_invalid_blocks`
rewinds only the first holder of a shared invalid block. The bad blocks and
every block computed on them therefore stay prefix-cached under their correct
hashes. A later request can extend them and save on top: a sharer's next turn,
or a hit on the freed blocks of a rewound loader that was preempted or aborted
before it recomputed. Per-request taint tracking cannot see these hits.

Fix (8a05b7c, `connector_v2` only): the first non-empty `invalid_block_ids` in
`update_connector_output` calls `Scheduler.on_load_errors`, which sets
`_saves_disabled` and logs an error once; `plan_saves` returns before planning
any save from then on (preemption flushes still run). The it2c taint sets
(`_failed_blocks`, `_to_check`, `_tainted`) are gone; the hold of unconfirmed
sync-loaded blocks for `max_concurrent_batches` plans stays, since it covers the
steps before vLLM reports the failure. `max_concurrent_batches` now falls back
to `pipeline_parallel_size + 1` when the config lacks it. Cost: no saves for the
rest of the process after one failed sync load; no load error was seen in any
run. Async failures stay as in it2c (vLLM resets the request before it runs).

Tests: `test_sync_load_outcome_gates_saves[sync-ok|sync-failed-sharer-next-turn|sync-failed-loader-preempted]`;
166 passed (`test_connector_v2_core.py`, `test_semantics_bindings.py`) on .44,
ruff clean. The teardown backport df01403 (gate owner, fast-forwarded) passes
`rsi/gates/test_teardown.py`.

### it2d official run

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2d` (rev 8a05b7c, GPU4, separate
remote tree `~/kexi/pegaflow-rsi-qwen3-it2d` on PYTHONPATH set outside the
gate; arms plain, baseline, old, v2; 32 x 12 replay, full gsm8k). vLLM did
not log which tree it imported. Corrected in it2e: the it2d review established
provenance from the it2d tree's
`connector_v2/__pycache__/connector.cpython-312.pyc` (mtime 17:42:05, the
second the v2 gsm8k vLLM started) and that tree's connector_v2 `.py` md5s
matching 8a05b7c. The connector now logs its import dir at init (`from=`).
`official: true`. Every v2 verdict passes. `old` fails G4.gsm8k_warm (+3.4%);
this branch's gate still counts `old` as a required verdict set, so the exit
code is 1 for that reason only. The first attempt (`...-it2d-aborted1`) died in
gate teardown (pegaflow-server with a 160 GB pool outlived `proc.wait(30)`
after SIGKILL); the unchanged relaunch is the official run, and df01403 fixes
the teardown for later runs.

| check | v2 | old | baseline | v2 verdict |
|---|---|---|---|---|
| replay warm TTFT p50 (warm2/3) | 0.563 / 0.538 s | 0.677 / 0.601 s | 2.544 / 2.547 s | pass |
| replay warm req/s (warm2/3) | 10.12 / 9.94 | 9.99 / 10.03 | 3.42 / 3.42 | pass |
| replay cold TTFT p50 | 1.287 s | 1.334 s | 2.804 s | pass |
| replay cold req/s | 6.70 | 6.75 | 3.26 | pass |
| gsm8k cold seconds | 165.96 (+0.9%) | 166.73 (+1.4%) | 164.45 | pass |
| gsm8k warm seconds | 164.29 (+0.0%) | 169.93 (+3.4%) | 164.28 | pass (old fails) |

plain gsm8k: 141.6 / 142.0 s. it2 official (rev 2ff447e) for comparison:
replay warm TTFT 0.541 / 0.552 s, gsm8k +0.9% / +0.0%.

| arm / pass | local | external | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|
| baseline gsm8k warm | 0.933 | 0 | 0.933 | 0.992 | 0 | 0 |
| old gsm8k warm | 0.928 | 0.065 | 0.993 | 0.992 | 1.0 | 1170 |
| v2 gsm8k warm | 0.920 | 0.072 | 0.992 | 0.992 | 1.0 | 0 |
| baseline replay warm/2/3 | 0.378 | 0 | 0.378 | 0.998 | 0 | 0 |
| old replay cold | 0.386 | 0.478 | 0.864 | 0.864 | 1.0 | 0 |
| old replay warm/2/3 | ~0.36 | ~0.64 | 0.998 | 0.998 | 1.0 | 195 |
| v2 replay cold | 0.381 | 0.484 | 0.864 | 0.864 | 1.0 | 0 |
| v2 replay warm/2/3 | 0.348 | 0.650 | 0.998 | 0.998 | 1.0 | 0 |

Digests: gsm8k 6643/6643 external (170178 local), replay 18545/18545 (8635
local), 0 mismatched; gsm8k accuracy 0.8582 = baseline; 0 load errors, G3
no_eviction passes. Save stats (second 500-step window): gsm8k 439 save steps,
submit 16.3 ms, rpc 776 ms, max job 60 ms; replay 310 save steps, 102240
blocks, submit 13.2 ms, rpc 1147 ms, max job 190 ms; both max 3 pending,
0 flushes, 0 failed. No `sync load failed` / `saves disabled` line. External
loads: replay 792 warm requests, 69 sync / 723 async, p50 3184 tokens; gsm8k
1300 sync / 19 async.

Status: G0-G4 accepted at 8a05b7c (it2d review ACCEPT-WITH-FIXES, no
must-fix, `docs/rsi/reviews/P3-qwen3-4b-it2d.md`); G5 waits for the G5 r3
rerun. `old` is a reference arm: its G4.gsm8k_warm failure is recorded, not
fixed, and it is the only reason for the gate's exit code 1.

### Native reference (record only, noisy)

vLLM's OffloadingConnector, `--arms baseline,native --workloads gsm8k,replay`,
GPU4, gate trees from `feat/rsi-g5` (not this branch's gate), with up to four
gate runs on the host at once, so timings are noisy and not comparable across
runs. The same-run v2 vs native comparison comes from the G5 r3 official run.

- `2026-10-23-qwen3-4b-p3-it2d-native-aborted1` (gate bb75843): died in
  baseline replay. aiohttp keep-alive race, the gate stopped vLLM itself;
  fixed by 8462ed9.
- `2026-10-23-qwen3-4b-p3-it2d-native` (gate bb75843): complete.

| check | native | baseline (same run) | native G4 |
|---|---|---|---|
| gsm8k cold / warm seconds | 176.8 / 171.6 | 164.4 / 165.0 | fail (+7.5% / +4.0%) |
| replay cold TTFT p50 / req/s | 1.581 s / 4.83 | 2.811 s / 3.27 | pass |
| replay warm TTFT p50 (warm/2/3) | 0.881 / 0.505 / 0.459 s | 2.81 / 2.56 / 2.56 s | pass |
| replay warm req/s (warm/2/3) | 6.81 / 9.63 / 9.76 | 3.26 / 3.42 / 3.41 | pass |

| native pass | local | ext | total | H* | E_ext | excess |
|---|---|---|---|---|---|---|
| gsm8k warm | 0.933 | 0.060 | 0.993 | 0.992 | 1.0 | 1170 |
| replay cold | 0.384 | 0.480 | 0.864 | 0.863 | 1.0 | 0 |
| replay warm / warm2 / warm3 | 0.400 / 0.378 / 0.378 | 0.599 / 0.621 / 0.621 | 0.998 | 0.998 | 1.0 | 195 |

Native fails G0.spec and G2.digest_vs_baseline.gsm8k (0 of 171348 local
blocks match). It runs vLLM with LBHNC instead of the gate's LBNHC, so its
page bytes are permuted. Its KV is not wrong: G2 accuracy 0.8582 and outputs
match, and digest_self passes. Its warm pass is slower than warm2/3 while
its CPU cache fills.

### it3 candidate: load-path cost (not started, no code change)

Trigger: native's warm2/3 TTFT p50 (0.505 / 0.459 s, noisy run above) is
below v2's (0.563 / 0.538 s, it2d official). If the G5 r3 same-run numbers
confirm it, load cost is the next Qwen3 hypothesis. The table measures from
the records named above. Native numbers come from vLLM's `kv_offload_*`
metrics (CUDA-event time on the transfer stream). v2 numbers come from
`pegaflow_load_*` (wall time on the server load thread, including the final
event sync). So the two are close but not identical measures.

| replay warm2 | v2 (it2d official) | native (record only) |
|---|---|---|
| loads / bytes | 265 / 122.4 GB | 264 / 116.7 GB |
| mean bytes per load | 462 MB (~196 blocks x 36 layers) | 442 MB |
| copy time per load, mean | 32.5 ms (histogram: 53 <= 10 ms, 196 <= 50 ms, all <= 100 ms) | 20.2 ms |
| effective H2D bandwidth | 14.2 GB/s | 21.9 GB/s |
| copy unit | one 64 KiB page per (block, layer); `MemcpyBackend` merges runs contiguous on both sides, then one `cuMemcpyAsync` per run | one 64 KiB page per (block, layer tensor), all submitted in one `cuMemcpyBatchAsync` (`ops.swap_blocks_batch`; Triton path only below 28 KiB pages) |
| CPU layout | per layer (layer-first slots) | 36 per-layer CPU tensors ("Allocating 36 CPU tensors") |
| mean TTFT / queue / prefill (vLLM metrics) | 0.750 / 0.671 / 0.057 s | 0.852 / 0.725 / 0.079 s |
| TTFT p50 / p90 | 0.563 / 1.749 s | 0.505 / 2.061 s |
| ms per engine step (pass seconds / steps) | 23.8 | 26.0 |

What the records show:
- Both arms copy the same unit: one 64 KiB page per (block, layer), up to
  ~7000 descriptors per load. Native's GPU layout (LBHNC) is still
  layer-first, and its CPU cache is per layer, so it does not get one
  contiguous span per block across layers. The coordinator's layout guess
  does not hold.
- The difference is submission. Native issues one batched driver call. v2
  issues one `cuMemcpyAsync` per merged run. How many runs v2 actually
  merges is not in the record: the per-load `copies=` line is debug level.
- v2 wins on mean TTFT, queue time and p90. Native wins only on p50. So the
  candidate is about bandwidth per load and the median, not the tail.

Not measured, because the record cannot show it:
- Per-request latency from the scheduler's query to load done.
- Steps between admission and the first computed step for async loads. The
  probe records `first_step` only when vLLM schedules the request after
  `finished_recving`. `progress` is recorded only at 16-token key
  boundaries, so its p50 of 7 steps to first progress is the same for both
  arms and says nothing about load wait.

Next measurements, before any code:
1. A replay warm slice with `RUST_LOG=pegaflow_core::gpu_worker=debug`, to
   get the merged `copies=` per load and the ms per load.
2. The same slice with the existing kernel transfer backend (one launch per
   load, a `LayerDecl.transfer_backend` choice), to see whether submission
   is the cost.
3. Only then consider a batched submit in the server
   (`cuMemcpyBatchAsync`; `memcpy.rs` avoids it for driver instability on
   some stacks).

Any change must keep v2's G4 win on gsm8k: native loses there (+7.5% / +4.0%)
because its store path costs step time.

### it3 measurement (2026-10-24): copies per load, direct vs kernel backend

Measurement only. Copy submission is pegaflow-core's and the server agent
owns it (`feat/rsi-server-reclaim`), so no code changed here. Run
`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it3-backend`:
- replay 32 x 12, GPU4, host busy, judged on in-run deltas only;
- arms baseline, v2 (direct) and v2 with `transfer_backend="kernel"`. The
  kernel arm is a test-only subclass in `~/kexi/rsi-tools/it3_kernel_arm.py`,
  not committed;
- server rev 056eb4b with `RUST_LOG=pegaflow_core::gpu_worker=debug`.

The run is non-official (replay only), so its gsm8k verdicts are "missing".

Per-load numbers come from the server's `Load task completed` lines, 955-957
loads per arm. Descriptors before `merge()` are blocks x 36 layers at
65536 B each (p50 7704 per load); merged runs are not logged.

| per load, >= 100 blocks (n = 767) | direct | kernel |
|---|---|---|
| elapsed p50 | 38.2 ms | 48.3 ms (+26%) |
| bandwidth p50 | 15.0 GB/s | 12.3 GB/s |
| all loads, elapsed p50 / mean | 32.3 / 31.7 ms | 40.6 / 39.0 ms |

| replay pass | direct TTFT p50, req/s | kernel | baseline |
|---|---|---|---|
| cold | 1.249 s, 6.77 | 1.527 s, 5.79 | 2.797 s, 3.28 |
| warm2 | 0.518 s, 10.06 | 0.709 s, 8.56 | 2.552 s, 3.42 |
| warm3 | 0.536 s, 9.99 | 0.705 s, 8.57 | 2.553 s, 3.42 |

Hit rates are the same in both arms: warm2 ext 0.65, total 0.998, E_ext 1.0,
excess 0.

Result:
- The kernel backend is 26% slower per load for Qwen3's 64 KiB pages and
  costs +37% warm2 TTFT, so it is not a candidate.
- The remaining lever is the direct path's submission. Native's one
  `cuMemcpyBatchAsync` per load reaches 21.9 GB/s (different, noisy run)
  against direct's 15.0. That is about 30% of load time, which clears the
  20% bar. But it is a server change, guarded by b96e80c ("cuMemcpyBatchAsync*
  has shown driver instability on some CUDA stacks"; no issue or stack named).
- Numbers sent to the server agent, which is reworking that path for H3 (one
  pinned region per block broke coalescing). If a backend choice ends up as
  a server default, the connector stays unchanged.
- Qwen3 it3 stops here on the P3 side.

## it2e (2026-10-24): it2d review fixes

Review `docs/rsi/reviews/P3-qwen3-4b-it2d.md` (ACCEPT-WITH-FIXES, no
must-fix). None of these changes touches the success path on Qwen3-4B (no
sync-load failure, no `min_hit_tokens` override), so the it2d official run
still stands for G0-G4. The next official run (it3 or the G5 rerun) covers
the final code.

- S1: after `on_load_errors`, every hit loads asynchronously
  (`hit.sync` also requires `not _saves_disabled`). A failed async load's
  blocks never enter vLLM's prefix cache, so a later failure adds no bad KV.
  The README records the residual risk: until the rewound holder
  recomputes, requests that hit the failed blocks in vLLM's prefix cache
  can produce wrong tokens. vLLM 0.32 has no connector hook to evict them.
- S2: `test_lifecycle_after_saves_disabled` covers three cases.
  `preempted-still-flushes` and `finished-still-held` cover a request that
  saved before the failure. `later-loads-async` covers S1. Each of the
  review's two mutants (the early return moved above the preemption loop,
  and `request_finished` returning False once disabled) fails exactly one
  case, and so does dropping the S1 condition.
- S3: the it2d provenance sentence above is corrected. The connector logs
  its import dir (`from=`) at init, the same line as P4's branch.
- S5: `min_hit_tokens` is deleted: no accepted model sets it, and it was a
  no-op at 0. `loading_wait_ms` stays, and its evidence gap is recorded.
  This setup has no SSD or RDMA tier, so the server never answers
  `QueryLoading`, and the it2d official run has 0 `prefetch wait expired`
  lines. Deleting it would change behaviour once a tier exists: a Loading
  answer would become an immediate miss. It needs gate evidence when
  SSD/RDMA is onboarded.
- S4: no action. The exit code is 1 only because of the reference `old`
  arm. Every v2 and baseline verdict passes.

Tests: 167 passed on .44 (`test_connector_v2_core.py`,
`test_semantics_bindings.py`); ruff clean.

### Non-regression slice

`~/kexi/rsi-runs/2026-10-23-qwen3-4b-p3-it2e-slice` (rev fc06c9b, GPU4, the
it2c-slice1 profile: 8 x 6 replay, full gsm8k). vLLM logged
`from=/root/kexi/pegaflow-rsi-qwen3-it2d/python/pegaflow/connector_v2`.

- Correctness passes: G0, G2 (digests, accuracy 0.8582, outputs) and G3
  (E_ext 1.0, excess 0, 0 load errors) all pass, with no `sync load failed`
  line.
- Replay G4 passes: warm TTFT 0.096 / 0.091 s vs baseline 0.213 / 0.212 s;
  warm req/s 6.84 vs 5.48; cold matches baseline.
- gsm8k warm passes: 165.3 s vs 165.4 s.
- gsm8k cold failed: 181.9 s vs 165.5 s (+9.9%).
  - The first save-stats window is 44.5 s (it2d and it2c: 29.3 / 29.1 s);
    later windows are the same as before (18.5 / 18.4 s).
  - vLLM logged a whole 10 s interval, 19:49:16-26, with 0 prompt and 0
    generation throughput while 31 requests were running.
  - That interval matches other runs' pegaflow-server events on the host,
    from the timestamps in every pegaflow.log:
    - P4 `dsv2-lite-p4-it3-rpc-01` EmptyRpcV2 replay: closed 19:47:45, last
      write (teardown end) 19:49:29;
    - P4 `dsv2-lite-p4-it3-rpc-23` Batch8V2 replay: pool alloc from 19:48:24,
      ready 19:50:29;
    - `qwen3-4b-g5-official-e9f3f33-r3` cap025: allocated at 19:46:36.

    A 160 GiB pool teardown and a pool allocation on the same host overlap
    the stall. This is consistent with P4's finding that pool teardown
    stalls other processes. Load average was 86 at the time.
  - The change itself does not touch this path: no sync failure, and
    `min_hit_tokens` was 0.
- gsm8k-only rerun `2026-10-23-qwen3-4b-p3-it2e-gsm8k-rerun` (baseline + v2,
  GPU4): cold 166.1 vs 165.2 s (+0.5%), warm 164.7 vs 165.0 s (-0.2%). Both
  pass, so the slice's cold failure does not reproduce.


## G5 official (r3, 2026-10-23)

Record `~/kexi/rsi-runs/2026-10-23-qwen3-4b-g5-official-e9f3f33-r3` (.44 GPU5,
18:53:55 to 22:00 UTC). `official: true`, `not_official_because: []`.

- Gate rev e9f3f33 (tree `~/kexi/pegaflow-rsi-g5-e9f3f33`, `.rsi-rev`).
- Connector under test: `pegaflow.connector_v2:PegaKVConnectorV2` at
  8a05b7c (`--under-test-path ~/kexi/pegaflow-rsi-g5-7bf8076-ut/python`, a
  `PYTHONPATH` prepend; the run predates the overlay). The tree was checked
  afterwards with `provenance.Trees` (144bfd8): it matches 8a05b7c
  blob-for-blob (63 files).
- Venv `pegaflow` (`old` arm, oracle): 056eb4b at
  `~/kexi/pegaflow-semantics-rsi/python`, also verified.
- pegaflow-server sha256 `d1970ead…c579d7`; its tree claims 056eb4b.
- Run settings: `--pool-size 160gb` (pre-64gb default), cap `auto` (64
  conversations), fractions 0.25/0.5/0.75.
- The run predates 91b6a8e..bf9ef3c, so `summary.provenance`, the warmup
  request and `G2.digest_groups` are absent. None of them changes a Qwen3-4B
  verdict: Qwen3-4B is split-invariant (FLASH_ATTN, `G2.split_control`
  passes), and it has one full-attention group, which is digested.
- Contamination, checked offline with the 325fbd0 rule (timed replay p99 TTFT
  `> 10x` the baseline's p99 in the same pass): none. The largest ratio is
  native `cold` (20.68 s vs 6.38 s, 3.2x); every other arm is below 1x.

Verdict: Qwen3-4B accepted on G0-G4. G5 fails: E_C(0.5) is 0.684 vs 0.9,
which is at LRU level (H_lru/H*_C 0.859).

### Uncapped hit table (tokens)

gsm8k: prompt 1,469,407 tokens per pass; H* is 0.9329 cold, 0.9925 warm.
Accuracy is 0.8582 in every pass of every arm.

| arm | pass | local | ext | total | hit | E_ext |
|---|---|---|---|---|---|---|
| baseline | warm | 1,370,784 | 0 | 1,370,784 | 0.9329 | 0.0 |
| old | warm | 1,361,424 | 98,098 | 1,459,522 | 0.9933 | 1.0 |
| native | warm | 1,370,784 | 88,738 | 1,459,522 | 0.9933 | 1.0 |
| v2 | warm | 1,338,544 | 119,808 | 1,458,352 | 0.9925 | 1.0 |

Cold equals baseline cold (1,370,784 local) for every arm.

Replay: prompt 1,275,208 tokens per pass; H* is 0.8641 cold (native 0.8631),
0.9983 warm.

| arm | pass | local | ext | total | hit | E_ext |
|---|---|---|---|---|---|---|
| baseline | cold / warm / warm2 / warm3 | 482,624 / 482,624 / 481,216 / 481,216 | 0 | same | 0.3785 / 0.3785 / 0.3774 / 0.3774 | 0.0 |
| old | cold | 488,272 | 613,664 | 1,101,936 | 0.8641 | 1.0 |
| old | warm / warm2 / warm3 | 456,672 / 475,248 / 454,224 | 816,515 / 797,939 / 818,963 | 1,273,187 | 0.9984 | 1.0 |
| native | cold | 489,904 | 612,032 | 1,101,936 | 0.8641 | 1.0 |
| native | warm / warm2 / warm3 | 482,512 / 481,856 / 481,856 | 790,675 / 791,331 / 791,331 | 1,273,187 | 0.9984 | 1.0 |
| v2 | cold | 481,648 | 620,288 | 1,101,936 | 0.8641 | 1.0 |
| v2 | warm / warm2 / warm3 | 443,216 / 443,168 / 443,168 | 829,776 / 829,824 / 829,824 | 1,272,992 | 0.9983 | 1.0 |

### G5 (cap trace, 64 conversations x 12 turns)

- U (measured on the baseline's `cap_cold`) is 54.43 GB; the cap_plan
  estimate was 54.43 GB against the 49.27 GB needed.
- GPU KV is 11.16 GB.
- Capacity floor passes: C(0.5)/GPU-KV is 2.44, above the 2.0 minimum.
- The gate is E_C(0.5) >= 0.9; 0.25 and 0.75 are reported only.
- H*_C, H_lru and hit are warm-timed hit rates. H*_C and H_lru are computed
  on each arm's own trace, so they differ slightly between arms.

| C/U | C (GB) | C/GPU-KV | arm | H*_C | H_lru | hit | E_C |
|---|---|---|---|---|---|---|---|
| 0.25 | 13.61 | 1.22 | v2 | 0.8645 | 0.5814 | 0.3987 | 0.084 |
| 0.25 | 13.61 | 1.22 | old | 0.8654 | 0.5815 | 0.3952 | 0.069 |
| 0.25 | 13.61 | 1.22 | native | 0.8653 | 0.5700 | 0.5285 | 0.287 |
| 0.5 | 27.22 | 2.44 | v2 | 0.9290 | 0.8569 | 0.8420 | **0.684** |
| 0.5 | 27.22 | 2.44 | old | 0.9300 | 0.8575 | 0.8324 | 0.681 |
| 0.5 | 27.22 | 2.44 | native | 0.9292 | 0.8579 | 0.8614 | 0.715 |
| 0.75 | 40.83 | 3.66 | v2 | 0.9707 | 0.9155 | 0.8758 | 0.669 |
| 0.75 | 40.83 | 3.66 | old | 0.9708 | 0.9153 | 0.8764 | 0.682 |
| 0.75 | 40.83 | 3.66 | native | 0.9711 | 0.9146 | 0.9269 | 0.770 |

v2 at C/U 0.5:

| metric | value |
|---|---|
| E_C cold | 0.971 |
| E_C own-local | 0.851 |
| H_lru/H*_C | 0.859 |
| freeze/H*_C | 0.623 (the trace separates a frozen store from H*_C) |
| evictions per pass | 14k-25k blocks |
| pool used at pass end | at most 26.4 of 27.2 GB |
| load errors / excess tokens | 0 / 0 |
| loads_ok, pool_ok | true |

`G2.digest_self.cap` passes for v2 at every fraction.

### G4 (vs the in-run baseline; medians)

Slack: replay warm must be strictly better than baseline; replay cold and
gsm8k may be at most 2% worse.

| check | baseline | v2 | old | native |
|---|---|---|---|---|
| replay warm TTFT p50 (s) | 2.780 / 2.785 | 0.538 / 0.542 | 0.606 / 0.633 | 0.499 / 0.491 |
| replay warm req/s | 3.169 / 3.166 | 10.033 / 10.023 | 10.056 / 10.233 | 5.855 / 9.767 |
| replay cold TTFT p50 (s) | 2.973 | 1.254 | 1.340 | 1.537 |
| replay cold req/s | 3.113 | 6.771 | 6.247 | 4.273 |
| gsm8k cold (s) | 164.2 | 165.6 | 166.2 | 168.2 (fail) |
| gsm8k warm (s) | 164.6 | 164.4 | 169.7 (fail) | 174.6 (fail) |

Replay p99 TTFT (s), cold / warm / warm2 / warm3:

| arm | p99 TTFT |
|---|---|
| baseline | 6.38 / 6.52 / 6.25 / 6.27 |
| v2 | 2.99 / 2.14 / 1.81 / 1.82 |
| old | 5.97 / 2.64 / 2.20 / 1.95 |
| native | 20.68 / 2.38 / 19.96 / 2.33 |

Verdicts:

| arm | result | failing checks |
|---|---|---|
| baseline | passes | none |
| v2 (required) | fails | `G5.replay_warm` only |
| old | fails (not required) | `G4.gsm8k_warm.seconds`, `G2.digest_self.cap`, `G5` |
| native | fails (not required) | `G0.spec` (LBNHC vs LBHNC layout), `G2.digest_vs_baseline.gsm8k`, G4 gsm8k, `G5` |
