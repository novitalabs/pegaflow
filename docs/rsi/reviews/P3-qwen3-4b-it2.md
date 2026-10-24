# Review: P3 Qwen3-4B it2 (async saves + sync short loads)

Reviewer: independent review agent (fresh context), 2026-10-23.
Scope: `0bb44bf`, `2ff447e`, `b783f5d`. Verdict: **APPROVE-WITH-FIXES**.
The measured numbers stand (no load failure occurred in any run); the fixes
are on failure paths and future layouts.

## Must-fix

- **M1** A failed sync load still saves blocks computed on top of it
  (`core/worker.py:208-230, 276-288`, `core/scheduler.py:358-430`). vLLM runs
  the step's forward, the new full blocks attend to garbage and are queued
  under correct chained hashes; the server keeps the first copy of a key
  (`pegaflow-core/src/storage/write_path.rs:144`), so later readers get a
  wrong hit. Fix: drop saves of a request whose sync load failed until the
  rewind is visible; test that no save goes out.

## Should-fix

- **S1** On vLLM's V2 model runner the preemption flush runs after block
  zeroing / CoW (`v1/worker/gpu/model_runner.py:1736, 1261, 1265, 2003`;
  V1 is ordered correctly). Safe for Qwen3-4B (no zeroing, hash block ==
  block); unsafe for layouts with zeroing or `hash_block_size < block_size`.
  Docs (README, ITERATIONS it2a, SKILL lesson) overstate the guarantee.
- **S2** A sync load that times out (30 s) is reported invalid while the
  server copy may still land into reused blocks. Wait without a deadline or
  fail hard.
- **S3** Held-out replay slice required by the policy rules not run; the
  "replay hits p50 ≈ 720 tokens" claim has no artifact.
- **S4** A dead saver thread or hung save RPC hangs `flush()` forever and
  leaks held blocks; advance `_completed` in `try/finally`.

## Nits

- `.rsi-rev` one commit behind the synced tests (harmless).
- `_wait_sync` spins with `time.sleep(0)`.
- `policy.py:12` docstring names a model class (review grep rule).

## Verified by the reviewer

Hold/release of finished requests, idle-engine draining, preemption bookkeeping,
prefix-cache sharing, event/stream ordering, RPC client thread safety,
MultiConnector + RsiProbe interplay, sync-load ordering and fallback condition,
isolation (no diff outside connector_v2/tests/docs/SKILL), unit tests (121
passed on .44), and every slice number in ITERATIONS/PROGRESS against
`summary.json`.

Not verified: runtime preemption flush (0 flushes), `Event.synchronize`
releasing the GIL, runtime behaviour of failed/timed-out sync loads.

## Resolution

Filled in by it2c (see `models/Qwen3-4B/ITERATIONS.md` it2c), amended by
it2d after the it2c re-review. Slice `2026-10-23-qwen3-4b-p3-it2c-slice1`
(rev 056eb4b): every verdict passes; gsm8k v2 cold 163.8 s (slice2 164.6),
warm 162.3 s (+0.2%, slice2 +0.3%). It did not show that G4 was unchanged:
replay warm TTFT moved +3.2% / +6.0% vs slice2 (+1.4 / +2.6 pp vs baseline)
and warm3 req/s -1.5 pp, so the final code got its own official run (it2d).

| ID | Resolution |
|---|---|
| M1 | Fixed (50e186b, 056eb4b). Saves of any request holding a sync-loaded block are held for `max_concurrent_batches` plans and re-offered from vLLM's rewound point; requests vLLM did not rewind although they computed on a failed block (shared-block case) never save again (`update_connector_output` -> `Scheduler.on_load_errors`). Async failures are safe: vLLM schedules the request only after processing `failed_recving` and resetting its computed tokens. Test: `test_sync_load_outcome_gates_saves[sync-ok\|sync-failed\|sync-failed-shared-later]`. |
| S1 | Fixed (50e186b). `core.may_zero_or_copy_blocks`: zeroing, `block_size != hash_block_size` (CoW) or unknown -> worker `sync_saves` (wait inside `wait_for_save`, before the next step on both runners). Qwen3-4B stays async. README, ITERATIONS it2a and SKILL lesson corrected. Test: `test_layouts_that_zero_or_copy_save_synchronously`. |
| S2 | Fixed (50e186b). No deadline; poll with 20-200 us backoff sleep, log every 10 s, invalid only on a final error state. Test: `test_sync_loads[sync-slow-ok]`. |
| S3 | Done. Held-out replay 8 x 6 (`r2egym-heldout-from33.jsonl`, conversations 33..) `2026-10-23-qwen3-4b-p3-it2c-heldout`: every replay verdict passes (warm TTFT 0.090 vs 0.236 s, cold req/s -0.4%). Load-size artifact `2026-10-23-qwen3-4b-p3-it2c-load-stats.json`: replay p50 720 (slice) / 864 (held-out) / 3152 (official) tokens, 70-91% async; gsm8k 98% sync. |
| S4 | Fixed in part (50e186b). Jobs complete in `finally`; a dead saver releases every hold, unblocks `flush` and drops later saves as failed. A hung RPC is not released: the PyO3 client has no per-call deadline and a client timeout would not stop the server's copy, so `flush` keeps waiting and logs every 10 s (a server-side deadline is a separate change). Test: `test_async_save_holds_blocks_until_copied[raises\|saver-dies]`. |
| nits | Fixed. `.rsi-rev` = synced commit; `time.sleep(0)` spin replaced (S2); `policy.py` and README name no model class. |

### it2c re-review (resolved in it2d)

| ID | Resolution |
|---|---|
| M1' | Fixed (8a05b7c): fail closed. The first `invalid_block_ids` vLLM reports (`update_connector_output` -> `Scheduler.on_load_errors`) disables every later save for the process and logs an error. Per-request taint tracking was removed: under the recompute policy vLLM evicts nothing, so the bad blocks and every block computed on them stay prefix-cached under correct hashes and any later request (a sharer's next turn, or a hit on a preempted/aborted loader's freed blocks) can extend them. The it2c hold of unconfirmed sync-loaded blocks stays (it covers the window before vLLM reports). Cost: no saves after the first sync-load failure (0 seen in any run). Test: `test_sync_load_outcome_gates_saves[sync-ok\|sync-failed-sharer-next-turn\|sync-failed-loader-preempted]`. |
| should-fix: rerun | Done. Official `2026-10-23-qwen3-4b-p3-it2d` (rev 8a05b7c, GPU4, arms plain/baseline/old/v2): `official: true`, every v2 verdict passes (G0, G2 digests/accuracy/outputs, G3 E_ext 1.0 excess 0, G4 x6). `old` fails G4.gsm8k_warm (+3.4%); this branch's gate still counts it as a required verdict set, so the exit code is 1 for that reason alone. Native ran separately as a record-only reference from the g5 gate tree (see ITERATIONS it2d). |
| should-fix: docs | Done. ITERATIONS it2c header says the M1 hold and S2 sleep-poll run on the measured path; the "within 0.5 pp" claims in PROGRESS and above are corrected. |
| nits | False-taint cases: moot, no taint tracking. `max_concurrent_batches` AttributeError: `getattr(vllm_config, "max_concurrent_batches", pipeline_parallel_size + 1)`. |
