# G5 gate re-review 2 (4797a05..e9f3f33)

Reviewer: independent agent, fresh context. Range: d646ac3 (teardown), bb75843
(split_invariant, G2.split_control, same-split computed_exact, gsm8k_outputs
bound), 01f3477 (GATES.md), 7bf8076 (`--under-test-path`), e9f3f33 (one
connection per workload request; scope added mid-review).

Verdict: **accept-with-fixes**. Qwen3-4B's gates are not weakened by this
range (it is split-invariant, so every changed path keeps its strict branch).
On split-variant models the new rules leave warm-pass computed KV unjudged
(M1). `--under-test-path` lets the tree under test replace the probe (M2).
Before any `split_invariant: false` model, or any `--under-test-path` run,
counts as official, M1 and M2 must be fixed.

Prior G5 re-review files: `docs/rsi/reviews/` holds only P0-P3, and no G5
review file exists in any branch or in the history. The prior must-fixes
below were checked against PROGRESS.md "G5 re-review (APPROVE-WITH-FIXES on
20e6b58..6510116) applied".

## Must-fix

### M1. Split-variant: warm-pass computed KV is never judged, and the outputs bound is loose

`digest_compare.py:191-199` judges a computed block only against copies with
the same split. In the gsm8k shape, the baseline's warm hits end at the
few-shot prefix and a connector's warm hits end at the last full prompt
block, so the baseline never has a same-split copy of any warm computed
block. I re-judged the DSV2-Lite TP2 self-test (rev 8af8fe4 records) at
e9f3f33 with `split_invariant=False`, on the host, read-only:

| arm | check | computed judged / unmatched / no-ref | coverage |
|---|---|---|---|
| honest | digest_self | 0 / 2136 / - | **0.000** |
| honest | vs_baseline | 2668 (all `cold`) / 1068 (all `warm`) / 538 (`warm`) | 0.714 |
| liar | vs_baseline | 0 / 28 | 0.000 |
| bitflip | digest_self | 0 / 28 | 0.000 |

So `computed_exact` judges none of the warm pass: coverage is 0 there. The
reported 0.714 comes entirely from the cold pass. No minimum coverage
exists, and judged == 0 is not a fail (`run_gate.py:886-893` only reports
`computed_coverage`). The check can be fully vacuous and still pass, as in
honest digest_self above.

Failure scenario. A connector loads its own blocks correctly, but a late
async load also writes a stale block into a block id that vLLM has already
reallocated to another running request. That id is a computed block of the
victim, not a loaded one, so:
- `external_*` and `local_*` stay clean;
- the victim's computed digest falls in the warm pass and is unjudged;
- with `split_invariant=true` this is caught by `computed_exact`; with false
  it is not.

`G2.gsm8k_outputs` is the only remaining check, and `run_gate.py:840-843`
bounds it as a rate. On full gsm8k that is `0.4905 + 0.02`, so about 26
extra differing completions are allowed. The differing set also need not be
the control's. A corruption that hits a few dozen requests passes →
**wrong connector, all G2 green**.

A much tighter check is already available, and the records show it holds.
On the same 128 prompts:
- honest warm vs plain `split` outputs: **0/128 differ** (vs baseline cold:
  60/128);
- liar, bitflip, offbyone: 128/128 differ.

Fix:
1. With `split_invariant=false`, `G2.gsm8k_outputs` for warm should require
   equality with the plain arm's `split` outputs on the same prompts. Use
   exact equality, or a margin only for the identified `prompt_len % bs == 0`
   requests. Cold should require equality with baseline cold: its split is
   the baseline's (all self-tests show `cold_differ` 0). Today cold gets the
   same 0.51 bound (`run_gate.py:840-843`), which is pure slack.
2. Give warm computed blocks a same-split reference. Run the `split` pass on
   the probed `baseline` arm, not only on `plain`, so the baseline probe dir
   holds same-split copies. Then require a minimum `computed_coverage` per
   pass (e.g. >= 0.9 of warm computed blocks judged), and fail when
   judged == 0 for the arm under test.

### M2. `--under-test-path` can replace the probe, the poison extension or vLLM itself

`run_gate.py:358` puts the under-test dir **first** on `PYTHONPATH` of that
arm's vLLM. Everything the measurement depends on is resolved through that
path: `rsi.gates.rsi_probe` (the probe connector), the
`--worker-extension-cls rsi.gates.rsi_probe.RsiWorkerExt` (poison), and
`vllm`/`torch`. CPython also imports `sitecustomize`/`usercustomize` from
`sys.path` at startup.

Failure scenario: the dir holds `rsi/gates/rsi_probe.py` with a fake
`RsiWorkerExt.rsi_poison` that reports bytes but writes nothing, and a fake
probe that writes consistent fake digests. Poison never happens, and
digests "match" → every G2 check passes for a connector that loads garbage.
A `sitecustomize.py` can monkeypatch the real probe just as easily. Gate
code in the gate process cannot be swapped this way, but the instrument
inside vLLM can.

Fix:
- Reject a path that has any top-level entry other than `pegaflow`. Better:
  stage a temp dir that contains only a symlink/copy of `<dir>/pegaflow`.
- Have the probe record `rsi_probe.__file__`, `vllm.__file__` and the
  extension's module file in `spec.json`. The gate then fails G0 unless they
  resolve under `REPO` and the shared venv.

Current host run: `~/kexi/pegaflow-rsi-g5-7bf8076-ut` contains only
`.rsi-rev` and `python/pegaflow/` (no `rsi`, no `sitecustomize`). That run is
not affected.

### M3. The run is official even when the under-test rev is unknown or unverified

`run_gate.py:1536` records `git_rev(<path>.parent)`, and `official`
(`run_gate.py:1541`) ignores it. Gaps in `git_rev` (`run_gate.py:1250-1255`):
- it returns `"unknown"` without adding a problem;
- `git rev-parse HEAD` searches **upward**, so a non-repo tree nested in any
  checkout records the enclosing repo's HEAD;
- it ignores uncommitted changes, so a dirty worktree records a clean rev;
- `.rsi-rev` is free text that nothing verifies.

Failure scenario: an edited, uncommitted tree runs and the record says
`under_test_rev = <clean commit>`, with `official: true`.

Fix: when `--under-test-path` is set, the run is official only if one of
these holds:
- the path is the top of a git work tree (`--show-toplevel` equals
  `parent`) and `git status --porcelain` is empty; or
- `.rsi-rev` is present and the gate verifies the tree. For example, compare
  `git ls-tree -r <rev> python/pegaflow` blob ids with the files and hash the
  `.so` into the record. The `.so` cannot be tied to a rev, so record its
  sha256.

Spot-check of the current host tree: all 63 tracked files under
`python/pegaflow` match 8a05b7c blob-for-blob, and the only extra file is
the `.so`. I compared hashes only and never displayed contents.

## Should-fix

- S1. The relaxation needs user sign-off. GATES.md:3-4 says agents "may not
  remove or relax anything". `split_invariant=false` relaxes
  `G2.gsm8k_outputs` and `computed_exact`. PROGRESS.md:495 says
  DeepSeek-V2-Lite's gate.json "must set split_invariant false (P4's file)",
  i.e. the onboarding agent writes it. GATES.md:274 calls gate.json
  "gate-owned" and GATES.md:352 says "not declared by onboarding". Resolve
  who commits the field. The claim cannot be abused, because
  `G2.split_control` rejects it both ways (verified below), but the process
  text contradicts itself.
- S2. Bitflip on a split-variant model has no unit test. The loaded-block
  path is unchanged and the records prove it still fails (below). Add a table
  case to `test_computed_exact_judges_same_split_only`: a mismatched
  external block with `split_invariant=False` must fail `_digest_verdict`.
  Also add M1's honest-vs-wrong outputs case once fixed.
- S3. A teardown failure on a reference arm lets the next stage start on a
  live GPU. `stop()` raises if the group survives (`run_gate.py:163-171`),
  but `stage()` (`run_gate.py:1469-1477`) catches every `Exception` for
  `old`/`native`/`lmcache`. Scenario: native's EngineCore is stuck past the
  300 s SIGKILL bound, the RuntimeError is recorded as a reference-arm
  error, and the under-test arm starts on the same GPU with less free
  memory. Its `num_gpu_blocks` and local hits drop, and G5 `E_C`/floor
  shift. Raise a distinct teardown exception that `stage()` re-raises.
- S4. Provenance of the shared components is not recorded, and GATES.md:267
  is wrong. "Reference arms stay at the gate tree's rev" does not hold on the
  host. vLLM's `PYTHONPATH` (`run_gate.py:338`) adds the gate repo root, not
  `REPO/python`. So `pegaflow` (the `old` arm, and `pegaflow.semantics` in
  the gate process: the G3/G5 oracle bar) resolves through the venv's
  `pegaflow_llm.pth` → `/root/kexi/pegaflow-semantics-rsi/python`. The r3
  run's `--pegaflow-server` is `/root/kexi/pegaflow-semantics-rsi/...`. Its
  rev is recorded nowhere, and `pegaflow_server()` (`run_gate.py:202`) puts
  the gate tree's `python/` first while vLLM uses the venv's. Record
  `pegaflow.__file__` + rev per process and the server binary's sha256, and
  fix the doc sentence.
- S5. The split key is lossy: it can cause false fails, never false passes.
  `rsi_probe.py:424-428` records a `progress` end only when a chunk crosses a
  `key_tokens` boundary. A prefill chunk that ends inside a window without
  crossing it (budget-limited) is not in the split key. Two computations can
  then get the same key with different MLA splits → a false `computed_exact`
  failure on an honest connector. Episode start plus chunk ends is otherwise
  right for chunked prefill and preemption: it uses the last episode,
  `num_computed_tokens` at its start, and `progress` from that episode's
  step on. Record every prefill chunk end, or document that the error is
  fail-closed.
- S6. The pre-warm test is weak. `warmed` (`run_gate.py:767-770`) only
  requires `split` to cache more than `split_ref`. A pre-warm that adds one
  block passes. Records show the hit is exactly the last full prompt block
  (1241/1319 Qwen, 1237/1319 DSV2), and one block short for the 78/82 prompts
  with `prompt_len % 16 == 0`. Require
  `cached >= (prompt_tokens - 1) // bs * bs`.

## Nits

- N1. `ref_equals_cold` (`run_gate.py:777`) is reported but never checked.
  If `split_ref` differs from plain cold, the control's basis is not the
  baseline the arm is compared to. Fail the control when the claim is true
  and `ref_equals_cold` is false.
- N2. `rsi/gates/test_oracle_capacity.py` imports `oracle` → native
  `pegaflow.semantics` at module top. The suggested source-only command
  errors at collection, and the e9f3f33 test also needs `aiohttp`, which the
  suggested venv does not install.
- N3. `test_workload_sessions_never_reuse_connections` only checks
  `session()`. It does not check that `run_gsm8k`/`run_replay`/`scrape` use
  it. A grep-style assertion (no `aiohttp.ClientSession(` outside `session`)
  would hold the contract.
- N4. `force_close` adds a TCP connect per request on every arm equally.
  G4 is unaffected, but sub-ms TTFT is not comparable with pre-e9f3f33
  records.

## Checks (what was verified)

1. split_invariant abuse
   - gate.json is read from `REPO/docs/rsi/models/<model>/gate.json`
     (`run_gate.py:1457`), the gate tree, never the under-test path.
   - `_split_control_verdict` (`run_gate.py:798-799`): measured invariant =
     0 output diffs and 0 logprob diffs. It passes only if the claim equals
     the measurement and prewarmed >= 0.9 n. So false on an invariant model
     fails, and true with any divergence (including logprob-only) fails.
     Tests: `false-claim`, `logprob-only`, `vacuous`, `missing`.
   - Required: it sits in `v["baseline"]`, which is required, and
     `v["baseline"]["pass"]` ANDs every sub-verdict. A missing control →
     `pass: False`. The plain arm is required for official
     (`official_problems`), and a plain crash aborts the run (plain is not
     in `_RECORD_ONLY`).
   - The control runs in the same run on the plain arm, so TP, GPU,
     scheduler flags, batch invariance and backend are the arms'. The
     offline controls on the host were plain-only, non-official runs.
   - Records reproduce the author's numbers. Both runs: rev bb75843, full
     gsm8k, prewarmed 1319/1319, `ref_equals_cold` true.

     | model | setup | outputs differ | logprobs differ | first 128 |
     |---|---|---|---|---|
     | Qwen3-4B | TP1 GPU5 FLASH_ATTN | 0/1319 | 0 | - |
     | DSV2-Lite | TP2 GPU2,3 TRITON_MLA | 647/1319 (0.4905) | 1319/1319 | 60/128 |

     Committed Qwen3-4B gate.json has no `split_invariant` → true → the
     control passes.
2. gsm8k_outputs bound
   - Applied per pass (cold and warm, `run_gate.py:840-843`).
   - The prompts are the same: the arm runs the first `n` of the same
     `gsm8k_prompts` list, and `control_rate` uses the first `n` control
     entries.
   - Slack: see M1.
   - The loaded-block path in `digest_compare.compare` is unchanged in this
     range. `local_*`/`external_*` matched/mismatched/unreferenced logic and
     `records()` tagging are byte-identical apart from the extra split
     field. `_digest_verdict`'s mismatched == 0, unreferenced == 0 and
     judged >= 0.9 checks are unchanged.
   - Re-judged at e9f3f33 with `split_invariant=False`, external mismatched
     in both digest checks: liar 2168, bitflip 256 (+254 local), offbyone
     828 (+17272 local). Honest: 1084/1084 matched. This confirms the
     author's claim.
3. Same-split computed_exact
   - The key is correct for chunked prefill and preemption modulo S5.
   - Coverage can collapse to 0 and pass: see M1.
4. `--under-test-path`
   - The `PYTHONPATH` prepend is under-test arm only (`arm_kind ==
     "under_test"`). Selftest, old, native, lmcache, plain, baseline and
     pegaflow-server never get it.
   - Shadowing: M2. Rev: M3.
4b. Teardown
   - `stop()` waits at most 120 s + 300 s, vs 60+30+120+60 before, so the
     bounds were only extended. It raises if the group survives, and
     `_group_gone` reaps the leader before probing. The test passes locally.
   - Ports come from `free_port()`, so a lingering bound port is never
     reused. The GPU can still be reused under S3.
5. Prior must-fixes (from PROGRESS.md). `oracle.py`, `rsi_probe.py` and
   `selftest/` are untouched in this range, and none regressed:
   - cap digests every request (`every = 1 if cap`);
   - exhaustive rule and vacuous-for-reference-arms-only unchanged;
   - `ext = max(ref, own_local)`, MLA `tp / kv_tp_replicas` copies and
     `gpu_kv_bytes` unchanged;
   - `cap_convs: "auto"` still in `OFFICIAL`.
6. Keep-alive fix (e9f3f33)
   - All three aiohttp paths (`run_gsm8k`, `run_replay`, `scrape`) build
     their session through `workloads.session()` with
     `TCPConnector(force_close=True)`. No other `ClientSession` exists under
     `rsi/gates`.
   - No workload request is retried. aiohttp (local 3.12.15, host venv
     3.14.5) retries a dropped persistent connection only for
     `IDEMPOTENT_METHODS` (no POST), so only the `/metrics` GET could be
     resent, which is harmless.
   - `run_gate.reset_and_poison` retries `/reset_prefix_cache` up to 30
     times. That is pre-existing, idempotent and not a probe-recorded
     request. `rsi_poison` and completions are never retried, and `requests`
     defaults to no retries.
   - Host evidence fits the stated cause. Both 7bf8076 official attempts
     (`...-7bf8076`, `...-7bf8076-r2`) died in `baseline/replay/cold` with
     `ServerDisconnectedError`. vLLM logs show `200 OK` up to the gate's own
     SIGTERM, with no server-side error.
7. Unit tests (`PYTHONPATH=. python -m pytest -q rsi/gates`, local macOS)
   - As instructed, collection errors: `oracle.py` needs native
     `pegaflow.semantics`, and there is no Rust toolchain on this machine.
   - With a throwaway stub for `pegaflow.semantics` (and aiohttp
     installed): 42 passed, 18 failed. All 18 are the stub's ImportError
     (oracle/verdict tests that need native semantics), so they were not
     run. Every test added in this range passes (15/15 for `split`, `stop`,
     `bounded`, `never_reuse`).
   - Tests exist for: claim mismatch both ways (yes), bound exceeded (yes,
     `variant-over-bound`), bitflip on a split-variant model (**no**, S2).

## Effect on the G5 official run in progress

The 7bf8076 attempts aborted (keep-alive). The run in progress is
`2026-10-23-qwen3-4b-g5-official-e9f3f33-r3`: gate tree e9f3f33, GPU5, arms
`plain,baseline,old,native,PegaKVConnectorV2`, `--under-test-path
~/kexi/pegaflow-rsi-g5-7bf8076-ut/python` (`.rsi-rev` 8a05b7c). None of
M1-M3 can change its verdicts:
- M1 needs `split_invariant=false`, and Qwen3-4B is true.
- M2: the under-test dir has no `rsi`/`vllm`/`sitecustomize`.
- M3: the rev is resolvable, and its tracked files match 8a05b7c.

S3 could change it only if a reference arm fails teardown. S4 affects the
record's provenance (server binary and `old`/oracle from the unrecorded
pegaflow-semantics-rsi tree), not a verdict.

## Resolution (implementer, 5bffd1e..HEAD)

| item | decision | commit | how |
|---|---|---|---|
| M1 | fixed | 91b6a8e | With `split_invariant=false` the baseline arm also runs `split`, so warm computed blocks have same-split references. `digest_vs_baseline.gsm8k` requires judged > 0 and `compared / (compared + unmatched_split + unreferenced) >= 0.9` separately in `cold` and `warm` (`computed_by_pass`). `G2.gsm8k_outputs`: cold == baseline cold exactly; warm == plain `split` exactly, except prompts with `prompt_len % cache_hit_alignment_tokens == 0` (reported as `warm_differ_vs_split_aligned`). The rate bound and `SPLIT_MARGIN` are gone. Pre-warm requests are tagged `<pass>_prewarm` and never scored. |
| M2 | fixed | 144bfd8 | `--under-test-path` is exposed through `<run>/under-test-overlay/` holding only `pegaflow -> DIR/pegaflow`. Every spec dump (scheduler and each worker) records `modules`: probe, worker extension, vllm, torch, pegaflow (+ native module, site/usercustomize if loaded), prefix. `G0.provenance` (every arm and baseline) fails unless probe/extension resolve under REPO, vllm/torch/prefix to the shared venv, pegaflow to the arm's expected dir. |
| M3 | fixed | 144bfd8 | `provenance.Trees`: verified iff clean git toplevel (`--show-toplevel == root`, `status --porcelain` empty), or `.rsi-rev` whose `git ls-tree -r <rev> python/pegaflow` blob ids equal the files on disk (blob-less fetch from GitHub; untracked native extensions allowed, sha256 recorded). A run with `--under-test-path` and an unverified tree is not official. Host check: 8a05b7c under-test tree verified (63 files). |
| S1 | resolved (process) + doc fixed | 91b6a8e.. | Who commits: the model's onboarding agent, on its own branch (DSV2: P4's 1bfb590); the gate owner never commits a model's gate.json. The claim is not trusted: `G2.split_control` fails a wrong claim either way, in-run. GATES.md now says the value is *checked* by the control. Sign-off: per the coordinator, agent review counts as acceptance and the user confirms in the final pass; listed in PROGRESS.md "Open questions" (GATES.md stays DRAFT). |
| S2 | fixed | 91b6a8e | `test_split_variant_still_fails_wrong_loaded_bytes`, plus M1's outputs table (`variant-one-wrong`, `variant-cold`, `variant-missing`). |
| S3 | fixed | 99146cc | `TeardownError` from `stop()`; `run_stage` re-raises it (also when chained during another error) for reference arms too. |
| S4 | fixed | 144bfd8 | `summary.provenance`: gate tree, venv `pegaflow` tree (`pegaflow_llm.pth` → pegaflow-semantics-rsi), under-test tree, each verified + native sha256; pegaflow-server sha256 and its tree's `.rsi-rev` (claim). Per-process `pegaflow` files in `G0.provenance.pegaflow_files`. GATES.md sentence replaced. |
| S5 | documented | 91b6a8e | Fail-closed (false fails only), documented in `digest_compare` and GATES.md; recording every chunk end not needed for a sound gate. |
| S6 | fixed | 91b6a8e | `cached >= (prompt_tokens - 1) // align * align`, align from the baseline spec dump. |
| N1 | fixed | 91b6a8e | Control fails when `split_ref != plain cold`, whatever the claim. |
| N2 | accepted | - | The gate tests are host-only by design (native `pegaflow.semantics`, aiohttp from the shared venv); the source-only command in CLAUDE.md is for `python/tests`, not `rsi/gates`. |
| N3 | fixed | 39cd826 | Grep assertion: no `ClientSession(`/`TCPConnector(` in `rsi/gates` outside `session()`. |
| N4 | noted | - | TTFT records from before e9f3f33 are not comparable with later ones at sub-ms level; G4 compares arms within one run only, so no verdict depends on it. |

Also in this range (coordinator requests): 940def4 (G2.digest_groups: a
transferred group without digest records fails G2; Mamba groups are listed
as not digested), 8a180e0 (SIGTERM/SIGHUP → `SystemExit`, stage groups torn down), cde6186 (`--pool-size` 64gb, `PoolTooSmall` before a store stage), dc70805 (untimed `_warmup` request per gsm8k/replay stage, isolation asserted), 325fbd0 (contamination: timed replay p99 > 10x baseline → not official).

### Re-judge evidence (offline, gate code 39cd826 + 940def4 rules)

References from other runs (no record has all of them):
- Baseline gsm8k (cold, warm, split; 1319):
  `2026-10-23-dsv2-lite-baseline-split-39cd826` (rev 39cd826, GPU 2,3).
  Same flags as every DSV2 record: TP2, TRITON_MLA, `VLLM_BATCH_INVARIANT=1`,
  max-model-len 8192, batched 2048, seqs 256, gpu-mem 0.8, concurrency 32,
  8 shots, seed 42. Its cold outputs equal each old record's baseline cold
  (0/128 self-test, 0/1319 slices), so batch invariance carries the
  references across runs.
- Plain split control: `2026-10-23-dsv2-lite-split-control` (bb75843).
- Gate settings: P4's `docs/rsi/models/DeepSeek-V2-Lite-Chat/gate.json` at
  1bfb590 (`split_invariant: false`), not committed on this branch.

| record | arm | digest_self | vs_baseline (warm coverage) | outputs: cold / warm vs split (aligned) | expected | result |
|---|---|---|---|---|---|---|
| selftest (8af8fe4) | honest | pass | pass, 1606/1606 = 1.0 | 0 / 0 (0) | pass all | pass |
| selftest | liar | fail | fail (2168 ext mismatched, 0.004) | 128 / 121 (7) | fail both digests | fail |
| selftest | bitflip | fail | fail (256 mismatched, 0.002) | 0 / 121 (7) | fail both digests | fail |
| selftest | offbyone | fail | fail (828 mismatched, 0.014) | 0 / 121 (7) | fail both digests | fail |
| p4-old-slice (b358ceb) | old | pass | pass, 16310/17492 = 0.932 | 0 / 0 (38) | pass G2 | pass G2 |
| p4-it1-slice (b358ceb) | v2 | pass | pass, 17482/17482 = 1.0 | 0 / 0 (0) | pass G2 | pass G2 |

Under the new rules the honest arm's warm computed KV is judged (coverage 1.0 vs 0
before). The negatives now also fail the coverage floor and the exact warm
outputs. The old connector's 38 warm diffs vs `split` are all aligned prompts
(reported only); its 1182 unjudged warm blocks are aligned-prompt splits
(964 other-split, 218 never computed by the baseline). Every arm also fails
`G0.spec`. That comes only from the `modules` field the new probe adds, which
old records lack (a spec-only diff). G4/G5 failures on the slices are the
records' own, unchanged. Qwen3-4B: re-judged all 24 recorded runs with
39cd826: 864/864 pass flags unchanged.
