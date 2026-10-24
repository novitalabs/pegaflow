# Review: P0 gate tooling (2026-10-23)

Reviewer: independent agent, fresh context, adversarial ("can a wrong or lazy connector pass?").
Scope: `rsi/gates/{rsi_probe,digest_compare,oracle,workloads,run_gate}.py`, GATES.md "Implementation",
`docs/rsi/models/Qwen3-4B/spec.json`, `pegaflow.semantics.resolve`.
vLLM cross-checks were made against a local vLLM main checkout (`.worktrees/vllm` @3de4b2b, Jul 2026), not
0.32.0. Anything marked **[unverified-0.32]** must be confirmed on .3 before the dependent gate is trusted.

## Verdict: APPROVE-WITH-FIXES

The architecture is right: a passive probe, Kinds taken from the spec rather than from the connector, a
byte digest on the raw buffer, and an output-identity check under batch invariance. But several routes still
give a vacuous pass, and the smoke evidence ("0 mismatches") cannot tell a working probe from a blind one.
M1-M9 must be fixed before any onboarding agent's run is accepted (P3). S4/S5/S9 must be fixed before P4/P5.

## Must-fix

**M1. A degenerate oracle gives E = 1.0.** `oracle.py:139`, `oracle.py:80-81`.
`efficiency = 1.0` when `best == 0 and actual == 0`. Scenario: Qwen3.5 with `mamba_cache_mode=none`
(the Mamba group's block_size can be very large, so the lcm step is greater than `prompt_len-1` and
`resolve` returns 0), or every group skipped (`enable_kv_transfer=False`). A connector that loads nothing
passes G3. Fix: G3 fails (or is marked void, which also fails) when `oracle_tokens == 0`, or when
`oracle_hit_rate` is below a floor set per workload, e.g. 0.5 for gsm8k warm.

**M2. G3 gates total reuse, not connector reuse. That relaxes the written gate.** `oracle.py:126-139`,
`run_gate.py:314-329`.
`E = first_computed / oracle`, and both sides include local GPU hits. In the warm passes most tokens are
local: the gsm8k 8-shot prefix is a local hit for every request after the first wave, and each replay turn
k+1 is a local hit on turn k. A connector that only serves the first wave, or only the common prefix, lands
near 0.9-0.95. GATES G2.3 ("warm external hit rate > 0.9 of oracle") and the G3 prose define H on
external hits; neither is implemented. The "Precise definitions" section claims to be narrower but is
actually looser. Fix: compute per request `ext_i = min(actual_i, oracle_i) - local_i` and
`ext*_i = max(0, oracle_i - local_i)`, then gate `E_ext = Σext/Σext*`. Report
`excess = Σ max(0, actual_i - oracle_i)` as its own number, and fail when it exceeds one block per
request. An honest connector cannot beat an unlimited oracle, so excess means over-reported
`num_matched` (or an oracle bug). The smoke result E = 1.0008 is already excess offsetting misses.

**M3. `hit_compared > 0` can be met by local hits compared against themselves.**
`digest_compare.py:58`, `run_gate.py:279`.
`hit = covered <= first_computed` mixes local prefix-cache hits with external loads. A local hit and its
"reference" are often the same physical block, so they trivially match. G2.digest_self therefore passes
for a connector that never loads a block. Fix: tag `local` (`covered <= local_hit`) and `external`
(`local_hit < covered <= first_computed`) separately. Gate `external_compared > 0` and
`external_mismatched == 0`. Also check that `external_compared` is at least a fixed fraction of the external
hit blocks of the sampled requests.

**M4. digest_vs_baseline passes vacuously when the keys do not line up.** `run_gate.py:289`,
`rsi_probe.py:250-256`.
The pass condition is only `hit_mismatched == 0`. When candidate keys `(rank, group, key, layer)` miss the
baseline, every hit lands in `hit_unreferenced` (ungated). This happens in three cases:
- `_rank_label` falls back to `pid<N>`;
- the arm runs with a different KV layout. If the connector under test lacks SupportsHMA, MultiConnector's
  `all_children_support_hma` is False and vLLM may disable HMA for that arm only, which changes groups and
  block sizes **[unverified-0.32]**;
- layer names change.

Fix: before any cross-arm comparison, require the arm's `spec.json` to equal the baseline's on groups,
spec classes, block sizes, layer→group map and tensor geometry, and fail G0 otherwise. Make
digest_vs_baseline require `hit_compared > 0` and `hit_unreferenced == 0`. Make `_rank_label` raise
instead of falling back.

**M5. The caller decides which verdicts exist and how strict they are.** `run_gate.py:267-342, 353-417`.
The required verdict set comes from caller flags:
- `--arms` without baseline drops G2.gsm8k, outputs, digest_vs_baseline and G4;
- `--digest off` drops both digest gates;
- `--no-batch-invariant` silently drops G2.gsm8k_outputs (`:304`);
- `--workloads gsm8k` drops replay G3/G4;
- `--gsm8k-limit`, `--acc-tol` and `--min-efficiency` relax thresholds.

`g["pass"]` covers only what was evaluated, and a baseline-only run exits 0 because `all([])` is true.
Agents run this tool, so in practice they choose the bar. Fix: define an `official` profile with a fixed
required verdict set and threshold constants in code (GATES.md is the source). Any missing or non-default
item makes the run `official: false`, and a verdict that is required but not evaluable is `fail`, not
absent.

**M6. Replay, the G3/G4 workload, has no correctness check.** `run_gate.py:200-201`.
Replay runs with digest off, without invariance, with `ignore_eos`, and its outputs are never compared. A
connector that over-reports `num_matched` on long multi-turn prompts gets E ≥ 1 and a faster TTFT while
running on garbage KV. Long prompts exercise paths that gsm8k (~1k-token prompts) never touches: chunked
prefill, many-block loads, and eviction. Fix: on replay, run digest_self on a sample and gate external hit
mismatches. With the "any computed occurrence" rule from S2 this needs no batch invariance. Do not count
G3/G4 replay results for a run whose replay digest fails.

**M7. A connector that loads nothing can still read stale KV that happens to be correct.**
`run_gate.py:181-187, 222-224`.
`reset_prefix_cache` only drops hashes (`block_pool.reset_prefix_cache`). GPU memory still holds the cold
pass's KV, including many duplicate physical copies of the shared prefix, because first-wave requests each
compute their own copy. Whether a fake hit is caught is therefore a matter of chance. Fix: after the reset
succeeds (no blocks held), poison the whole KV buffer through a `--worker-extension-cls` method called via
`/collective_rpc` (`VLLM_SERVER_DEV_MODE=1` is already set). Use a **finite** non-zero pattern, because
NaN × 0 in masked attention lanes can poison honest runs. Self-validate: baseline warm outputs must not
change.

**M8. No negative control. The smoke evidence is consistent with a blind probe.**
Every smoke number is "0 mismatched". A probe that digests the wrong region (M9) or a vacuous comparison
(M3/M4) produces the same numbers. Fix: add gate self-test arms under `rsi/gates/selftest/`, to be run on
each new model at onboarding:
- (a) "liar": reports `prompt_len-1` hits and loads nothing;
- (b) "bitflip": loads correctly, then flips one byte in one layer of one loaded block of each group;
- (c) "off-by-one": loads block i+1 into slot i.

Each must fail G2.digest_* and G2.gsm8k_outputs. Record the results in PROGRESS.md.

**M9. Digest geometry is assumed, never checked.** `rsi_probe.py:258-268, 385-401, 422`.
The probe takes the storage of the *first* layer only, trusts `offset + li*layer_stride + b*block_stride`,
and assumes a block's page is `page` contiguous bytes. Two cases would silently digest other memory, which
stays self-consistent so nothing fails:
- vLLM allocates KVCacheTensors as separate storages (each with `offset=0`): every tensor after the first
  reads tensor 0;
- a backend view splits K and V (`(2, num_blocks, …)`): block b's region covers K of other blocks and no V.

This applies to FLASH_ATTN under batch invariance, MLA and Mamba **[unverified-0.32]**. Fix: in
`register_kv_caches`, check each layer view (and each part for list/tuple caches): its `data_ptr` must lie
in the raw storage, and the byte span of `view[block b]` (block-dim stride × index, `numel*elem`) must equal
`[region_base + b*bstride, +page)` for b = 1 and b = num_blocks-1. Raise on mismatch.

## Should-fix

**S1. Gate computed-block exactness under batch invariance.** `run_gate.py:288-295`.
Today `computed_exact` is reported only, and several cases fall into "computed":
- a partial-block load (unaligned `first_computed`, e.g. `prompt_len-1`), where the block straddling the
  boundary is tagged computed;
- loads after preemption/resume (only the first schedule is recorded, `rsi_probe.py:314`);
- a connector writing into other requests' blocks.

None of these are gated. The smoke run shows 37296/37296 exact, so gate `computed_exact == computed_compared`
for digest_vs_baseline whenever batch invariance is on.

**S2. The "first computed occurrence" reference is too strict.** `digest_compare.py:65-66`.
Without byte-invariant compute (no batch invariance, Mamba states that depend on chunking, some MLA
backends), the connector may legitimately save the second computed copy, which gives a false mismatch.
Rule: a hit must equal *some* computed occurrence of the key in REF; this stays sound. Gate the hits of
digest_vs_baseline only when both runs are batch-invariant; otherwise report them.

**S3. Oracle Kind-mapping gaps.** `oracle.py:46-65`.
- `UniformTypeKVCacheSpecs` (likely for gemma-4 because of mixed head sizes **[unverified-0.32]**) raises.
  Map it from the inner `kv_cache_specs`, all of which must yield the same Kind.
- `sliding_window == 1` produces `Window{0}`, which raises.
- Add the spec class to the error message for every unmapped inner spec.

**S4. The Checkpoint oracle is unreachable (P5 blocker).** `oracle.py:86-108`.
The oracle stores a Checkpoint block at *every* block boundary of the computed tokens. vLLM only
materializes Mamba/GDN state at step (chunk) ends (see `saveable`). The checkpoint at
`floor(L/bt)*bt` for the shared gsm8k prefix usually never exists, so H* is too high, an honest Qwen3.5
connector fails E ≥ 0.95, and there is pressure to relax the gate. Fix: in `build_connector_meta`, the probe
records per request the `(step, num_computed_after_step)` pairs (from `num_scheduled_tokens`). The oracle then
stores a Checkpoint block only where a step ended on a boundary (or per vLLM's align-mode chunking rule).
The same data fixes S11.

**S5. The Mamba digest includes padding, and its block→position mapping is unverified (P5 blocker).**
`rsi_probe.py:399, 359-361`.
- `page_size_bytes` includes `page_size_padded`. Hybrid groups share physical memory, so the padding can
  hold stale attention bytes and give false mismatches. Digest only the state content: for each
  `(shape, dtype)`, its sub-range.
- The assumption "block i holds the state at (i+1)*bt" is unverified for align mode **[unverified-0.32]**.
  In `none` mode a single running-state block could be digested under the wrong key.

Verify on Qwen3.5 and digest Mamba groups only in modes where the mapping holds.

**S6. Probe memory blow-up on large pages.** `rsi_probe.py:429-435`.
64 blocks × page × int64, plus the product, is a temporary of about 16×(64×page). With MB-scale Mamba pages
that is GB-scale and can OOM at `--gpu-mem 0.8`. Chunk by bytes (≤64 MiB) instead of 64 blocks.

**S7. No plain-vLLM control.** The probe makes `connector != None`, which changes scheduler paths:
- on vLLM main, hybrid-Mamba local lookup goes through `find_longest_cache_hit_per_group` with
  `local = max(per_group_hits)` (scheduler.py ~715-749), which also changes what `local_hit` means;
- frees are delayed for sampled requests.

Add a `plain` arm (no kv-transfer-config) and require baseline outputs to equal plain outputs on gsm8k.
Verify the hybrid local-hit semantics on 0.32 before P5.

**S8. Load failures still count as hits.** With `kv_load_failure_policy=recompute`, failed sync loads roll
`num_computed_tokens` back *after* the probe has recorded `first_computed`, so G3 is inflated. Fix: in the
scheduler-side probe, implement `update_connector_output`, count `invalid_block_ids`, and fail G3 when any
are seen (or subtract them).

**S9. Batch invariance is per-model knowledge but is set by env (P4/P5 blocker).** `run_gate.py:142-146`.
FLASH_ATTN is wrong for MLA, and invariance for GDN kernels is unknown **[unverified-0.32]**. Move
backend/invariance/`mamba_cache_mode`/prefix-caching flags to `docs/rsi/models/<model>/gate.json` (gate-owned).
Add a control: baseline warm outputs must equal baseline cold outputs. If they do not, G2.gsm8k_outputs is
void, and void counts as fail under M5.

**S10. Unimplemented hard gates.** The following are not implemented:
- G3 "no H drop vs last accepted";
- G4 "no >5% regression vs last accepted";
- G0 arm spec vs committed `docs/rsi/models/<model>/spec.json`;
- G1.

Add `--accepted RUN_DIR`. Until then, state in GATES.md that exit 0 does not cover G0, G1 or the regression
clauses.

**S11. The oracle is optimistic under chunked prefill.** `oracle.py:89-90`.
Prompt blocks become visible right after the first step, but replay prompts (up to 8k) are larger than the
default `max_num_batched_tokens`. This deflates E for honest connectors, mainly through the R2EGym system
prompt shared across conversations. Fix with the per-step data from S4. With M2's per-request cap,
same-step sharing (vLLM caches at allocation) can no longer inflate E.

**S12. G4 is noisy and under-specified.** It is one run of p50 over 48 requests with no repeats or margin.
Baseline warm still reuses turn k locally, so the comparison mostly measures turn 1. Pin
`--max-num-batched-tokens`, `--max-num-seqs` and `--enable-prefix-caching` in the gate config, use ≥3
repeats (or a larger `--replay-convs`), and report the spread.

## Nits

- n1 `rsi_probe.py:12` is over 100 columns. `rsi/` is not covered by `scripts/check.sh` ruff, and `logger`
  (`rsi_probe.py:50`) is unused.
- n2 GATES says `local_hit` comes from the "first schedule attempt", but the code keeps the *last*
  attempt before the first schedule (`rsi_probe.py:304-305`). Align the doc.
- n3 `--gpu` length is not checked against `--tp`.
- n4 `digest_compare.load`: if a digested `req_id` is missing from `requests.jsonl`, `fc=0` and all of its
  blocks silently count as computed. Raise instead.
- n5 The probe's default `prefer_cross_layer_blocks=False` forces MultiConnector to False, so the gate
  forbids cross-layer layouts for the connector under test. Document this or let the probe follow the
  connector.
- n6 The committed `spec.json` was dumped at `max_model_len=4096`, while the gate default is 8192.
  `num_blocks`/`model` path are volatile; exclude them from the G0 comparison (M4).
- n7 `run_gate.py:239-245`: move the 2 s sleep before the after-scrape and before `count_lines`, so late
  finish records land in the right pass.
- n8 Document the contract that the connector under test must read `PEGAFLOW_PORT` (`run_gate.py:150`).

## Resolution (2026-10-23, main session)

| Finding | Fix |
|---|---|
| M1 | G3 fails when `efficiency is None` (no external oracle tokens) or warm `oracle_hit_rate` < 0.5 |
| M2 | `oracle.summarize`: per-request `ext`/`ext*`, `E_ext`, `excess` with a one-block-per-request limit; all gated |
| M3 | blocks tagged `local` / `external` / `computed` from the last episode; gates need `external_matched > 0` and judged >= 0.9 of external blocks on gsm8k |
| M4 | G0.spec compares arm vs baseline (scheduler dump + per-layer view layouts) and baseline vs committed; `unreferenced == 0` gated; `_rank_label` raises |
| M5 | `OFFICIAL` constants; `official: false` with reasons for any deviation; missing required verdict = fail; exit 0 only for official all-pass |
| M6 | replay `cold`/`warm` digest every 4th conversation; G2.digest_self.replay gated; replay G3/G4 void when it fails |
| M7 | `RsiWorkerExt.rsi_poison` via `/collective_rpc` after every reset (0x3F, finite); control `baseline warm == cold` |
| M8 | `rsi/gates/selftest/connector.py` honest/liar/bitflip/offbyone arms; negatives must fail both digest gates (outputs reported only: a single flipped byte need not change greedy text) |
| M9 | `_check_geometry`: every view in the shared allocation, block 1 and N-1 byte spans equal the region math; raises |
| S1 | `computed_exact == computed_compared` gated under invariance |
| S2 | any computed copy is a valid reference; mismatch only when the reference is exhaustive, else `uncertain` |
| S3 | `UniformTypeKVCacheSpecs` -> inner Kind; `sw == 1` -> Scratch; class named in errors |
| S4/S11 | probe records `progress`; oracle stores at the step reaching the block end (checkpoint: exactly on it), visible to requests first scheduled after that step |
| S5 | Mamba digested only in align mode; unpadded `real_page_size_bytes`. Block->position mapping still to verify on Qwen3.5 (P5) |
| S6 | digest chunked by 64 MiB |
| S7 | `plain` arm; `plain_equals_baseline` gated |
| S8 | `update_connector_output` -> `load_errors.jsonl`; G3 fails on any entry |
| S9 | `docs/rsi/models/<model>/gate.json` |
| S10 | G0 vs committed spec implemented; GATES.md states exit 0 excludes G1 and regression clauses (`--accepted` deferred) |
| S12 | scheduler flags pinned; 3 replay passes (1 digested + 2 timed), median of p50 with every value reported |
| n1-n8 | ruff over `rsi/`; docs aligned (local_hit = last attempt); `--gpu`/`--tp` check; missing req raises; cross-layer + `PEGAFLOW_PORT` documented; volatile fields excluded; sleep before scrape |
