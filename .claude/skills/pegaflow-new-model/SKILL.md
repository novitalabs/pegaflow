---
name: pegaflow-new-model
description: Onboard a new model to the PegaFlow vLLM connector in RSI mode (spec dump -> adapter/policy -> gates -> review). Use when adding support for a new model or KV layout, or when resuming the RSI effort in docs/rsi/.
---

# PegaFlow new-model onboarding (RSI loop)

This skill is the main process for supporting a model. Keep improving it: when a
step was missing, wrong, or slow, edit this file in the same iteration and log
the change in `docs/rsi/PROGRESS.md`.

## 0. Resume first

1. Read `docs/rsi/PROGRESS.md` (state, last iteration, open questions).
2. Read `docs/rsi/GATES.md` (what "done" means). Never relax a gate.
3. Read `docs/rsi/models/<model>/ITERATIONS.md` if it exists.
4. Write what you are about to do into PROGRESS.md before doing it.

## 1. Isolation

Do not read, grep, or copy:
- `python/pegaflow/connector/**` (old connector)
- model branches in `pegaflow-server/src/service.rs`, `pegaflow-core/src/instance.rs`,
  `pegaflow-core/src/offload.rs`
- git history of past model adaptations

Derive everything from: the vLLM 0.32 source, the model's spec dump, the
`pegaflow-semantics` API, the new connector package, and gate output. The old
connector is a black-box baseline (run it, read its numbers, not its code).

## 2. Loop for one model

1. **Spec dump + gate self-test** (G0): any gate run writes the dump (the
   `RsiProbe` stacked in every arm). Run the self-test command from GATES.md
   ("Self-test at onboarding"): plain, baseline and the four `selftest:` arms on
   128 gsm8k questions. Copy `RUN_DIR/baseline/gsm8k/probe/spec.json` to
   `docs/rsi/models/<model>/spec.json`, and have the gate owner write
   `docs/rsi/models/<model>/gate.json` (batch invariance, backend,
   `mamba_cache_mode`). Accept the model's gate only if the baseline controls
   pass, honest passes and every negative arm fails both digest checks;
   otherwise the gate is blind on this model and must be fixed first. Note
   groups, spec classes, block sizes, page bytes, TP shard info. If
   `oracle.py` raises "no Kind mapping", or the self-test connector raises
   NotImplementedError (non-full groups), stop: the gate owner maps the spec
   class first (gates are not the agent's to edit).
2. **Map to semantics**: each group -> (Kind, Shard, unit). If no existing Kind
   fits, propose a new one in `pegaflow-semantics` with V1 vectors, log it in
   PROGRESS.md "Semantics changelog" (no need to ask the user), and have the
   review agent audit it.
3. **Adapter/policy**: add the minimum adapter code for unknown spec classes;
   put model knobs in `models/<model>`. Prefer policy changes over adapter changes.
4. **Gates**: on .3 from `~/kexi/pegaflow-semantics-rsi`, run
   `python -m rsi.gates.run_gate --model ... --tag <model>-it<N> --arms baseline,old,<module>:<Class>`
   (full command in GATES.md "Implementation"; always include `plain`). Run
   record lands in `~/kexi/rsi-runs/<date>-<tag>/summary.json`; paste verdicts +
   hit table into ITERATIONS.md. Iterate on small slices first
   (`--gsm8k-limit 64 --replay-convs 8 --replay-turns 6`); such runs are
   `official: false` and exit 1 by design. Accept only an official run
   (`summary.json` "official": true, empty `not_official_because`).
   The connector under test must read `PEGAFLOW_PORT` and tolerate an empty
   `kv_connector_extra_config`; vLLM is started with
   `kv_load_failure_policy=recompute`.
5. **Optimize**: hit-rate levels (E_ext, E_C) are reported, not gated (user,
   2026-10-24; GATES.md "Hit rate"). Write a "hit-rate judgment" in
   ITERATIONS.md for every accepted run: measured hit rate vs H* / H*_C and
   the reference arms (H_lru, old, native), every gap explained, and whether
   it is close enough. If it is not, or perf is below target, form a hypothesis from metrics (fallbacks per group, unused
   loads, write amplification, stranded bytes per group, save/load time), change
   the lowest layer that fixes it, rerun. One hypothesis per iteration, record
   result even if negative. Hypothesis checklist (extend it when you find more):
   - window groups: save only the trailing window, not every block;
   - checkpoint groups: save only at boundaries that requests resume from;
   - state that cannot be resumed (`prefix_cacheable=False`, ring buffers): never save;
   - eviction must be group-coherent: a resume point needs every group's data,
     so evict a prefix's groups together and prefer evicting the tail;
   - scan resistance: one-pass sweeps must not flush reused prefixes;
   - save cost: no synchronous wait on the model step.
   Do not block a model on hit-rate level alone; correctness checks stay hard.
   Do not chase the highest hit rate: no eviction policy is globally optimal,
   so H*_C is a ceiling, not a target.
   The server is out of scope (user, 2026-10-24): gates run on the base
   pegaflow-server built from `feat/semantics-rsi`, and agents do not change
   or tune it. When the evidence points at the server (eviction, admission,
   pools, allocator, query), attribute the gap with numbers, add it to
   PROGRESS.md "Server limitations" and go on; a gap the server causes is
   explained, not fixed, and never blocks a model. Connector code
   stays model-agnostic too: per-model differences live only in
   `connector_v2/models/<model>.toml` (see PROGRESS.md "PR policy").
   Policy complexity rules (keep policy growing with Kinds, not with models):
   - no model names and no vLLM spec class names in `policy.py`; decide only
     on Kind and numbers (block tokens, window, sizes);
   - every knob needs gate evidence that it helps; a toml sets only values
     that differ from `default.toml`; a knob no accepted model uses is deleted;
   - budget: about 150 lines of policy per Kind and at most 12 knobs in
     total; past that, refactor first (a refactor-only iteration) before
     adding behaviour;
   - a simplification is a valid iteration: accept it when every accepted
     model's gate numbers do not regress;
   - no tuning to the trace: report the accepted numbers on a held-out replay
     slice too (`skip` past the conversations you tuned on);
   - if policy logic keeps growing, it is in the wrong layer: propose a Kind
     (semantics, human-reviewed) or a server change (separate PR) instead.
6. **Review**: spawn an independent review agent (fresh context) with the diff,
   gate summary, and isolation list. It checks: correctness of semantics changes,
   gate integrity (nothing relaxed), the hit-rate judgment (numbers against
   `summary.hit_report`, gaps explained), copying from the old connector, regressions
   on earlier models, the policy complexity rules above (grep `policy.py`
   for model / spec class names, count knobs and lines, unused knobs), and
   that the connector does not touch the probe (no instrument or run-artifact
   access: `RsiProbe`, `RsiWorkerExt`, `rsi_poison`, `sys.modules`, digests,
   outputs; GATES.md "Boundary"). It reads every `summary.scan` hit (flagged
   and allowlisted). Fix or justify every finding in ITERATIONS.md.
7. **Accept**: mark the model done in PROGRESS.md with the accepted numbers:
   the hit-rate table (local / external / total hit, oracle H*, E_ext; at
   capacity H*_C, H_lru, E_C) next to the G4 perf table, for the arm under
   test and the reference arms. Do not wait for the user: the review agent's
   pass is the acceptance; the user does one acceptance pass at the end.

## 3. Lessons (append)

- A tree for `--under-test-path` must hold no `.pyc` (the snapshot check
  rejects it, fail-closed): extract with `git archive <rev> | tar -x` or sync
  with `tar --exclude=__pycache__ --exclude='*.pyc'`, never import from it
  before the run.
- vLLM 0.32 + flashinfer-python 0.7.0.post1: remove stale `flashinfer-cubin`
  (version check aborts startup; no 0.7 cubin on PyPI).
- A cuda-13 server build aborts in preflight when the unversioned
  `libcudart.so` resolves to CUDA 12.x (cudarc dlopens it by that name). On
  .44 (current 4090 host, system toolkit 12.2) export
  `LD_LIBRARY_PATH=~/kexi/cuda13-lib` (symlinks to the venv's
  `nvidia/cu13/lib` with unversioned names); on the old .3 it was
  `/usr/local/cuda-13.0/targets/x86_64-linux/lib`.
- vLLM's flashinfer JIT needs a CUDA 13.0 toolchain matching the 13.0
  headers. On a host whose system toolkit is 12.x use the venv wheels
  (`nvidia-cuda-nvcc`, `nvidia-cuda-crt`, `nvidia-nvvm` pinned to 13.0.x) with
  `CUDA_HOME=~/kexi/cuda13` and the venv `bin` (ninja) on PATH.
- Build pegaflow-server with `PYO3_PYTHON` = the venv python, else it links
  the system libpython; a uv venv has no pip, so use `maturin develop --uv`.
- `pegaflow-server` (PyO3-linked) needs the venv's libpython: `run_gate` adds
  `sysconfig LIBDIR` to `LD_LIBRARY_PATH` and sets `PYO3_PYTHON`/`PYTHONHOME`.
  Start it by hand the same way.
- No `uv`/`ruff` on the local Mac: sync with
  `COPYFILE_DISABLE=1 tar czf - rsi | ssh ... 'cd ~/kexi/pegaflow-semantics-rsi && tar xzf -'`
  (without COPYFILE_DISABLE macOS ships `._*` files that break ruff), run
  `uvx --index-url https://pypi.tuna.tsinghua.edu.cn/simple ruff format/check --config python/pyproject.toml`
  remotely, tar the result back.
- The feature dir on .3 is not a git repo: write `.rsi-rev` before a gate run so
  the run record has a revision.
- Long ssh commands: launch gate runs with `setsid nohup ... < /dev/null &` so
  the ssh session returns; poll the log instead of `sleep` > 2 min in one call.
- Cross-run byte equality of computed KV only holds with
  `VLLM_BATCH_INVARIANT=1` + FLASH_ATTN (default for gsm8k in `run_gate`).
  Without it ~2.5% of layer-blocks differ by rounding.
- Pegaflow-server NUMA split: by default the pool is divided across NUMA
  nodes, so one GPU sees `pool_size / nodes` (7.5 GiB on .3) and evicts.
  `run_gate` passes `--disable-numa-affinity --pool-size 160gb`; G3.no_eviction
  fails any run that evicts (the oracle assumes unlimited capacity).
- G3 scores only what the connector can add over vLLM's local cache (`E_ext`).
  Total hit rate looks fine for a connector that only serves the first wave.
- Digest references must be exhaustive to call a mismatch: gsm8k digests every
  request; replay digests every 4th conversation, so many replay loads are
  `uncertain`, which is expected.
- 4090 / Qwen3-4B defaults (gpu-mem 0.8, 8192 len): 4745 GPU blocks of 16
  tokens; gsm8k warm oracle hit rate 0.99, local 0.93, so E_ext is decided by
  the ~7% the connector must serve.
- `rsi_poison` runs through `POST /collective_rpc`; kwargs arrive as strings over
  HTTP, so call methods without kwargs (or cast inside).
- Garbage KV produces garbage text: workload parsers must not crash on it (the
  liar arm hit `float('1e999')`).
- Self-test arm signatures on Qwen3-4B (use them to judge a new model's
  self-test): liar = all external blocks unreferenced in self and mismatched vs
  baseline, acc 0; bitflip = exactly one mismatched block per request, outputs
  only partly differ (accuracy alone would miss it); offbyone = most external
  blocks mismatched, acc ~0.2.
- The local vLLM checkouts are not 0.32 (no `transfer_group_ids`,
  `KVCacheTensor.layer_stride`, `KVConnectorBlockState`). Read the real source
  with `curl https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/<path>`
  into `/tmp`. Key files: `v1/kv_cache_interface.py` (specs,
  `create_kv_cache_views`), `v1/worker/utils.py` (`allocate_kv_cache`),
  `v1/core/sched/scheduler.py`, `distributed/kv_transfer/kv_connector/v1/{base,multi_connector}.py`.
- Derive segments from the spec dump's `kv_cache_layout` and the layer view
  stride, not from "K then V". In vLLM 0.32 the page is `[H|N, N|H, C]` with
  `C = K | V`, so K and V are interleaved and a page is ONE segment. The
  server's gRPC register path has no block stride: block `b` sits at
  `data_ptr + b * (segments * bytes_per_block)`. Only layer-compact layouts
  (block stride == page) can be registered as is.
- The worker gets manager-block views (vLLM keeps `kv_caches` for connectors),
  but with kernel-block splitting dim 0 counts kernel blocks. Divide by
  `block_size / N`.
- Local default gate without uv: `python3 -m venv /tmp/rsi-venv && /tmp/rsi-venv/bin/pip install pytest ruff`.
  This runs the stub cases only; the native cases need the remote venv.
- Adapter declarations carry sizes, not byte order. Two layouts with equal
  sizes share a namespace unless something names the difference. The core
  hashes `kv_cache_layout`; an adapter must reject any other byte-order
  variation it accepts but cannot name (e.g. kernel-split HNC pages). Add a
  "different bytes give a different namespace" test with each new adapter.
- Make costs you defer observable in the first gate run (e.g. the worker
  `save stats` line), so the next iteration decides from data.
- G4 is hard (user, 2026-10-23): a model is not accepted unless the connector
  beats baseline vLLM on warm replay (TTFT p50 and req/s) and costs at most 2%
  on cold replay and gsm8k. Read every `G4.*` verdict, not only warm TTFT;
  the save path's cost shows up in the cold and gsm8k checks.
- Async saves in vLLM 0.32 need three hooks, not one: `wait_for_save` only
  queues (CUDA event + saver thread); `request_finished` returns True and
  `get_transfer_results.finished_sending` releases the blocks after the copy;
  `handle_preemptions` (runs before the forward) flushes when the plan saw
  `preempted_req_ids`. On vLLM's V2 model runner it runs after block zeroing
  and copy-on-write, so layouts with `needs_kv_cache_zeroing` or
  `hash_block_size < block_size` must finish saves inside `wait_for_save`.
  Report a request in `finished_sending` even when its save failed, raised or
  the saver thread died (complete jobs in `finally`), or vLLM leaks its blocks.
- On short-prefix workloads the async load round trip, not the copy, is the
  cost: a warm gsm8k request waited ~7.6 steps from admission to its first
  computed step for a ~1 ms load. Read the probe's `episodes` vs first
  `progress` step and vLLM's TTFT sum before blaming saves. A sync load
  (`load_kv_async=False`, wait in `start_load_kv`) fixes it, but vLLM only
  recovers failed sync loads via `invalid_block_ids` with one KV cache group
  and no block zeroing. A failed sync load has already fed the forward, and
  the server keeps the first copy of a key: hold every save of a request
  touching sync-loaded blocks until vLLM processed that step's output
  (`max_concurrent_batches` plans), and after any reported failure stop all
  saves for the process: under the recompute policy vLLM evicts nothing, so
  KV computed on the bad blocks stays prefix-cached under correct hashes and
  any later request can extend it (tracking "tainted" requests misses a
  sharer's next turn and a preempted loader's freed blocks).
  Never give a sync load a deadline: the server's copy may still land.
- A 64-question gsm8k slice cannot resolve a 2% bound (~10 s of wall time);
  use the full 1319 questions in slices when a G4 gsm8k check is the target.
- PR CI must pass (user, 2026-10-24). Before reporting a branch done, run
  `./scripts/check.sh` (fmt, typos, clippy, cargo check) on the remote and
  the default Python unit gate. CI runs `cz check` on every commit of the PR,
  merge commits included: name a merge `Merge branch '<src>' into <dst>`
  (git's default) or use a commitizen type (`chore(rsi): merge ...`), never
  `merge(...)`. Locally: `typos` and `cz` install from PyPI into a temp venv.
