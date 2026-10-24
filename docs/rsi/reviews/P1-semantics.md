# Review: P1 semantics (2026-10-23)

Reviewer: independent agent, fresh context, isolation list applied.
Scope: `pegaflow-semantics/**`, `python/src/semantics.rs`, `python/pegaflow/semantics.{py,pyi}`,
`python/tests/test_semantics_bindings.py`.
Verdict: APPROVE-WITH-FIXES (no blockers). All original vectors hand-verified.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | `Group.window(tokens, block_tokens)` argument order is the reverse of the design doc's `window(block_size, sliding_window)`; both are ints, so a swap fails silently | should-fix | Binding changed to `window(block_tokens, tokens)` to match the doc |
| 2 | `Group` fields are `pub`, so a struct literal can skip the zero check and cause a divide-by-zero | should-fix | Fields made private, `Group::new` returns `Result<_, InvalidGroup>`; Python raises ValueError |
| 3 | The lcm fold can overflow u64 with large coprime block sizes, giving a garbage step | should-fix | The fold stops as soon as the step exceeds `cap`; test `huge_coprime_block_sizes_do_not_overflow` |
| 4 | Scratch groups count toward the lcm step and lose hits; `Window{0}` gets past the all-scratch guard | should-fix | Scratch is excluded from the step; `Window{0}` is rejected; vector `scratch-excluded-from-step` |
| 5 | seal_plan checks segment sizes only within one part, not across all parts of a layer | should-fix | All parts of a layer must have identical segments; zero-byte segments are rejected (`EmptySegment`) |
| 6 | Tokens (DCP) shard is untested; seal_plan has no JSON vectors; several resolve/saveable cases missing | should-fix | Added `vectors/seal_plan.json` (12 cases, including dcp-tokens and every error kind), 7 more resolve cases (window off-by-one, window > prefix, unaligned `before`, `after <= before`, checkpoint at block 0, window block size different from the step), plus a Tokens unit test |
| n1 | Window convention is undocumented (`sliding_window - 1` is exact) | nit | Documented on `Kind::Window` and in the binding |
| n2 | Window saves read recycled memory if `before` is stale | nit | Documented as a precondition on `saveable`; the connector must pass the step-start count |
| n3 | Lowest rank as writer vs. the doc's "first writer wins" | nit | Documented as a de-dup hint only: the store must accept the copy from any holder |
| n4 | Ranks are not checked to be dense; nothing ties plan groups to resolve groups | nit | Deferred: checked by G0 layout gate (adapter vs spec dump) |
| n5 | DCP `block_tokens` (per-rank or logical) is unspecified | nit | Deferred to the first DCP model; noted on `Shard::Tokens` |
| n6 | `keys.rs` duplicates `pegaflow-common::group_hash` | nit | `group_hash` now delegates to `pegaflow_semantics::key` (common depends on semantics) |
| n7 | `RunIndex` allocates `prompt_len / bt` entries | nit | Bounded by largest present index + 1 |
| n8 | `resolve` binding rejects Python sets | nit | The `pegaflow.semantics.resolve` wrapper accepts any iterable |
| n9 | `needed` and `saveable` return different shapes for empty ranges | nit | Both return `(0, 0)` |
| n10 | Binding test silently skips on an old native build that lacks `semantics` | nit | Skips only when `pegaflow.pegaflow` is missing |
| n11 | Layout fingerprint for the namespace is not implemented | gap | Tracked for P2 (connector computes it from adapter declarations) |

Post-fix evidence (remote .3):
- `cargo test -p pegaflow-semantics`: 13 unit tests and 3 vector tests pass.
- `cargo test -p pegaflow-common`: 48 pass.
- `cargo clippy --workspace --no-default-features --features cuda-13,rdma --all-targets -D warnings`: clean.
- `cargo fmt --all --check`, ruff format/check and typos on the changed paths: clean.
- `pytest tests/test_semantics_bindings.py`: 35 pass.
