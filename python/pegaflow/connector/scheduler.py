"""
Scheduler-side connector logic.
"""

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pegaflow.connector.common import (
    CacheGroupLayout,
    ConnectorContext,
    LoadIntent,
    LocalRecurrentLoad,
    LocalRecurrentSave,
    LocalSaveRef,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    PegaWorkerMetadata,
    RecurrentLoadHold,
    SaveIntent,
    logger,
    reconcile_hybrid_hit,
)
from pegaflow.connector.connector_metrics import PrefetchTracker
from pegaflow.connector.linear_state_cache import LinearStateCache, LinearStateSlot
from pegaflow.connector.tp_shards import ShardedQueryReady, TpShardQueryClient
from pegaflow.pegaflow import EngineRpcClient

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass(slots=True)
class _QueryProbe:
    """One remote prefix-query snapshot.

    A probe snapshots:

    * ``computed_blocks`` — blocks already computed locally when the query was
      issued
    * ``query_hashes`` — remaining full-block hashes plus at most one derived
      tail key sent to the backend
    * ``tail_tokens`` — valid rows in that tail block, used for token accounting
      and request-drift validation

    If the request keeps making local progress while the backend is loading,
    the current query key may drift.  A *Ready* result is accepted only if the
    current key still matches this snapshot.
    """

    computed_blocks: int
    query_hashes: tuple[bytes, ...]
    tail_tokens: int = 0

    # ``None`` means the backend is still loading.
    hit_blocks: int | None = None
    leases: tuple[bytes, ...] = ()
    # Hybrid (HMA): pinned recurrent checkpoints from the membership queries,
    # set together with `leases` when the hybrid reconcile found a boundary.
    recurrent_hold: RecurrentLoadHold | None = None
    # Sorted query positions a hybrid hit may end at (intersection over all
    # recurrent groups and shards, below the attention prefix). Used to
    # re-derive a legal boundary when the token budget shrinks the hit.
    usable_positions: frozenset[int] = frozenset()
    # Connector-owned recurrent checkpoint pinned until local D2D load completes.
    local_recurrent: LinearStateSlot | None = None

    @property
    def is_ready(self) -> bool:
        return self.hit_blocks is not None

    def matches(
        self,
        computed_blocks: int,
        query_hashes: tuple[bytes, ...],
        tail_tokens: int,
    ) -> bool:
        return (
            self.computed_blocks == computed_blocks
            and self.query_hashes == query_hashes
            and self.tail_tokens == tail_tokens
        )

    def mark_ready(self, ready: ShardedQueryReady) -> None:
        hit_blocks = ready.num_hit_blocks
        if hit_blocks > len(self.query_hashes):
            raise RuntimeError(
                f"invariant violated: server returned {hit_blocks} hits for "
                f"{len(self.query_hashes)} hashes"
            )
        self.hit_blocks = hit_blocks
        self.leases = ready.leases
        self.recurrent_hold = ready.recurrent_hold
        self.usable_positions = frozenset(ready.usable_positions)
        self.local_recurrent = ready.local_recurrent

    def require_hit_blocks(self) -> int:
        if self.hit_blocks is None:
            raise RuntimeError("query probe is still loading")
        return self.hit_blocks


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    def __init__(
        self,
        context: ConnectorContext,
        engine_clients: tuple[EngineRpcClient, ...] | None = None,
        pd_tail_save: bool = False,
        pd_tail_load: bool = False,
        vllm_config=None,
        kv_cache_config=None,
    ):
        self._ctx = context
        engine_clients = tuple(engine_clients or (context.engine_client,))
        expected_shards = context.tp_shards.shard_count if context.tp_shards is not None else 1
        if len(engine_clients) != expected_shards:
            raise ValueError(
                f"scheduler has {len(engine_clients)} PegaFlow clients for "
                f"{expected_shards} TP shards"
            )
        self._tp_shard_client = TpShardQueryClient(engine_clients)
        self._cache_groups = CacheGroupLayout.from_config(kv_cache_config)
        if self._cache_groups.has_recurrent_state and (pd_tail_save or pd_tail_load):
            raise ValueError("P/D tail-block caching is not supported with HMA")
        self._get_local_cached_blocks = None
        self._linear_state_cache: LinearStateCache | None = None
        if context.linear_state_cache_size_bytes:
            compound_bytes = self._cache_groups.recurrent_compound_page_bytes
            if compound_bytes <= 0:
                raise ValueError("enabled linear-state cache requires MambaSpec.page_size_bytes")
            self._linear_state_cache = LinearStateCache(
                context.linear_state_cache_size_bytes // compound_bytes
            )

        # P/D tail-block extension (`pegaflow.pd_tail_save`): vLLM only hashes
        # full blocks, so a prompt's partial tail block never enters the tier
        # and a strict no-prefill decode peer would have to recompute it.
        # When enabled, the step that schedules the final prompt chunk also
        # saves the partial tail block under a key derived with vLLM's OWN
        # hash function over (last_full_hash, tail_prompt_token_ids, None) —
        # well-defined, and independently derivable by the decode peer.
        # vLLM derives NONE_HASH from PYTHONHASHSEED when it is set. A fixed
        # seed makes the configured hash function reproducible across the
        # scheduler processes participating in the transfer.
        self._tail_save_enabled = pd_tail_save
        self._tail_load_enabled = pd_tail_load
        self._tail_hash_fn = None
        if pd_tail_save or pd_tail_load:
            assert vllm_config is not None
            algo = vllm_config.cache_config.prefix_caching_hash_algo
            if os.environ.get("PYTHONHASHSEED") is None:
                enabled_options = ", ".join(
                    option
                    for enabled, option in (
                        (pd_tail_save, "pegaflow.pd_tail_save"),
                        (pd_tail_load, "pegaflow.pd_tail_load"),
                    )
                    if enabled
                )
                raise ValueError(
                    "P/D tail-block caching requires a fixed PYTHONHASHSEED "
                    f"across vLLM processes; enabled options: {enabled_options}"
                )
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core import kv_cache_utils

            self._tail_hash_fn = get_hash_fn_by_name(algo)
            self._hash_block_tokens = kv_cache_utils.hash_block_tokens
            # NONE_HASH is only assigned by vLLM's init_none_hash(), which
            # runs after connector construction — it must be read lazily
            # through the module, never imported by name here.
            self._kv_cache_utils = kv_cache_utils
            logger.info(
                "[PegaKVConnector] P/D tail-block cache enabled (save=%s load=%s algo=%s)",
                pd_tail_save,
                pd_tail_load,
                algo,
            )
        self._tail_saved: set[str] = set()

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._prefetch_start_times: dict[str, float] = {}
        self._pending_query_probes: dict[str, _QueryProbe] = {}

        # Prefetch tracking (for metrics)
        self._prefetch_tracker = PrefetchTracker()

        # Save state (per-request)
        self._block_hashes: dict[str, tuple[bytes, ...]] = {}
        self._external_matched_blocks: dict[str, int] = {}
        self._block_index_offsets: dict[str, int] = {}
        self._allocated_blocks: dict[str, list[list[int]]] = {}
        self._scheduled_tokens: dict[str, int] = {}
        self._next_stored_block_idx: dict[str, int] = {}

        # Live Request references – used to refresh block_hashes during decode
        # so that newly completed blocks can be saved, not just prefill blocks.
        self._requests: dict[str, Request] = {}

        # Completion tracking
        self._deferred_save_intents: dict[str, SaveIntent] = {}
        self._pending_saves: set[str] = set()
        self._pending_local_saves: dict[LocalSaveRef, LinearStateSlot] = {}
        self._local_saves_by_req: dict[str, set[LocalSaveRef]] = {}
        self._local_save_ack_counts: dict[LocalSaveRef, int] = {}
        self._pending_local_loads: dict[str, LinearStateSlot] = {}
        self._held_requests: set[str] = set()

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self._get_local_cached_blocks = gpu_block_pool.get_cached_block
        if self._cache_groups.group_count <= 1:
            return

        # vLLM can cache an async-loaded dense group and expose it to a sibling
        # before the sparse group has a usable state. PegaFlow is the sole HMA
        # prefix index until vLLM exposes an atomic all-group cache hook.
        def no_local_hma_prefix_hit(*_args, **_kwargs) -> None:
            return None

        gpu_block_pool.get_cached_block = no_local_hma_prefix_hit

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        req_id = request.request_id

        if not self._ctx.read_enabled:
            logger.debug(
                "[PegaKVConnector] req=%s cache_lookup_skipped: mode=%s",
                req_id,
                self._ctx.mode.value,
            )
            return (0, False)

        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size
        query_hashes, tail_tokens = self._build_query(request, computed_blocks)

        # Nothing remains to query remotely.
        if not query_hashes:
            self._release_pending_query_probe(req_id)
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        probe = self._pending_query_probes.get(req_id)

        # Ready result already cached.  Reuse it only if the request identity
        # has not drifted since the query was issued.
        if probe is not None and probe.is_ready:
            if probe.matches(computed_blocks, query_hashes, tail_tokens):
                return self._finish_cache_lookup(
                    req_id=req_id,
                    num_tokens=request.num_tokens,
                    probe=probe,
                    lookup_us=None,
                    reused=True,
                )

            # Cached Ready is stale.  It has a lease, so release it.
            self._release_pending_query_probe(req_id)
            probe = None

        # A Loading task is keyed by req_id server-side. If the current query
        # drifted, finish polling with the original identity so the TP layer can
        # validate the stale Ready before we release it below.
        backend_query_hashes = query_hashes
        if probe is not None and not probe.matches(computed_blocks, query_hashes, tail_tokens):
            backend_query_hashes = probe.query_hashes

        # No reusable Ready result. Ask backend.
        lookup_start = time.perf_counter()
        ready = self._count_available_block_prefix(backend_query_hashes, req_id)
        lookup_us = (time.perf_counter() - lookup_start) * 1e6

        # Backend is still loading.  Keep the original snapshot.
        if ready is None:
            if probe is None:
                self._pending_query_probes[req_id] = _QueryProbe(
                    computed_blocks=computed_blocks,
                    query_hashes=query_hashes,
                    tail_tokens=tail_tokens,
                )
            return (None, False)

        # A previous Loading probe exists, but the request has moved on.
        # This Ready belongs to the old query.  Do not consume it.
        if probe is not None and not probe.matches(computed_blocks, query_hashes, tail_tokens):
            logger.warning(
                "[PegaKVConnector] req=%s query identity drifted: "
                "snapshot computed=%d/%d hashes, current computed=%d/%d hashes "
                "- discarding stale Ready",
                req_id,
                probe.computed_blocks,
                len(probe.query_hashes),
                computed_blocks,
                len(query_hashes),
            )
            self._release_leases(ready.leases, req_id)
            if ready.recurrent_hold is not None:
                for group_index, group_leases in enumerate(ready.recurrent_hold.leases):
                    self._tp_shard_client.release(group_leases, f"{req_id}:g{group_index}")
            if ready.local_recurrent is not None:
                assert self._linear_state_cache is not None
                self._linear_state_cache.unpin(ready.local_recurrent)
            self._pending_query_probes.pop(req_id, None)
            return (None, False)

        # Either:
        #   1. IDLE -> Ready
        #   2. Loading probe -> Ready (identity matched above)
        if probe is None:
            probe = _QueryProbe(
                computed_blocks=computed_blocks,
                query_hashes=query_hashes,
                tail_tokens=tail_tokens,
            )
            self._pending_query_probes[req_id] = probe

        probe.mark_ready(ready)
        return self._finish_cache_lookup(
            req_id=req_id,
            num_tokens=request.num_tokens,
            probe=probe,
            lookup_us=lookup_us,
            reused=False,
        )

    def _finish_cache_lookup(
        self,
        *,
        req_id: str,
        num_tokens: int,
        probe: _QueryProbe,
        lookup_us: float | None,
        reused: bool,
    ) -> tuple[int, bool]:
        hit_blocks = probe.require_hit_blocks()
        computed_blocks = probe.computed_blocks
        vbs = self._ctx.virtual_block_size
        tail_hit = probe.tail_tokens > 0 and hit_blocks == len(probe.query_hashes)
        # _build_query appends the tail key last and the backend reports prefix
        # hits, so a full query hit makes the final block the partial tail.
        last_block_tokens = probe.tail_tokens if tail_hit else vbs
        hit_tokens = (hit_blocks - 1) * vbs + last_block_tokens

        # A request still needs one forward token to produce logits. The last
        # loaded page may contain that token's KV, but vLLM must recompute and
        # overwrite it unless a P/D router supplied a separate decode token.
        locally_computed_tokens = computed_blocks * vbs
        hit_tokens = min(hit_tokens, max(0, num_tokens - locally_computed_tokens - 1))

        if probe.recurrent_hold is not None or probe.local_recurrent is not None:
            # A mamba checkpoint is valid only at its own block boundary. If
            # the token budget cut inside the reconciled span, fall back to
            # the best earlier boundary; if none survives, drop the hit.
            boundary_blocks = hit_tokens // vbs
            usable = [p for p in probe.usable_positions if p < boundary_blocks]
            if not usable:
                if self._pending_query_probes.get(req_id) is probe:
                    self._release_pending_query_probe(req_id)
                return (0, False)
            checkpoint = max(usable)
            reduced_hit_blocks = checkpoint + 1
            if reduced_hit_blocks < hit_blocks:
                exact = self._tp_shard_client.query(
                    self._ctx.instance_id,
                    list(probe.query_hashes[:reduced_hit_blocks]),
                    f"{req_id}:hma-budget-exact-{reduced_hit_blocks}",
                    False,
                )
                self._release_leases(probe.leases, req_id)
                probe.leases = ()
                if exact is None or exact.num_hit_blocks != reduced_hit_blocks:
                    if exact is not None:
                        self._release_leases(exact.leases, req_id)
                    self._release_pending_query_probe(req_id)
                    return (0, False)
                probe.leases = exact.leases
            hit_blocks = reduced_hit_blocks
            hit_tokens = hit_blocks * vbs
            probe.hit_blocks = hit_blocks
            if probe.recurrent_hold is not None:
                probe.recurrent_hold = replace(probe.recurrent_hold, checkpoint=checkpoint)
            elif probe.local_recurrent is not None:
                assert self._linear_state_cache is not None
                if probe.local_recurrent.block_hash != probe.query_hashes[checkpoint]:
                    replacement = self._linear_state_cache.lookup(
                        probe.query_hashes[checkpoint], pin=True
                    )
                    if replacement is None:
                        self._release_pending_query_probe(req_id)
                        return (0, False)
                    self._linear_state_cache.unpin(probe.local_recurrent)
                    probe.local_recurrent = replacement

        # Cacheable tails contain at least two tokens, so recomputing the final
        # prompt token cannot remove the last leased block from the load.
        loaded_blocks = (hit_tokens + vbs - 1) // vbs
        self._external_matched_blocks[req_id] = computed_blocks + loaded_blocks

        if reused:
            logger.debug(
                "[PegaKVConnector] req=%s cache_lookup_reuse: hit_blocks=%d "
                "computed_blocks=%d hit_tokens=%d num_tokens=%d total_query_hashes=%d "
                "tail_hit=%s tail_tokens=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                len(probe.query_hashes),
                tail_hit,
                probe.tail_tokens,
            )
        else:
            logger.info(
                "[PegaKVConnector] req=%s cache_lookup: hit_blocks=%d computed_blocks=%d "
                "hit_tokens=%d num_tokens=%d lookup_us=%.0f total_query_hashes=%d "
                "tail_hit=%s tail_tokens=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                lookup_us or 0.0,
                len(probe.query_hashes),
                tail_hit,
                probe.tail_tokens,
            )

        if hit_tokens <= 0:
            # No external load will consume this lease.
            self._release_pending_query_probe(req_id)
            return (0, False)

        return (hit_tokens, True)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id

        # Keep a live reference so we can refresh block_hashes during decode
        # (Request.block_hashes grows as new full blocks are completed).
        self._requests[req_id] = request

        # request.block_hashes are already at virtual_block_size granularity
        # (1 hash per scheduler block =
        # block_size * dcp_world_size * pcp_world_size tokens).
        # They are 1-to-1 with block_ids from the scheduler.
        self._block_hashes[req_id] = tuple(request.block_hashes)
        if req_id not in self._allocated_blocks:
            # The first locally allocated block may be after an external-hit
            # prefix. Track that global block index explicitly.
            base_block_idx = self._external_matched_blocks.get(req_id, 0)
            self._block_index_offsets[req_id] = base_block_idx
            self._allocated_blocks[req_id] = [[] for _ in range(self._cache_groups.group_count)]
            self._scheduled_tokens[req_id] = 0
            self._next_stored_block_idx[req_id] = base_block_idx

        if num_external_tokens > 0:
            block_ids_by_group = (
                self._copy_block_ids_by_group(blocks.get_block_ids())
                if blocks
                else tuple(() for _ in range(self._cache_groups.group_count))
            )
            hash_group_index = self._cache_groups.hash_group_index
            num_computed_blocks = (
                sum(block.block_hash is not None for block in blocks.blocks[hash_group_index])
                if blocks
                else 0
            )
            start_block_idx = num_computed_blocks
            vbs = self._ctx.virtual_block_size
            num_load_blocks = (num_external_tokens + vbs - 1) // vbs
            try:
                load_block_ids_by_group = self._load_block_ids_by_group(
                    block_ids_by_group,
                    start_block_idx,
                    num_load_blocks,
                )
            except RuntimeError:
                self._release_pending_query_probe(req_id)
                raise

            pending_probe = self._pending_query_probes.get(req_id)
            local_recurrent = None
            if pending_probe is not None and pending_probe.local_recurrent is not None:
                destinations = tuple(
                    (group_index, block_ids_by_group[group_index][-1])
                    for group_index in sorted(self._cache_groups.recurrent_group_indices)
                    if block_ids_by_group[group_index]
                    and block_ids_by_group[group_index][-1] is not None
                )
                if len(destinations) != len(self._cache_groups.recurrent_group_indices):
                    self._release_pending_query_probe(req_id)
                    raise RuntimeError(f"req {req_id} missing local recurrent destination")
                local_recurrent = LocalRecurrentLoad(
                    source=pending_probe.local_recurrent,
                    destination_block_ids=destinations,
                )
                load_block_ids_by_group = tuple(
                    (None,) * len(group)
                    if index in self._cache_groups.recurrent_group_indices
                    else group
                    for index, group in enumerate(load_block_ids_by_group)
                )
            load_intent = LoadIntent(
                block_ids_by_group=load_block_ids_by_group,
                leases=pending_probe.leases if pending_probe is not None else (),
                num_tokens=num_external_tokens,
                recurrent_hold=(
                    pending_probe.recurrent_hold if pending_probe is not None else None
                ),
                local_recurrent=local_recurrent,
            )
            if pending_probe is not None:
                query_hashes, tail_tokens = self._build_query(request, num_computed_blocks)
                if not pending_probe.matches(num_computed_blocks, query_hashes, tail_tokens):
                    self._release_pending_query_probe(req_id)
                    raise RuntimeError(f"req {req_id} query identity changed before external load")
                leased_blocks = pending_probe.require_hit_blocks()
                if leased_blocks != num_load_blocks:
                    self._release_pending_query_probe(req_id)
                    raise RuntimeError(
                        f"req {req_id} leased block mismatch: "
                        f"leased={leased_blocks} load={num_load_blocks}"
                    )
            if not load_intent.leases or any(not lease for lease in load_intent.leases):
                raise RuntimeError(f"req {req_id} missing query lease for external load")
            self._pending_load_intents[req_id] = load_intent
            if local_recurrent is not None:
                self._pending_local_loads[req_id] = local_recurrent.source
            self._pending_query_probes.pop(req_id, None)
            logger.debug(
                "[PegaKVConnector] req=%s alloc: total_blocks=%d computed_blocks=%d "
                "load_blocks=%d start_block_idx=%d load_tokens=%d pending_loads=%d",
                req_id,
                len(block_ids_by_group[hash_group_index]),
                num_computed_blocks,
                len(load_intent.block_ids_by_group[hash_group_index]),
                start_block_idx,
                load_intent.num_tokens,
                len(self._pending_load_intents),
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        reservations_before = set(self._pending_local_saves)
        try:
            return self._build_connector_meta(scheduler_output)
        except Exception:
            for ref in set(self._pending_local_saves) - reservations_before:
                self._cancel_local_save(ref)
            raise

    def _build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        # Leave deferred work in place until metadata construction succeeds.
        ready_save_intents = dict(self._deferred_save_intents)
        potential_saves: dict[str, SaveIntent] = {}
        load_intents = dict(self._pending_load_intents)

        # Process new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)

            # Verify update_state_after_alloc was called for this request
            assert req_id in self._block_hashes, (
                f"req {req_id} not initialized in update_state_after_alloc"
            )

            # Populate block IDs from scheduler_output — single source of
            # truth for the save path (consistent with offloading connector).
            if req.block_ids:
                self._allocated_blocks[req_id] = [
                    list(group) for group in self._copy_block_ids_by_group(req.block_ids)
                ]

            if self._ctx.read_enabled:
                self._scheduled_tokens[req_id] += num_tokens
            else:
                self._scheduled_tokens[req_id] = max(
                    self._scheduled_tokens.get(req_id, 0),
                    req.num_computed_tokens + num_tokens,
                )

            # Positions with valid KV after this step, from the scheduler's
            # own invariant (num_computed_tokens covers prefix-cache hits and
            # is reset on preemption — no connector-side bookkeeping can be
            # trusted across a preempt/resume cycle).
            written = req.num_computed_tokens + num_tokens
            if save_intent := self._consume_save_intent(
                req_id,
                written,
                req.num_computed_tokens,
            ):
                potential_saves[req_id] = save_intent

        # Process cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._block_hashes:
                continue

            # Refresh block hashes from the live Request object so that
            # newly completed blocks during decode are also saved.
            req = self._requests.get(req_id)
            if req is not None:
                self._block_hashes[req_id] = tuple(req.block_hashes)

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)

            # Append newly allocated blocks
            new_block_ids = cached_reqs.new_block_ids[idx]
            if req_id in cached_reqs.resumed_req_ids:
                self._allocated_blocks[req_id] = (
                    [list(group) for group in self._copy_block_ids_by_group(new_block_ids)]
                    if new_block_ids
                    else [[] for _ in range(self._cache_groups.group_count)]
                )
            elif new_block_ids:
                for allocated, new_group in zip(
                    self._allocated_blocks[req_id],
                    self._copy_block_ids_by_group(new_block_ids),
                    strict=True,
                ):
                    allocated.extend(new_group)

            if self._ctx.read_enabled:
                self._scheduled_tokens[req_id] += num_tokens
            else:
                prior_computed_tokens = cached_reqs.num_computed_tokens[idx]
                self._scheduled_tokens[req_id] = max(
                    self._scheduled_tokens.get(req_id, 0),
                    prior_computed_tokens + num_tokens,
                )

            written = cached_reqs.num_computed_tokens[idx] + num_tokens
            if save_intent := self._consume_save_intent(
                req_id,
                written,
                cached_reqs.num_computed_tokens[idx],
            ):
                potential_saves[req_id] = save_intent

        save_intents = potential_saves
        self._pending_saves.update(save_intents.keys())
        self._pending_load_intents.clear()
        self._deferred_save_intents.clear()

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves",
            len(load_intents),
            len(save_intents),
        )

        return PegaConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
            ready_save_intents=ready_save_intents,
            preempted_req_ids=scheduler_output.preempted_req_ids or None,
        )

    def _consume_save_intent(
        self,
        req_id: str,
        written: int,
        computed_before_step: int = 0,
    ) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving.

        `written` = positions with valid KV once this step's schedule runs
        (scheduler-authoritative num_computed_tokens + this step's tokens).
        """
        regular = self._consume_full_block_saves(req_id, written, computed_before_step)
        tail = self._consume_tail_save(req_id, written)
        if tail is None:
            return regular
        if regular is None:
            return tail
        return SaveIntent(
            block_ids_by_group=tuple(
                regular_ids + tail_ids
                for regular_ids, tail_ids in zip(
                    regular.block_ids_by_group,
                    tail.block_ids_by_group,
                    strict=True,
                )
            ),
            block_hashes=regular.block_hashes + tail.block_hashes,
        )

    def _consume_tail_save(self, req_id: str, written: int) -> SaveIntent | None:
        """P/D tail extension: save the prompt's partial tail block once its
        prompt rows are final (the step scheduling the final prompt chunk).

        The saved page may contain rows past the prompt (the first generated
        token lands in it on the next step, racing the async D2H) — harmless:
        the key covers only the tail *prompt* tokens, and the decode peer
        recomputes every position past them anyway.
        """
        if not self._tail_save_enabled or req_id in self._tail_saved:
            return None
        req = self._requests.get(req_id)
        if req is None:
            return None
        vbs = self._ctx.virtual_block_size
        prompt_len = req.num_prompt_tokens
        tail = self._derive_tail_block(req)
        if tail is None:
            return None
        tail_key, tail_len = tail
        if written < prompt_len:
            return None  # tail prompt rows not written yet
        tail_idx = prompt_len // vbs
        allocated = self._allocated_blocks.get(req_id, [])
        block_hashes = self._block_hashes.get(req_id) or ()
        if (
            not allocated
            or any(tail_idx >= len(group) for group in allocated)
            or tail_idx > len(block_hashes)
        ):
            return None  # tail block not allocated / full-block hashes lagging
        self._tail_saved.add(req_id)
        logger.info(
            "[PegaKVConnector] req=%s pd_tail_save: block_id=%d tail_tokens=%d key=%s",
            req_id,
            allocated[self._cache_groups.hash_group_index][tail_idx],
            tail_len,
            tail_key.hex(),
        )
        return SaveIntent(
            block_ids_by_group=tuple((group[tail_idx],) for group in allocated),
            block_hashes=(tail_key,),
        )

    def _derive_tail_block(self, request: "Request") -> tuple[bytes, int] | None:
        if self._tail_hash_fn is None:
            return None
        # The tail key carries no extra_keys. Reusing it for salted, LoRA, or
        # multimodal requests would alias distinct vLLM cache identities.
        if request.lora_request is not None or request.cache_salt or request.mm_features:
            return None
        vbs = self._ctx.virtual_block_size
        prompt_len = request.num_prompt_tokens
        tail_len = prompt_len % vbs
        # vLLM must recompute the final prompt token to produce logits. A
        # one-token tail therefore cannot reduce local work; treating it as a
        # hit would also lease one more hash than vLLM allocates load blocks.
        if tail_len <= 1:
            return None
        tail_idx = prompt_len // vbs
        block_hashes = tuple(request.block_hashes)
        if tail_idx > len(block_hashes):
            raise RuntimeError(
                f"req {request.request_id} missing parent hash for tail block: "
                f"tail_idx={tail_idx} full_hashes={len(block_hashes)}"
            )
        parent = block_hashes[tail_idx - 1] if tail_idx > 0 else self._kv_cache_utils.NONE_HASH
        tail_tokens = list(request.prompt_token_ids[tail_idx * vbs : prompt_len])
        tail_key = bytes(self._hash_block_tokens(self._tail_hash_fn, parent, tail_tokens, None))
        return tail_key, tail_len

    def _build_query(
        self, request: "Request", computed_blocks: int
    ) -> tuple[tuple[bytes, ...], int]:
        query_hashes = tuple(request.block_hashes[computed_blocks:])
        if not self._tail_load_enabled:
            return query_hashes, 0

        tail = self._derive_tail_block(request)
        if tail is None:
            return query_hashes, 0
        return query_hashes + (tail[0],), tail[1]

    def _consume_full_block_saves(
        self,
        req_id: str,
        written: int,
        computed_before_step: int | None = None,
    ) -> SaveIntent | None:
        # block_hashes are at virtual_block_size granularity, 1-to-1 with block_ids.
        block_hashes = self._block_hashes.get(req_id)
        if block_hashes is None:
            return None

        allocated = self._allocated_blocks.get(req_id, [])
        scheduled = self._scheduled_tokens.get(req_id, 0)
        base_block_idx = self._block_index_offsets.get(req_id, 0)
        start_block_idx = self._next_stored_block_idx.get(req_id, base_block_idx)

        # _allocated_blocks tracks request block IDs in global request order.
        # In external-hit cases, the prefix-loaded block IDs are still present at
        # the front, so save intents must slice by global block index rather than
        # rebasing to a local-only view.
        if self._cache_groups.has_recurrent_state:
            if computed_before_step is None:
                computed_before_step = written
            saveable_block_idx = min(
                len(block_hashes),
                computed_before_step // self._ctx.virtual_block_size,
            )
        else:
            local_saveable = min(
                min((len(group) for group in allocated), default=0),
                scheduled // self._ctx.virtual_block_size,
            )
            saveable_block_idx = min(len(block_hashes), base_block_idx + local_saveable)
        new_blocks = saveable_block_idx - start_block_idx
        if new_blocks <= 0:
            return None

        hash_start = start_block_idx
        save_hashes = block_hashes[hash_start : hash_start + new_blocks]
        if self._cache_groups.has_recurrent_state:
            save_block_ids_by_group = self._local_cached_block_ids(save_hashes)
            if save_block_ids_by_group is None:
                return None
        else:
            save_block_ids_by_group = tuple(
                tuple(group[hash_start : hash_start + new_blocks]) for group in allocated
            )
        self._next_stored_block_idx[req_id] = saveable_block_idx

        logger.debug(
            "[PegaKVConnector] req=%s save_intent: start=%d hash_start=%d "
            "base_block_idx=%d saveable_block_idx=%d new_blocks=%d total_hashes=%d",
            req_id,
            hash_start,
            hash_start,
            base_block_idx,
            saveable_block_idx,
            new_blocks,
            len(block_hashes),
        )

        return self._build_hma_save_intent(req_id, save_block_ids_by_group, save_hashes)

    def _build_hma_save_intent(
        self,
        req_id: str,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        block_hashes: tuple[bytes, ...],
    ) -> SaveIntent:
        if self._linear_state_cache is None:
            return SaveIntent(block_ids_by_group=block_ids_by_group, block_hashes=block_hashes)

        local_saves: list[LocalRecurrentSave] = []
        for position, block_hash in enumerate(block_hashes):
            ref = self._linear_state_cache.reserve(block_hash)
            if ref is None:
                continue
            save_ref = LocalSaveRef(req_id, ref.slot, ref.generation, ref.block_hash)
            self._pending_local_saves[save_ref] = ref
            self._local_saves_by_req.setdefault(req_id, set()).add(save_ref)
            sources = tuple(
                (group_index, block_ids_by_group[group_index][position])
                for group_index in sorted(self._cache_groups.recurrent_group_indices)
                if block_ids_by_group[group_index][position] != 0
            )
            if len(sources) != len(self._cache_groups.recurrent_group_indices):
                self._cancel_local_save(save_ref)
                continue
            local_saves.append(LocalRecurrentSave(target=ref, source_block_ids=sources))

        remote_groups = tuple(
            (0,) * len(block_hashes)
            if index in self._cache_groups.recurrent_group_indices
            else group
            for index, group in enumerate(block_ids_by_group)
        )
        return SaveIntent(
            block_ids_by_group=remote_groups,
            block_hashes=block_hashes,
            local_recurrent_saves=tuple(local_saves),
        )

    def _local_cached_block_ids(
        self,
        block_hashes: tuple[bytes, ...],
    ) -> tuple[tuple[int, ...], ...] | None:
        if self._get_local_cached_blocks is None:
            raise RuntimeError("HMA block pool was not bound before building save metadata")

        group_ids = list(range(self._cache_groups.group_count))
        block_ids_by_group = [[] for _ in group_ids]
        for block_hash in block_hashes:
            cached_blocks = self._get_local_cached_blocks(block_hash, group_ids)
            if cached_blocks is None:
                return None
            for block_ids, block in zip(block_ids_by_group, cached_blocks, strict=True):
                block_ids.append(block.block_id)
        return tuple(tuple(block_ids) for block_ids in block_ids_by_group)

    def _consume_finished_hma_save(
        self,
        req_id: str,
        block_ids: tuple[list[int], ...],
        written: int,
    ) -> SaveIntent | None:
        block_hashes = self._block_hashes.get(req_id)
        if block_hashes is None:
            return None

        start_block_idx = self._next_stored_block_idx.get(
            req_id, self._block_index_offsets.get(req_id, 0)
        )
        saveable_block_idx = min(len(block_hashes), written // self._ctx.virtual_block_size)
        if saveable_block_idx <= start_block_idx:
            return None

        groups = self._copy_block_ids_by_group(block_ids)
        available = tuple(len(group) for group in groups)
        required = tuple(
            saveable_block_idx + 1
            if group_index in self._cache_groups.recurrent_group_indices
            else saveable_block_idx
            for group_index in range(self._cache_groups.group_count)
        )
        if any(length < minimum for length, minimum in zip(available, required, strict=True)):
            raise RuntimeError(
                f"req {req_id} final HMA block table is shorter than its hash prefix: "
                f"saveable={saveable_block_idx} available_by_group={available} "
                f"required_by_group={required}"
            )

        num_new_blocks = saveable_block_idx - start_block_idx
        save_block_ids_by_group = []
        for group_index, group in enumerate(groups):
            if group_index in self._cache_groups.recurrent_group_indices:
                save_block_ids_by_group.append(
                    (0,) * (num_new_blocks - 1) + (group[saveable_block_idx],)
                )
            else:
                save_block_ids_by_group.append(group[start_block_idx:saveable_block_idx])

        self._next_stored_block_idx[req_id] = saveable_block_idx
        return self._build_hma_save_intent(
            req_id,
            tuple(save_block_ids_by_group),
            block_hashes[start_block_idx:saveable_block_idx],
        )

    def _copy_block_ids_by_group(self, block_ids) -> tuple[tuple[int, ...], ...]:
        groups = tuple(tuple(group) for group in block_ids)
        if len(groups) != self._cache_groups.group_count:
            raise RuntimeError(
                "KV cache group count mismatch: "
                f"expected={self._cache_groups.group_count} actual={len(groups)}"
            )
        if any(block_id < 0 for group in groups for block_id in group):
            raise RuntimeError("KV cache block IDs must be non-negative")
        return groups

    def _load_block_ids_by_group(
        self,
        block_ids_by_group: tuple[tuple[int, ...], ...],
        start_block_idx: int,
        num_load_blocks: int,
    ) -> tuple[tuple[int | None, ...], ...]:
        end_block_idx = start_block_idx + num_load_blocks
        available = [len(group) for group in block_ids_by_group]
        if any(length < end_block_idx for length in available):
            raise RuntimeError(
                f"load block mismatch: start={start_block_idx} count={num_load_blocks} "
                f"available_by_group={available}"
            )

        result: list[tuple[int | None, ...]] = []
        for group_index, block_ids in enumerate(block_ids_by_group):
            destinations = block_ids[start_block_idx:end_block_idx]
            if group_index in self._cache_groups.recurrent_group_indices and destinations:
                destinations = (None,) * (len(destinations) - 1) + (destinations[-1],)
            result.append(destinations)
        return tuple(result)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in getattr(connector_output, "finished_recving", None) or []:
            ref = self._pending_local_loads.pop(req_id, None)
            if ref is not None:
                assert self._linear_state_cache is not None
                self._linear_state_cache.unpin(ref)

        worker_meta = getattr(connector_output, "kv_connector_worker_meta", None)
        if worker_meta is not None:
            if not isinstance(worker_meta, PegaWorkerMetadata):
                raise TypeError(f"unexpected PegaFlow worker metadata {type(worker_meta)!r}")
            self._apply_local_save_acks(worker_meta)

        for req_id in connector_output.finished_sending or []:
            self._pending_saves.discard(req_id)
            logger.debug("[PegaKVConnector] Request %s save completed", req_id)

            # Clean up if request already finished
            if req_id in self._held_requests and req_id not in self._deferred_save_intents:
                self._cleanup_request(req_id)
                self._held_requests.discard(req_id)

    def _apply_local_save_acks(self, metadata: PegaWorkerMetadata) -> None:
        for ref, count in metadata.failed.items():
            if count > 0 and ref in self._pending_local_saves:
                logger.error(
                    "[PegaKVConnector] local recurrent save failed: req=%s slot=%d generation=%d",
                    ref.req_id,
                    ref.slot,
                    ref.generation,
                )
                self._cancel_local_save(ref)

        for ref, count in metadata.succeeded.items():
            if ref not in self._pending_local_saves:
                continue
            total = self._local_save_ack_counts.get(ref, 0) + count
            if total > self._ctx.tp_size:
                raise RuntimeError(
                    f"duplicate local recurrent ACKs for {ref}: "
                    f"received={total} expected={self._ctx.tp_size}"
                )
            if total == self._ctx.tp_size:
                assert self._linear_state_cache is not None
                slot_ref = self._pending_local_saves.pop(ref)
                self._linear_state_cache.commit(slot_ref)
                self._forget_local_save(ref)
            else:
                self._local_save_ack_counts[ref] = total

    def _cancel_local_save(self, ref: LocalSaveRef) -> None:
        slot_ref = self._pending_local_saves.pop(ref, None)
        if slot_ref is not None:
            assert self._linear_state_cache is not None
            self._linear_state_cache.cancel(slot_ref)
        self._forget_local_save(ref)

    def _forget_local_save(self, ref: LocalSaveRef) -> None:
        self._local_save_ack_counts.pop(ref, None)
        request_refs = self._local_saves_by_req.get(ref.req_id)
        if request_refs is None:
            return
        request_refs.discard(ref)
        if not request_refs:
            self._local_saves_by_req.pop(ref.req_id, None)

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict | None]:
        req_id = request.request_id

        if self._cache_groups.has_recurrent_state and req_id in self._block_hashes:
            self._block_hashes[req_id] = tuple(request.block_hashes)
            final_save = self._consume_finished_hma_save(
                req_id,
                block_ids,
                request.num_computed_tokens,
            )
            if final_save is not None:
                self._deferred_save_intents[req_id] = final_save
                self._pending_saves.add(req_id)

        # Check if there are pending saves for this request
        if req_id in self._pending_saves:
            self._held_requests.add(req_id)
            logger.debug(
                "[PegaKVConnector] Request %s blocks held for async save",
                req_id,
            )
            return (True, None)

        # No pending saves, clean up immediately
        self._cleanup_request(req_id)
        return (False, None)

    def _cleanup_request(self, req_id: str) -> None:
        """Clean up all state for a completed request."""
        self._release_pending_query_probe(req_id)
        self._requests.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._deferred_save_intents.pop(req_id, None)
        self._pending_saves.discard(req_id)
        if self._linear_state_cache is not None:
            # A committed local load remains pinned even if the request aborts;
            # only finished_recving proves every worker's D2D read is complete.
            for save_ref in tuple(self._local_saves_by_req.get(req_id, ())):
                self._cancel_local_save(save_ref)
        self._tail_saved.discard(req_id)

    def _count_available_block_prefix(
        self, block_hashes: Iterable[bytes], req_id: str
    ) -> ShardedQueryReady | None:
        """Query available blocks with prefetch support.

        Returns:
            ShardedQueryReady: Common ready block count and one lease per TP shard
            None: Blocks are being prefetched from DFS, retry later
        """
        block_hash_list = list(block_hashes)
        ready = self._tp_shard_client.query(
            self._ctx.instance_id,
            block_hash_list,
            req_id,
            self._ctx.wait_for_full_prefix,
        )
        if ready is None:
            if req_id not in self._prefetch_start_times:
                self._prefetch_start_times[req_id] = time.perf_counter()
                self._prefetch_tracker.on_prefetch_start()
                logger.debug(
                    "[PegaKVConnector] Prefetch started: req=%s pending_prefetches=%d",
                    req_id,
                    self._prefetch_tracker.pending_prefetches,
                )
            return None

        if req_id in self._prefetch_start_times:
            prefetch_duration_ms = (
                time.perf_counter() - self._prefetch_start_times.pop(req_id)
            ) * 1000
            self._prefetch_tracker.on_prefetch_complete(prefetch_duration_ms, ready.num_hit_blocks)

            logger.debug(
                "[PegaKVConnector] Prefetch completed: req=%s hit_blocks=%d "
                "prefetch_duration_ms=%.2f pending_prefetches=%d",
                req_id,
                ready.num_hit_blocks,
                prefetch_duration_ms,
                self._prefetch_tracker.pending_prefetches,
            )

        if self._cache_groups.has_recurrent_state:
            return self._reconcile_hybrid(block_hash_list, ready, req_id)
        return ready

    def _reconcile_hybrid(
        self,
        block_hashes: list[bytes],
        ready: ShardedQueryReady,
        req_id: str,
    ) -> ShardedQueryReady:
        """Gate an attention-prefix hit on a usable recurrent boundary.

        HMA can resume only where every recurrent group cached its state on
        every shard: attention KV alone cannot skip mamba's sequential
        prefill. On success returns the reduced hit with the membership
        leases attached; on failure every lease acquired is released and the
        result degrades to a plain miss.
        """
        if ready.num_hit_blocks == 0:
            return ready
        if self._linear_state_cache is not None:
            usable = tuple(
                index
                for index, block_hash in enumerate(block_hashes[: ready.num_hit_blocks])
                if self._linear_state_cache.lookup(block_hash) is not None
            )
            if not usable:
                self._release_leases(ready.leases, req_id)
                logger.info(
                    "[PegaKVConnector] req=%s attention prefix of %d blocks has no "
                    "committed local recurrent checkpoint; recomputing instead",
                    req_id,
                    ready.num_hit_blocks,
                )
                return ShardedQueryReady(0, tuple(b"" for _ in ready.leases))
            checkpoint = max(usable)
            local_ref = self._linear_state_cache.lookup(block_hashes[checkpoint], pin=True)
            assert local_ref is not None
            hybrid_hit = checkpoint + 1
            if hybrid_hit < ready.num_hit_blocks:
                exact = self._tp_shard_client.query(
                    self._ctx.instance_id,
                    block_hashes[:hybrid_hit],
                    f"{req_id}:hma-local-exact-{hybrid_hit}",
                    False,
                )
                self._release_leases(ready.leases, req_id)
                if exact is None or exact.num_hit_blocks != hybrid_hit:
                    if exact is not None:
                        self._release_leases(exact.leases, req_id)
                    self._linear_state_cache.unpin(local_ref)
                    return ShardedQueryReady(0, tuple(b"" for _ in ready.leases))
                ready = exact
            return ShardedQueryReady(
                hybrid_hit,
                ready.leases,
                usable_positions=usable,
                local_recurrent=local_ref,
            )

        group_ids = tuple(
            self._cache_groups.storage_group_ids[index]
            for index in sorted(self._cache_groups.recurrent_group_indices)
        )
        per_group: list[list[tuple[tuple[int, ...], bytes]]] = []
        try:
            for group_id in group_ids:
                per_group.append(
                    self._tp_shard_client.query_group_membership(
                        self._ctx.instance_id,
                        block_hashes,
                        f"{req_id}:g{group_id}",
                        group_id,
                    )
                )
            hybrid_hit, checkpoint, usable = reconcile_hybrid_hit(
                ready.num_hit_blocks,
                tuple(
                    tuple(positions for positions, _ in group_results)
                    for group_results in per_group
                ),
            )
        except Exception:
            self._release_leases(ready.leases, req_id)
            for group_id, group_results in zip(group_ids, per_group, strict=False):
                self._tp_shard_client.release(
                    tuple(lease for _, lease in group_results), f"{req_id}:g{group_id}"
                )
            raise

        if hybrid_hit == 0 or checkpoint is None:
            self._release_leases(ready.leases, req_id)
            for group_id, group_results in zip(group_ids, per_group, strict=True):
                self._tp_shard_client.release(
                    tuple(lease for _, lease in group_results), f"{req_id}:g{group_id}"
                )
            logger.info(
                "[PegaKVConnector] req=%s HMA attention prefix of %d blocks has no "
                "common recurrent checkpoint; recomputing instead",
                req_id,
                ready.num_hit_blocks,
            )
            return ShardedQueryReady(0, tuple(b"" for _ in ready.leases))

        if hybrid_hit < ready.num_hit_blocks:
            # The hit shrank behind the prefix lease: re-lease the exact
            # shortened attention prefix so lease count and load agree.
            exact = self._tp_shard_client.query(
                self._ctx.instance_id,
                block_hashes[:hybrid_hit],
                f"{req_id}:hma-exact-{hybrid_hit}",
                False,
            )
            self._release_leases(ready.leases, req_id)
            if exact is None or exact.num_hit_blocks != hybrid_hit:
                if exact is not None:
                    self._release_leases(exact.leases, req_id)
                for group_id, group_results in zip(group_ids, per_group, strict=True):
                    self._tp_shard_client.release(
                        tuple(lease for _, lease in group_results), f"{req_id}:g{group_id}"
                    )
                logger.warning(
                    "[PegaKVConnector] req=%s could not re-lease the reconciled "
                    "%d-block HMA prefix; recomputing instead",
                    req_id,
                    hybrid_hit,
                )
                return ShardedQueryReady(0, tuple(b"" for _ in ready.leases))
            ready = exact

        return ShardedQueryReady(
            hybrid_hit,
            ready.leases,
            RecurrentLoadHold(
                leases=tuple(
                    tuple(lease for _, lease in group_results) for group_results in per_group
                ),
                hit_positions=tuple(
                    tuple(positions for positions, _ in group_results)
                    for group_results in per_group
                ),
                checkpoint=checkpoint,
            ),
            usable_positions=tuple(sorted(usable)),
        )

    def _cancel_prefetch_tracking(self, req_id: str) -> None:
        """Drop in-flight prefetch metrics when polling stops before QueryReady."""
        if req_id not in self._prefetch_start_times:
            return

        started_at = self._prefetch_start_times.pop(req_id)
        self._prefetch_tracker.on_prefetch_cancel()
        waited_ms = (time.perf_counter() - started_at) * 1000
        logger.warning(
            "[PegaKVConnector] Prefetch aborted before ready: req=%s waited_ms=%.2f "
            "pending_prefetches=%d",
            req_id,
            waited_ms,
            self._prefetch_tracker.pending_prefetches,
        )

    def get_stats(self) -> PegaKVConnectorStats | None:
        """Get current connector stats for metrics exposure."""
        # Get stats from prefetch tracker
        prefetch_stats = self._prefetch_tracker.get_stats()

        data: dict = {
            "pending_prefetches": prefetch_stats["pending_prefetches"],
            "prefetch_duration": prefetch_stats["prefetch_duration"],
            "prefetch_blocks": prefetch_stats["prefetch_blocks"],
        }

        stats = PegaKVConnectorStats(data=data)
        if stats.is_empty():
            return None
        return stats

    def shutdown(self) -> None:
        for req_id in list(self._pending_query_probes):
            self._release_pending_query_probe(req_id)
        if self._linear_state_cache is not None:
            self._pending_local_loads.clear()
            self._pending_local_saves.clear()
            self._local_saves_by_req.clear()
            self._local_save_ack_counts.clear()
            self._linear_state_cache.clear()

    def _release_pending_query_probe(self, req_id: str) -> bool:
        probe = self._pending_query_probes.pop(req_id, None)
        if probe is None:
            return True

        return self._release_query_probe(req_id, probe)

    def _release_query_probe(self, req_id: str, probe: _QueryProbe) -> bool:
        released = True
        if probe.leases and any(probe.leases):
            released = self._release_leases(probe.leases, req_id)
        else:
            self._cancel_prefetch_tracking(req_id)
        hold = probe.recurrent_hold
        if hold is not None:
            for group_index, group_leases in enumerate(hold.leases):
                if not self._tp_shard_client.release(group_leases, f"{req_id}:g{group_index}"):
                    released = False
        if probe.local_recurrent is not None:
            assert self._linear_state_cache is not None
            self._linear_state_cache.unpin(probe.local_recurrent)
        return released

    def _release_leases(self, leases: tuple[bytes, ...], req_id: str) -> bool:
        return self._tp_shard_client.release(leases, req_id)


__all__ = ["SchedulerConnector"]
