"""
Scheduler-side connector logic.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pegaflow.connector.common import (
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    SaveIntent,
    logger,
)
from pegaflow.connector.connector_metrics import PrefetchTracker
from pegaflow.pegaflow import QueryLoading, QueryReady

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


@dataclass(slots=True)
class _QueryProbe:
    """One remote prefix-query snapshot.

    A probe is keyed by:

    * ``computed_blocks`` — blocks already computed locally when the query was
      issued
    * ``query_hashes`` — remaining block hashes sent to the backend

    If the request keeps making local progress while the backend is loading,
    the current query key may drift.  A *Ready* result is accepted only if the
    current key still matches this snapshot.
    """

    computed_blocks: int
    query_hashes: tuple[bytes, ...]

    # ``None`` means the backend is still loading.
    hit_blocks: int | None = None
    lease: bytes = b""

    @property
    def is_ready(self) -> bool:
        return self.hit_blocks is not None

    def matches(self, computed_blocks: int, query_hashes: tuple[bytes, ...]) -> bool:
        return self.computed_blocks == computed_blocks and self.query_hashes == query_hashes

    def mark_ready(self, ready: QueryReady) -> None:
        hit_blocks = ready.num_hit_blocks
        if hit_blocks > len(self.query_hashes):
            raise RuntimeError(
                f"invariant violated: server returned {hit_blocks} hits for "
                f"{len(self.query_hashes)} hashes"
            )
        self.hit_blocks = hit_blocks
        self.lease = ready.lease

    def require_hit_blocks(self) -> int:
        if self.hit_blocks is None:
            raise RuntimeError("query probe is still loading")
        return self.hit_blocks

    @property
    def leased_hashes(self) -> tuple[bytes, ...]:
        """Hashes covered by the lease. Only valid after Ready."""
        return self.query_hashes[: self.require_hit_blocks()]


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    def __init__(self, context: ConnectorContext, kv_cache_config=None):
        self._ctx = context

        # Group layout from KVCacheConfig (populated on scheduler role only).
        # A Gemma-style SWA hybrid (5 sliding : 1 full) yields SEVERAL sliding-window groups plus
        # one full-attention group, and vLLM orders them by first-layer index
        # (FA is typically LAST) — never assume 2 groups or that group 0 is
        # full attention.
        self._group_layer_names: list[list[str]] = []
        self._layer_to_group: dict[str, int] = {}
        self._num_groups: int = 1
        self._fa_group_idx: int = 0
        # GPU block pool, bound by the vLLM scheduler via
        # bind_gpu_block_pool. Multi-group saves need it to detect
        # sliding-window blocks that were freed (and possibly reused)
        # before their save was cut.
        self._gpu_block_pool = None
        self._pool_hash_mismatch_logged = False

        if kv_cache_config is not None:
            groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
            if groups:
                self._num_groups = len(groups)
                self._group_layer_names = [list(g.layer_names) for g in groups]
                fa_groups: list[int] = []
                for g_idx, g in enumerate(groups):
                    for ln in g.layer_names:
                        self._layer_to_group[ln] = g_idx
                    spec_name = type(getattr(g, "kv_cache_spec", None)).__name__
                    if spec_name == "FullAttentionSpec":
                        fa_groups.append(g_idx)
                    elif self._num_groups > 1 and spec_name != "SlidingWindowSpec":
                        # Every positional assumption below (1 block per
                        # scheduler block, content hashes, window masking)
                        # is attention-specific. Mamba/SSM, cross-attention
                        # and other state groups must fail loudly, not be
                        # saved/loaded as if they were positional KV.
                        raise ValueError(
                            f"PegaKVConnector does not support KV cache group "
                            f"spec {spec_name} in hybrid (multi-group) models; "
                            f"only FullAttentionSpec + SlidingWindowSpec groups "
                            f"are supported."
                        )
                if self._num_groups > 1 and not fa_groups:
                    raise ValueError(
                        "PegaKVConnector needs at least one FullAttentionSpec "
                        "group in hybrid models (it is the computed-prefix "
                        "reference; sliding-window groups under-count via "
                        "masked positions)."
                    )
                # The FA group's blocks are never masked out, so it is the
                # reference for computed-prefix accounting. Never assume 2
                # groups or that group 0 is full attention: a Gemma-style
                # 5:1 hybrid yields 5 SWA groups + 1 FA group, FA last.
                self._fa_group_idx = fa_groups[0] if fa_groups else 0
                logger.info(
                    "[PegaKVConnector] KV cache groups: %d groups, fa_group=%d, layer_counts=%s",
                    self._num_groups,
                    self._fa_group_idx,
                    [len(g.layer_names) for g in groups],
                )

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._prefetch_start_times: dict[str, float] = {}
        self._pending_query_probes: dict[str, _QueryProbe] = {}

        # Prefetch tracking (for metrics)
        self._prefetch_tracker = PrefetchTracker()

        # Save state (per-request)
        # _allocated_blocks[req_id][g] = list of block IDs for kv_cache_group g
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
        self._pending_saves: set[str] = set()
        self._held_requests: set[str] = set()

        # GPU blocks ref-pinned for in-flight multi-group saves: the save
        # copy runs asynchronously after the scheduler step, so without a
        # pin a sliding-window block can slide out, be reused by another
        # request, and get stored under the original hash (poisoning the
        # cache). Pinned at intent cut, released on save completion or
        # request cleanup.
        self._pinned_save_blocks: dict[str, list] = {}

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        """Called unconditionally by the vLLM scheduler after the KV cache
        manager is built. Multi-group saves consult the pool to skip
        sliding-window blocks that were freed (and possibly reused by
        another request) before their save was cut."""
        self._gpu_block_pool = gpu_block_pool

    def _null_block_id(self) -> int:
        pool = self._gpu_block_pool
        null_block = getattr(pool, "null_block", None) if pool is not None else None
        return getattr(null_block, "block_id", 0)

    def _classify_block_for_save(self, block_id: int, expected_hash: bytes, group_idx: int) -> str:
        """One of:
        "valid"     — the block still holds `expected_hash` content;
        "transient" — the block exists but is not sealed/hashed yet (a
                      freshly scheduled position): retry next round;
        "dead"      — slid out (null placeholder) or reused by another
                      request (foreign hash): this position can never be
                      saved again.
        """
        pool = self._gpu_block_pool
        if pool is None:
            return "transient"
        try:
            blk = pool.blocks[block_id]
        except (KeyError, IndexError, TypeError, AttributeError):
            return "dead"
        if getattr(blk, "is_null", False):
            return "dead"
        bh = getattr(blk, "block_hash", None)
        if bh is None:
            # Allocated but not yet committed to the prefix cache.
            return "transient"
        raw = getattr(bh, "block_hash", bh)
        if raw == expected_hash:
            return "valid"
        # vLLM >= 0.22 stores BlockHashWithGroupId: the plain content hash
        # with a 4-byte big-endian group id appended. Strip and verify both
        # parts — comparing the packed bytes against the plain request hash
        # can never match and silently disabled every hybrid save.
        if (
            isinstance(raw, (bytes, bytearray))
            and isinstance(expected_hash, (bytes, bytearray))
            and len(raw) == len(expected_hash) + 4
            and bytes(raw[:-4]) == bytes(expected_hash)
            and int.from_bytes(raw[-4:], "big") == group_idx
        ):
            return "valid"
        if not self._pool_hash_mismatch_logged:
            self._pool_hash_mismatch_logged = True
            logger.warning(
                "[PegaKVConnector] pool hash mismatch (types %s vs %s, lengths %s vs %s)"
                " — treating block as reused",
                type(raw).__name__,
                type(expected_hash).__name__,
                len(raw) if isinstance(raw, (bytes, bytearray)) else "?",
                len(expected_hash),
            )
        return "dead"

    def _pin_save_blocks(self, req_id: str, blocks: list) -> bool:
        """Ref-pin GPU blocks so the async save copy cannot race a
        slide-out + reuse. Returns False when the pool lacks the ref API
        (version drift): the caller then refuses the save rather than
        risking a poisoned store."""
        if not all(hasattr(b, "incr_ref") for b in blocks):
            return False
        for b in blocks:
            b.incr_ref()
        self._pinned_save_blocks.setdefault(req_id, []).extend(blocks)
        return True

    def _unpin_save_blocks(self, req_id: str) -> None:
        blocks = self._pinned_save_blocks.pop(req_id, None)
        if not blocks:
            return
        pool = self._gpu_block_pool
        free_blocks = getattr(pool, "free_blocks", None)
        if free_blocks is not None:
            # Decrements ref counts and returns fully-released blocks to
            # the pool's free list.
            free_blocks(blocks)
        else:
            for b in blocks:
                if hasattr(b, "decr_ref"):
                    b.decr_ref()

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
        query_hashes = tuple(request.block_hashes[computed_blocks:])

        # Everything is already computed locally.
        if not query_hashes:
            self._release_pending_query_probe(req_id)
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        probe = self._pending_query_probes.get(req_id)

        # Ready result already cached.  Reuse it only if the request identity
        # has not drifted since the query was issued.
        if probe is not None and probe.is_ready:
            if probe.matches(computed_blocks, query_hashes):
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

        # No reusable Ready result.  Ask backend.
        lookup_start = time.perf_counter()
        ready = self._count_available_block_prefix(query_hashes, req_id)
        lookup_us = (time.perf_counter() - lookup_start) * 1e6

        # Backend is still loading.  Keep the original snapshot.
        if ready is None:
            if probe is None:
                self._pending_query_probes[req_id] = _QueryProbe(
                    computed_blocks=computed_blocks,
                    query_hashes=query_hashes,
                )
            return (None, False)

        # A previous Loading probe exists, but the request has moved on.
        # This Ready belongs to the old query.  Do not consume it.
        if probe is not None and not probe.matches(computed_blocks, query_hashes):
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
            if ready.lease:
                self._ctx.engine_client.release(ready.lease)
            self._pending_query_probes.pop(req_id, None)
            return (None, False)

        # Either:
        #   1. IDLE -> Ready
        #   2. Loading probe -> Ready (identity matched above)
        if probe is None:
            probe = _QueryProbe(
                computed_blocks=computed_blocks,
                query_hashes=query_hashes,
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
        hit_tokens = hit_blocks * self._ctx.virtual_block_size

        self._external_matched_blocks[req_id] = computed_blocks + hit_blocks

        if reused:
            logger.debug(
                "[PegaKVConnector] req=%s cache_lookup_reuse: hit_blocks=%d "
                "computed_blocks=%d hit_tokens=%d num_tokens=%d total_query_hashes=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                len(probe.query_hashes),
            )
        else:
            logger.info(
                "[PegaKVConnector] req=%s cache_lookup: hit_blocks=%d computed_blocks=%d "
                "hit_tokens=%d num_tokens=%d lookup_us=%.0f total_query_hashes=%d",
                req_id,
                hit_blocks,
                computed_blocks,
                hit_tokens,
                num_tokens,
                lookup_us or 0.0,
                len(probe.query_hashes),
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
            self._allocated_blocks[req_id] = []
            self._scheduled_tokens[req_id] = 0
            self._next_stored_block_idx[req_id] = base_block_idx

        if num_external_tokens > 0:
            if num_external_tokens % self._ctx.virtual_block_size != 0:
                self._release_pending_query_probe(req_id)
                raise RuntimeError(
                    f"req {req_id} external tokens {num_external_tokens} not "
                    f"block-aligned (block={self._ctx.virtual_block_size})"
                )
            num_load_blocks = num_external_tokens // self._ctx.virtual_block_size

            # Destination ids per group, positionally aligned across groups
            # (uniform block granularity, enforced at connector init). The
            # locally-computed prefix is measured on the FULL-ATTENTION
            # group: its blocks are never masked, so hashed == computed.
            # SWA groups pad out-of-window positions with the null block
            # and would undercount.
            block_ids_by_group = blocks.get_block_ids() if blocks else ()
            fa = min(self._fa_group_idx, max(0, len(block_ids_by_group) - 1))
            num_computed_blocks = (
                sum(blk.block_hash is not None for blk in blocks.blocks[fa]) if blocks else 0
            )
            start_block_idx = num_computed_blocks

            fa_ids = block_ids_by_group[fa] if block_ids_by_group else []
            if len(fa_ids) < start_block_idx + num_load_blocks:
                self._release_pending_query_probe(req_id)
                raise RuntimeError(
                    f"req {req_id} load block mismatch: external={num_load_blocks} "
                    f"allocated={len(fa_ids)} computed={num_computed_blocks}"
                )

            pending_probe = self._pending_query_probes.get(req_id)
            if (
                pending_probe is not None
                and tuple(
                    self._block_hashes[req_id][start_block_idx : start_block_idx + num_load_blocks]
                )
                != pending_probe.leased_hashes[:num_load_blocks]
            ):
                self._release_pending_query_probe(req_id)
                raise RuntimeError(f"req {req_id} load hashes do not match pending query probe")

            # The load RPC requires len(lease hashes) == len(destination ids)
            # per (lease, block_ids) pair. When vLLM accepts fewer blocks than
            # the leased hit (e.g. full-prompt hits are trimmed by one block so
            # the last token is computed for logits), pad the destinations up
            # to the lease length with the null block id — vLLM reserves that
            # slot and never reads it, so the extra DMA is harmless.
            lease_blocks = (
                pending_probe.require_hit_blocks() if pending_probe is not None else num_load_blocks
            )
            null_id = self._null_block_id()
            group_block_ids: list[tuple[int, ...]] = []
            for g_ids in block_ids_by_group:
                dest = list(g_ids[start_block_idx : start_block_idx + num_load_blocks])
                dest += [null_id] * (lease_blocks - len(dest))
                group_block_ids.append(tuple(dest))

            primary_lease = pending_probe.lease if pending_probe is not None else b""
            if not primary_lease:
                raise RuntimeError(f"req {req_id} missing query lease for external load")

            # The server consumes a lease on first load, so every group's
            # load RPC needs its own. Mint (num_groups - 1) extra leases
            # over the same hash range — the initial probe made those
            # blocks resident, so these normally return Ready instantly.
            # A failed mint leaves an empty lease: the worker reports that
            # group's destinations as load errors and vLLM recomputes.
            # Known trade-offs, deliberate for now:
            # - the (num_groups - 1) extra mints are sequential blocking
            #   RPCs in the scheduler step; they only run on an external
            #   hit and the target is a local socket, but batching them
            #   into one multi-lease RPC is the obvious follow-up.
            # - a mint answered with QueryLoading counts as failed and may
            #   leave a server-side prefetch running under the synthetic
            #   "-hma<g>" request id; the server GC reaps it (it is
            #   rare — the primary probe just proved residency).
            group_leases: list[bytes] = [primary_lease]
            if pending_probe is not None and len(group_block_ids) > 1:
                leased = list(pending_probe.leased_hashes)
                for g_idx in range(1, len(group_block_ids)):
                    lease = b""
                    try:
                        result = self._ctx.engine_client.query_prefetch(
                            self._ctx.instance_id,
                            leased,
                            req_id=f"{req_id}-hma{g_idx}",
                        )
                        if isinstance(result, QueryReady) and result.num_hit_blocks == lease_blocks:
                            lease = result.lease
                        elif isinstance(result, QueryReady) and result.lease:
                            self._ctx.engine_client.release(result.lease)
                    except Exception:
                        logger.exception(
                            "[PegaKVConnector] req=%s group %d lease mint failed",
                            req_id,
                            g_idx,
                        )
                    if not lease:
                        logger.warning(
                            "[PegaKVConnector] req=%s group %d has no lease; its "
                            "blocks will be recomputed",
                            req_id,
                            g_idx,
                        )
                    group_leases.append(lease)

            load_intent = LoadIntent(
                group_block_ids=tuple(group_block_ids),
                group_leases=tuple(group_leases),
                num_tokens=num_external_tokens,
            )
            self._pending_load_intents[req_id] = load_intent
            self._pending_query_probes.pop(req_id, None)
            logger.debug(
                "[PegaKVConnector] req=%s alloc: computed_blocks=%d load_blocks=%d "
                "lease_blocks=%d load_tokens=%d pending_loads=%d num_groups=%d",
                req_id,
                num_computed_blocks,
                num_load_blocks,
                lease_blocks,
                load_intent.num_tokens,
                len(self._pending_load_intents),
                len(group_block_ids),
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        # Collect all save intents that became available this scheduler step.
        potential_saves: dict[str, SaveIntent] = {}

        load_intents = self._pending_load_intents
        self._pending_load_intents = {}

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
            # req.block_ids is tuple[list[int], ...]: one list per kv_cache_group.
            if req.block_ids:
                self._allocated_blocks[req_id] = [list(g) for g in req.block_ids]

            if self._ctx.read_enabled:
                self._scheduled_tokens[req_id] += num_tokens
            else:
                self._scheduled_tokens[req_id] = max(
                    self._scheduled_tokens.get(req_id, 0),
                    req.num_computed_tokens + num_tokens,
                )

            if save_intent := self._consume_save_intent(req_id):
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

            # Append newly allocated blocks (new_block_ids: tuple[list[int], ...])
            new_block_ids = cached_reqs.new_block_ids[idx]
            if req_id in cached_reqs.resumed_req_ids:
                self._allocated_blocks[req_id] = (
                    [list(g) for g in new_block_ids] if new_block_ids else []
                )
            elif new_block_ids:
                for g_idx, g_ids in enumerate(new_block_ids):
                    if g_idx < len(self._allocated_blocks[req_id]):
                        self._allocated_blocks[req_id][g_idx].extend(g_ids)
                    else:
                        self._allocated_blocks[req_id].append(list(g_ids))

            if self._ctx.read_enabled:
                self._scheduled_tokens[req_id] += num_tokens
            else:
                prior_computed_tokens = cached_reqs.num_computed_tokens[idx]
                self._scheduled_tokens[req_id] = max(
                    self._scheduled_tokens.get(req_id, 0),
                    prior_computed_tokens + num_tokens,
                )

            if save_intent := self._consume_save_intent(req_id):
                potential_saves[req_id] = save_intent

        save_intents = potential_saves

        # Track requests with pending saves
        self._pending_saves.update(save_intents.keys())

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves",
            len(load_intents),
            len(save_intents),
        )

        return PegaConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
            preempted_req_ids=scheduler_output.preempted_req_ids or None,
            # Only for true multi-group configs: shipping layer_to_group on
            # single-group models flips the worker into HMA save mode and
            # silently discards page-first (MLA) block striping.
            layer_to_group=self._layer_to_group if self._num_groups > 1 else None,
            group_layer_names=self._group_layer_names if self._num_groups > 1 else None,
        )

    def _consume_save_intent(self, req_id: str) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving."""
        # block_hashes are at virtual_block_size granularity, 1-to-1 with block_ids.
        block_hashes = self._block_hashes.get(req_id)
        if block_hashes is None:
            return None

        # allocated is list[list[int]]: one list per kv_cache_group
        allocated = self._allocated_blocks.get(req_id, [])
        scheduled = self._scheduled_tokens.get(req_id, 0)
        base_block_idx = self._block_index_offsets.get(req_id, 0)
        start_block_idx = self._next_stored_block_idx.get(req_id, base_block_idx)

        if not allocated:
            return None

        # Use virtual_block_size because block_hashes are at the scheduler hash
        # granularity. In external-hit cases, scheduled tokens are local-only,
        # so add the first local block index tracked during allocation.
        local_ready_blocks = scheduled // self._ctx.virtual_block_size
        saveable_block_idx = min(len(block_hashes), base_block_idx + local_ready_blocks)

        # All groups allocate one block per scheduler block (uniform block
        # granularity, enforced at init), so the per-group lists are
        # POSITIONALLY parallel: entry p of every group covers the same
        # token range as block_hashes[p]. Never save past what every group
        # has an entry for.
        saveable_block_idx = min(saveable_block_idx, min(len(g) for g in allocated))
        new_blocks = saveable_block_idx - start_block_idx
        if new_blocks <= 0:
            return None

        multi_group = self._num_groups > 1
        if multi_group and self._gpu_block_pool is None:
            # Without the pool we cannot tell whether a sliding-window
            # block slid out (and its slot got reused) before this save —
            # saving it would poison the store under a valid hash.
            logger.warning(
                "[PegaKVConnector] req=%s multi-group save deferred: GPU block "
                "pool not bound (cursor kept; retried once it binds)",
                req_id,
            )
            return None

        # A position is saved only when EVERY group still holds a live,
        # hash-matching block for it, so every stored hash is complete
        # across all layers (a partially-covered hash could serve garbage
        # to the uncovered layers on a later load).
        #   dead      (slid out / reused): skipped for all groups together
        #             and never revisited — an unavoidable, correct hole.
        #   transient (not sealed yet):    STOP here; the cursor does not
        #             advance, so the position is retried next round
        #             instead of being punched out of the hash chain.
        # Positions that pass are ref-pinned until the async save copy
        # completes (finished_sending) — otherwise a sliding-window block
        # can slide out and be reused between this check and the DMA read.
        save_hashes: list[bytes] = []
        group_ids: list[list[int]] = [[] for _ in allocated]
        skipped = 0
        stop_at = saveable_block_idx
        for p in range(start_block_idx, saveable_block_idx):
            ids_p = [g[p] for g in allocated]
            if multi_group:
                states = [
                    self._classify_block_for_save(gid, block_hashes[p], g_idx)
                    for g_idx, gid in enumerate(ids_p)
                ]
                if "transient" in states and "dead" not in states:
                    stop_at = p
                    break
                if "dead" in states:
                    skipped += 1
                    continue
                blks = [self._gpu_block_pool.blocks[gid] for gid in ids_p]
                if not self._pin_save_blocks(req_id, blks):
                    logger.warning(
                        "[PegaKVConnector] req=%s save skipped: block pool "
                        "lacks ref-pin API; refusing unpinned async saves",
                        req_id,
                    )
                    stop_at = p
                    break
            save_hashes.append(block_hashes[p])
            for g_idx, gid in enumerate(ids_p):
                group_ids[g_idx].append(gid)

        self._next_stored_block_idx[req_id] = stop_at
        if not save_hashes:
            return None

        logger.debug(
            "[PegaKVConnector] req=%s save_intent: start=%d saveable=%d "
            "saved=%d skipped_stale=%d total_hashes=%d num_groups=%d",
            req_id,
            start_block_idx,
            saveable_block_idx,
            len(save_hashes),
            skipped,
            len(block_hashes),
            len(allocated),
        )

        return SaveIntent(
            group_block_ids=tuple(tuple(g) for g in group_ids),
            block_hashes=tuple(save_hashes),
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in connector_output.finished_sending or []:
            self._unpin_save_blocks(req_id)
            self._pending_saves.discard(req_id)
            logger.debug("[PegaKVConnector] Request %s save completed", req_id)

            # Clean up if request already finished
            if req_id in self._held_requests:
                self._cleanup_request(req_id)
                self._held_requests.discard(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],  # noqa: ARG002 - required by vLLM interface
    ) -> tuple[bool, dict | None]:
        req_id = request.request_id

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
        self._unpin_save_blocks(req_id)
        self._release_pending_query_probe(req_id)
        self._requests.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._pending_saves.discard(req_id)

    def _count_available_block_prefix(
        self, block_hashes: Iterable[bytes], req_id: str
    ) -> QueryReady | None:
        """Query available blocks with prefetch support.

        Returns:
            QueryReady: Ready block count and lease
            None: Blocks are being prefetched from DFS, retry later
        """
        block_hash_list = list(block_hashes)
        result = self._ctx.engine_client.query_prefetch(
            self._ctx.instance_id,
            block_hash_list,
            req_id=req_id,
        )

        if isinstance(result, QueryLoading):
            if req_id not in self._prefetch_start_times:
                self._prefetch_start_times[req_id] = time.perf_counter()
                self._prefetch_tracker.on_prefetch_start()
                logger.debug(
                    "[PegaKVConnector] Prefetch started: req=%s pending_prefetches=%d",
                    req_id,
                    self._prefetch_tracker.pending_prefetches,
                )
            return None

        if not isinstance(result, QueryReady):
            raise TypeError(f"query_prefetch returned unexpected outcome {type(result)!r}")

        hit_blocks = result.num_hit_blocks
        if req_id in self._prefetch_start_times:
            prefetch_duration_ms = (
                time.perf_counter() - self._prefetch_start_times.pop(req_id)
            ) * 1000
            self._prefetch_tracker.on_prefetch_complete(prefetch_duration_ms, hit_blocks)

            logger.debug(
                "[PegaKVConnector] Prefetch completed: req=%s hit_blocks=%d "
                "prefetch_duration_ms=%.2f pending_prefetches=%d",
                req_id,
                hit_blocks,
                prefetch_duration_ms,
                self._prefetch_tracker.pending_prefetches,
            )

        return result

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

    def _release_pending_query_probe(self, req_id: str) -> bool:
        probe = self._pending_query_probes.pop(req_id, None)
        if probe is None:
            return True

        return self._release_query_probe(req_id, probe)

    def _release_query_probe(self, req_id: str, probe: _QueryProbe) -> bool:
        if not probe.lease:
            self._cancel_prefetch_tracking(req_id)
            return True  # nothing leased server-side (still loading, or zero-hit Ready)
        try:
            self._ctx.engine_client.release(probe.lease)
        except Exception:
            logger.exception(
                "[PegaKVConnector] pending query lease release exception: req=%s",
                req_id,
            )
            return False
        return True


__all__ = ["SchedulerConnector"]
