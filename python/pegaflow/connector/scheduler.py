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
    tail_tokens: int = 0

    # ``None`` means the backend is still loading.
    hit_blocks: int | None = None
    lease: bytes = b""

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

    def __init__(
        self,
        context: ConnectorContext,
        pd_tail_save: bool = False,
        pd_tail_load: bool = False,
        vllm_config=None,
    ):
        self._ctx = context

        # P/D tail-block extension (`pegaflow.pd_tail_save`): vLLM only hashes
        # full blocks, so a prompt's partial tail block never enters the tier
        # and a strict no-prefill decode peer would have to recompute it.
        # When enabled, the step that schedules the final prompt chunk also
        # saves the partial tail block under a key derived with vLLM's OWN
        # hash function over (last_full_hash, tail_prompt_token_ids, None) —
        # well-defined, and independently derivable by the decode peer.
        # Only xxhash_cbor is validated cross-engine; everything else is
        # rejected (the pickle-based algos aren't even derivable elsewhere).
        self._tail_save_enabled = pd_tail_save
        self._tail_load_enabled = pd_tail_load
        self._tail_hash_fn = None
        if pd_tail_save or pd_tail_load:
            assert vllm_config is not None
            algo = vllm_config.cache_config.prefix_caching_hash_algo
            if algo != "xxhash_cbor":
                enabled_option = (
                    "pegaflow.pd_tail_save" if pd_tail_save else "pegaflow.pd_tail_load"
                )
                raise ValueError(
                    f"{enabled_option} requires prefix_caching_hash_algo "
                    f"'xxhash_cbor' (cross-engine derivable, validated); got {algo!r} — "
                    f"run vLLM with --prefix-caching-hash-algo xxhash_cbor"
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
        self._allocated_blocks: dict[str, list[int]] = {}
        self._scheduled_tokens: dict[str, int] = {}
        self._next_stored_block_idx: dict[str, int] = {}

        # Live Request references – used to refresh block_hashes during decode
        # so that newly completed blocks can be saved, not just prefill blocks.
        self._requests: dict[str, Request] = {}

        # Completion tracking
        self._pending_saves: set[str] = set()
        self._held_requests: set[str] = set()

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

        # Everything is already computed locally.
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
        hit_tokens = (hit_blocks - 1) * vbs + probe.tail_tokens if tail_hit else hit_blocks * vbs

        # A request still needs one forward token to produce logits. The last
        # loaded page may contain that token's KV, but vLLM must recompute and
        # overwrite it unless a P/D router supplied a separate decode token.
        locally_computed_tokens = computed_blocks * vbs
        hit_tokens = min(hit_tokens, max(0, num_tokens - locally_computed_tokens - 1))

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
            self._allocated_blocks[req_id] = []
            self._scheduled_tokens[req_id] = 0
            self._next_stored_block_idx[req_id] = base_block_idx

        if num_external_tokens > 0:
            block_ids = list(blocks.get_block_ids()[0]) if blocks else []
            num_computed_blocks = (
                sum(block.block_hash is not None for block in blocks.blocks[0]) if blocks else 0
            )
            start_block_idx = num_computed_blocks
            vbs = self._ctx.virtual_block_size
            num_load_blocks = (num_external_tokens + vbs - 1) // vbs
            expected_load_blocks = len(block_ids) - num_computed_blocks
            if num_load_blocks != expected_load_blocks:
                self._release_pending_query_probe(req_id)
                raise RuntimeError(
                    f"req {req_id} load block mismatch: external={num_load_blocks} "
                    f"expected={expected_load_blocks}"
                )

            pending_probe = self._pending_query_probes.get(req_id)
            load_intent = LoadIntent(
                block_ids=tuple(block_ids[start_block_idx:]),
                lease=pending_probe.lease if pending_probe is not None else b"",
                num_tokens=num_external_tokens,
            )
            if pending_probe is not None:
                query_hashes, tail_tokens = self._build_query(request, num_computed_blocks)
                if (
                    not pending_probe.matches(num_computed_blocks, query_hashes, tail_tokens)
                    or pending_probe.leased_hashes != query_hashes[:num_load_blocks]
                ):
                    self._release_pending_query_probe(req_id)
                    raise RuntimeError(f"req {req_id} load hashes do not match pending query probe")
            if not load_intent.lease:
                raise RuntimeError(f"req {req_id} missing query lease for external load")
            self._pending_load_intents[req_id] = load_intent
            self._pending_query_probes.pop(req_id, None)
            logger.debug(
                "[PegaKVConnector] req=%s alloc: total_blocks=%d computed_blocks=%d "
                "load_blocks=%d start_block_idx=%d load_tokens=%d pending_loads=%d",
                req_id,
                len(block_ids),
                num_computed_blocks,
                len(load_intent.block_ids),
                start_block_idx,
                load_intent.num_tokens,
                len(self._pending_load_intents),
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
            if req.block_ids:
                self._allocated_blocks[req_id] = list(req.block_ids[0])

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
            if save_intent := self._consume_save_intent(req_id, written):
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
                self._allocated_blocks[req_id] = list(new_block_ids[0]) if new_block_ids else []
            elif new_block_ids:
                self._allocated_blocks[req_id].extend(new_block_ids[0])

            if self._ctx.read_enabled:
                self._scheduled_tokens[req_id] += num_tokens
            else:
                prior_computed_tokens = cached_reqs.num_computed_tokens[idx]
                self._scheduled_tokens[req_id] = max(
                    self._scheduled_tokens.get(req_id, 0),
                    prior_computed_tokens + num_tokens,
                )

            written = cached_reqs.num_computed_tokens[idx] + num_tokens
            if save_intent := self._consume_save_intent(req_id, written):
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
        )

    def _consume_save_intent(self, req_id: str, written: int) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving.

        `written` = positions with valid KV once this step's schedule runs
        (scheduler-authoritative num_computed_tokens + this step's tokens).
        """
        regular = self._consume_full_block_saves(req_id)
        tail = self._consume_tail_save(req_id, written)
        if tail is None:
            return regular
        if regular is None:
            return tail
        return SaveIntent(
            block_ids=regular.block_ids + tail.block_ids,
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
        if tail_idx >= len(allocated) or tail_idx > len(block_hashes):
            return None  # tail block not allocated / full-block hashes lagging
        self._tail_saved.add(req_id)
        logger.info(
            "[PegaKVConnector] req=%s pd_tail_save: block_id=%d tail_tokens=%d key=%s",
            req_id,
            allocated[tail_idx],
            tail_len,
            tail_key.hex(),
        )
        return SaveIntent(block_ids=(allocated[tail_idx],), block_hashes=(tail_key,))

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
        if tail_len == 0:
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
        # Loading a one-token tail cannot reduce local work: vLLM must always
        # execute at least the final prompt token to produce logits.
        if tail is None or tail[1] <= 1:
            return query_hashes, 0
        return query_hashes + (tail[0],), tail[1]

    def _consume_full_block_saves(self, req_id: str) -> SaveIntent | None:
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
        local_saveable = min(
            len(allocated),
            scheduled // self._ctx.virtual_block_size,
        )
        saveable_block_idx = min(len(block_hashes), base_block_idx + local_saveable)
        new_blocks = saveable_block_idx - start_block_idx
        if new_blocks <= 0:
            return None

        self._next_stored_block_idx[req_id] = saveable_block_idx
        hash_start = start_block_idx
        save_hashes = block_hashes[hash_start : hash_start + new_blocks]
        save_block_ids = allocated[hash_start : hash_start + new_blocks]

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

        return SaveIntent(
            block_ids=tuple(save_block_ids),
            block_hashes=save_hashes,
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in connector_output.finished_sending or []:
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
        self._release_pending_query_probe(req_id)
        self._requests.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._pending_saves.discard(req_id)
        self._tail_saved.discard(req_id)

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
            wait_for_full_prefix=self._ctx.wait_for_full_prefix,
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
