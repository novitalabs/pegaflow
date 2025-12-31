"""
Scheduler-side connector logic.
"""

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pegaflow.connector.common import (
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
    RequestTracker,
    SaveIntent,
    logger,
)
from pegaflow.logging_utils import timing_wrapper

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    def __init__(self, context: ConnectorContext):
        self._ctx = context
        self._trackers: dict[str, RequestTracker] = {}
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._held_requests: set[str] = set()

    @timing_wrapper
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        num_tokens = request.num_tokens
        block_hashes = request.block_hashes

        tracker = self._get_or_create_tracker(request)

        lookup_start = time.perf_counter()
        hit_blocks = self._count_available_block_prefix(block_hashes)
        lookup_end = time.perf_counter()
        elapsed_us = (lookup_end - lookup_start) * 1e6

        computed_blocks = num_computed_tokens // self._ctx.block_size

        tracker.on_lookup(hit_blocks, computed_blocks)

        num_hit_tokens = hit_blocks * self._ctx.block_size - num_computed_tokens
        if num_hit_tokens <= 0:
            return (0, False)

        if num_hit_tokens >= num_tokens:
            num_hit_tokens = num_tokens - 1

        need_to_compute_tokens = num_tokens - num_hit_tokens

        logger.info(
            "[PegaKVConnector] hit_blocks=%d computed_blocks=%d need_to_compute_tokens=%d "
            "hit_tokens=%d elapsed_us=%.0f for request %s",
            hit_blocks,
            computed_blocks,
            need_to_compute_tokens,
            num_hit_tokens,
            elapsed_us,
            req_id,
        )

        return (num_hit_tokens, True)

    @timing_wrapper
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        tracker = self._trackers.get(req_id)
        if tracker is None:
            logger.warning(
                "[PegaKVConnector] No tracker for request %s in update_state_after_alloc",
                req_id,
            )
            return

        block_ids = list(blocks.get_block_ids()[0]) if blocks else []
        tracker.on_alloc(block_ids, num_external_tokens)

        # Always consume to clear _load state, avoiding stale state on preemption
        load_intent = tracker.consume_load_intent()
        if load_intent is not None:
            self._pending_load_intents[req_id] = load_intent
            logger.debug(
                "[PegaKVConnector] update_state_after_alloc req=%s created LoadIntent: "
                "%d blocks, %d tokens",
                req_id,
                len(load_intent.block_ids),
                load_intent.num_tokens,
            )

        logger.debug(
            "[PegaKVConnector] update_state_after_alloc req=%s blocks=%d external_tokens=%d phase=%s",
            req_id,
            len(block_ids),
            num_external_tokens,
            tracker.phase.value,
        )

    @timing_wrapper
    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> PegaConnectorMetadata:
        save_intents: dict[str, SaveIntent] = {}

        load_intents = self._pending_load_intents
        self._pending_load_intents = {}

        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            tracker = self._trackers.get(req_id)
            if tracker is None:
                continue

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            tracker.on_scheduled(num_tokens)

            if req_id not in load_intents:
                if load_intent := tracker.consume_load_intent():
                    load_intents[req_id] = load_intent

            if save_intent := tracker.consume_save_intent():
                save_intents[req_id] = save_intent

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            tracker = self._trackers.get(req_id)
            if tracker is None:
                continue

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            tracker.on_scheduled(num_tokens)

            new_block_ids = cached_reqs.new_block_ids[idx]
            if new_block_ids:
                tracker.on_alloc(list(new_block_ids[0]), 0)

            if save_intent := tracker.consume_save_intent():
                save_intents[req_id] = save_intent

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves",
            len(load_intents),
            len(save_intents),
        )

        return PegaConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
        )

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        for req_id in connector_output.finished_sending or []:
            tracker = self._trackers.get(req_id)
            if tracker:
                while tracker._saved_layers < tracker._total_layers:
                    tracker.on_layer_saved()
                logger.debug(
                    "[PegaKVConnector] Request %s save completed, phase=%s",
                    req_id,
                    tracker.phase.value,
                )

                if tracker.is_done():
                    del self._trackers[req_id]
                    logger.debug("[PegaKVConnector] Cleaned up tracker for %s", req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict | None]:
        req_id = request.request_id
        tracker = self._trackers.get(req_id)

        if tracker:
            tracker.on_finished()

            if tracker.should_hold_blocks():
                self._held_requests.add(req_id)
                logger.debug(
                    "[PegaKVConnector] Request %s blocks held for async save",
                    req_id,
                )
                return (True, None)

        return (False, None)

    def _get_or_create_tracker(self, request: "Request") -> RequestTracker:
        req_id = request.request_id
        if req_id not in self._trackers:
            self._trackers[req_id] = RequestTracker(
                request_id=req_id,
                block_hashes=list(request.block_hashes),
                block_size=self._ctx.block_size,
                num_layers=self._ctx.num_layers,
            )
        return self._trackers[req_id]

    def _count_available_block_prefix(self, block_hashes: Iterable[bytes]) -> int:
        ok, message, hit_blocks = self._ctx.engine_client.query(
            self._ctx.instance_id, list(block_hashes)
        )
        if not ok:
            raise RuntimeError(f"Query failed: {message}")
        return hit_blocks


__all__ = ["SchedulerConnector"]
