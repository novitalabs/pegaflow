"""
Scheduler-side connector logic.
"""

import os
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pegaflow.connector.common import (
    ConnectorContext,
    LoadIntent,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    SaveIntent,
    logger,
)
from pegaflow.pegaflow import PegaFlowBusinessError, PegaFlowServiceError

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


def _parse_env_int(name: str, default: int) -> int:
    """Parse an integer from environment variable with fallback to default."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %d", name, value, default)
        return default


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    # Recompute bypass thresholds (configurable via environment variables)
    # When pending loads >= this, consider bypass for short requests
    BYPASS_PENDING_LOAD_THRESHOLD: int = _parse_env_int("PEGA_BYPASS_PENDING_LOAD_THRESHOLD", 4)
    # Requests with remaining blocks <= this are considered "short"
    BYPASS_SHORT_REQUEST_BLOCKS: int = _parse_env_int("PEGA_BYPASS_SHORT_REQUEST_BLOCKS", 8)

    def __init__(self, context: ConnectorContext):
        self._ctx = context

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._prefetch_start_times: dict[str, float] = {}

        # Fallback statistics
        self._fallback_count: int = 0

        # Save state (per-request)
        self._block_hashes: dict[str, tuple[bytes, ...]] = {}
        self._allocated_blocks: dict[str, list[int]] = {}
        self._scheduled_tokens: dict[str, int] = {}
        self._stored_blocks: dict[str, int] = {}

        # Completion tracking
        self._pending_saves: set[str] = set()
        self._held_requests: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        num_tokens = request.num_tokens
        block_hashes = request.block_hashes

        computed_blocks = num_computed_tokens // self._ctx.block_size
        remaining_hashes = block_hashes[computed_blocks:]

        if not remaining_hashes:
            return (0, False)

        # Recompute fallback: when many requests are waiting for remote KV,
        # let short requests bypass cache lookup and compute locally to
        # improve GPU utilization.
        num_remaining_blocks = len(remaining_hashes)
        num_pending_loads = len(self._pending_load_intents)
        if (
            num_pending_loads >= self.BYPASS_PENDING_LOAD_THRESHOLD
            and num_remaining_blocks <= self.BYPASS_SHORT_REQUEST_BLOCKS
        ):
            self._fallback_count += 1
            logger.info(
                "[PegaKVConnector] req=%s recompute_fallback: "
                "pending_loads=%d remaining_blocks=%d fallback_total=%d",
                req_id,
                num_pending_loads,
                num_remaining_blocks,
                self._fallback_count,
            )
            return (0, False)

        lookup_start = time.perf_counter()
        hit_blocks = self._count_available_block_prefix(remaining_hashes, req_id)
        lookup_end = time.perf_counter()
        elapsed_us = (lookup_end - lookup_start) * 1e6

        # Prefetch in progress - tell scheduler to retry later
        if hit_blocks is None:
            return (None, False)

        # hit_blocks now represents hits in remaining (non-computed) blocks only
        num_hit_tokens = hit_blocks * self._ctx.block_size

        logger.info(
            "[PegaKVConnector] req=%s cache_lookup: hit_blocks=%d computed_blocks=%d "
            "hit_tokens=%d num_tokens=%d lookup_us=%.0f",
            req_id,
            hit_blocks,
            computed_blocks,
            num_hit_tokens,
            num_tokens,
            elapsed_us,
        )

        if num_hit_tokens <= 0:
            return (0, False)

        return (num_hit_tokens, True)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids = list(blocks.get_block_ids()[0]) if blocks else []

        # Reset state for this request (handles preemption correctly)
        self._block_hashes[req_id] = tuple(request.block_hashes)
        self._allocated_blocks[req_id] = block_ids
        self._scheduled_tokens[req_id] = 0
        self._stored_blocks[req_id] = 0

        if num_external_tokens > 0:
            num_load_blocks = num_external_tokens // self._ctx.block_size
            start = len(block_ids) - num_load_blocks

            load_intent = LoadIntent(
                block_ids=tuple(block_ids[start:]),
                block_hashes=tuple(request.block_hashes[start : start + num_load_blocks]),
                num_tokens=num_external_tokens,
            )
            self._pending_load_intents[req_id] = load_intent
            logger.info(
                "[PegaKVConnector] req=%s alloc: total_blocks=%d load_blocks=%d load_tokens=%d",
                req_id,
                len(block_ids),
                len(load_intent.block_ids),
                load_intent.num_tokens,
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        save_intents: dict[str, SaveIntent] = {}

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

            self._scheduled_tokens[req_id] += num_tokens

            if save_intent := self._consume_save_intent(req_id):
                save_intents[req_id] = save_intent

        # Process cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._block_hashes:
                continue

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            self._scheduled_tokens[req_id] += num_tokens

            # Append newly allocated blocks
            new_block_ids = cached_reqs.new_block_ids[idx]
            if new_block_ids:
                self._allocated_blocks[req_id].extend(new_block_ids[0])

            if save_intent := self._consume_save_intent(req_id):
                save_intents[req_id] = save_intent

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
        )

    def _consume_save_intent(self, req_id: str) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving."""
        block_hashes = self._block_hashes.get(req_id)
        if block_hashes is None:
            return None

        allocated = self._allocated_blocks.get(req_id, [])
        scheduled = self._scheduled_tokens.get(req_id, 0)
        stored = self._stored_blocks.get(req_id, 0)

        saveable = min(len(block_hashes), len(allocated), scheduled // self._ctx.block_size)
        new_blocks = saveable - stored
        if new_blocks <= 0:
            return None

        start = stored
        self._stored_blocks[req_id] = stored + new_blocks
        return SaveIntent(
            block_ids=tuple(allocated[start : start + new_blocks]),
            block_hashes=block_hashes[start : start + new_blocks],
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
        self._block_hashes.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._stored_blocks.pop(req_id, None)
        self._pending_saves.discard(req_id)

    def _count_available_block_prefix(
        self, block_hashes: Iterable[bytes], req_id: str
    ) -> int | None:
        """Query available blocks with prefetch support and fault tolerance.

        Returns:
            int: Number of blocks ready in cache (proceed with this)
            None: Blocks are being prefetched from DFS, retry later

        Fault tolerance:
            - If service unavailable, returns 0 (no cache hits)
            - Any exception marks service unavailable and returns 0
        """
        # Check service availability first
        if not self._ctx.state_manager.is_available():
            return 0

        block_hash_list = list(block_hashes)
        try:
            result = self._ctx.engine_client.query(self._ctx.instance_id, block_hash_list)
        except PegaFlowServiceError as e:
            # Service error (network/internal) - mark unavailable
            self._ctx.state_manager.mark_unavailable(str(e))
            return 0
        except PegaFlowBusinessError as e:
            # Business error (invalid args, etc.) - log details and propagate
            logger.error(
                "[PegaKVConnector] Query business error: %s, "
                "req_id=%s, instance_id=%s, num_blocks=%d",
                e,
                req_id,
                self._ctx.instance_id,
                len(block_hash_list),
            )
            raise

        # Handle new dict response format
        if isinstance(result, dict):
            if not result.get("ok", False):
                # Response-level errors are treated as business errors
                error_msg = result.get("message", "unknown error")
                logger.error(
                    "[PegaKVConnector] Query failed: %s, req_id=%s, instance_id=%s, num_blocks=%d",
                    error_msg,
                    req_id,
                    self._ctx.instance_id,
                    len(block_hash_list),
                )
                raise RuntimeError(f"Query failed: {error_msg}")

            prefetch_state = result.get("prefetch_state", "done")
            hit_blocks = result.get("hit_blocks", 0)

            if prefetch_state == "loading":
                # Record first time we see loading state
                if req_id not in self._prefetch_start_times:
                    self._prefetch_start_times[req_id] = time.perf_counter()
                return None  # Signal scheduler to retry later

            # Prefetch done - log duration if we were tracking
            if req_id in self._prefetch_start_times:
                prefetch_duration_ms = (
                    time.perf_counter() - self._prefetch_start_times.pop(req_id)
                ) * 1000
                logger.info(
                    "[PegaKVConnector] Prefetch completed: req=%s hit_blocks=%d prefetch_duration_ms=%.2f",
                    req_id,
                    hit_blocks,
                    prefetch_duration_ms,
                )

            return hit_blocks

        # Legacy tuple response format (ok, message, hit_blocks)
        _, _, hit_blocks = result
        return hit_blocks

    def get_stats(self) -> PegaKVConnectorStats | None:
        """Get current connector stats for metrics exposure."""
        # Calculate pending load blocks
        pending_load_blocks = sum(
            len(intent.block_ids) for intent in self._pending_load_intents.values()
        )

        stats = PegaKVConnectorStats(
            data={
                "pending_load_requests": len(self._pending_load_intents),
                "pending_load_blocks": pending_load_blocks,
                "fallback_count": self._fallback_count,
            }
        )

        # Reset fallback count after reporting (it's a counter)
        self._fallback_count = 0

        if stats.is_empty():
            return None
        return stats


__all__ = ["SchedulerConnector"]
