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
    PegaKVConnectorStats,
    SaveIntent,
    logger,
    parse_env_int,
)
from pegaflow.connector.connector_metrics import PrefetchTracker
from pegaflow.pegaflow import PegaFlowBusinessError, PegaFlowServiceError

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    # Bypass thresholds (configurable via environment variables).
    # Default 0 means disabled - bypass strategy only activates when explicitly set.
    # NOTE: Read from environment at module import time.
    BYPASS_BLOCKS: int = parse_env_int("PEGA_BYPASS_BLOCKS", 0)
    HIGH_LOAD_THRESHOLD: int = parse_env_int("PEGA_HIGH_LOAD_THRESHOLD", 0)

    # Maximum number of requests that can have pending saves simultaneously.
    # Default 0 means unlimited - drop strategy only activates when explicitly set.
    # When this limit is reached, new save intents will be dropped (shorter first).
    MAX_PENDING_SAVE_REQUESTS: int = parse_env_int("PEGA_MAX_PENDING_SAVE_REQUESTS", 0)

    def __init__(self, context: ConnectorContext):
        self._ctx = context

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._prefetch_start_times: dict[str, float] = {}

        # Prefetch tracking (for metrics and bypass decisions)
        self._prefetch_tracker = PrefetchTracker()

        # Bypass statistics
        self._bypass_count: int = 0

        # Save state (per-request)
        self._block_hashes: dict[str, tuple[bytes, ...]] = {}
        self._allocated_blocks: dict[str, list[int]] = {}
        self._scheduled_tokens: dict[str, int] = {}
        self._stored_blocks: dict[str, int] = {}

        # Completion tracking
        self._pending_saves: set[str] = set()
        self._held_requests: set[str] = set()

        # Save drop statistics (for metrics)
        self._save_dropped_count: int = 0

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

        # Check if request should bypass remote cache lookup
        # Bypass short requests when queue is busy to avoid blocking long-running queries
        num_remaining_blocks = len(remaining_hashes)
        pending = self._prefetch_tracker.pending_prefetches
        if num_remaining_blocks < self.BYPASS_BLOCKS and pending >= self.HIGH_LOAD_THRESHOLD:
            self._bypass_count += 1
            logger.info(
                "[PegaKVConnector] req=%s bypass: remaining_blocks=%d "
                "pending_prefetches=%d bypass_count=%d",
                req_id,
                num_remaining_blocks,
                pending,
                self._bypass_count,
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
                "[PegaKVConnector] req=%s alloc: total_blocks=%d load_blocks=%d "
                "load_tokens=%d pending_loads=%d",
                req_id,
                len(block_ids),
                len(load_intent.block_ids),
                load_intent.num_tokens,
                len(self._pending_load_intents),
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        # Collect potential save intents first, then apply drop decision
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

            self._scheduled_tokens[req_id] += num_tokens

            if save_intent := self._consume_save_intent(req_id):
                potential_saves[req_id] = save_intent

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
                potential_saves[req_id] = save_intent

        # Apply save limit: drop new save intents if pending saves exceed limit
        # Priority: longer requests (more blocks) are kept, shorter ones are dropped
        # When MAX_PENDING_SAVE_REQUESTS <= 0, no limit is applied (all saves allowed)
        save_intents: dict[str, SaveIntent] = {}

        if self.MAX_PENDING_SAVE_REQUESTS <= 0:
            # No limit configured - save all requests
            save_intents = potential_saves
        else:
            # Apply limit with length-based priority
            current_pending = len(self._pending_saves)
            available_slots = max(0, self.MAX_PENDING_SAVE_REQUESTS - current_pending)

            # Separate continuing saves (already in pending) from new requests
            continuing_saves: dict[str, SaveIntent] = {}
            new_saves: list[tuple[str, SaveIntent, int]] = []  # (req_id, intent, block_count)

            for req_id, intent in potential_saves.items():
                if req_id in self._pending_saves:
                    # Continuing saves are always allowed
                    continuing_saves[req_id] = intent
                else:
                    # New request - record its total block count for sorting
                    block_count = len(self._block_hashes.get(req_id, ()))
                    new_saves.append((req_id, intent, block_count))

            # Sort new requests by block count (descending) - longer requests first
            new_saves.sort(key=lambda x: x[2], reverse=True)

            # Add all continuing saves
            save_intents.update(continuing_saves)

            # Add new saves up to available slots, prioritizing longer requests
            for req_id, intent, block_count in new_saves:
                if len(save_intents) - len(continuing_saves) < available_slots:
                    save_intents[req_id] = intent
                else:
                    # Drop this save intent due to limit (shorter requests dropped first)
                    self._save_dropped_count += 1
                    logger.warning(
                        "[PegaKVConnector] Save limit reached (%d/%d), dropping req=%s (blocks=%d)",
                        current_pending,
                        self.MAX_PENDING_SAVE_REQUESTS,
                        req_id,
                        block_count,
                    )

        # Track requests with pending saves
        self._pending_saves.update(save_intents.keys())

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves (dropped %d)",
            len(load_intents),
            len(save_intents),
            len(potential_saves) - len(save_intents),
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
                    self._prefetch_tracker.on_prefetch_start()
                    logger.info(
                        "[PegaKVConnector] Prefetch started: req=%s pending_prefetches=%d",
                        req_id,
                        self._prefetch_tracker.pending_prefetches,
                    )
                return None  # Signal scheduler to retry later

            # Prefetch done - log duration if we were tracking
            if req_id in self._prefetch_start_times:
                prefetch_duration_ms = (
                    time.perf_counter() - self._prefetch_start_times.pop(req_id)
                ) * 1000
                self._prefetch_tracker.on_prefetch_complete(prefetch_duration_ms, hit_blocks)

                logger.info(
                    "[PegaKVConnector] Prefetch completed: req=%s hit_blocks=%d "
                    "prefetch_duration_ms=%.2f pending_prefetches=%d",
                    req_id,
                    hit_blocks,
                    prefetch_duration_ms,
                    self._prefetch_tracker.pending_prefetches,
                )

            return hit_blocks

        # Legacy tuple response format (ok, message, hit_blocks)
        _, _, hit_blocks = result
        return hit_blocks

    def get_stats(self) -> PegaKVConnectorStats | None:
        """Get current connector stats for metrics exposure."""
        # Get stats from prefetch tracker
        prefetch_stats = self._prefetch_tracker.get_stats()

        data: dict = {
            "pending_prefetches": prefetch_stats["pending_prefetches"],
            "bypass_count": self._bypass_count,
            "prefetch_duration": prefetch_stats["prefetch_duration"],
            "prefetch_blocks": prefetch_stats["prefetch_blocks"],
        }

        # Add save_dropped_count if there were any drops
        if self._save_dropped_count > 0:
            data["save_dropped_count"] = self._save_dropped_count
            self._save_dropped_count = 0

        # Reset bypass count after reporting (it's a counter)
        self._bypass_count = 0

        stats = PegaKVConnectorStats(data=data)
        if stats.is_empty():
            return None
        return stats


__all__ = ["SchedulerConnector"]
