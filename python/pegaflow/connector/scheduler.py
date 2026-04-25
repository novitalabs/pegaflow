"""
Scheduler-side connector logic.
"""

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pegaflow.connector.common import (
    ConnectorContext,
    KvEgressIntent,
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
        self._pd_receive_prepares: dict[tuple[str, str], dict] = {}
        self._pd_receive_prepare_keys_by_req: dict[str, set[tuple[str, str]]] = {}

        # Prefetch tracking (for metrics and bypass decisions)
        self._prefetch_tracker = PrefetchTracker()

        # Bypass statistics
        self._bypass_count: int = 0

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

        pd_params = self._pd_push_params(request)
        if pd_params is not None:
            return self._prepare_pd_receive(request, num_computed_tokens, pd_params)

        # request.block_hashes are already at virtual_block_size granularity
        # (vLLM hashes every scheduler_block_size =
        # block_size * dcp_world_size * pcp_world_size).
        # Skip blocks that are already computed locally.
        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size
        remaining_hashes = block_hashes[computed_blocks:]

        if not remaining_hashes:
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        # Check if request should bypass remote cache lookup
        # Bypass short requests when queue is busy to avoid blocking long-running queries
        num_remaining_blocks = len(remaining_hashes)
        pending = self._prefetch_tracker.pending_prefetches
        if num_remaining_blocks < self.BYPASS_BLOCKS and pending >= self.HIGH_LOAD_THRESHOLD:
            self._external_matched_blocks[req_id] = computed_blocks
            self._bypass_count += 1
            logger.debug(
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

        self._external_matched_blocks[req_id] = computed_blocks + hit_blocks

        # Each hit block = 1 virtual block = virtual_block_size global tokens.
        num_hit_tokens = hit_blocks * self._ctx.virtual_block_size

        logger.debug(
            "[PegaKVConnector] req=%s cache_lookup: hit_blocks=%d computed_blocks=%d "
            "hit_tokens=%d num_tokens=%d lookup_us=%.0f total_query_hashes=%d",
            req_id,
            hit_blocks,
            computed_blocks,
            num_hit_tokens,
            num_tokens,
            elapsed_us,
            len(remaining_hashes),
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
            pd_request_id, pd_handle = self._pd_receive_load_ref(request)
            block_ids = list(blocks.get_block_ids()[0]) if blocks else []
            num_computed_blocks = (
                sum(block.block_hash is not None for block in blocks.blocks[0]) if blocks else 0
            )
            start_block_idx = num_computed_blocks
            num_load_blocks = num_external_tokens // self._ctx.virtual_block_size
            expected_load_blocks = len(block_ids) - num_computed_blocks
            assert num_load_blocks == expected_load_blocks, (
                f"req {req_id} load block mismatch: external={num_load_blocks} "
                f"expected={expected_load_blocks}"
            )

            load_intent = LoadIntent(
                block_ids=tuple(block_ids[start_block_idx:]),
                block_hashes=tuple(
                    self._block_hashes[req_id][start_block_idx : start_block_idx + num_load_blocks]
                ),
                num_tokens=num_external_tokens,
                pd_request_id=pd_request_id,
                pd_handle=pd_handle,
            )
            self._pending_load_intents[req_id] = load_intent
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

            # Populate block IDs from scheduler_output — single source of
            # truth for the save path (consistent with offloading connector).
            if req.block_ids:
                self._allocated_blocks[req_id] = list(req.block_ids[0])

            self._scheduled_tokens[req_id] += num_tokens

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
        egress_intents = self._build_egress_intents(save_intents)

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves, %d egress "
            "(dropped %d)",
            len(load_intents),
            len(save_intents),
            len(egress_intents),
            len(potential_saves) - len(save_intents),
        )

        return PegaConnectorMetadata(
            load_intents=load_intents,
            save_intents=save_intents,
            egress_intents=egress_intents,
            preempted_req_ids=scheduler_output.preempted_req_ids or None,
        )

    def _consume_save_intent(self, req_id: str) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving."""
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
        self._requests.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._pending_saves.discard(req_id)
        for key in self._pd_receive_prepare_keys_by_req.pop(req_id, set()):
            self._pd_receive_prepares.pop(key, None)

    def _pd_push_params(self, request: "Request") -> dict | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return None
        if _is_source_role(params):
            return None

        nested = params.get("pegaflow_pd")
        if isinstance(nested, dict):
            enabled = nested.get("enabled", True)
            mode = nested.get("mode")
            if not enabled or _is_source_role(nested):
                return None
            if mode in (None, "push", "cpu_staging_push") or nested.get("pd_push"):
                merged = dict(params)
                merged.update(nested)
                return merged

        if params.get("pegaflow_pd_push") or params.get("pd_push"):
            return params
        return None

    def _pd_receive_load_ref(self, request: "Request") -> tuple[str | None, str | None]:
        params = self._pd_push_params(request)
        if params is None:
            return (None, None)

        req_id = request.request_id
        pd_request_id = str(
            params.get("pd_request_id") or params.get("request_id") or req_id
        )
        dst_instance_id = str(
            params.get("dst_instance_id")
            or params.get("d_instance_id")
            or params.get("instance_id")
            or self._ctx.instance_id
        )
        response = self._pd_receive_prepares.get((dst_instance_id, pd_request_id), {})
        handle = _param_str(params, ("handle", "pd_handle")) or response.get("handle")
        return (pd_request_id, str(handle) if handle else None)

    def _kv_egress_params(self, request: "Request") -> dict | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return None

        egress = params.get("pegaflow_kv_egress")
        if isinstance(egress, dict):
            if not egress.get("enabled", True):
                return None
            merged = dict(params)
            merged.update(egress)
            return merged
        if egress:
            return params

        nested = params.get("pegaflow_pd")
        if isinstance(nested, dict):
            if not nested.get("enabled", True):
                return None
            if (
                _is_source_role(params)
                or _is_source_role(nested)
                or nested.get("egress")
                or nested.get("source")
            ):
                merged = dict(params)
                merged.update(nested)
                return merged

        if _is_source_role(params) and (params.get("pegaflow_pd_push") or params.get("pd_push")):
            return params

        return None

    def _build_egress_intents(
        self,
        save_intents: dict[str, SaveIntent],
    ) -> dict[str, KvEgressIntent]:
        egress_intents: dict[str, KvEgressIntent] = {}
        for req_id, save_intent in save_intents.items():
            request = self._requests.get(req_id)
            if request is None:
                continue

            params = self._kv_egress_params(request)
            if params is None:
                continue

            d_pegaflow_addr = _param_str(
                params,
                ("d_pegaflow_addr", "dst_pegaflow_addr", "remote_addr"),
            )
            if not d_pegaflow_addr:
                logger.warning(
                    "[PegaKVConnector] req=%s P/D egress ignored: missing d_pegaflow_addr",
                    req_id,
                )
                continue

            dst_instance_id = _param_str(
                params,
                ("dst_instance_id", "d_instance_id", "instance_id"),
            )
            if not dst_instance_id:
                logger.warning(
                    "[PegaKVConnector] req=%s P/D egress ignored: missing dst_instance_id",
                    req_id,
                )
                continue

            pd_request_id = str(
                params.get("pd_request_id") or params.get("request_id") or req_id
            )
            handle = _param_str(params, ("handle", "pd_handle"))

            egress_intents[req_id] = KvEgressIntent(
                pd_request_id=pd_request_id,
                d_pegaflow_addr=d_pegaflow_addr,
                dst_instance_id=dst_instance_id,
                block_ids=save_intent.block_ids,
                block_hashes=save_intent.block_hashes,
                handle=handle or None,
            )

        return egress_intents

    def _prepare_pd_receive(
        self,
        request: "Request",
        num_computed_tokens: int,
        params: dict,
    ) -> tuple[int | None, bool]:
        req_id = request.request_id
        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size
        num_blocks = max(0, len(request.block_hashes) - computed_blocks)
        if num_blocks == 0:
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        pd_request_id = str(
            params.get("pd_request_id") or params.get("request_id") or req_id
        )
        dst_instance_id = str(
            params.get("dst_instance_id")
            or params.get("d_instance_id")
            or params.get("instance_id")
            or self._ctx.instance_id
        )
        expected_imm_count = _param_int(
            params,
            ("expected_imm_count", "expected_contributors"),
            0,
        )
        expire_after_ms = _param_int(params, ("expire_after_ms", "ttl_ms"), 0)

        prepare_key = (dst_instance_id, pd_request_id)
        if prepare_key in self._pd_receive_prepares:
            return self._poll_pd_receive_ready(
                req_id,
                pd_request_id,
                dst_instance_id,
                num_blocks,
                self._pd_receive_prepares[prepare_key],
            )

        remaining_hashes = list(
            request.block_hashes[computed_blocks : computed_blocks + num_blocks]
        )
        block_hashes = [bytes(h) for h in remaining_hashes]

        try:
            response = self._ctx.engine_client.prepare_pd_receive(
                dst_instance_id,
                pd_request_id,
                block_hashes,
                num_blocks,
                expected_imm_count,
                expire_after_ms,
            )
        except PegaFlowServiceError as e:
            self._ctx.state_manager.mark_unavailable(str(e))
            logger.warning(
                "[PegaKVConnector] req=%s pd_receive_prepare service error: %s",
                req_id,
                e,
            )
            return (None, False)
        except PegaFlowBusinessError as e:
            logger.error(
                "[PegaKVConnector] req=%s pd_receive_prepare rejected: %s",
                req_id,
                e,
            )
            raise

        if not response.get("ok", False):
            raise RuntimeError(
                "PreparePdReceive failed: "
                f"{response.get('message', 'unknown error')}"
            )

        self._pd_receive_prepares[prepare_key] = response
        self._pd_receive_prepare_keys_by_req.setdefault(req_id, set()).add(prepare_key)
        logger.info(
            "[PegaKVConnector] req=%s pd_receive_prepare accepted: pd_request_id=%s "
            "instance=%s blocks=%d expected_imm_count=%d handle=%s imm=%s",
            req_id,
            pd_request_id,
            dst_instance_id,
            num_blocks,
            expected_imm_count,
            response.get("handle"),
            response.get("imm_data"),
        )

        # First prepare creates the CPU staging lease. Later scheduler polls
        # return (N, True) only after D observes WRITE_WITH_IMM.
        return (None, False)

    def _poll_pd_receive_ready(
        self,
        req_id: str,
        pd_request_id: str,
        dst_instance_id: str,
        num_blocks: int,
        prepare_response: dict,
    ) -> tuple[int | None, bool]:
        try:
            descriptor = self._ctx.engine_client.get_pd_receive_descriptor(
                dst_instance_id,
                pd_request_id,
                -1,
                prepare_response.get("handle"),
            )
        except PegaFlowServiceError as e:
            self._ctx.state_manager.mark_unavailable(str(e))
            logger.warning(
                "[PegaKVConnector] req=%s pd_receive_status service error: %s",
                req_id,
                e,
            )
            return (None, False)
        except PegaFlowBusinessError as e:
            logger.error(
                "[PegaKVConnector] req=%s pd_receive_status rejected: %s",
                req_id,
                e,
            )
            raise

        if not descriptor.get("ok", False):
            raise RuntimeError(
                "GetPdReceiveDescriptor failed: "
                f"{descriptor.get('message', 'unknown error')}"
            )

        state = descriptor.get("state", "pending")
        if state in {"failed", "expired"}:
            raise RuntimeError(
                f"P/D receive lease is not usable: request_id={pd_request_id} state={state}"
            )

        if state == "ready" and descriptor.get("data_ready", False):
            ready_tokens = num_blocks * self._ctx.virtual_block_size
            logger.info(
                "[PegaKVConnector] req=%s pd_receive ready: pd_request_id=%s "
                "instance=%s blocks=%d tokens=%d ranks=%d",
                req_id,
                pd_request_id,
                dst_instance_id,
                num_blocks,
                ready_tokens,
                len(descriptor.get("ranks") or ()),
            )
            return (ready_tokens, True)

        logger.debug(
            "[PegaKVConnector] req=%s pd_receive pending: pd_request_id=%s "
            "state=%s data_ready=%s",
            req_id,
            pd_request_id,
            state,
            descriptor.get("data_ready", False),
        )
        return (None, False)

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
            result = self._ctx.engine_client.query_prefetch(
                self._ctx.instance_id,
                block_hash_list,
                req_id=req_id,
            )
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
                    logger.debug(
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

                logger.debug(
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


def _param_int(params: dict, keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        if key in params and params[key] is not None:
            return int(params[key])
    return int(default)


def _param_str(params: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = params.get(key)
        if value is not None and value != "":
            return str(value)
    return ""


def _is_source_role(params: dict) -> bool:
    for key in ("role", "pd_role", "side", "phase"):
        value = params.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"p", "prefill", "producer", "source", "egress", "sender"}:
            return True
    return False


__all__ = ["SchedulerConnector"]
