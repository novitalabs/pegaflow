"""
Scheduler-side connector logic.
"""

import time
from typing import TYPE_CHECKING

from pegaflow import (
    LoadPlan,
    PrepareLoadResult,
)
from pegaflow.connector.common import (
    ConnectorContext,
    KvEgressIntent,
    LoadIntent,
    PegaConnectorMetadata,
    PegaKVConnectorStats,
    SaveIntent,
    logger,
)
from pegaflow.kv_transfer import (
    prefill_push_from_request,
    prepare_load_request_from_request,
)
from pegaflow.pegaflow import PegaFlowBusinessError, PegaFlowServiceError

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from pegaflow import PrepareLoadRequest


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    def __init__(self, context: ConnectorContext):
        self._ctx = context

        # Load state
        self._pending_load_intents: dict[str, LoadIntent] = {}
        self._load_plans: dict[str, LoadPlan] = {}
        self._prefill_push_submitted: set[str] = set()

        # Save state (per-request)
        self._block_hashes: dict[str, tuple[bytes, ...]] = {}
        self._external_matched_blocks: dict[str, int] = {}
        self._block_index_offsets: dict[str, int] = {}
        self._allocated_blocks: dict[str, list[int]] = {}
        self._computed_blocks: dict[str, int] = {}
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
        num_tokens = request.num_tokens

        # request.block_hashes are already at virtual_block_size granularity
        # (vLLM hashes every scheduler_block_size =
        # block_size * dcp_world_size * pcp_world_size).
        # Only the scheduler-visible index is needed here; source-specific
        # matching and P/D polling are handled by prepare_load.
        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size

        prepare_start = time.perf_counter()
        result = self._prepare_load(request, num_computed_tokens)
        prepare_end = time.perf_counter()
        elapsed_us = (prepare_end - prepare_start) * 1e6

        if result.preparing:
            return (None, False)

        plan = result.plan
        if plan is None or plan.num_tokens <= 0:
            self._external_matched_blocks[req_id] = computed_blocks
            return (0, False)

        self._load_plans[req_id] = plan
        self._external_matched_blocks[req_id] = computed_blocks + plan.num_blocks

        logger.debug(
            "[PegaKVConnector] req=%s prepare_load: source=%s blocks=%d "
            "computed_blocks=%d external_tokens=%d num_tokens=%d prepare_us=%.0f",
            req_id,
            plan.source.value,
            plan.num_blocks,
            computed_blocks,
            plan.num_tokens,
            num_tokens,
            elapsed_us,
        )

        return (plan.num_tokens, True)

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
        self._computed_blocks[req_id] = max(
            self._computed_blocks.get(req_id, 0),
            _count_computed_blocks(blocks),
        )
        if req_id not in self._allocated_blocks:
            # The first locally allocated block may be after an external-hit
            # prefix. Track that global block index explicitly.
            base_block_idx = self._external_matched_blocks.get(req_id, 0)
            self._block_index_offsets[req_id] = base_block_idx
            self._allocated_blocks[req_id] = []
            self._scheduled_tokens[req_id] = 0
            self._next_stored_block_idx[req_id] = base_block_idx

        if num_external_tokens > 0:
            plan = self._load_plans.get(req_id)
            if plan is None:
                raise RuntimeError(f"req {req_id} missing ready load plan")
            block_ids = list(blocks.get_block_ids()[0]) if blocks else []
            num_computed_blocks = self._computed_blocks.get(req_id, 0)
            start_block_idx = num_computed_blocks
            num_load_blocks = _ceil_div(num_external_tokens, self._ctx.virtual_block_size)
            expected_load_blocks = len(block_ids) - num_computed_blocks
            assert num_load_blocks == expected_load_blocks, (
                f"req {req_id} load block mismatch: external={num_load_blocks} "
                f"expected={expected_load_blocks}"
            )

            load_intent = LoadIntent(
                block_ids=tuple(block_ids[start_block_idx:]),
                plan=plan,
                num_tokens=num_external_tokens,
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

            # Populate block IDs from scheduler_output — single source of
            # truth for the save path (consistent with offloading connector).
            if req.block_ids:
                self._allocated_blocks[req_id] = list(req.block_ids[0])

            self._scheduled_tokens[req_id] += num_tokens

            if save_intent := self._consume_save_intent(req_id):
                save_intents[req_id] = save_intent

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
                save_intents[req_id] = save_intent

        # Track requests with pending saves
        egress_intents = self._build_egress_intents()
        self._pending_saves.update(save_intents.keys())
        self._pending_saves.update(egress_intents.keys())

        logger.debug(
            "[PegaKVConnector] build_connector_meta: %d loads, %d saves, %d egress",
            len(load_intents),
            len(save_intents),
            len(egress_intents),
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
        self._load_plans.pop(req_id, None)
        self._block_hashes.pop(req_id, None)
        self._external_matched_blocks.pop(req_id, None)
        self._block_index_offsets.pop(req_id, None)
        self._allocated_blocks.pop(req_id, None)
        self._computed_blocks.pop(req_id, None)
        self._scheduled_tokens.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        self._pending_saves.discard(req_id)
        self._prefill_push_submitted.discard(req_id)

    def _build_egress_intents(self) -> dict[str, KvEgressIntent]:
        egress_intents: dict[str, KvEgressIntent] = {}
        for req_id, request in list(self._requests.items()):
            if req_id in self._prefill_push_submitted:
                continue

            allocated = self._allocated_blocks.get(req_id, [])
            request = self._requests.get(req_id)
            if request is None:
                continue

            push_tokens = _transferable_prompt_tokens(request, 0)
            expected_blocks = _ceil_div(push_tokens, self._ctx.virtual_block_size)
            if push_tokens <= 0 or expected_blocks == 0:
                self._prefill_push_submitted.add(req_id)
                continue

            ready_blocks = self._prefill_push_ready_blocks(
                req_id,
                expected_blocks,
                push_tokens,
            )
            if ready_blocks < expected_blocks or len(allocated) < expected_blocks:
                continue

            transfer = prefill_push_from_request(request)
            if transfer is None:
                continue

            block_hashes = self._block_hashes.get(req_id, ())
            hash_count = min(len(block_hashes), expected_blocks)

            egress_intents[req_id] = KvEgressIntent(
                request_id=transfer.request_id,
                remote_endpoint=transfer.decode_endpoint,
                remote_instance_id=transfer.decode_instance_id,
                block_ids=tuple(allocated[:expected_blocks]),
                block_hashes=tuple(block_hashes[:hash_count]),
                handle=transfer.handle,
            )
            self._prefill_push_submitted.add(req_id)
            logger.info(
                "[PegaKVConnector] req=%s prefill push intent ready: "
                "transfer_request_id=%s tokens=%d blocks=%d hashes=%d ready_blocks=%d",
                req_id,
                transfer.request_id,
                push_tokens,
                expected_blocks,
                hash_count,
                ready_blocks,
            )

        return egress_intents

    def _prefill_push_ready_blocks(
        self,
        req_id: str,
        total_blocks: int,
        ready_token_limit: int | None = None,
    ) -> int:
        external_blocks = self._external_matched_blocks.get(req_id, 0)
        computed_blocks = self._computed_blocks.get(req_id, 0)
        allocated = self._allocated_blocks.get(req_id, [])
        scheduled = self._scheduled_tokens.get(req_id, 0)
        base_block_idx = self._block_index_offsets.get(req_id, 0)
        token_ready_blocks = _ceil_div(scheduled, self._ctx.virtual_block_size)
        if ready_token_limit is not None:
            token_ready_blocks = min(
                token_ready_blocks,
                _ceil_div(ready_token_limit, self._ctx.virtual_block_size),
            )
        local_ready = min(
            len(allocated),
            token_ready_blocks,
        )
        ready = max(external_blocks, computed_blocks, base_block_idx + local_ready)
        return min(total_blocks, ready)

    def _prepare_load(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> PrepareLoadResult:
        req_id = request.request_id
        if not self._ctx.state_manager.is_available():
            return PrepareLoadResult.done(None)

        prepare_request = self._build_prepare_load_request(request, num_computed_tokens)
        try:
            return self._ctx.engine_client.prepare_load(prepare_request)
        except PegaFlowServiceError as e:
            self._ctx.state_manager.mark_unavailable(str(e))
            logger.warning(
                "[PegaKVConnector] req=%s prepare_load service error: %s",
                req_id,
                e,
            )
            return PrepareLoadResult.done(None)
        except PegaFlowBusinessError as e:
            logger.error(
                "[PegaKVConnector] prepare_load business error: %s, "
                "req_id=%s instance_id=%s blocks=%d",
                e,
                req_id,
                self._ctx.instance_id,
                len(prepare_request.block_hashes),
            )
            raise

    def _build_prepare_load_request(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> "PrepareLoadRequest":
        return prepare_load_request_from_request(
            request,
            self._ctx.instance_id,
            num_computed_tokens,
            self._ctx.virtual_block_size,
        )

    def get_stats(self) -> PegaKVConnectorStats | None:
        return None


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (int(value) + int(divisor) - 1) // int(divisor)


def _prompt_token_count(request: "Request") -> int:
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    try:
        return len(prompt_token_ids)
    except TypeError:
        pass

    num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if num_prompt_tokens is not None:
        return int(num_prompt_tokens)

    return int(getattr(request, "num_tokens", 0) or 0)


def _transferable_prompt_tokens(request: "Request", num_computed_tokens: int) -> int:
    # Match vLLM's P2P remote-prefill behavior: D loads KV up to the token
    # before the final prompt token, then recomputes the final prompt token to
    # produce logits locally.
    return max(0, _prompt_token_count(request) - 1 - int(num_computed_tokens))


def _count_computed_blocks(blocks: "KVCacheBlocks") -> int:
    try:
        first_group = blocks.blocks[0]
    except (AttributeError, IndexError, TypeError):
        return 0
    return sum(block.block_hash is not None for block in first_group)


__all__ = ["SchedulerConnector"]
