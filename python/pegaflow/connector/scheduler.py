"""
Scheduler-side connector logic.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pegaflow import (
    LoadPlan,
    PrepareLoadHandle,
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


@dataclass
class _RequestState:
    request: "Request"
    handled_num_preemptions: int = 0
    external_matched_blocks: int = 0
    allocated_blocks: list[int] = field(default_factory=list)
    scheduled_tokens: int = 0
    next_stored_block_idx: int | None = None
    load_plan: LoadPlan | None = None
    prepare_handle: PrepareLoadHandle | None = None
    pending_load_intent: LoadIntent | None = None
    prefill_push_submitted: bool = False
    pending_save: bool = False
    hold_after_save: bool = False


class SchedulerConnector:
    """Holds scheduler-only state and behaviors."""

    def __init__(self, context: ConnectorContext):
        self._ctx = context
        self._request_states: dict[str, _RequestState] = {}

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        state = self._get_or_create_state(request)

        # request.block_hashes are already at virtual_block_size granularity
        # (vLLM hashes every scheduler_block_size =
        # block_size * dcp_world_size * pcp_world_size).
        # Only the scheduler-visible index is needed here; source-specific
        # matching and P/D polling are handled by prepare_load.
        computed_blocks = num_computed_tokens // self._ctx.virtual_block_size

        result = self._prepare_load(state, request, num_computed_tokens)

        if result.preparing:
            return (None, False)

        plan = result.plan
        if plan is None:
            state.load_plan = None
            state.external_matched_blocks = computed_blocks
            return (0, False)

        state.load_plan = plan
        prepared_blocks = _ceil_div(plan.num_tokens, self._ctx.virtual_block_size)
        state.external_matched_blocks = computed_blocks + prepared_blocks

        return (plan.num_tokens, True)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        state = self._require_state(req_id)
        assert state.request is request, f"req {req_id} request object unexpectedly replaced"

        if state.next_stored_block_idx is None:
            assert not state.allocated_blocks, (
                f"req {req_id} save cursor initialized before block ids were tracked"
            )
            assert state.scheduled_tokens == 0, (
                f"req {req_id} save cursor initialized after scheduling started"
            )
            # The first locally allocated block may be after an external-hit
            # prefix. Track that global block index explicitly.
            state.next_stored_block_idx = state.external_matched_blocks

        if num_external_tokens > 0:
            plan = state.load_plan
            assert plan is not None, f"req {req_id} missing ready load plan"
            block_ids = blocks.get_block_ids()[0]
            num_load_blocks = _ceil_div(num_external_tokens, self._ctx.virtual_block_size)
            assert num_load_blocks <= len(block_ids), (
                f"req {req_id} load blocks exceed allocation: "
                f"load={num_load_blocks} total={len(block_ids)}"
            )
            start_block_idx = len(block_ids) - num_load_blocks
            load_block_ids = tuple(block_ids[start_block_idx:])
            if not load_block_ids:
                raise RuntimeError(f"req {req_id} prepared load has no allocated block ids")

            state.pending_load_intent = LoadIntent(
                block_ids=load_block_ids,
                plan=plan,
                num_tokens=num_external_tokens,
            )
            state.load_plan = None
            logger.debug(
                "[PegaKVConnector] req=%s alloc: total_blocks=%d local_prefix_blocks=%d "
                "load_blocks=%d start_block_idx=%d load_tokens=%d pending_loads=%d",
                req_id,
                len(block_ids),
                start_block_idx,
                len(state.pending_load_intent.block_ids),
                start_block_idx,
                state.pending_load_intent.num_tokens,
                self._pending_load_count(),
            )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> PegaConnectorMetadata:
        save_intents: dict[str, SaveIntent] = {}
        for req_id in scheduler_output.preempted_req_ids or ():
            state = self._require_state(req_id)
            self._sync_preemption_state(state)
        load_intents = self._drain_pending_load_intents()
        scheduled_req_ids: set[str] = set()

        # Process new requests
        for req in scheduler_output.scheduled_new_reqs:
            req_id = req.req_id
            scheduled_req_ids.add(req_id)
            state = self._require_state(req_id)
            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)

            # Populate block IDs from scheduler_output — single source of
            # truth for the save path (consistent with offloading connector).
            if req.block_ids:
                state.allocated_blocks = list(req.block_ids[0])

            state.scheduled_tokens += num_tokens

            if save_intent := self._consume_save_intent(req_id, state):
                save_intents[req_id] = save_intent

        # Process cached (running) requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            scheduled_req_ids.add(req_id)
            state = self._request_states.get(req_id)
            if state is None:
                continue

            num_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            state.scheduled_tokens += num_tokens

            # Append newly allocated blocks
            new_block_ids = cached_reqs.new_block_ids[idx]
            if new_block_ids:
                state.allocated_blocks.extend(new_block_ids[0])

            if save_intent := self._consume_save_intent(req_id, state):
                save_intents[req_id] = save_intent

        # Track requests with pending saves
        egress_intents = self._build_egress_intents(scheduled_req_ids.difference(load_intents))
        for req_id in [*save_intents.keys(), *egress_intents.keys()]:
            state = self._request_states.get(req_id)
            if state is not None:
                state.pending_save = True

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

    def _consume_save_intent(self, req_id: str, state: _RequestState) -> SaveIntent | None:
        """Calculate and return SaveIntent for new blocks that need saving."""
        block_hashes = tuple(state.request.block_hashes)
        if not block_hashes:
            return None
        allocated = state.allocated_blocks
        start_block_idx = state.next_stored_block_idx
        assert start_block_idx is not None, f"req {req_id} save cursor used before initialization"

        # _allocated_blocks tracks request block IDs in global request order.
        # In external-hit cases, the prefix-loaded block IDs are still present at
        # the front, so save intents must slice by global block index rather than
        # rebasing to a local-only view.
        local_saveable = min(
            len(allocated),
            state.scheduled_tokens // self._ctx.virtual_block_size,
        )
        saveable_block_idx = min(
            len(block_hashes),
            state.external_matched_blocks + local_saveable,
        )
        new_blocks = saveable_block_idx - start_block_idx
        if new_blocks <= 0:
            return None

        state.next_stored_block_idx = saveable_block_idx
        save_hashes = block_hashes[start_block_idx : start_block_idx + new_blocks]
        save_block_ids = allocated[start_block_idx : start_block_idx + new_blocks]

        logger.debug(
            "[PegaKVConnector] req=%s save_intent: start_block_idx=%d "
            "base_block_idx=%d saveable_block_idx=%d new_blocks=%d total_hashes=%d",
            req_id,
            start_block_idx,
            state.external_matched_blocks,
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
            state = self._request_states.get(req_id)
            if state is None:
                continue
            state.pending_save = False
            logger.debug("[PegaKVConnector] Request %s save completed", req_id)

            # Clean up if request already finished
            if state.hold_after_save:
                self._cleanup_request(req_id)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],  # noqa: ARG002 - required by vLLM interface
    ) -> tuple[bool, dict | None]:
        req_id = request.request_id
        state = self._request_states.get(req_id)

        # Check if there are pending saves for this request
        if state is not None and state.pending_save:
            state.hold_after_save = True
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
        self._request_states.pop(req_id, None)

    def _build_egress_intents(self, candidate_req_ids: Iterable[str]) -> dict[str, KvEgressIntent]:
        egress_intents: dict[str, KvEgressIntent] = {}
        for req_id in candidate_req_ids:
            state = self._request_states.get(req_id)
            if state is None:
                continue
            if state.prefill_push_submitted:
                continue

            request = state.request
            push_tokens = _transferable_prompt_tokens(request, 0)
            expected_blocks = _ceil_div(push_tokens, self._ctx.virtual_block_size)
            if push_tokens <= 0 or expected_blocks == 0:
                state.prefill_push_submitted = True
                continue

            ready_blocks = self._prefill_push_ready_blocks(
                state,
                expected_blocks,
                push_tokens,
            )
            if ready_blocks < expected_blocks or len(state.allocated_blocks) < expected_blocks:
                continue

            transfer = prefill_push_from_request(request)
            if transfer is None:
                continue

            block_hashes = tuple(request.block_hashes)
            hash_count = min(len(block_hashes), expected_blocks)

            egress_intents[req_id] = KvEgressIntent(
                request_id=transfer.request_id,
                remote_endpoint=transfer.decode_endpoint,
                remote_instance_id=transfer.decode_instance_id,
                block_ids=tuple(state.allocated_blocks[:expected_blocks]),
                block_hashes=block_hashes[:hash_count],
                handle=transfer.handle,
            )
            state.prefill_push_submitted = True
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
        state: _RequestState,
        total_blocks: int,
        ready_token_limit: int | None = None,
    ) -> int:
        token_ready_blocks = _ceil_div(state.scheduled_tokens, self._ctx.virtual_block_size)
        if ready_token_limit is not None:
            token_ready_blocks = min(
                token_ready_blocks,
                _ceil_div(ready_token_limit, self._ctx.virtual_block_size),
            )
        local_ready = min(
            len(state.allocated_blocks),
            token_ready_blocks,
        )
        ready = state.external_matched_blocks + local_ready
        return min(total_blocks, ready)

    def _prepare_load(
        self,
        state: _RequestState,
        request: "Request",
        num_computed_tokens: int,
    ) -> PrepareLoadResult:
        prepare_start = time.perf_counter()
        req_id = request.request_id
        if not self._ctx.state_manager.is_available():
            state.prepare_handle = None
            return PrepareLoadResult.done(None)

        if state.prepare_handle is None:
            prepare_request = prepare_load_request_from_request(
                request,
                self._ctx.instance_id,
                num_computed_tokens,
                self._ctx.virtual_block_size,
            )
            try:
                state.prepare_handle = self._ctx.client.begin_prepare_load(prepare_request)
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

        # TODO: add a prepare-handle timeout. If pegaflow-server dies after
        # accepting PrepareLoad, this shm can stay pending forever; the
        # scheduler should eventually fall back locally and mark service
        # unavailable.
        result = state.prepare_handle.result()

        if not result.preparing:
            plan = result.plan
            state.prepare_handle = None
            elapsed_us = (time.perf_counter() - prepare_start) * 1e6
            prepared_blocks = (
                _ceil_div(plan.num_tokens, self._ctx.virtual_block_size) if plan is not None else 0
            )
            logger.info(
                "[PegaKVConnector] req=%s prepare_load ready: result=%s "
                "external_tokens=%d external_blocks=%d prepare_us=%.0f",
                req_id,
                "prepared" if plan is not None else "no_plan",
                plan.num_tokens if plan is not None else 0,
                prepared_blocks,
                elapsed_us,
            )

        return result

    def get_stats(self) -> PegaKVConnectorStats | None:
        return None

    def _get_or_create_state(self, request: "Request") -> _RequestState:
        state = self._request_states.get(request.request_id)
        if state is None:
            state = _RequestState(
                request=request,
                handled_num_preemptions=request.num_preemptions,
            )
            self._request_states[request.request_id] = state
        else:
            assert state.request is request, (
                f"req {request.request_id} request object unexpectedly replaced"
            )
            self._sync_preemption_state(state)
        return state

    def _require_state(self, req_id: str) -> _RequestState:
        state = self._request_states.get(req_id)
        if state is None:
            raise RuntimeError(f"req {req_id} missing scheduler connector state")
        return state

    def _sync_preemption_state(self, state: _RequestState) -> None:
        current_num_preemptions = state.request.num_preemptions
        handled_num_preemptions = state.handled_num_preemptions
        assert current_num_preemptions >= handled_num_preemptions, (
            f"req {state.request.request_id} preemption counter went backwards: "
            f"{current_num_preemptions} < {handled_num_preemptions}"
        )
        if current_num_preemptions == handled_num_preemptions:
            return

        req_id = state.request.request_id
        logger.info(
            "[PegaKVConnector] req=%s resetting scheduling state after preemption: %d -> %d",
            req_id,
            handled_num_preemptions,
            current_num_preemptions,
        )
        state.handled_num_preemptions = current_num_preemptions
        state.external_matched_blocks = 0
        state.allocated_blocks.clear()
        state.scheduled_tokens = 0
        state.next_stored_block_idx = None
        state.load_plan = None
        state.pending_load_intent = None

    def _drain_pending_load_intents(self) -> dict[str, LoadIntent]:
        load_intents: dict[str, LoadIntent] = {}
        for req_id, state in self._request_states.items():
            if state.pending_load_intent is None:
                continue
            load_intents[req_id] = state.pending_load_intent
            state.pending_load_intent = None
        return load_intents

    def _pending_load_count(self) -> int:
        return sum(state.pending_load_intent is not None for state in self._request_states.values())


def _ceil_div(value: int, divisor: int) -> int:
    if value <= 0:
        return 0
    return (int(value) + int(divisor) - 1) // int(divisor)


def _transferable_prompt_tokens(request: "Request", num_computed_tokens: int) -> int:
    # Match vLLM's P2P remote-prefill behavior: D loads KV up to the token
    # before the final prompt token, then recomputes the final prompt token to
    # produce logits locally.
    return max(0, request.num_prompt_tokens - 1 - int(num_computed_tokens))


__all__ = ["SchedulerConnector"]
