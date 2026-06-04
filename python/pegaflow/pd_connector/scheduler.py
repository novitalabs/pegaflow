"""Scheduler-side logic for the experimental P/D connector."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.kv_params import (
    is_consumer,
    is_producer,
    parse_consumer,
    parse_producer,
)
from pegaflow.pd_connector.metadata import (
    PdConnectorMetadata,
    PushReqMeta,
    WaitReqMeta,
    normalize_block_ids,
)

logger = get_connector_logger()


def _request_id(request: Any) -> str:
    return str(request.request_id)


def _kv_params(request: Any) -> dict[str, Any]:
    return getattr(request, "kv_transfer_params", None) or {}


def _num_prompt_tokens(request: Any) -> int:
    return int(request.num_prompt_tokens)


def _prompt_token_ids(request: Any) -> tuple[int, ...]:
    return tuple(int(token_id) for token_id in request.prompt_token_ids)


def _request_was_aborted(request: Any) -> bool:
    status = getattr(request, "status", None)
    if status is not None and "ABORT" in str(status).upper():
        return True
    finish_reason = getattr(request, "finish_reason", None)
    if finish_reason is not None and str(finish_reason).upper() == "ABORT":
        return True
    get_finished_reason = getattr(request, "get_finished_reason", None)
    if get_finished_reason is None:
        return False
    reason = get_finished_reason()
    return reason is not None and str(reason).upper() == "ABORT"


class PdSchedulerConnector:
    def __init__(self, vllm_config: Any, **_kwargs: Any) -> None:
        self.vllm_config = vllm_config
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""

        # per-step buffers, cleared after build_connector_meta
        self._reqs_to_wait: dict[str, WaitReqMeta] = {}
        self._reqs_to_push: dict[str, PushReqMeta] = {}
        self._reqs_to_release: set[str] = set()

        # D-side cross-step state
        self._active_waits: dict[str, WaitReqMeta] = {}
        self._completed_waits: set[str] = set()
        self._matched_ts_ns: dict[str, int] = {}

        # P-side cross-step state
        self._active_pushes: dict[str, PushReqMeta] = {}

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        req_id = _request_id(request)
        params = _kv_params(request)

        consumer = parse_consumer(params)
        if consumer is not None:
            matched_ts_ns = time.time_ns()
            self._matched_ts_ns[req_id] = matched_ts_ns
            count = _num_prompt_tokens(request) - num_computed_tokens
            logger.info(
                "[PdConnector] D matched_tokens req=%s prompt=%d computed=%d count=%d prefill_url=%s remote_req=%s done_req=%s proxy_to_matched_ms=%.3f ts_ns=%d",
                req_id,
                _num_prompt_tokens(request),
                num_computed_tokens,
                count,
                consumer.prefill_url,
                consumer.remote_request_id or req_id,
                consumer.done_request_id or req_id,
                _elapsed_ms(consumer.proxy_start_ts_ns, matched_ts_ns),
                matched_ts_ns,
            )
            return (count, True) if count > 0 else (0, False)

        producer = parse_producer(params)
        if producer is not None:
            logger.info(
                "[PdConnector] P matched_tokens req=%s computed=%d target_engine=%s target_req=%s handshakes=%d",
                req_id,
                num_computed_tokens,
                producer.target_engine_id,
                producer.target_request_id,
                len(producer.handshakes),
            )
            return 0, False

        logger.info(
            "[PdConnector] matched_tokens req=%s computed=%d (no kv_transfer_params)",
            req_id,
            num_computed_tokens,
        )
        return 0, False

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        params = _kv_params(request)
        req_id = _request_id(request)
        local_block_ids = normalize_block_ids(blocks)

        consumer = parse_consumer(params)
        if consumer is not None:
            scheduler_wait_ts_ns = time.time_ns()
            assert req_id not in self._active_waits, (
                f"update_state_after_alloc called while req={req_id} is still waiting for RDMA transfer"
            )
            if req_id in self._completed_waits:
                return
            matched_ts_ns = self._matched_ts_ns.get(req_id, 0)
            wait_req = WaitReqMeta(
                local_block_ids=local_block_ids,
                remote_request_id=consumer.remote_request_id or req_id,
                done_request_id=consumer.done_request_id or req_id,
                prompt_token_ids=_prompt_token_ids(request),
                prefill_url=consumer.prefill_url,
                model=str(getattr(request, "model", "") or ""),
                prefill_max_tokens=consumer.prefill_max_tokens,
                proxy_start_ts_ns=consumer.proxy_start_ts_ns,
                matched_ts_ns=matched_ts_ns,
                scheduler_wait_ts_ns=scheduler_wait_ts_ns,
            )
            self._active_waits[req_id] = wait_req
            self._reqs_to_wait[req_id] = wait_req
            logger.info(
                "[PdConnector] scheduler wait req=%s blocks=%d prefill_url=%s proxy_to_wait_ms=%.3f matched_to_wait_ms=%.3f ts_ns=%d",
                req_id,
                _count(local_block_ids),
                wait_req.prefill_url,
                _elapsed_ms(consumer.proxy_start_ts_ns, scheduler_wait_ts_ns),
                _elapsed_ms(matched_ts_ns, scheduler_wait_ts_ns),
                scheduler_wait_ts_ns,
            )
            return

        producer = parse_producer(params)
        if producer is not None:
            push_req = PushReqMeta(
                local_block_ids=local_block_ids,
                target_request_id=producer.target_request_id or req_id,
                handshakes=producer.handshakes,
            )
            self._reqs_to_push[req_id] = push_req
            self._active_pushes[req_id] = push_req
            logger.info(
                "[PdConnector] scheduler push req=%s blocks=%d target_req=%s handshakes=%d",
                req_id,
                _count(local_block_ids),
                push_req.target_request_id,
                len(producer.handshakes),
            )

    def build_connector_meta(self, scheduler_output: Any) -> PdConnectorMetadata:
        self._add_cached_producer_chunks(scheduler_output)
        meta = PdConnectorMetadata(
            reqs_to_wait=self._reqs_to_wait,
            reqs_to_push=self._reqs_to_push,
            reqs_to_release=self._reqs_to_release,
        )
        self._reqs_to_wait = {}
        self._reqs_to_push = {}
        self._reqs_to_release = set()
        return meta

    def update_connector_output(self, connector_output: Any) -> None:
        for req_id in connector_output.finished_sending or ():
            self._active_pushes.pop(req_id, None)
            self._reqs_to_release.add(req_id)
            logger.info("[PdConnector] scheduler finished sending req=%s", req_id)

        for req_id in connector_output.finished_recving or ():
            finished_ts_ns = time.time_ns()
            wait_req = self._active_waits.get(req_id)
            self._active_waits.pop(req_id, None)
            self._completed_waits.add(req_id)
            matched_ts_ns = self._matched_ts_ns.pop(req_id, None)
            logger.info(
                "[PdConnector] scheduler finished recving req=%s proxy_to_finished_ms=%.3f matched_to_finished_ms=%.3f wait_to_finished_ms=%.3f ts_ns=%d",
                req_id,
                _elapsed_ms(wait_req.proxy_start_ts_ns, finished_ts_ns) if wait_req else -1.0,
                _elapsed_ms(matched_ts_ns or 0, finished_ts_ns),
                _elapsed_ms(wait_req.scheduler_wait_ts_ns, finished_ts_ns) if wait_req else -1.0,
                finished_ts_ns,
            )

    def request_finished(
        self,
        request: Any,
        block_ids: Any,
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = _request_id(request)
        params = _kv_params(request)
        is_prod = is_producer(params)

        if is_consumer(params):
            self._reqs_to_release.add(req_id)
            self._active_waits.pop(req_id, None)
            self._completed_waits.discard(req_id)
            self._matched_ts_ns.pop(req_id, None)

        if is_prod and _request_was_aborted(request):
            self._reqs_to_release.add(req_id)
            self._active_pushes.pop(req_id, None)
            return False, None

        if is_prod and _has_blocks(block_ids):
            self._active_pushes.setdefault(
                req_id,
                PushReqMeta(
                    local_block_ids=normalize_block_ids(block_ids),
                    target_request_id=req_id,
                ),
            )
            return True, None
        if is_prod:
            self._reqs_to_release.add(req_id)
            self._active_pushes.pop(req_id, None)
        return False, None

    def _add_cached_producer_chunks(self, scheduler_output: Any) -> None:
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached is None:
            return
        req_ids = tuple(str(req_id) for req_id in (getattr(cached, "req_ids", None) or ()))
        new_block_ids = tuple(getattr(cached, "new_block_ids", None) or ())
        for req_id, blocks in zip(req_ids, new_block_ids, strict=False):
            if req_id in self._reqs_to_push:
                continue
            push_req = self._active_pushes.get(req_id)
            if push_req is None:
                continue
            local_block_ids = normalize_block_ids(blocks)
            if not _has_blocks(local_block_ids):
                continue
            self._reqs_to_push[req_id] = replace(push_req, local_block_ids=local_block_ids)

    def shutdown(self) -> None:
        pass


def _count(block_ids: tuple[list[int], ...]) -> int:
    return sum(len(group) for group in block_ids)


def _has_blocks(block_ids: Any) -> bool:
    return any(len(group) > 0 for group in normalize_block_ids(block_ids))


def _elapsed_ms(start_ts_ns: int, end_ts_ns: int) -> float:
    if start_ts_ns <= 0 or end_ts_ns <= 0 or end_ts_ns < start_ts_ns:
        return -1.0
    return (end_ts_ns - start_ts_ns) / 1_000_000
