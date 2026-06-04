"""D-side async HTTP trigger for remote prefill."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from pegaflow.logging_utils import get_connector_logger

logger = get_connector_logger()


@dataclass(frozen=True)
class PrefillHttpTask:
    request_id: str
    prefill_url: str
    model: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    kv_transfer_params: dict[str, Any]


class AsyncPrefillSender:
    def __init__(self, worker_count: int = 16) -> None:
        self._worker_count = max(1, int(worker_count))
        self._closed = False
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="pd-prefill-sender",
            daemon=True,
        )
        self._thread.start()
        self._loop_ready.wait()

    def submit(self, task: PrefillHttpTask) -> None:
        if self._closed:
            return
        asyncio.run_coroutine_threadsafe(self._submit(task), self._loop)

    def cancel(self, request_id: str) -> None:
        if self._closed:
            return
        asyncio.run_coroutine_threadsafe(self._cancel(request_id), self._loop)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        future.result(timeout=30)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        self._semaphore = asyncio.Semaphore(self._worker_count)
        self._tasks: set[asyncio.Task[None]] = set()
        self._tasks_by_req: dict[str, asyncio.Task[None]] = {}
        self._loop_ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._client.aclose())
        self._loop.close()

    async def _submit(self, task: PrefillHttpTask) -> None:
        old_task = self._tasks_by_req.pop(task.request_id, None)
        if old_task is not None:
            old_task.cancel()
        async_task = asyncio.create_task(self._run_task(task))
        self._tasks.add(async_task)
        self._tasks_by_req[task.request_id] = async_task

        def _discard(done_task: asyncio.Task[None]) -> None:
            self._tasks.discard(done_task)
            if self._tasks_by_req.get(task.request_id) is done_task:
                self._tasks_by_req.pop(task.request_id, None)

        async_task.add_done_callback(_discard)

    async def _cancel(self, request_id: str) -> None:
        task = self._tasks_by_req.pop(request_id, None)
        if task is not None:
            task.cancel()

    async def _shutdown(self) -> None:
        if self._tasks:
            for task in tuple(self._tasks):
                task.cancel()
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks_by_req.clear()

    async def _run_task(self, task: PrefillHttpTask) -> None:
        try:
            async with self._semaphore:
                await post_prefill_request_async(task, self._client)
        except asyncio.CancelledError:
            logger.info("[PdConnector] D -> P prefill cancelled req=%s", task.request_id)
            raise


def _prefill_request_body(task: PrefillHttpTask) -> dict[str, Any]:
    return {
        "model": task.model,
        "prompt": list(task.prompt_token_ids),
        "max_tokens": task.max_tokens,
        "temperature": 0,
        "stream": False,
        "request_id": task.request_id,
        "kv_transfer_params": task.kv_transfer_params,
    }


async def post_prefill_request_async(
    task: PrefillHttpTask,
    client: httpx.AsyncClient | None = None,
) -> None:
    url = task.prefill_url.rstrip("/") + "/v1/completions"
    start_ts_ns = time.time_ns()
    body = _prefill_request_body(task)
    payload = json.dumps(body, separators=(",", ":")).encode()
    logger.info(
        "[PdConnector] D -> P prefill request req=%s url=%s tokens=%d payload_bytes=%d target_req=%s ts_ns=%d",
        task.request_id,
        url,
        len(task.prompt_token_ids),
        len(payload),
        task.kv_transfer_params.get("target_request_id"),
        start_ts_ns,
    )
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
    try:
        response = await client.post(
            url,
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        response_body = response.content
        end_ts_ns = time.time_ns()
        if response.status_code >= 400:
            logger.error(
                "[PdConnector] D -> P prefill failed req=%s status=%s body=%s",
                task.request_id,
                response.status_code,
                response_body[:512],
            )
            return
        logger.info(
            "[PdConnector] D -> P prefill completed req=%s status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
            task.request_id,
            response.status_code,
            len(response_body),
            (end_ts_ns - start_ts_ns) / 1_000_000,
            end_ts_ns,
        )
    except httpx.RequestError:
        logger.exception(
            "[PdConnector] D -> P prefill connection failed req=%s url=%s",
            task.request_id,
            url,
        )
    finally:
        if owns_client:
            await client.aclose()


def post_prefill_request(task: PrefillHttpTask) -> None:
    asyncio.run(post_prefill_request_async(task))
