"""D-side async HTTP trigger for remote prefill."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
        self._queue: queue.Queue[PrefillHttpTask | None] = queue.Queue()
        self._worker_count = max(1, int(worker_count))
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"pd-prefill-sender-{idx}",
                daemon=True,
            )
            for idx in range(self._worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, task: PrefillHttpTask) -> None:
        self._queue.put(task)

    def close(self) -> None:
        for _ in self._threads:
            self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                post_prefill_request(task)
            finally:
                self._queue.task_done()


def post_prefill_request(task: PrefillHttpTask) -> None:
    url = task.prefill_url.rstrip("/") + "/v1/completions"
    start_ts_ns = time.time_ns()
    body = {
        "model": task.model,
        "prompt": list(task.prompt_token_ids),
        "max_tokens": task.max_tokens,
        "temperature": 0,
        "stream": False,
        "request_id": task.request_id,
        "kv_transfer_params": task.kv_transfer_params,
    }
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
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            response_body = response.read()
            end_ts_ns = time.time_ns()
            logger.info(
                "[PdConnector] D -> P prefill completed req=%s status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
                task.request_id,
                response.status,
                len(response_body),
                (end_ts_ns - start_ts_ns) / 1_000_000,
                end_ts_ns,
            )
    except HTTPError as exc:
        response_body = exc.read()
        logger.error(
            "[PdConnector] D -> P prefill failed req=%s status=%s body=%s",
            task.request_id,
            exc.code,
            response_body[:512],
        )
    except URLError:
        logger.exception(
            "[PdConnector] D -> P prefill connection failed req=%s url=%s",
            task.request_id,
            url,
        )
