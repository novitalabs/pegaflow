#!/usr/bin/env python3
"""Minimal OpenAI-compatible P/D push proxy for vLLM benchmark runs."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


class ProxyConfig:
    def __init__(self) -> None:
        self.model = "pd-test"
        self.p_url = "http://127.0.0.1:18100/v1/completions"
        self.d_url = "http://127.0.0.1:18200/v1/completions"
        self.d_pegaflow_addr = "127.0.0.1:51155"
        self.dst_instance_id = "pd-d0"
        self.p_delay_s = 0.02


config = ProxyConfig()
app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    app.state.client = httpx.AsyncClient(timeout=None)


@app.on_event("shutdown")
async def shutdown() -> None:
    await app.state.client.aclose()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": config.model,
                "object": "model",
                "created": 0,
                "owned_by": "pegaflow",
            }
        ],
    }


def _pd_payloads(payload: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    request_id = f"pd-bench-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    d_payload = dict(payload)
    p_payload = dict(payload)
    d_payload["kv_transfer_params"] = {
        "pegaflow_pd_push": True,
        "pd_request_id": request_id,
        "dst_instance_id": config.dst_instance_id,
    }
    p_payload["stream"] = False
    p_payload.pop("stream_options", None)
    p_payload["kv_transfer_params"] = {
        "role": "source",
        "pegaflow_pd_push": True,
        "pd_request_id": request_id,
        "d_pegaflow_addr": config.d_pegaflow_addr,
        "dst_instance_id": config.dst_instance_id,
    }
    return request_id, d_payload, p_payload


async def _run_p_push(client: httpx.AsyncClient, p_payload: dict[str, Any]) -> None:
    if config.p_delay_s > 0:
        await asyncio.sleep(config.p_delay_s)
    response = await client.post(config.p_url, json=p_payload)
    if response.status_code >= 400:
        body = response.text[:1000]
        raise RuntimeError(f"P request failed: HTTP {response.status_code}: {body}")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    payload = await request.json()
    _, d_payload, p_payload = _pd_payloads(payload)
    client: httpx.AsyncClient = app.state.client

    if not payload.get("stream"):
        d_task = asyncio.create_task(client.post(config.d_url, json=d_payload))
        p_task = asyncio.create_task(_run_p_push(client, p_payload))
        d_response, p_result = await asyncio.gather(d_task, p_task, return_exceptions=True)
        if isinstance(p_result, Exception):
            return JSONResponse({"error": str(p_result)}, status_code=502)
        if isinstance(d_response, Exception):
            return JSONResponse({"error": str(d_response)}, status_code=502)
        return Response(
            content=d_response.content,
            status_code=d_response.status_code,
            media_type=d_response.headers.get("content-type", "application/json"),
        )

    async def stream_d() -> Any:
        p_task = asyncio.create_task(_run_p_push(client, p_payload))
        try:
            async with client.stream("POST", config.d_url, json=d_payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")[:1000]
                    raise RuntimeError(f"D request failed: HTTP {response.status_code}: {body}")
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
            await p_task
        except Exception as exc:
            p_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await p_task
            error = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            yield f"data: {error}\n\n".encode()
            yield b"data: [DONE]\n\n"

    return StreamingResponse(stream_d(), media_type="text/event-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--served-model-name", default="pd-test")
    parser.add_argument("--p-url", default="http://127.0.0.1:18100/v1/completions")
    parser.add_argument("--d-url", default="http://127.0.0.1:18200/v1/completions")
    parser.add_argument("--d-pegaflow-addr", default="127.0.0.1:51155")
    parser.add_argument("--dst-instance-id", default="pd-d0")
    parser.add_argument("--p-delay-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config.model = args.served_model_name
    config.p_url = args.p_url
    config.d_url = args.d_url
    config.d_pegaflow_addr = args.d_pegaflow_addr
    config.dst_instance_id = args.dst_instance_id
    config.p_delay_s = args.p_delay_ms / 1000.0
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
