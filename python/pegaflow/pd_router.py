"""PegaFlow P/D Disaggregation Router.

Simple router that coordinates prefill (P) and decode (D) nodes.
Flow:
1. Receive request
2. Send to P node (max_tokens=1)
3. Forward to D node (P response means KV is ready)
"""

import logging
import time
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pd_router")

# Global state
_prefill_clients: list[httpx.AsyncClient] = []
_decode_clients: list[httpx.AsyncClient] = []
_p_index = 0
_d_index = 0

app = FastAPI()


@app.on_event("shutdown")
async def shutdown():
    for client in _prefill_clients:
        await client.aclose()
    for client in _decode_clients:
        await client.aclose()


def configure(prefill_endpoints: list[str], decode_endpoints: list[str]):
    """Configure P and D endpoints with persistent clients."""
    global _prefill_clients, _decode_clients
    _prefill_clients = [
        httpx.AsyncClient(timeout=None, base_url=url) for url in prefill_endpoints
    ]
    _decode_clients = [
        httpx.AsyncClient(timeout=None, base_url=url) for url in decode_endpoints
    ]


def _get_next_p() -> httpx.AsyncClient:
    global _p_index
    client = _prefill_clients[_p_index % len(_prefill_clients)]
    _p_index += 1
    return client


def _get_next_d() -> httpx.AsyncClient:
    global _d_index
    client = _decode_clients[_d_index % len(_decode_clients)]
    _d_index += 1
    return client


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion request with P/D disaggregation."""
    return await _handle_completion(request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    """Handle completion request with P/D disaggregation."""
    return await _handle_completion(request, "/v1/completions")


async def _handle_completion(request: Request, api_path: str):
    """Common handler for both chat/completions and completions APIs."""
    body = await request.json()
    arrive_time = time.time()

    # Use existing request_id or generate new one
    req_id = body.get("request_id") or str(uuid.uuid4())

    logger.info(f"request arrived: req={req_id}")

    # Save original values to restore for D request
    org_max_tokens = body.get("max_tokens")
    org_stream = body.get("stream", False)
    stream_options = body.pop("stream_options", None)

    # Prepare P request (max_tokens=1, non-streaming)
    p_body = dict(body)
    p_body["max_tokens"] = 1
    p_body["stream"] = False
    p_body["request_id"] = req_id
    # Ensure min_tokens <= max_tokens to avoid 400 from P node
    if "min_tokens" in p_body:
        p_body["min_tokens"] = min(p_body.get("min_tokens") or 0, 1)
    else:
        p_body["min_tokens"] = 0

    headers = {"X-Request-Id": req_id}
    p_client = _get_next_p()

    # Send to P node
    p_response = await p_client.post(api_path, json=p_body, headers=headers)
    p_result = p_response.json()

    if p_response.status_code != 200:
        logger.error(f"P error: req={req_id} status={p_response.status_code} body={p_result}")
        return p_result

    prefill_latency = (time.time() - arrive_time) * 1000
    logger.info(f"prefill done: req={req_id} latency={prefill_latency:.1f}ms")

    # Prepare D request (restore original settings)
    d_body = dict(body)
    d_body["max_tokens"] = org_max_tokens
    d_body["stream"] = org_stream
    d_body["request_id"] = req_id
    if stream_options is not None:
        d_body["stream_options"] = stream_options

    d_headers = {"X-Request-Id": req_id}
    d_client = _get_next_d()

    if org_stream:
        # Streaming response
        async def stream_from_d():
            try:
                async with d_client.stream(
                    "POST", api_path, json=d_body, headers=d_headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                total = (time.time() - arrive_time) * 1000
                logger.info(f"done (stream): req={req_id} total={total:.1f}ms")

        return StreamingResponse(stream_from_d(), media_type="text/event-stream")
    else:
        # Non-streaming response
        d_response = await d_client.post(api_path, json=d_body, headers=d_headers)
        total = (time.time() - arrive_time) * 1000
        logger.info(f"done: req={req_id} total={total:.1f}ms")
        return d_response.json()


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="PegaFlow P/D Router")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--prefill", required=True, nargs="+", help="Prefill endpoints")
    parser.add_argument("--decode", required=True, nargs="+", help="Decode endpoints")
    args = parser.parse_args()

    configure(args.prefill, args.decode)

    logger.info(f"Starting on {args.host}:{args.port}")
    logger.info(f"Prefill nodes: {args.prefill}")
    logger.info(f"Decode nodes: {args.decode}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
