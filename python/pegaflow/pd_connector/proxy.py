"""Small async P/D proxy for exercising PdConnector with two local vLLM servers.

The proxy accepts OpenAI-compatible completion requests, injects the
``kv_transfer_params`` expected by ``PdConnector``, and sends the request only
to D. D allocates KV blocks, then uses the P hint from those params to trigger
the prefill side. D begins decoding after its connector observes the RDMA IMM
notification from P.
"""

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, AsyncIterator

import httpx

from pegaflow.pd_connector.kv_params import ConsumerKvParams

logger = logging.getLogger("pegaflow.pd_proxy")

SUPPORTED_PATHS = {"/v1/completions", "/v1/chat/completions"}


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    timeout_s: float
    prefill_max_tokens: int


@dataclass(frozen=True)
class PdProxyRequest:
    request_id: str
    decode_body: dict[str, Any]


def build_pd_proxy_request(
    body: dict[str, Any],
    config: ProxyConfig,
    request_id: str | None = None,
    proxy_start_ts_ns: int = 0,
) -> PdProxyRequest:
    req_id = request_id or str(body.get("request_id") or f"pd-{uuid.uuid4().hex}")
    prefill_req_id = f"{req_id}-p"
    decode_req_id = f"{req_id}-d"

    decode_body = {
        **body,
        "request_id": decode_req_id,
    }
    decode_body["kv_transfer_params"] = ConsumerKvParams(
        prefill_url=config.prefill_url,
        remote_request_id=prefill_req_id,
        done_request_id=decode_req_id,
        prefill_max_tokens=config.prefill_max_tokens,
        proxy_start_ts_ns=proxy_start_ts_ns,
    ).to_dict()

    return PdProxyRequest(
        request_id=req_id,
        decode_body=decode_body,
    )


class PdProxy:
    def __init__(self, config: ProxyConfig, client: httpx.AsyncClient) -> None:
        self.config = config
        self.client = client

    async def handle_openai_request(
        self,
        path: str,
        body: dict[str, Any],
    ) -> Any:
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            from fastapi.responses import Response

            return Response(
                content=json.dumps(payload).encode(),
                status_code=HTTPStatus.NOT_FOUND,
                media_type="application/json",
            )

        start_ts_ns = time.time_ns()
        req = build_pd_proxy_request(body, self.config, proxy_start_ts_ns=start_ts_ns)
        if body.get("stream"):
            req.decode_body["stream"] = True
            logger.info(
                "[PdProxy] request=%s accepted streaming path=%s ts_ns=%d",
                req.request_id,
                path,
                start_ts_ns,
            )
            return await self._open_decode_stream(path, req, start_ts_ns)

        logger.info(
            "[PdProxy] request=%s accepted path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )
        return await self._post_decode(path, req, start_ts_ns)

    async def _post_decode(
        self,
        path: str,
        req: PdProxyRequest,
        start_ts_ns: int,
    ) -> Any:
        from fastapi.responses import Response

        payload = _compact_json_bytes(req.decode_body)
        url = self.config.decode_url + path
        logger.info(
            "[PdProxy] request=%s -> D url=%s body_bytes=%d kv=%s",
            req.request_id,
            url,
            len(payload),
            req.decode_body.get("kv_transfer_params"),
        )
        response = await self.client.post(
            url,
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        response_body = response.content
        end_ts_ns = time.time_ns()
        logger.info(
            "[PdProxy] request=%s D completed status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
            req.request_id,
            response.status_code,
            len(response_body),
            (end_ts_ns - start_ts_ns) / 1_000_000,
            end_ts_ns,
        )
        return Response(
            content=response_body,
            status_code=response.status_code,
            media_type=response.headers.get("Content-Type", "application/json"),
        )

    async def _open_decode_stream(
        self,
        path: str,
        req: PdProxyRequest,
        start_ts_ns: int,
    ) -> Any:
        from fastapi.responses import StreamingResponse

        payload = _compact_json_bytes(req.decode_body)
        url = self.config.decode_url + path
        logger.info(
            "[PdProxy] request=%s -> D stream url=%s body_bytes=%d kv=%s",
            req.request_id,
            url,
            len(payload),
            req.decode_body.get("kv_transfer_params"),
        )

        async def generate() -> AsyncIterator[bytes]:
            first_chunk = True
            async with self.client.stream(
                "POST",
                url,
                content=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                header_ts_ns = time.time_ns()
                logger.info(
                    "[PdProxy] request=%s D stream headers status=%s latency_ms=%.3f ts_ns=%d",
                    req.request_id,
                    response.status_code,
                    (header_ts_ns - start_ts_ns) / 1_000_000,
                    header_ts_ns,
                )
                async for chunk in response.aiter_bytes():
                    if first_chunk:
                        first_chunk = False
                        now_ns = time.time_ns()
                        logger.info(
                            "[PdProxy] request=%s first stream chunk bytes=%d ttft_ms=%.3f ts_ns=%d",
                            req.request_id,
                            len(chunk),
                            (now_ns - start_ts_ns) / 1_000_000,
                            now_ns,
                        )
                    yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    def close(self) -> None:
        return None


def create_app(config: ProxyConfig):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.decode_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_s),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
        yield
        await app.state.decode_client.aclose()

    app = FastAPI(lifespan=lifespan)

    async def handle(path: str, request: Request) -> Response:
        try:
            body = await request.json()
            proxy = PdProxy(config, request.app.state.decode_client)
            return await proxy.handle_openai_request(path, body)
        except httpx.HTTPStatusError as exc:
            logger.exception("[PdProxy] D returned HTTP error")
            return Response(
                content=await exc.response.aread(),
                status_code=exc.response.status_code,
                media_type=exc.response.headers.get("Content-Type", "application/json"),
            )
        except Exception as exc:
            logger.exception("[PdProxy] request handling failed")
            return Response(
                content=json.dumps({"error": str(exc)}).encode(),
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                media_type="application/json",
            )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await handle("/v1/completions", request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await handle("/v1/chat/completions", request)

    @app.get("/health")
    @app.get("/healthcheck")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _compact_json_bytes(body: dict[str, Any]) -> bytes:
    return json.dumps(body, separators=(",", ":")).encode()


def iter_http_stream_chunks(response) -> Any:
    pending = bytearray()
    while True:
        line = response.readline()
        if not line:
            if pending:
                yield bytes(pending)
            return
        pending.extend(line)
        if line in {b"\n", b"\r\n"}:
            yield bytes(pending)
            pending.clear()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PegaFlow P/D local proxy")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8100)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8001")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8002")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    parser.add_argument("--log-file")
    args = parser.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
    )

    config = ProxyConfig(
        prefill_url=args.prefill_url.rstrip("/"),
        decode_url=args.decode_url.rstrip("/"),
        timeout_s=args.timeout_s,
        prefill_max_tokens=args.prefill_max_tokens,
    )
    logger.info(
        "[PdProxy] listening http://%s:%d prefill=%s decode=%s",
        args.listen_host,
        args.listen_port,
        config.prefill_url,
        config.decode_url,
    )
    import uvicorn

    uvicorn.run(create_app(config), host=args.listen_host, port=args.listen_port)


if __name__ == "__main__":
    main()
