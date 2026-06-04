"""Small P/D proxy for exercising PdConnector with two local vLLM servers.

The proxy accepts OpenAI-compatible completion requests, injects the
``kv_transfer_params`` expected by ``PdConnector``, and sends the request only
to D. D allocates KV blocks, then uses the P hint from those params to trigger
the prefill side. D begins decoding after its connector observes the RDMA IMM
notification from P.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pegaflow.pd_connector.kv_params import ConsumerKvParams

logger = logging.getLogger("pegaflow.pd_proxy")


SUPPORTED_PATHS = {"/v1/completions", "/v1/chat/completions"}


@dataclass(frozen=True)
class PdEndpoint:
    url: str
    instance_id: str = ""


@dataclass(frozen=True)
class PdRoute:
    prefill: PdEndpoint
    decode: PdEndpoint

    def as_tuple(self) -> tuple[str, str]:
        return (self.prefill.url, self.decode.url)


class PdRouter(Protocol):
    def select(self) -> PdRoute:
        ...


class RoundRobinPairRouter:
    def __init__(
        self,
        *,
        prefill_endpoints: tuple[PdEndpoint, ...],
        decode_endpoints: tuple[PdEndpoint, ...],
    ) -> None:
        if not prefill_endpoints:
            raise ValueError("PdProxy requires at least one prefill endpoint")
        if not decode_endpoints:
            raise ValueError("PdProxy requires at least one decode endpoint")
        self._prefill_endpoints = tuple(prefill_endpoints)
        self._decode_endpoints = tuple(decode_endpoints)
        self._next_index = 0
        self._lock = threading.Lock()

    def select(self) -> PdRoute:
        with self._lock:
            index = self._next_index
            self._next_index += 1
        return PdRoute(
            prefill=self._prefill_endpoints[index % len(self._prefill_endpoints)],
            decode=self._decode_endpoints[index % len(self._decode_endpoints)],
        )


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    timeout_s: float
    prefill_max_tokens: int
    router: PdRouter | None = None


@dataclass(frozen=True)
class PdProxyRequest:
    request_id: str
    decode_url: str
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
    route = config.router.select() if config.router is not None else None
    prefill_url = (route.prefill.url if route is not None else config.prefill_url).rstrip("/")
    decode_url = (route.decode.url if route is not None else config.decode_url).rstrip("/")

    decode_body = {
        **body,
        "request_id": decode_req_id,
    }
    decode_body["kv_transfer_params"] = ConsumerKvParams(
        prefill_url=prefill_url,
        remote_request_id=prefill_req_id,
        done_request_id=decode_req_id,
        prefill_max_tokens=config.prefill_max_tokens,
        proxy_start_ts_ns=proxy_start_ts_ns,
    ).to_dict()

    return PdProxyRequest(
        request_id=req_id,
        decode_url=decode_url,
        decode_body=decode_body,
    )


class PdProxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config

    def handle_openai_request(self, path: str, body: dict[str, Any]) -> tuple[int, bytes, str]:
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return HTTPStatus.NOT_FOUND, json.dumps(payload).encode(), "application/json"

        start_ts_ns = time.time_ns()
        req = build_pd_proxy_request(body, self.config, proxy_start_ts_ns=start_ts_ns)
        logger.info(
            "[PdProxy] request=%s accepted path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )

        decode_status, decode_body, decode_content_type = _post_json(
            req.decode_url + path,
            req.decode_body,
            self.config.timeout_s,
            req.request_id,
            "D",
        )
        logger.info(
            "[PdProxy] request=%s D completed status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
            req.request_id,
            decode_status,
            len(decode_body),
            (end_ts_ns := time.time_ns() - start_ts_ns) / 1_000_000,
            start_ts_ns + end_ts_ns,
        )
        return decode_status, decode_body, decode_content_type

    def open_openai_stream(self, path: str, body: dict[str, Any]):
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return (
                HTTPStatus.NOT_FOUND,
                "application/json",
                None,
                "",
                time.time_ns(),
                json.dumps(payload).encode(),
            )

        start_ts_ns = time.time_ns()
        req = build_pd_proxy_request(body, self.config, proxy_start_ts_ns=start_ts_ns)
        req.decode_body["stream"] = True
        logger.info(
            "[PdProxy] request=%s accepted streaming path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )
        response = _open_json(
            req.decode_url + path,
            req.decode_body,
            self.config.timeout_s,
            req.request_id,
            "D",
        )
        logger.info(
            "[PdProxy] request=%s D stream headers status=%s latency_ms=%.3f ts_ns=%d",
            req.request_id,
            int(response.status),
            (header_delta_ns := time.time_ns() - start_ts_ns) / 1_000_000,
            start_ts_ns + header_delta_ns,
        )
        return (
            int(response.status),
            response.headers.get("Content-Type", "text/event-stream"),
            response,
            req.request_id,
            start_ts_ns,
            None,
        )

    def close(self) -> None:
        return None


def _post_json(
    url: str,
    body: dict[str, Any],
    timeout_s: float,
    request_id: str,
    role: str,
) -> tuple[int, bytes, str]:
    payload = json.dumps(body, separators=(",", ":")).encode()
    logger.info(
        "[PdProxy] request=%s -> %s url=%s body_bytes=%d kv=%s",
        request_id,
        role,
        url,
        len(payload),
        body.get("kv_transfer_params"),
    )
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            response_body = response.read()
            return (
                int(response.status),
                response_body,
                response.headers.get("Content-Type", "application/json"),
            )
    except HTTPError as exc:
        response_body = exc.read()
        logger.error(
            "[PdProxy] request=%s %s returned status=%s body=%s",
            request_id,
            role,
            exc.code,
            response_body[:512],
        )
        return int(exc.code), response_body, exc.headers.get("Content-Type", "application/json")
    except URLError:
        logger.exception("[PdProxy] request=%s %s connection failed url=%s", request_id, role, url)
        raise


def _open_json(
    url: str,
    body: dict[str, Any],
    timeout_s: float,
    request_id: str,
    role: str,
):
    payload = json.dumps(body, separators=(",", ":")).encode()
    logger.info(
        "[PdProxy] request=%s -> %s stream url=%s body_bytes=%d kv=%s",
        request_id,
        role,
        url,
        len(payload),
        body.get("kv_transfer_params"),
    )
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        return urlopen(request, timeout=timeout_s)
    except HTTPError:
        raise
    except URLError:
        logger.exception("[PdProxy] request=%s %s connection failed url=%s", request_id, role, url)
        raise


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


class _Handler(BaseHTTPRequestHandler):
    server: _PdHttpServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body)
            if body.get("stream"):
                (
                    status,
                    content_type,
                    response,
                    request_id,
                    start_ts_ns,
                    payload,
                ) = self.server.proxy.open_openai_stream(
                    self.path,
                    body,
                )
                self.send_response(int(status))
                self.send_header("Content-Type", content_type)
                if payload is not None:
                    self.send_header("Content-Length", str(len(payload)))
                else:
                    self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if payload is not None:
                    self.wfile.write(payload)
                    return
                with response:
                    first_chunk = True
                    for chunk in iter_http_stream_chunks(response):
                        if first_chunk:
                            first_chunk = False
                            now_ns = time.time_ns()
                            logger.info(
                                "[PdProxy] request=%s first stream chunk bytes=%d ttft_ms=%.3f ts_ns=%d",
                                request_id,
                                len(chunk),
                                (now_ns - start_ts_ns) / 1_000_000,
                                now_ns,
                            )
                        self.wfile.write(chunk)
                        self.wfile.flush()
                return
            status, payload, content_type = self.server.proxy.handle_openai_request(
                self.path,
                body,
            )
        except HTTPError as exc:
            payload = exc.read()
            status = int(exc.code)
            content_type = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:
            logger.exception("[PdProxy] request handling failed")
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            payload = json.dumps({"error": str(exc)}).encode()
            content_type = "application/json"

        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        logger.info("[PdProxyHTTP] " + format, *args)


class _PdHttpServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], proxy: PdProxy) -> None:
        super().__init__(server_address, _Handler)
        self.proxy = proxy


def parse_endpoint_urls(raw: str) -> tuple[str, ...]:
    urls = tuple(url.strip().rstrip("/") for url in raw.split(",") if url.strip())
    if not urls:
        raise argparse.ArgumentTypeError("expected at least one endpoint URL")
    return urls


def build_router(
    *,
    prefill_urls: tuple[str, ...],
    decode_urls: tuple[str, ...],
    routing_policy: str,
) -> RoundRobinPairRouter:
    if routing_policy != "round_robin":
        raise ValueError(f"unsupported routing policy {routing_policy!r}")
    normalized_prefill_urls = tuple(url.rstrip("/") for url in prefill_urls)
    normalized_decode_urls = tuple(url.rstrip("/") for url in decode_urls)
    return RoundRobinPairRouter(
        prefill_endpoints=tuple(
            PdEndpoint(url=url, instance_id=f"p{index}")
            for index, url in enumerate(normalized_prefill_urls)
        ),
        decode_endpoints=tuple(
            PdEndpoint(url=url, instance_id=f"d{index}")
            for index, url in enumerate(normalized_decode_urls)
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PegaFlow P/D local proxy")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8100)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8001")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8002")
    parser.add_argument("--prefill-urls", type=parse_endpoint_urls)
    parser.add_argument("--decode-urls", type=parse_endpoint_urls)
    parser.add_argument("--routing-policy", choices=["round_robin"], default="round_robin")
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

    prefill_urls = args.prefill_urls or (args.prefill_url.rstrip("/"),)
    decode_urls = args.decode_urls or (args.decode_url.rstrip("/"),)
    router = build_router(
        prefill_urls=prefill_urls,
        decode_urls=decode_urls,
        routing_policy=args.routing_policy,
    )
    config = ProxyConfig(
        prefill_url=prefill_urls[0],
        decode_url=decode_urls[0],
        timeout_s=args.timeout_s,
        prefill_max_tokens=args.prefill_max_tokens,
        router=router,
    )
    proxy = PdProxy(config)
    server = _PdHttpServer((args.listen_host, args.listen_port), proxy)
    logger.info(
        "[PdProxy] listening http://%s:%d policy=%s prefill=%s decode=%s",
        args.listen_host,
        args.listen_port,
        args.routing_policy,
        list(prefill_urls),
        list(decode_urls),
    )
    try:
        server.serve_forever()
    finally:
        proxy.close()
        server.server_close()


if __name__ == "__main__":
    main()
