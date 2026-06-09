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
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import httpx

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
    def select(self) -> PdRoute: ...


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
        self._route_counts: dict[tuple[str, str], int] = {}

    def select(self) -> PdRoute:
        with self._lock:
            index = self._next_index
            self._next_index += 1
            route = PdRoute(
                prefill=self._prefill_endpoints[index % len(self._prefill_endpoints)],
                decode=self._decode_endpoints[index % len(self._decode_endpoints)],
            )
            key = (route.prefill.instance_id, route.decode.instance_id)
            self._route_counts[key] = self._route_counts.get(key, 0) + 1
            return route

    def route_counts(self) -> dict[tuple[str, str], int]:
        with self._lock:
            return dict(self._route_counts)


@dataclass
class ProxyMetrics:
    request_count: int = 0
    stream_request_count: int = 0
    inflight_requests: int = 0
    error_count: int = 0
    decode_request_durations_s: list[float] = field(default_factory=list)
    first_chunk_durations_s: list[float] = field(default_factory=list)
    stream_durations_s: list[float] = field(default_factory=list)
    stream_chunks: int = 0
    stream_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def start_request(self, *, stream: bool) -> None:
        with self._lock:
            self.request_count += 1
            if stream:
                self.stream_request_count += 1
            self.inflight_requests += 1

    def finish_request(self, duration_s: float | None = None, *, error: bool = False) -> None:
        with self._lock:
            self.inflight_requests = max(0, self.inflight_requests - 1)
            if duration_s is not None:
                self.decode_request_durations_s.append(max(0.0, duration_s))
            if error:
                self.error_count += 1

    def record_first_chunk(self, duration_s: float) -> None:
        with self._lock:
            self.first_chunk_durations_s.append(max(0.0, duration_s))

    def record_stream_chunk(self, size: int) -> None:
        with self._lock:
            self.stream_chunks += 1
            self.stream_bytes += max(0, int(size))

    def record_stream_duration(self, duration_s: float) -> None:
        with self._lock:
            self.stream_durations_s.append(max(0.0, duration_s))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_count": self.request_count,
                "stream_request_count": self.stream_request_count,
                "inflight_requests": self.inflight_requests,
                "error_count": self.error_count,
                "decode_request_durations_s": list(self.decode_request_durations_s),
                "first_chunk_durations_s": list(self.first_chunk_durations_s),
                "stream_durations_s": list(self.stream_durations_s),
                "stream_chunks": self.stream_chunks,
                "stream_bytes": self.stream_bytes,
            }


def render_proxy_metrics(config: ProxyConfig) -> bytes:
    metrics = config.metrics.snapshot()
    lines = [
        "# TYPE pega_pd_proxy_requests_total counter",
        f"pega_pd_proxy_requests_total {metrics['request_count']}",
        "# TYPE pega_pd_proxy_stream_requests_total counter",
        f"pega_pd_proxy_stream_requests_total {metrics['stream_request_count']}",
        "# TYPE pega_pd_proxy_inflight_requests gauge",
        f"pega_pd_proxy_inflight_requests {metrics['inflight_requests']}",
        "# TYPE pega_pd_proxy_errors_total counter",
        f"pega_pd_proxy_errors_total {metrics['error_count']}",
        "# TYPE pega_pd_proxy_stream_chunks_total counter",
        f"pega_pd_proxy_stream_chunks_total {metrics['stream_chunks']}",
        "# TYPE pega_pd_proxy_stream_bytes_total counter",
        f"pega_pd_proxy_stream_bytes_total {metrics['stream_bytes']}",
    ]
    _extend_summary(
        lines,
        "pega_pd_proxy_decode_request_duration_seconds",
        metrics["decode_request_durations_s"],
    )
    _extend_summary(
        lines,
        "pega_pd_proxy_first_chunk_duration_seconds",
        metrics["first_chunk_durations_s"],
    )
    _extend_summary(
        lines,
        "pega_pd_proxy_stream_duration_seconds",
        metrics["stream_durations_s"],
    )
    route_counts = (
        config.router.route_counts()
        if config.router is not None and hasattr(config.router, "route_counts")
        else {}
    )
    lines.append("# TYPE pega_pd_proxy_route_total counter")
    for (prefill_instance, decode_instance), count in sorted(route_counts.items()):
        lines.append(
            "pega_pd_proxy_route_total{"
            f'prefill_instance="{_escape_label(prefill_instance)}",'
            f'decode_instance="{_escape_label(decode_instance)}"'
            f"}} {count}"
        )
    return ("\n".join(lines) + "\n").encode()


def _extend_summary(lines: list[str], name: str, values: list[float]) -> None:
    count = len(values)
    total = sum(values)
    lines.append(f"# TYPE {name} summary")
    lines.append(f"{name}_count {count}")
    lines.append(f"{name}_sum {total:.9f}")


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    timeout_s: float
    prefill_max_tokens: int
    router: PdRouter | None = None
    metrics: ProxyMetrics = field(default_factory=ProxyMetrics)


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
        self._stream_client: httpx.Client | None = None
        self._stream_client_lock = threading.Lock()

    def handle_openai_request(self, path: str, body: dict[str, Any]) -> tuple[int, bytes, str]:
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return HTTPStatus.NOT_FOUND, json.dumps(payload).encode(), "application/json"

        start_ts_ns = time.time_ns()
        self.config.metrics.start_request(stream=False)
        req = build_pd_proxy_request(body, self.config, proxy_start_ts_ns=start_ts_ns)
        try:
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
            end_ts_ns = time.time_ns()
            self.config.metrics.finish_request(
                (end_ts_ns - start_ts_ns) / 1_000_000_000,
                error=decode_status >= 400,
            )
            logger.info(
                "[PdProxy] request=%s D completed status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
                req.request_id,
                decode_status,
                len(decode_body),
                (end_ts_ns - start_ts_ns) / 1_000_000,
                end_ts_ns,
            )
            return decode_status, decode_body, decode_content_type
        except Exception:
            self.config.metrics.finish_request(error=True)
            raise

    def open_openai_stream(self, path: str, body: dict[str, Any]):
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return (
                HTTPStatus.NOT_FOUND,
                "application/json",
                None,
                "",
                0,
                json.dumps(payload).encode(),
            )

        start_ts_ns = time.time_ns()
        self.config.metrics.start_request(stream=True)
        req = build_pd_proxy_request(body, self.config, proxy_start_ts_ns=start_ts_ns)
        req.decode_body["stream"] = True
        logger.info(
            "[PdProxy] request=%s accepted streaming path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )
        try:
            response = _open_json(
                req.decode_url + path,
                req.decode_body,
                self.config.timeout_s,
                req.request_id,
                "D",
                client=self._get_stream_client(),
            )
            header_ts_ns = time.time_ns()
            logger.info(
                "[PdProxy] request=%s D stream headers status=%s latency_ms=%.3f ts_ns=%d",
                req.request_id,
                int(response.status),
                (header_ts_ns - start_ts_ns) / 1_000_000,
                header_ts_ns,
            )
            return (
                int(response.status),
                response.headers.get("Content-Type", "text/event-stream"),
                response,
                req.request_id,
                start_ts_ns,
                None,
            )
        except Exception:
            self.config.metrics.finish_request(error=True)
            raise

    def close(self) -> None:
        with self._stream_client_lock:
            client = self._stream_client
            self._stream_client = None
        if client is not None:
            client.close()

    def _get_stream_client(self) -> httpx.Client:
        with self._stream_client_lock:
            if self._stream_client is None:
                self._stream_client = httpx.Client(
                    timeout=self.config.timeout_s,
                    limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
                )
            return self._stream_client


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
        logger.exception(
            "[PdProxy] request=%s %s connection failed url=%s",
            request_id,
            role,
            url,
        )
        raise


def _open_json(
    url: str,
    body: dict[str, Any],
    timeout_s: float,
    request_id: str,
    role: str,
    *,
    client: httpx.Client | None = None,
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
    stream_cm: Any | None = None
    owns_client = client is None
    try:
        if client is None:
            client = httpx.Client(
                timeout=timeout_s,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            )
        stream_cm = client.stream(
            "POST",
            url,
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        response = stream_cm.__enter__()
        return _OpenHttpxStream(
            client=client,
            stream_cm=stream_cm,
            response=response,
            owns_client=owns_client,
        )
    except Exception:
        if stream_cm is not None:
            stream_cm.__exit__(None, None, None)
        if client is not None and owns_client:
            client.close()
        logger.exception(
            "[PdProxy] request=%s %s connection failed url=%s",
            request_id,
            role,
            url,
        )
        raise


@dataclass
class _OpenHttpxStream:
    client: httpx.Client
    stream_cm: Any
    response: httpx.Response
    owns_client: bool = True

    @property
    def status(self) -> int:
        return self.response.status_code

    @property
    def headers(self) -> httpx.Headers:
        return self.response.headers

    def iter_bytes(self) -> Any:
        yield from self.response.iter_bytes()

    def __enter__(self) -> "_OpenHttpxStream":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.stream_cm.__exit__(exc_type, exc, traceback)
        finally:
            if self.owns_client:
                self.client.close()


def iter_http_stream_bytes(response, chunk_size: int | None = None) -> Any:
    iter_bytes = getattr(response, "iter_bytes", None)
    if iter_bytes is not None:
        if chunk_size is None:
            yield from iter_bytes()
        else:
            yield from iter_bytes(chunk_size=chunk_size)
        return

    if chunk_size is None:
        chunk_size = 64 * 1024
    read_chunk = getattr(response, "read1", None)
    if read_chunk is None:
        read_chunk = response.read
    while True:
        chunk = read_chunk(chunk_size)
        if not chunk:
            return
        yield chunk


class _Handler(BaseHTTPRequestHandler):
    server: _PdHttpServer

    def do_GET(self) -> None:
        if self.path != "/metrics":
            payload = b"not found\n"
            self.send_response(HTTPStatus.NOT_FOUND)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        payload = render_proxy_metrics(self.server.proxy.config)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
                    if start_ts_ns > 0:
                        self.server.proxy.config.metrics.finish_request(
                            (time.time_ns() - start_ts_ns) / 1_000_000_000,
                            error=int(status) >= 400,
                        )
                    return
                try:
                    with response:
                        first_chunk = True
                        for chunk in iter_http_stream_bytes(response):
                            self.server.proxy.config.metrics.record_stream_chunk(len(chunk))
                            if first_chunk:
                                first_chunk = False
                                now_ns = time.time_ns()
                                self.server.proxy.config.metrics.record_first_chunk(
                                    (now_ns - start_ts_ns) / 1_000_000_000
                                )
                                logger.info(
                                    "[PdProxy] request=%s first stream chunk bytes=%d ttft_ms=%.3f ts_ns=%d",
                                    request_id,
                                    len(chunk),
                                    (now_ns - start_ts_ns) / 1_000_000,
                                    now_ns,
                                )
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    stream_duration_s = (time.time_ns() - start_ts_ns) / 1_000_000_000
                    self.server.proxy.config.metrics.record_stream_duration(stream_duration_s)
                    self.server.proxy.config.metrics.finish_request(stream_duration_s)
                except Exception:
                    self.server.proxy.config.metrics.finish_request(error=True)
                    raise
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
