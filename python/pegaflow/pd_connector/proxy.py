"""Small P/D proxy for exercising PdConnector with two local vLLM servers.

The proxy accepts OpenAI-compatible completion requests, injects the
``kv_transfer_params`` expected by ``PdConnector``, and sends the request only
to D. D allocates KV blocks, then uses the P hint from those params to trigger
the prefill side. D begins decoding after its connector observes the RDMA IMM
notification from P.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
) -> PdProxyRequest:
    req_id = request_id or str(body.get("request_id") or f"pd-{uuid.uuid4().hex}")
    prefill_req_id = f"{req_id}-p"
    decode_req_id = f"{req_id}-d"

    decode_body = copy.deepcopy(body)

    decode_body["request_id"] = decode_req_id
    decode_body["kv_transfer_params"] = ConsumerKvParams(
        prefill_url=config.prefill_url,
        remote_request_id=prefill_req_id,
        done_request_id=decode_req_id,
    ).to_dict()

    return PdProxyRequest(
        request_id=req_id,
        decode_body=decode_body,
    )


class PdProxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config

    def handle_openai_request(self, path: str, body: dict[str, Any]) -> tuple[int, bytes, str]:
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return HTTPStatus.NOT_FOUND, json.dumps(payload).encode(), "application/json"

        req = build_pd_proxy_request(body, self.config)
        start_ts_ns = time.time_ns()
        logger.info(
            "[PdProxy] request=%s accepted path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )

        decode_status, decode_body, decode_content_type = _post_json(
            self.config.decode_url + path,
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

        req = build_pd_proxy_request(body, self.config)
        req.decode_body["stream"] = True
        start_ts_ns = time.time_ns()
        logger.info(
            "[PdProxy] request=%s accepted streaming path=%s ts_ns=%d",
            req.request_id,
            path,
            start_ts_ns,
        )
        response = _open_json(
            self.config.decode_url + path,
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
    payload = json.dumps(body).encode()
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
    payload = json.dumps(body).encode()
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
                    while chunk := response.read(65536):
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
    proxy = PdProxy(config)
    server = _PdHttpServer((args.listen_host, args.listen_port), proxy)
    logger.info(
        "[PdProxy] listening http://%s:%d prefill=%s decode=%s",
        args.listen_host,
        args.listen_port,
        config.prefill_url,
        config.decode_url,
    )
    try:
        server.serve_forever()
    finally:
        proxy.close()
        server.server_close()


if __name__ == "__main__":
    main()
