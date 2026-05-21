"""Small P/D proxy for exercising PdConnector with two local vLLM servers.

The proxy accepts OpenAI-compatible completion requests, injects the
``kv_transfer_params`` expected by ``PdConnector``, starts the decode request
against D, then starts the prefill request against P. D only begins decoding
after its connector observes the async fake-RDMA done notification from P.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("pegaflow.pd_proxy")


SUPPORTED_PATHS = {"/v1/completions", "/v1/chat/completions"}


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    done_endpoint: str
    timeout_s: float
    prefill_max_tokens: int


@dataclass(frozen=True)
class PdProxyRequest:
    request_id: str
    prefill_body: dict[str, Any]
    decode_body: dict[str, Any]


def build_pd_proxy_request(
    body: dict[str, Any],
    config: ProxyConfig,
    request_id: str | None = None,
) -> PdProxyRequest:
    req_id = request_id or str(body.get("request_id") or f"pd-{uuid.uuid4().hex}")
    prefill_req_id = f"{req_id}-p"
    decode_req_id = f"{req_id}-d"

    prefill_body = copy.deepcopy(body)
    decode_body = copy.deepcopy(body)

    prefill_body["request_id"] = prefill_req_id
    prefill_body["stream"] = False
    prefill_body["max_tokens"] = int(body.get("pd_prefill_max_tokens", config.prefill_max_tokens))
    prefill_body["kv_transfer_params"] = {
        "do_remote_prefill_sender": True,
        "target_engine_id": "decode",
        "target_request_id": decode_req_id,
        "done_endpoint": config.done_endpoint,
    }

    decode_body["request_id"] = decode_req_id
    decode_body["stream"] = False
    decode_body["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "remote_engine_id": "prefill",
        "remote_request_id": prefill_req_id,
        "done_endpoint": config.done_endpoint,
    }

    return PdProxyRequest(
        request_id=req_id,
        prefill_body=prefill_body,
        decode_body=decode_body,
    )


class PdProxy:
    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="pd-proxy")

    def handle_openai_request(self, path: str, body: dict[str, Any]) -> tuple[int, bytes, str]:
        if path not in SUPPORTED_PATHS:
            payload = {"error": f"unsupported path {path}"}
            return HTTPStatus.NOT_FOUND, json.dumps(payload).encode(), "application/json"
        if body.get("stream"):
            payload = {"error": "Pd proxy currently runs non-streaming requests"}
            return HTTPStatus.BAD_REQUEST, json.dumps(payload).encode(), "application/json"

        req = build_pd_proxy_request(body, self.config)
        logger.info(
            "[PdProxy] request=%s accepted path=%s done_endpoint=%s",
            req.request_id,
            path,
            self.config.done_endpoint,
        )

        decode_future = self.executor.submit(
            _post_json,
            self.config.decode_url + path,
            req.decode_body,
            self.config.timeout_s,
            req.request_id,
            "D",
        )

        # Give D a small head start so its connector can allocate blocks and
        # bind the fake-RDMA done endpoint before P finishes the prefill pass.
        time.sleep(0.05)
        prefill_future = self.executor.submit(
            _post_json,
            self.config.prefill_url + path,
            req.prefill_body,
            self.config.timeout_s,
            req.request_id,
            "P",
        )

        prefill_status, prefill_body, _ = prefill_future.result(timeout=self.config.timeout_s)
        logger.info(
            "[PdProxy] request=%s P completed status=%s bytes=%d",
            req.request_id,
            prefill_status,
            len(prefill_body),
        )

        decode_status, decode_body, decode_content_type = decode_future.result(
            timeout=self.config.timeout_s
        )
        logger.info(
            "[PdProxy] request=%s D completed status=%s bytes=%d",
            req.request_id,
            decode_status,
            len(decode_body),
        )
        return decode_status, decode_body, decode_content_type

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


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


class _Handler(BaseHTTPRequestHandler):
    server: _PdHttpServer

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            body = json.loads(raw_body)
            status, payload, content_type = self.server.proxy.handle_openai_request(
                self.path,
                body,
            )
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
    parser.add_argument("--done-endpoint", default="tcp://127.0.0.1:7200")
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
        done_endpoint=args.done_endpoint,
        timeout_s=args.timeout_s,
        prefill_max_tokens=args.prefill_max_tokens,
    )
    proxy = PdProxy(config)
    server = _PdHttpServer((args.listen_host, args.listen_port), proxy)
    logger.info(
        "[PdProxy] listening http://%s:%d prefill=%s decode=%s done_endpoint=%s",
        args.listen_host,
        args.listen_port,
        config.prefill_url,
        config.decode_url,
        config.done_endpoint,
    )
    try:
        server.serve_forever()
    finally:
        proxy.close()
        server.server_close()


if __name__ == "__main__":
    main()
