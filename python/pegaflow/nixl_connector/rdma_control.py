# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side control channel for Pega NIXL RDMA v1 handshakes."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import msgspec
import zmq
from vllm import envs
from vllm.utils.network_utils import make_zmq_path

from pegaflow.nixl_connector.metadata import PEGA_RDMA_HANDSHAKE_MSG
from pegaflow.nixl_connector.utils import zmq_ctx

logger = logging.getLogger(__name__)


class RdmaHandshakeTransport(Protocol):
    def prepare_rdma_peer(self, peer_key: str) -> str: ...

    def complete_rdma_peer(self, peer_key: str, remote_metadata: str) -> None: ...


def rdma_control_port(vllm_config: object, tp_rank: int) -> int:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    dp_index = int(getattr(parallel_config, "data_parallel_index", 0) or 0)
    return int(envs.VLLM_NIXL_SIDE_CHANNEL_PORT) + 10000 + dp_index * 1024 + int(tp_rank)


def rdma_control_host() -> str:
    return str(envs.VLLM_NIXL_SIDE_CHANNEL_HOST)


def rdma_peer_key(engine_id: str, tp_rank: int) -> str:
    return f"{engine_id}:{int(tp_rank)}"


class PegaRdmaControlServer:
    def __init__(
        self,
        *,
        transport: RdmaHandshakeTransport,
        host: str,
        port: int,
    ) -> None:
        self.transport = transport
        self.host = host
        self.port = int(port)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        ready_event = threading.Event()
        self._thread = threading.Thread(
            target=self._serve,
            args=(ready_event,),
            daemon=True,
            name="pega-rdma-control",
        )
        self._thread.start()
        ready_event.wait()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _serve(self, ready_event: threading.Event) -> None:
        path = make_zmq_path("tcp", self.host, self.port)
        with zmq_ctx(zmq.ROUTER, path) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 1000)
            ready_event.set()
            while not self._stop_event.is_set():
                try:
                    identity, _, msg = sock.recv_multipart()
                except zmq.Again:
                    continue
                try:
                    msg_type, payload = msgspec.msgpack.decode(msg)
                    if msg_type != PEGA_RDMA_HANDSHAKE_MSG:
                        raise RuntimeError(f"unexpected RDMA control message {msg_type!r}")
                    peer_key = str(payload["peer_key"])
                    remote_metadata = str(payload["metadata"])
                    local_metadata = self.transport.prepare_rdma_peer(peer_key)
                    self.transport.complete_rdma_peer(peer_key, remote_metadata)
                    reply = msgspec.msgpack.encode(
                        {"ok": True, "metadata": local_metadata}
                    )
                except Exception as exc:
                    logger.exception("Failed to handle Pega RDMA v1 handshake")
                    reply = msgspec.msgpack.encode({"ok": False, "error": str(exc)})
                sock.send_multipart((identity, b"", reply))


def exchange_rdma_handshake(
    *,
    transport: RdmaHandshakeTransport,
    local_peer_key: str,
    remote_peer_key: str,
    remote_host: str,
    remote_port: int,
    timeout_ms: int = 5000,
) -> None:
    local_metadata = transport.prepare_rdma_peer(remote_peer_key)
    path = make_zmq_path("tcp", remote_host, int(remote_port))
    with zmq_ctx(zmq.REQ, path) as sock:
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.send(
            msgspec.msgpack.encode(
                (
                    PEGA_RDMA_HANDSHAKE_MSG,
                    {
                        "peer_key": local_peer_key,
                        "metadata": local_metadata,
                    },
                )
            )
        )
        reply = msgspec.msgpack.decode(sock.recv())
    if not reply.get("ok"):
        raise RuntimeError(f"Pega RDMA v1 handshake failed: {reply.get('error')}")
    transport.complete_rdma_peer(remote_peer_key, str(reply["metadata"]))
