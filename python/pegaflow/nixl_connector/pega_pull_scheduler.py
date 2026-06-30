# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Scheduler-side Pega RDMA v1 hooks for the NIXL pull connector."""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm.logger import init_logger

from pegaflow.nixl_connector.metadata import NixlHandshakePayload
from pegaflow.nixl_connector.pega_rdma_v1 import (
    PEGA_RDMA_V1_ACCEPT_ACK,
    PEGA_RDMA_V1_ACCEPT_ENDPOINT,
    PEGA_RDMA_V1_ACCEPT_REGISTER,
    PEGA_RDMA_V1_EXTENSION,
    PegaRdmaV1Config,
    accept_handshake_via_zmq,
    make_accept_broker_endpoint,
    reverse_peer_key,
)
from pegaflow.nixl_connector.pull_scheduler import NixlPullConnectorScheduler
from pegaflow.nixl_connector.utils import zmq_ctx

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class PegaNixlPullConnectorScheduler(NixlPullConnectorScheduler):
    """Fold Pega RDMA v1 accept metadata into NIXL's GET_META response."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self._pega_rdma_config = PegaRdmaV1Config.from_extra_config(extra_config)
        self._pega_accept_broker_endpoint = make_accept_broker_endpoint(
            self.engine_id,
            self.side_channel_port,
        )
        self._pega_accept_broker_stop = threading.Event()
        self._pega_accept_broker_thread: threading.Thread | None = None

    def set_xfer_handshake_metadata(self, metadata):
        """Advertise one scheduler-owned broker endpoint to all remote workers."""
        self._start_pega_rdma_accept_broker()
        for payload in metadata.values():
            if not isinstance(payload, NixlHandshakePayload):
                continue
            extensions = dict(payload.extensions or {})
            extensions[PEGA_RDMA_V1_ACCEPT_ENDPOINT] = self._pega_accept_broker_endpoint
            payload.extensions = extensions
        super().set_xfer_handshake_metadata(metadata)

    def _start_pega_rdma_accept_broker(self) -> None:
        """Start the scheduler-owned ROUTER that workers register with."""
        if self._pega_accept_broker_thread is not None:
            return
        if self._pega_accept_broker_endpoint.startswith("ipc://"):
            with suppress(FileNotFoundError):
                os.unlink(self._pega_accept_broker_endpoint[len("ipc://") :])
        ready_event = threading.Event()
        thread = threading.Thread(
            target=self._pega_rdma_accept_broker_loop,
            args=(
                self._pega_accept_broker_endpoint,
                ready_event,
                self._pega_accept_broker_stop,
            ),
            daemon=True,
            name=f"pega-rdma-v1-accept-broker-{self.engine_id}",
        )
        thread.start()
        if not ready_event.wait(timeout=self._pega_rdma_config.handshake_timeout_s):
            self._pega_accept_broker_stop.set()
            raise RuntimeError(
                f"Pega RDMA v1 accept broker failed to bind "
                f"{self._pega_accept_broker_endpoint}"
            )
        self._pega_accept_broker_thread = thread

    def _pega_rdma_accept_broker_loop(
        self,
        endpoint: str,
        ready_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        """Route scheduler accept requests to the worker owning target TP rank."""
        workers: dict[int, bytes] = {}
        deadline_by_request: dict[int, float] = {}
        pending: dict[int, tuple[tuple[bytes, ...], int]] = {}
        payload_by_request: dict[int, dict[str, Any]] = {}
        waiting_by_rank: dict[int, list[int]] = {}
        next_request_id = 1
        timeout_s = self._pega_rdma_config.handshake_timeout_s
        with zmq_ctx(zmq.ROUTER, endpoint) as sock:
            sock.setsockopt(zmq.RCVTIMEO, 100)
            ready_event.set()
            while not stop_event.is_set():
                now = time.monotonic()
                for request_id, deadline in list(deadline_by_request.items()):
                    if now <= deadline:
                        continue
                    pending_entry = pending.pop(request_id, None)
                    deadline_by_request.pop(request_id, None)
                    if pending_entry is not None:
                        client_reply_prefix, target_tp_rank = pending_entry
                        if waiting := waiting_by_rank.get(target_tp_rank):
                            with suppress(ValueError):
                                waiting.remove(request_id)
                        payload_by_request.pop(request_id, None)
                        response = {
                            "ok": False,
                            "error": "Pega RDMA v1 worker accept timed out",
                        }
                        sock.send_multipart(
                            (*client_reply_prefix, msgspec.msgpack.encode(response))
                        )
                try:
                    frames = sock.recv_multipart()
                except zmq.Again:
                    continue
                try:
                    identity, reply_prefix, msg = self._decode_broker_frames(frames)
                except ValueError:
                    logger.debug("Pega RDMA v1 broker got malformed frames: %s", frames)
                    continue
                try:
                    payload = msgspec.msgpack.decode(msg)
                    if not isinstance(payload, dict):
                        raise ValueError("broker payload must be a dict")
                    kind = payload.get("kind")
                    if kind == PEGA_RDMA_V1_ACCEPT_REGISTER:
                        tp_rank = payload.get("tp_rank")
                        if not isinstance(tp_rank, int):
                            raise ValueError("worker register missing tp_rank")
                        workers[tp_rank] = identity
                        sock.send_multipart(
                            (
                                identity,
                                msgspec.msgpack.encode(
                                    {"ok": True, "kind": PEGA_RDMA_V1_ACCEPT_ACK}
                                ),
                            )
                        )
                        for request_id in list(waiting_by_rank.pop(tp_rank, [])):
                            pending_entry = pending.get(request_id)
                            if pending_entry is None:
                                continue
                            _client_reply_prefix, _target_tp_rank = pending_entry
                            payload_to_send = payload_by_request.pop(request_id)
                            sock.send_multipart(
                                (identity, msgspec.msgpack.encode(payload_to_send))
                            )
                    elif "request_id" in payload:
                        request_id = payload.get("request_id")
                        pending_entry = pending.pop(request_id, None)
                        deadline_by_request.pop(request_id, None)
                        payload_by_request.pop(request_id, None)
                        if pending_entry is None:
                            logger.warning(
                                "Pega RDMA v1 broker got response for unknown request %s",
                                request_id,
                            )
                            continue
                        client_reply_prefix, _target_tp_rank = pending_entry
                        sock.send_multipart(
                            (*client_reply_prefix, msgspec.msgpack.encode(payload))
                        )
                    else:
                        target_tp_rank = payload.get("target_tp_rank")
                        if not isinstance(target_tp_rank, int):
                            raise ValueError("accept request missing target_tp_rank")
                        request_id = next_request_id
                        next_request_id += 1
                        payload["request_id"] = request_id
                        pending[request_id] = (reply_prefix, target_tp_rank)
                        deadline_by_request[request_id] = time.monotonic() + timeout_s
                        worker = workers.get(target_tp_rank)
                        if worker is None:
                            payload_by_request[request_id] = payload
                            waiting_by_rank.setdefault(target_tp_rank, []).append(request_id)
                        else:
                            sock.send_multipart((worker, msgspec.msgpack.encode(payload)))
                except Exception as exc:
                    logger.debug("Pega RDMA v1 broker request failed", exc_info=True)
                    response = {"ok": False, "error": str(exc)}
                    sock.send_multipart((*reply_prefix, msgspec.msgpack.encode(response)))

    @staticmethod
    def _decode_broker_frames(frames: list[bytes]) -> tuple[bytes, tuple[bytes, ...], bytes]:
        """Decode ROUTER frames from either a REQ client or DEALER worker.

        ``accept_handshake_via_zmq`` uses REQ, so ROUTER receives
        ``identity, empty, payload`` and must include the empty delimiter when
        replying.  Worker threads use DEALER identities and exchange a single
        payload frame after the identity.  Keeping those shapes separate avoids
        leaking an empty frame into worker ``recv()``.
        """
        if len(frames) == 2:
            identity, msg = frames
            return identity, (identity,), msg
        if len(frames) == 3 and frames[1] == b"":
            identity, _empty, msg = frames
            return identity, (identity, b""), msg
        raise ValueError(f"unexpected ROUTER frame shape with {len(frames)} frames")

    def _handle_handshake_extensions(
        self,
        target_tp_rank: int,
        payload: NixlHandshakePayload,
        request_extensions: dict[str, Any],
    ) -> NixlHandshakePayload:
        """Accept D-side RDMA metadata before replying to NIXL GET_META."""
        rdma_request = request_extensions.get(PEGA_RDMA_V1_EXTENSION)
        if rdma_request is None:
            return payload
        if not isinstance(rdma_request, dict):
            raise ValueError("Pega RDMA v1 request extension must be a dict")

        peer_key = rdma_request.get("peer_key")
        metadata = rdma_request.get("metadata")
        if not isinstance(peer_key, str) or not isinstance(metadata, bytes):
            raise ValueError("Pega RDMA v1 request extension missing peer_key/metadata")

        endpoint = (payload.extensions or {}).get(PEGA_RDMA_V1_ACCEPT_ENDPOINT)
        if not isinstance(endpoint, str):
            raise RuntimeError(
                "Pega RDMA v1 accept endpoint missing from target rank handshake payload"
            )

        timeout_ms = int(self._pega_rdma_config.handshake_timeout_s * 1000)
        response_extensions = dict(payload.extensions or {})
        try:
            response_metadata = accept_handshake_via_zmq(
                endpoint,
                target_tp_rank,
                reverse_peer_key(peer_key),
                metadata,
                timeout_ms,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "metadata": response_metadata,
            }
        except Exception as exc:
            logger.warning(
                "Failed to accept Pega RDMA v1 handshake target_rank=%s peer=%s: %s",
                target_tp_rank,
                peer_key,
                exc,
            )
            response_extensions[PEGA_RDMA_V1_EXTENSION] = {
                "error": str(exc),
            }
        logger.debug(
            "Accepted Pega RDMA v1 handshake target_rank=%s peer=%s",
            target_tp_rank,
            peer_key,
        )
        return NixlHandshakePayload(
            compatibility_hash=payload.compatibility_hash,
            agent_metadata_bytes=payload.agent_metadata_bytes,
            extensions=response_extensions,
        )

    def shutdown(self):
        """Stop the Pega RDMA broker before the base scheduler listener exits."""
        self._pega_accept_broker_stop.set()
        if self._pega_accept_broker_thread is not None:
            self._pega_accept_broker_thread.join(timeout=2.0)
            self._pega_accept_broker_thread = None
        if self._pega_accept_broker_endpoint.startswith("ipc://"):
            with suppress(FileNotFoundError):
                os.unlink(self._pega_accept_broker_endpoint[len("ipc://") :])
        super().shutdown()
