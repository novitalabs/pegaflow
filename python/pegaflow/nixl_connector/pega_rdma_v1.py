# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""Pega RDMA v1 helpers for the NIXL pull connector."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
from vllm import envs

from pegaflow.nixl_connector.utils import zmq_ctx
from pegaflow.pegaflow import PegaRdmaV1Engine

if TYPE_CHECKING:
    from vllm.config import VllmConfig


PEGA_RDMA_V1_EXTENSION = "pegaflow_rdma_v1"
PEGA_RDMA_V1_ACCEPT_ENDPOINT = "pegaflow_rdma_v1_accept_endpoint"
PEGA_RDMA_V1_ACCEPT_REGISTER = "register"
PEGA_RDMA_V1_ACCEPT_REQUEST = "accept"
PEGA_RDMA_V1_ACCEPT_RESPONSE = "accept_response"
PEGA_RDMA_V1_ACCEPT_ACK = "registered"


@dataclass(frozen=True)
class PegaRdmaV1Config:
    nics: list[str]
    qps_per_peer: int
    handshake_timeout_s: float
    perf_enabled: bool
    perf_log_every: int

    @classmethod
    def from_extra_config(
        cls,
        extra_config: dict[str, Any],
        *,
        tp_rank: int | None = None,
    ) -> PegaRdmaV1Config:
        """Parse Pega RDMA v1 settings from vLLM's connector extra config."""
        nics = _parse_rank_nics(
            extra_config.get("pegaflow.nixl.rdma_v1.nics_by_rank"),
            tp_rank,
        )
        if nics is None:
            raw_nics = extra_config.get("pegaflow.nixl.rdma_v1.nics")
            if raw_nics is None:
                raw_nics = extra_config.get("pegaflow.pd.rdma.nics")
            nics = _parse_nics(raw_nics)
        if not nics:
            raise ValueError(
                "PegaNixlPullConnector requires kv_connector_extra_config "
                "'pegaflow.nixl.rdma_v1.nics' or "
                "'pegaflow.nixl.rdma_v1.nics_by_rank' with non-bond RDMA NIC names"
            )

        qps_per_peer = int(extra_config.get("pegaflow.nixl.rdma_v1.qps_per_peer", 4))
        if qps_per_peer <= 0:
            raise ValueError("pegaflow.nixl.rdma_v1.qps_per_peer must be positive")

        handshake_timeout_s = float(
            extra_config.get("pegaflow.nixl.rdma_v1.handshake_timeout_s", 5.0)
        )
        if handshake_timeout_s <= 0:
            raise ValueError("pegaflow.nixl.rdma_v1.handshake_timeout_s must be positive")

        perf_enabled = _config_bool(
            extra_config.get("pegaflow.nixl.rdma_v1.perf"),
            default=os.getenv("PEGA_NIXL_RDMA_V1_PERF", "").lower() in {"1", "true", "yes"},
        )
        perf_log_every = int(
            extra_config.get(
                "pegaflow.nixl.rdma_v1.perf_log_every",
                os.getenv("PEGA_NIXL_RDMA_V1_PERF_LOG_EVERY", 64),
            )
        )
        if perf_log_every <= 0:
            raise ValueError("pegaflow.nixl.rdma_v1.perf_log_every must be positive")
        return cls(
            nics=nics,
            qps_per_peer=qps_per_peer,
            handshake_timeout_s=handshake_timeout_s,
            perf_enabled=perf_enabled,
            perf_log_every=perf_log_every,
        )


@dataclass(frozen=True)
class PegaRdmaV1Read:
    remote_addr: str
    handle: int
    notif_agent: str
    notif_msg: bytes
    submitted_at: float
    bytes_transferred: int
    num_descriptors: int


@dataclass(frozen=True)
class PegaRdmaV1BrokerRequest:
    reply_prefix: tuple[bytes, ...]
    target_tp_rank: int
    payload: dict[str, Any]
    deadline: float


class PegaRdmaV1BrokerState:
    """In-memory routing state for scheduler-owned RDMA accept broker."""

    def __init__(self, timeout_s: float):
        self._timeout_s = timeout_s
        self._next_request_id = 1
        self._workers: dict[int, bytes] = {}
        self._pending: dict[int, PegaRdmaV1BrokerRequest] = {}
        self._waiting_by_rank: dict[int, list[int]] = defaultdict(list)

    def add_request(
        self,
        *,
        reply_prefix: tuple[bytes, ...],
        target_tp_rank: int,
        payload: dict[str, Any],
        now: float,
    ) -> tuple[bytes | None, dict[str, Any]]:
        """Add a scheduler request and return the worker to send to, if ready."""
        request_id = self._next_request_id
        self._next_request_id += 1
        payload = dict(payload)
        payload["request_id"] = request_id
        self._pending[request_id] = PegaRdmaV1BrokerRequest(
            reply_prefix=reply_prefix,
            target_tp_rank=target_tp_rank,
            payload=payload,
            deadline=now + self._timeout_s,
        )
        worker = self._workers.get(target_tp_rank)
        if worker is None:
            self._waiting_by_rank[target_tp_rank].append(request_id)
        return worker, payload

    def register_worker(self, tp_rank: int, identity: bytes) -> list[dict[str, Any]]:
        """Remember a worker identity and return queued requests for that rank."""
        self._workers[tp_rank] = identity
        queued_payloads = []
        for request_id in self._waiting_by_rank.pop(tp_rank, []):
            request = self._pending.get(request_id)
            if request is not None:
                queued_payloads.append(request.payload)
        return queued_payloads

    def complete_request(
        self,
        request_id: int,
    ) -> tuple[bytes, ...] | None:
        """Remove a completed worker request and return the client reply route."""
        request = self._pending.pop(request_id, None)
        if request is None:
            return None
        self._remove_waiting_request(request.target_tp_rank, request_id)
        return request.reply_prefix

    def pop_timed_out(self, now: float) -> list[tuple[tuple[bytes, ...], int]]:
        """Remove expired requests and return reply routes with request IDs."""
        expired = []
        for request_id, request in list(self._pending.items()):
            if now <= request.deadline:
                continue
            self._pending.pop(request_id, None)
            self._remove_waiting_request(request.target_tp_rank, request_id)
            expired.append((request.reply_prefix, request_id))
        return expired

    def _remove_waiting_request(self, tp_rank: int, request_id: int) -> None:
        waiting = self._waiting_by_rank.get(tp_rank)
        if not waiting:
            return
        try:
            waiting.remove(request_id)
        except ValueError:
            return
        if not waiting:
            self._waiting_by_rank.pop(tp_rank, None)


class PegaRdmaV1Perf:
    def __init__(self, *, enabled: bool, log_every: int, logger_name: str):
        """Create an opt-in lightweight stage timer for the RDMA v1 path."""
        self.enabled = enabled
        self.log_every = log_every
        self.logger_name = logger_name
        self._count_by_stage: dict[str, int] = defaultdict(int)
        self._total_ns_by_stage: dict[str, int] = defaultdict(int)
        self._max_ns_by_stage: dict[str, int] = defaultdict(int)
        self._events = 0

    @contextmanager
    def measure(self, stage: str):
        """Measure one named stage when perf probes are enabled."""
        if not self.enabled:
            yield
            return
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_ns(stage, time.perf_counter_ns() - start_ns)

    def record_ns(self, stage: str, elapsed_ns: int) -> None:
        """Record a measured stage duration in nanoseconds."""
        if not self.enabled:
            return
        self._count_by_stage[stage] += 1
        self._total_ns_by_stage[stage] += elapsed_ns
        self._max_ns_by_stage[stage] = max(self._max_ns_by_stage[stage], elapsed_ns)
        self._events += 1
        if self._events % self.log_every == 0:
            self.log_summary()

    def log_summary(self) -> None:
        """Log aggregate timing counters collected so far."""
        if not self.enabled or not self._count_by_stage:
            return
        import logging

        parts = []
        for stage in sorted(self._count_by_stage):
            count = self._count_by_stage[stage]
            total_ms = self._total_ns_by_stage[stage] / 1_000_000
            avg_ms = total_ms / count
            max_ms = self._max_ns_by_stage[stage] / 1_000_000
            parts.append(f"{stage}:n={count},avg={avg_ms:.3f}ms,max={max_ms:.3f}ms")
        logging.getLogger(self.logger_name).warning(
            "Pega RDMA v1 perf summary %s", "; ".join(parts)
        )

    def log_final_summary(self) -> None:
        """Emit the final timing summary during worker shutdown."""
        if not self.enabled:
            return
        import logging

        logging.getLogger(self.logger_name).warning("Pega RDMA v1 perf final summary follows")
        self.log_summary()


def _config_bool(value: Any, *, default: bool) -> bool:
    """Coerce a connector config value to bool with a caller-provided default."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _parse_nics(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return []


def _parse_rank_nics(value: Any, tp_rank: int | None) -> list[str] | None:
    if value is None:
        return None
    if tp_rank is None:
        if isinstance(value, dict):
            rank_values = value.values()
        elif isinstance(value, list):
            rank_values = value
        else:
            raise ValueError("pegaflow.nixl.rdma_v1.nics_by_rank must be a dict or list")
        nics: list[str] = []
        for rank_value in rank_values:
            for nic in _parse_nics(rank_value):
                if nic not in nics:
                    nics.append(nic)
        return nics
    raw_rank_nics: Any
    if isinstance(value, dict):
        raw_rank_nics = value.get(str(tp_rank))
        if raw_rank_nics is None:
            raw_rank_nics = value.get(tp_rank)
    elif isinstance(value, list):
        raw_rank_nics = value[tp_rank] if tp_rank < len(value) else None
    else:
        raise ValueError("pegaflow.nixl.rdma_v1.nics_by_rank must be a dict or list")
    nics = _parse_nics(raw_rank_nics)
    if not nics:
        raise ValueError(
            "pegaflow.nixl.rdma_v1.nics_by_rank missing non-empty NIC list "
            f"for tp_rank {tp_rank}"
        )
    return nics


def make_peer_key(local_engine_id: str, local_tp_rank: int, remote_engine_id: str, remote_rank: int):
    """Build a stable directional key for one local TP rank to one remote rank."""
    return f"{local_engine_id}:{local_tp_rank}->{remote_engine_id}:{remote_rank}"


def parse_peer_endpoint(endpoint: str) -> tuple[str, int]:
    """Parse the ``engine_id:rank`` endpoint encoded in a peer key."""
    engine_id, rank = endpoint.rsplit(":", 1)
    return engine_id, int(rank)


def peer_key_source(peer_key: str) -> tuple[str, int]:
    """Return the source endpoint of a directional Pega RDMA peer key."""
    source, _target = peer_key.split("->", 1)
    return parse_peer_endpoint(source)


def reverse_peer_key(peer_key: str) -> str:
    """Reverse a directional peer key for the opposite side of the connection."""
    local, remote = peer_key.split("->", 1)
    return f"{remote}->{local}"


def worker_identity(engine_id: str, tp_rank: int) -> bytes:
    """Return the stable broker identity for one Pega RDMA worker rank."""
    return f"{engine_id}:{tp_rank}".encode()


def make_accept_broker_endpoint(engine_id: str, side_channel_port: int) -> str:
    """Build the scheduler-owned local IPC endpoint for Pega RDMA accepts."""
    base_path = Path(envs.VLLM_RPC_BASE_PATH)
    base_path.mkdir(parents=True, exist_ok=True)
    filename = f"pegaflow-nixl-rdma-v1-{engine_id}-{side_channel_port}.sock"
    return f"ipc://{base_path / filename}"


def make_accept_broker_endpoint_from_config(engine_id: str, vllm_config: VllmConfig) -> str:
    """Build the Pega RDMA accept broker endpoint using vLLM's NIXL port rule."""
    side_channel_port = (
        envs.VLLM_NIXL_SIDE_CHANNEL_PORT
        + vllm_config.parallel_config.data_parallel_index
    )
    return make_accept_broker_endpoint(engine_id, side_channel_port)


def build_memory_regions(blocks_data: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    """Build native registration regions from NIXL block descriptors.

    NIXL may describe the same backing allocation through many logical KV
    block entries.  Pega RDMA v1 only needs each base address registered once,
    so duplicate addresses are merged before crossing into PyO3.
    """
    merged: dict[int, int] = {}
    for addr, size, _device_id in blocks_data:
        if size <= 0:
            continue
        merged[addr] = max(merged.get(addr, 0), size)
    return [{"addr": addr, "len": size} for addr, size in merged.items()]


def build_handshake_request_extension(peer_key: str, metadata: bytes) -> dict[str, Any]:
    """Build the D->P RDMA extension carried by a NIXL GET_META request."""
    return {
        PEGA_RDMA_V1_EXTENSION: {
            "peer_key": peer_key,
            "metadata": metadata,
        }
    }


def parse_handshake_response_extension(
    response_extensions: dict[str, Any] | None,
) -> bytes | None:
    """Return P-side RDMA metadata carried by a NIXL handshake response."""
    if response_extensions is None:
        return None
    payload = response_extensions.get(PEGA_RDMA_V1_EXTENSION)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("Pega RDMA v1 response extension must be a dict")
    error = payload.get("error")
    if error is not None:
        raise RuntimeError(f"Pega RDMA v1 handshake failed: {error}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, bytes):
        raise ValueError("Pega RDMA v1 response extension missing metadata bytes")
    return metadata


def accept_handshake_via_zmq(
    endpoint: str,
    target_tp_rank: int,
    peer_key: str,
    metadata: bytes,
    timeout_ms: int,
) -> bytes:
    """Ask the scheduler-owned broker to route one RDMA accept to a P worker."""
    request = msgspec.msgpack.encode(
        {
            "kind": PEGA_RDMA_V1_ACCEPT_REQUEST,
            "target_tp_rank": target_tp_rank,
            "peer_key": peer_key,
            "metadata": metadata,
        }
    )
    with zmq_ctx(zmq.REQ, endpoint) as sock:
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.send(request)
        response = msgspec.msgpack.decode(sock.recv())
    if not isinstance(response, dict):
        raise ValueError("Pega RDMA v1 accept response must be a dict")
    if response.get("kind") != PEGA_RDMA_V1_ACCEPT_RESPONSE:
        raise ValueError(f"Pega RDMA v1 accept got unexpected response {response.get('kind')!r}")
    if not response.get("ok"):
        error = response.get("error")
        raise RuntimeError(f"Pega RDMA v1 accept failed: {error}")
    response_metadata = response.get("metadata")
    if not isinstance(response_metadata, bytes):
        raise ValueError("Pega RDMA v1 accept response missing metadata bytes")
    return response_metadata


def create_rdma_engine(config: PegaRdmaV1Config) -> PegaRdmaV1Engine:
    """Create the native Pega RDMA v1 engine from parsed connector config."""
    return PegaRdmaV1Engine(nics=config.nics, qps_per_peer=config.qps_per_peer)
