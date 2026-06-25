# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""PegaFlow RDMA v1 adapter for the vendored NIXL pull connector.

The vendored NIXL connector still owns scheduler metadata, TP/block mapping,
heartbeats, and completion notifications.  This module only defines the small
Pega RDMA v1 side channel used to exchange native RDMA handshake metadata and
submit the KV READ data plane from the pull worker.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import msgspec

from pegaflow.pegaflow import PegaRdmaV1Engine

# Pega RDMA v1 handshake messages share NIXL's notification channel, so they
# need an explicit prefix to stay distinguishable from regular request/HB
# notifications handled by the upstream NIXL worker.
PEGA_RDMA_V1_HANDSHAKE_PREFIX = b"PEGA_RDMA_V1_HS:"


@dataclass(frozen=True)
class PegaRdmaV1Config:
    nics: list[str]
    qps_per_peer: int
    handshake_timeout_s: float
    perf_enabled: bool
    perf_log_every: int

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> PegaRdmaV1Config:
        """Parse Pega RDMA v1 settings from vLLM's connector extra config."""
        raw_nics = extra_config.get("pegaflow.nixl.rdma_v1.nics")
        if raw_nics is None:
            raw_nics = extra_config.get("pegaflow.pd.rdma.nics")
        if isinstance(raw_nics, str):
            nics = [item.strip() for item in raw_nics.split(",") if item.strip()]
        elif isinstance(raw_nics, list):
            nics = [str(item) for item in raw_nics]
        else:
            nics = []
        if not nics:
            raise ValueError(
                "PegaNixlPullConnector requires kv_connector_extra_config "
                "'pegaflow.nixl.rdma_v1.nics' with non-bond RDMA NIC names"
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


def encode_handshake_request(
    peer_key: str,
    metadata: bytes,
    response_agent_metadata: bytes,
) -> bytes:
    """Encode the D->P Pega RDMA handshake request.

    The extra ``response_agent_metadata`` is the decode worker's NIXL agent
    metadata.  The prefill worker cannot assume the decode agent is already in
    its ``_remote_agents`` table while replying to this transport-level
    handshake, so it registers a temporary response agent from this metadata.
    """
    return PEGA_RDMA_V1_HANDSHAKE_PREFIX + msgspec.msgpack.encode(
        {
            "kind": "request",
            "peer_key": peer_key,
            "metadata": metadata,
            "response_agent_metadata": response_agent_metadata,
        }
    )


def encode_handshake_response(peer_key: str, metadata: bytes) -> bytes:
    """Encode the P->D Pega RDMA metadata response."""
    return PEGA_RDMA_V1_HANDSHAKE_PREFIX + msgspec.msgpack.encode(
        {
            "kind": "response",
            "peer_key": peer_key,
            "metadata": metadata,
        }
    )


def decode_handshake_notif(notif: bytes) -> dict[str, Any] | None:
    """Return a Pega RDMA handshake payload, or ``None`` for normal NIXL notifs."""
    if not notif.startswith(PEGA_RDMA_V1_HANDSHAKE_PREFIX):
        return None
    payload = msgspec.msgpack.decode(notif[len(PEGA_RDMA_V1_HANDSHAKE_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("Pega RDMA v1 handshake notification payload must be a dict")
    return payload


def create_rdma_engine(config: PegaRdmaV1Config) -> PegaRdmaV1Engine:
    """Create the native Pega RDMA v1 engine from parsed connector config."""
    return PegaRdmaV1Engine(nics=config.nics, qps_per_peer=config.qps_per_peer)
