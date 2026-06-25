# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by PegaFlow contributors in 2026.
"""PegaFlow RDMA v1 adapter for the vendored NIXL pull connector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgspec

from pegaflow.pegaflow import PegaRdmaV1Engine

PEGA_RDMA_V1_HANDSHAKE_PREFIX = b"PEGA_RDMA_V1_HS:"


@dataclass(frozen=True)
class PegaRdmaV1Config:
    nics: list[str]
    qps_per_peer: int
    handshake_timeout_s: float

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> PegaRdmaV1Config:
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
        return cls(
            nics=nics,
            qps_per_peer=qps_per_peer,
            handshake_timeout_s=handshake_timeout_s,
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


def make_peer_key(local_engine_id: str, local_tp_rank: int, remote_engine_id: str, remote_rank: int):
    return f"{local_engine_id}:{local_tp_rank}->{remote_engine_id}:{remote_rank}"


def parse_peer_endpoint(endpoint: str) -> tuple[str, int]:
    engine_id, rank = endpoint.rsplit(":", 1)
    return engine_id, int(rank)


def peer_key_source(peer_key: str) -> tuple[str, int]:
    source, _target = peer_key.split("->", 1)
    return parse_peer_endpoint(source)


def reverse_peer_key(peer_key: str) -> str:
    local, remote = peer_key.split("->", 1)
    return f"{remote}->{local}"


def build_memory_regions(blocks_data: list[tuple[int, int, int]]) -> list[dict[str, int]]:
    merged: dict[int, int] = {}
    for addr, size, _device_id in blocks_data:
        if size <= 0:
            continue
        merged[addr] = max(merged.get(addr, 0), size)
    return [{"addr": addr, "len": size} for addr, size in merged.items()]


def build_read_descs(
    local_blocks_data: list[tuple[int, int, int]],
    remote_blocks_data: list[tuple[int, int, int]],
    local_desc_ids,
    remote_desc_ids,
) -> list[dict[str, int]]:
    descs: list[dict[str, int]] = []
    for local_idx, remote_idx in zip(local_desc_ids, remote_desc_ids, strict=True):
        local_addr, local_len, _local_dev = local_blocks_data[int(local_idx)]
        remote_addr, remote_len, _remote_dev = remote_blocks_data[int(remote_idx)]
        transfer_len = min(local_len, remote_len)
        if transfer_len <= 0:
            continue
        descs.append(
            {
                "local_addr": int(local_addr),
                "remote_addr": int(remote_addr),
                "len": int(transfer_len),
            }
        )
    return descs


def encode_handshake_request(
    peer_key: str,
    metadata: bytes,
    response_agent_metadata: bytes,
) -> bytes:
    return PEGA_RDMA_V1_HANDSHAKE_PREFIX + msgspec.msgpack.encode(
        {
            "kind": "request",
            "peer_key": peer_key,
            "metadata": metadata,
            "response_agent_metadata": response_agent_metadata,
        }
    )


def encode_handshake_response(peer_key: str, metadata: bytes) -> bytes:
    return PEGA_RDMA_V1_HANDSHAKE_PREFIX + msgspec.msgpack.encode(
        {
            "kind": "response",
            "peer_key": peer_key,
            "metadata": metadata,
        }
    )


def decode_handshake_notif(notif: bytes) -> dict[str, Any] | None:
    if not notif.startswith(PEGA_RDMA_V1_HANDSHAKE_PREFIX):
        return None
    payload = msgspec.msgpack.decode(notif[len(PEGA_RDMA_V1_HANDSHAKE_PREFIX) :])
    if not isinstance(payload, dict):
        raise ValueError("Pega RDMA v1 handshake notification payload must be a dict")
    return payload


def create_rdma_engine(config: PegaRdmaV1Config) -> PegaRdmaV1Engine:
    return PegaRdmaV1Engine(nics=config.nics, qps_per_peer=config.qps_per_peer)
