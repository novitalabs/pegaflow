"""
Worker-side outbound KV transfer runtime helpers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pegaflow.connector.common import KvEgressIntent
from pegaflow.connector.common import logger

if TYPE_CHECKING:
    from pegaflow.pegaflow import KvEgressRuntime


_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


@dataclass(frozen=True)
class KvEgressConfig:
    enabled: bool
    nic_names: tuple[str, ...]


@dataclass(frozen=True)
class EgressLayerRegistration:
    base_ptr: int
    size_bytes: int
    num_blocks: int
    bytes_per_block: int
    kv_stride_bytes: int
    segments: int


@dataclass(frozen=True)
class KvEgressStats:
    bytes_written: int
    write_descs: int
    coalesced_write_descs: int
    descriptor_ms: float
    connect_ms: float
    build_desc_ms: float
    rdma_write_ms: float
    imm_ms: float
    total_ms: float
    write_nics_active: int
    imm_nics_active: int
    preferred_nic_idx: int | None

    @property
    def rdma_write_gbps(self) -> float:
        if self.rdma_write_ms <= 0:
            return 0.0
        return (self.bytes_written * 8) / (self.rdma_write_ms / 1000.0) / 1e9


def resolve_kv_egress_config() -> KvEgressConfig:
    enabled = _truthy_env("PEGAFLOW_KV_EGRESS")
    nic_names = (
        _parse_env_list("PEGAFLOW_KV_EGRESS_NICS")
        or _parse_env_list("PEGAFLOW_RDMA_NICS")
    )
    if enabled and not nic_names:
        nic_names = ("mlx5_0",)
    return KvEgressConfig(enabled=enabled, nic_names=nic_names)


def create_kv_egress_runtime() -> "KvEgressRuntime | None":
    config = resolve_kv_egress_config()
    if not config.enabled:
        return None

    from pegaflow.pegaflow import KvEgressRuntime

    try:
        runtime = KvEgressRuntime(list(config.nic_names))
    except Exception as e:
        logger.error(
            "[PegaKVConnector] Failed to initialize KV egress runtime with nics=%s: %s",
            list(config.nic_names),
            e,
        )
        raise

    logger.info(
        "[PegaKVConnector] Initialized KV egress runtime in worker process: nics=%s",
        list(config.nic_names),
    )
    return runtime


def execute_kv_egress(
    runtime: "KvEgressRuntime",
    intent: KvEgressIntent,
    layers: dict[str, EgressLayerRegistration],
    requester_id: str,
    receive_rank: int,
    preferred_nic_idx: int | None,
) -> KvEgressStats:
    total_start = time.perf_counter()
    client = _engine_client(intent.remote_endpoint)
    descriptor_start = time.perf_counter()
    descriptor = _wait_for_descriptor(intent, client, receive_rank)
    descriptor_ms = _elapsed_ms(descriptor_start)
    remote_addr = _transfer_peer_key(intent.remote_endpoint)

    connect_start = time.perf_counter()
    runtime._ensure_connected(remote_addr, requester_id, client._native)
    connect_ms = _elapsed_ms(connect_start)

    build_start = time.perf_counter()
    descs = _build_write_descs(intent, descriptor, layers, receive_rank)
    write_descs = len(descs)
    descs = _coalesce_adjacent_descs(descs)
    coalesced_write_descs = len(descs)
    build_desc_ms = _elapsed_ms(build_start)

    write_start = time.perf_counter()
    if preferred_nic_idx is None:
        bytes_written, write_nics_active = runtime._write_registered(remote_addr, descs)
    else:
        bytes_written, write_nics_active = runtime._write_registered_on_nic(
            remote_addr, descs, int(preferred_nic_idx)
        )
    rdma_write_ms = _elapsed_ms(write_start)

    imm_start = time.perf_counter()
    _, imm_nics_active = runtime._write_imm(remote_addr, int(descriptor["imm_data"]))
    imm_ms = _elapsed_ms(imm_start)

    return KvEgressStats(
        bytes_written=int(bytes_written),
        write_descs=write_descs,
        coalesced_write_descs=coalesced_write_descs,
        descriptor_ms=descriptor_ms,
        connect_ms=connect_ms,
        build_desc_ms=build_desc_ms,
        rdma_write_ms=rdma_write_ms,
        imm_ms=imm_ms,
        total_ms=_elapsed_ms(total_start),
        write_nics_active=int(write_nics_active),
        imm_nics_active=int(imm_nics_active),
        preferred_nic_idx=(
            int(preferred_nic_idx) if preferred_nic_idx is not None else None
        ),
    )


def _wait_for_descriptor(intent: KvEgressIntent, client, receive_rank: int) -> dict:
    timeout_s = _env_float("PEGAFLOW_KV_EGRESS_DESCRIPTOR_TIMEOUT_SECONDS", 30.0)
    interval_s = _env_float("PEGAFLOW_KV_EGRESS_DESCRIPTOR_POLL_SECONDS", 0.01)
    deadline = time.monotonic() + timeout_s

    while True:
        descriptor = client._get_staged_load_descriptor(
            intent.remote_instance_id,
            intent.request_id,
            int(receive_rank),
            intent.handle,
        )
        if not descriptor.get("ok", False):
            raise RuntimeError(
                "GetPdReceiveDescriptor failed: "
                f"{descriptor.get('message', 'unknown error')}"
            )

        state = descriptor.get("state")
        if state == "ready":
            return descriptor
        if state in {"failed", "expired"}:
            raise RuntimeError(
                "staged load descriptor is not usable: "
                f"state={state} request_id={intent.request_id}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for staged load descriptor: "
                f"request_id={intent.request_id} dst={intent.remote_instance_id} "
                f"state={state}"
            )
        time.sleep(interval_s)


def _build_write_descs(
    intent: KvEgressIntent,
    descriptor: dict,
    layers: dict[str, EgressLayerRegistration],
    receive_rank: int,
) -> list[tuple[int, int, int]]:
    slabs = list(descriptor.get("slabs") or [])
    if not slabs:
        raise RuntimeError(f"staged load descriptor has no slabs: request_id={intent.request_id}")

    ranks = list(descriptor.get("ranks") or [])
    if ranks:
        if len(ranks) != 1:
            raise RuntimeError(
                f"staged load rank descriptor must contain exactly one receive rank, got {len(ranks)}"
            )
        actual_rank = int(ranks[0].get("receive_rank", -1))
        if actual_rank != receive_rank:
            raise RuntimeError(
                f"staged load descriptor receive_rank mismatch: expected={receive_rank} got={actual_rank}"
            )

    descs: list[tuple[int, int, int]] = []
    for remote_layer in descriptor.get("layers") or []:
        if "receive_rank" in remote_layer and int(remote_layer["receive_rank"]) != receive_rank:
            continue
        layer_name = str(remote_layer["layer_name"])

        local = layers.get(layer_name)
        if local is None:
            raise RuntimeError(f"KV egress layer is not registered locally: {layer_name}")

        slab_index = int(remote_layer["slab_index"])
        if slab_index < 0 or slab_index >= len(slabs):
            raise RuntimeError(f"staged load descriptor references invalid slab_index={slab_index}")
        slab = slabs[slab_index]

        block_count = len(intent.block_ids)
        remote_num_blocks = int(remote_layer["num_blocks"])
        if block_count > remote_num_blocks:
            raise RuntimeError(
                f"staged load descriptor has {remote_num_blocks} blocks for {layer_name}, "
                f"but intent needs {block_count}"
            )

        segment_count = int(remote_layer["segment_count"])
        segment_size = int(remote_layer["segment_size"])
        padded_segment_stride = int(remote_layer["padded_segment_stride"])
        block_stride = int(remote_layer["block_stride"])
        layer_offset = int(remote_layer["layer_offset"])
        remote_base = int(slab["base_ptr"]) + layer_offset

        if segment_count != local.segments:
            raise RuntimeError(
                f"KV egress segment count mismatch for {layer_name}: "
                f"local={local.segments} remote={segment_count}"
            )
        if segment_size != local.bytes_per_block:
            raise RuntimeError(
                f"KV egress segment size mismatch for {layer_name}: "
                f"local={local.bytes_per_block} remote={segment_size}"
            )

        for remote_block_idx, local_block_id in enumerate(intent.block_ids):
            local_block = int(local_block_id)
            if local_block < 0 or local_block >= local.num_blocks:
                raise RuntimeError(
                    f"KV egress local block id out of range for {layer_name}: "
                    f"block_id={local_block} capacity={local.num_blocks}"
                )
            for segment_idx in range(segment_count):
                local_ptr = _local_segment_ptr(local, local_block, segment_idx)
                remote_ptr = (
                    remote_base
                    + remote_block_idx * block_stride
                    + segment_idx * padded_segment_stride
                )
                descs.append((local_ptr, remote_ptr, segment_size))

    if not descs:
        raise RuntimeError(f"KV egress has no write descriptors: request_id={intent.request_id}")
    return descs


def _coalesce_adjacent_descs(descs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    if len(descs) <= 1:
        return descs

    max_len = (1 << 32) - 1
    merged: list[tuple[int, int, int]] = []
    cur_local, cur_remote, cur_len = descs[0]
    for local_ptr, remote_ptr, length in descs[1:]:
        if (
            cur_len + length <= max_len
            and cur_local + cur_len == local_ptr
            and cur_remote + cur_len == remote_ptr
        ):
            cur_len += length
            continue
        merged.append((cur_local, cur_remote, cur_len))
        cur_local, cur_remote, cur_len = local_ptr, remote_ptr, length
    merged.append((cur_local, cur_remote, cur_len))
    return merged


def _local_segment_ptr(
    layer: EgressLayerRegistration,
    block_id: int,
    segment_idx: int,
) -> int:
    if layer.segments == 1:
        if segment_idx != 0:
            raise RuntimeError(f"invalid segment_idx={segment_idx} for single-segment layer")
        return layer.base_ptr + block_id * layer.bytes_per_block
    if segment_idx >= layer.segments:
        raise RuntimeError(f"invalid segment_idx={segment_idx} for segments={layer.segments}")
    return layer.base_ptr + block_id * layer.bytes_per_block + segment_idx * layer.kv_stride_bytes


def _engine_client(addr: str):
    from pegaflow import PegaClient

    return PegaClient(_grpc_endpoint(addr))


def _grpc_endpoint(addr: str) -> str:
    return addr if "://" in addr else f"http://{addr}"


def _transfer_peer_key(addr: str) -> str:
    return addr


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %.3f", name, value, default)
        return default
    return max(0.001, parsed)


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _parse_env_list(name: str) -> tuple[str, ...]:
    value = os.environ.get(name, "")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _truthy_env(*names: str) -> bool:
    return any(os.environ.get(name) in _TRUTHY for name in names)


__all__ = [
    "EgressLayerRegistration",
    "KvEgressConfig",
    "KvEgressStats",
    "create_kv_egress_runtime",
    "execute_kv_egress",
    "resolve_kv_egress_config",
]
