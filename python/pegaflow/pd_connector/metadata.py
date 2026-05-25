"""Metadata exchanged by the experimental P/D RDMA push connector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

BlockIds = tuple[list[int], ...]


def normalize_block_ids(block_ids: Any) -> BlockIds:
    """Convert vLLM block-id containers to a tuple of mutable group lists."""
    if block_ids is None:
        return ()
    if hasattr(block_ids, "get_unhashed_block_ids_all_groups"):
        block_ids = block_ids.get_unhashed_block_ids_all_groups()
    elif hasattr(block_ids, "get_block_ids_all_groups"):
        block_ids = block_ids.get_block_ids_all_groups()
    elif hasattr(block_ids, "get_block_ids"):
        block_ids = block_ids.get_block_ids()

    if isinstance(block_ids, tuple):
        return tuple(list(group) for group in block_ids)
    if isinstance(block_ids, list):
        if not block_ids:
            return ()
        if all(isinstance(item, int) for item in block_ids):
            return (list(block_ids),)
        return tuple(list(group) for group in block_ids)
    raise TypeError(f"unsupported block id container: {type(block_ids)!r}")


def flatten_block_ids(block_ids: BlockIds) -> set[int]:
    return {block_id for group in block_ids for block_id in group}


@dataclass(frozen=True)
class WaitReqMeta:
    local_block_ids: BlockIds
    remote_request_id: str
    done_request_id: str
    prompt_token_ids: tuple[int, ...]
    prefill_url: str
    model: str = ""
    prefill_max_tokens: int = 1


@dataclass(frozen=True)
class PushReqMeta:
    local_block_ids: BlockIds
    target_request_id: str
    handshakes: tuple[PdHandshake, ...] = ()


@dataclass(frozen=True)
class LinearBlockAddrLayout:
    block_id_start: int
    block_id_stride: int
    num_blocks: int
    k_addr_start: int
    v_addr_start: int
    addr_stride: int


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    base_addr: int
    block_bytes: int
    block_ids: tuple[int, ...]
    k_block_addrs: tuple[int, ...]
    v_block_addrs: tuple[int, ...]
    mr_desc: Any | None = None
    linear: LinearBlockAddrLayout | None = None


@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int
    tp_size: int
    block_size: int
    layers: tuple[LayerRemoteLayout, ...]
    imm_id: int | None = None


def layer_layout_from_dict(data: dict[str, Any]) -> LayerRemoteLayout:
    if data.get("block_addr_format") == "linear":
        num_blocks = int(data["num_blocks"])
        block_id_start = int(data["block_id_start"])
        block_id_stride = int(data.get("block_id_stride", 1))
        addr_stride = int(data["addr_stride"])
        linear = LinearBlockAddrLayout(
            block_id_start=block_id_start,
            block_id_stride=block_id_stride,
            num_blocks=num_blocks,
            k_addr_start=int(data["k_addr_start"]),
            v_addr_start=int(data["v_addr_start"]),
            addr_stride=addr_stride,
        )
        block_ids = tuple(block_id_start + idx * block_id_stride for idx in range(num_blocks))
        k_block_addrs = tuple(
            int(data["k_addr_start"]) + idx * addr_stride for idx in range(num_blocks)
        )
        v_block_addrs = tuple(
            int(data["v_addr_start"]) + idx * addr_stride for idx in range(num_blocks)
        )
    else:
        linear = None
        block_ids = tuple(int(block_id) for block_id in data["block_ids"])
        k_block_addrs = tuple(int(addr) for addr in data["k_block_addrs"])
        v_block_addrs = tuple(int(addr) for addr in data["v_block_addrs"])
    return LayerRemoteLayout(
        layer_name=str(data["layer_name"]),
        layer_idx=int(data["layer_idx"]),
        base_addr=int(data["base_addr"]),
        block_bytes=int(data["block_bytes"]),
        block_ids=block_ids,
        k_block_addrs=k_block_addrs,
        v_block_addrs=v_block_addrs,
        mr_desc=data.get("mr_desc"),
        linear=linear,
    )


def handshake_from_dict(data: dict[str, Any] | None) -> PdHandshake | None:
    if data is None:
        return None
    return PdHandshake(
        request_id=str(data["request_id"]),
        engine_id=str(data["engine_id"]),
        tp_rank=int(data["tp_rank"]),
        tp_size=int(data["tp_size"]),
        block_size=int(data["block_size"]),
        layers=tuple(layer_layout_from_dict(layer) for layer in data["layers"]),
        imm_id=int(data["imm_id"]) if data.get("imm_id") is not None else None,
    )


def handshakes_from_dicts(data: Any) -> tuple[PdHandshake, ...]:
    if data is None:
        return ()
    iterable = data.values() if isinstance(data, dict) else data
    handshakes = tuple(handshake_from_dict(item) for item in iterable)
    assert all(handshake is not None for handshake in handshakes)
    return handshakes  # type: ignore[return-value]


def layer_layout_to_dict(layer: LayerRemoteLayout) -> dict[str, Any]:
    common = {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "base_addr": layer.base_addr,
        "block_bytes": layer.block_bytes,
        "mr_desc": layer.mr_desc,
    }
    compact = _linear_layer_layout_to_dict(layer)
    if compact is not None:
        return {**common, **compact}
    return {
        **common,
        "block_ids": list(layer.block_ids),
        "k_block_addrs": list(layer.k_block_addrs),
        "v_block_addrs": list(layer.v_block_addrs),
    }


def _linear_layer_layout_to_dict(layer: LayerRemoteLayout) -> dict[str, Any] | None:
    if layer.linear is not None:
        return {
            "block_addr_format": "linear",
            "block_id_start": layer.linear.block_id_start,
            "block_id_stride": layer.linear.block_id_stride,
            "num_blocks": layer.linear.num_blocks,
            "k_addr_start": layer.linear.k_addr_start,
            "v_addr_start": layer.linear.v_addr_start,
            "addr_stride": layer.linear.addr_stride,
        }
    num_blocks = len(layer.block_ids)
    if num_blocks == 0:
        return None
    if not (num_blocks == len(layer.k_block_addrs) == len(layer.v_block_addrs)):
        return None
    block_id_stride = _constant_stride(layer.block_ids)
    k_stride = _constant_stride(layer.k_block_addrs)
    v_stride = _constant_stride(layer.v_block_addrs)
    if block_id_stride is None or k_stride is None or v_stride is None or k_stride != v_stride:
        return None
    return {
        "block_addr_format": "linear",
        "block_id_start": layer.block_ids[0],
        "block_id_stride": block_id_stride,
        "num_blocks": num_blocks,
        "k_addr_start": layer.k_block_addrs[0],
        "v_addr_start": layer.v_block_addrs[0],
        "addr_stride": k_stride,
    }


def _constant_stride(values: tuple[int, ...]) -> int | None:
    if len(values) <= 1:
        return 1
    stride = values[1] - values[0]
    for prev, current in zip(values[:-1], values[1:], strict=True):
        if current - prev != stride:
            return None
    return stride


def handshake_to_dict(handshake: PdHandshake) -> dict[str, Any]:
    return {
        "request_id": handshake.request_id,
        "engine_id": handshake.engine_id,
        "tp_rank": handshake.tp_rank,
        "tp_size": handshake.tp_size,
        "block_size": handshake.block_size,
        "layers": [layer_layout_to_dict(layer) for layer in handshake.layers],
        "imm_id": handshake.imm_id,
    }


class PdConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for one vLLM scheduler step."""

    def __init__(
        self,
        reqs_to_wait: dict[str, WaitReqMeta] | None = None,
        reqs_to_push: dict[str, PushReqMeta] | None = None,
        reqs_to_release: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.reqs_to_wait = reqs_to_wait or {}
        self.reqs_to_push = reqs_to_push or {}
        self.reqs_to_release = reqs_to_release or set()


@dataclass
class PdWorkerMetadata(KVConnectorWorkerMetadata):
    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        return self


@dataclass
class WorkerDoneState:
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    failed_blocks: set[int] = field(default_factory=set)
