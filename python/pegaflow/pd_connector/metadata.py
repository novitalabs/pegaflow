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
class TransferRegionLayout:
    region_idx: int
    base_addr: int
    block_len: int

    def __post_init__(self) -> None:
        assert self.region_idx >= 0
        assert self.base_addr > 0
        assert self.block_len > 0


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    block_ids: tuple[int, ...]
    regions: tuple[TransferRegionLayout, ...]
    mr_desc: Any | None = None

    def __post_init__(self) -> None:
        assert self.layer_name
        assert self.layer_idx >= 0
        assert self.block_ids
        assert self.block_ids == tuple(sorted(self.block_ids))
        assert self.regions
        assert all(block_id >= 0 for block_id in self.block_ids)
        assert tuple(region.region_idx for region in self.regions) == tuple(
            range(len(self.regions))
        )

    @property
    def base_addr(self) -> int:
        return min(region.base_addr for region in self.regions)

    @property
    def byte_len(self) -> int:
        max_end = max(
            region.base_addr + (self.block_ids[-1] + 1) * region.block_len
            for region in self.regions
        )
        return max_end - self.base_addr

    def region_block_addrs(self, region_idx: int) -> tuple[int, ...]:
        region = self.regions[region_idx]
        return tuple(region.base_addr + block_id * region.block_len for block_id in self.block_ids)


@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int
    tp_size: int
    block_size: int
    layers: tuple[LayerRemoteLayout, ...]
    imm_id: int | None = None


def layer_layout_from_dict(
    data: dict[str, Any],
    *,
    block_ids: tuple[int, ...] | None = None,
) -> LayerRemoteLayout:
    assert "regions" in data
    raw_block_ids = data.get("block_ids")
    if raw_block_ids is None:
        if block_ids is None:
            raise KeyError("block_ids")
        raw_block_ids = block_ids
    return LayerRemoteLayout(
        layer_name=str(data["layer_name"]),
        layer_idx=int(data["layer_idx"]),
        block_ids=tuple(int(block_id) for block_id in raw_block_ids),
        regions=tuple(
            TransferRegionLayout(
                region_idx=int(region["region_idx"]),
                base_addr=int(region["base_addr"]),
                block_len=int(region["block_len"]),
            )
            for region in data["regions"]
        ),
        mr_desc=data.get("mr_desc"),
    )


def handshake_from_dict(data: dict[str, Any] | None) -> PdHandshake | None:
    if data is None:
        return None
    block_ids = data.get("block_ids")
    shared_block_ids = (
        tuple(int(block_id) for block_id in block_ids) if block_ids is not None else None
    )
    return PdHandshake(
        request_id=str(data["request_id"]),
        engine_id=str(data["engine_id"]),
        tp_rank=int(data["tp_rank"]),
        tp_size=int(data["tp_size"]),
        block_size=int(data["block_size"]),
        layers=tuple(
            layer_layout_from_dict(layer, block_ids=shared_block_ids) for layer in data["layers"]
        ),
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
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "block_ids": list(layer.block_ids),
        "regions": [
            {
                "region_idx": region.region_idx,
                "base_addr": region.base_addr,
                "block_len": region.block_len,
            }
            for region in layer.regions
        ],
        "mr_desc": layer.mr_desc,
    }


def layer_layout_to_compact_dict(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "regions": [
            {
                "region_idx": region.region_idx,
                "base_addr": region.base_addr,
                "block_len": region.block_len,
            }
            for region in layer.regions
        ],
        "mr_desc": layer.mr_desc,
    }


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


def handshake_to_compact_dict(handshake: PdHandshake) -> dict[str, Any]:
    if not handshake.layers:
        return handshake_to_dict(handshake)
    block_ids = handshake.layers[0].block_ids
    if any(layer.block_ids != block_ids for layer in handshake.layers):
        return handshake_to_dict(handshake)
    return {
        "request_id": handshake.request_id,
        "engine_id": handshake.engine_id,
        "tp_rank": handshake.tp_rank,
        "tp_size": handshake.tp_size,
        "block_size": handshake.block_size,
        "block_ids": list(block_ids),
        "layers": [layer_layout_to_compact_dict(layer) for layer in handshake.layers],
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
