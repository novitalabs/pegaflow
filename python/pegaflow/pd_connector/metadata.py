"""Metadata exchanged by the experimental P/D RDMA push connector."""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

BlockIds = tuple[list[int], ...]

RELEASE_CONSUMER_ABORT = "consumer_abort"
RELEASE_PRODUCER_ABORT = "producer_abort"
RELEASE_PRODUCER_FINISHED = "producer_finished"
RELEASE_PRODUCER_PREEMPTED = "producer_preempted"


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


def _is_strictly_increasing(values: tuple[int, ...]) -> bool:
    return all(prev < current for prev, current in zip(values, values[1:], strict=False))


@dataclass(frozen=True)
class WaitReqMeta:
    local_block_ids: BlockIds
    remote_request_id: str
    done_request_id: str
    prompt_token_ids: tuple[int, ...]
    prefill_url: str
    model: str = ""
    prefill_max_tokens: int = 1
    proxy_start_ts_ns: int = 0
    matched_ts_ns: int = 0
    scheduler_wait_ts_ns: int = 0


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
    block_stride: int | None = None

    def __post_init__(self) -> None:
        assert self.region_idx >= 0
        assert self.base_addr > 0
        assert self.block_len > 0
        assert self.block_stride is None or self.block_stride >= self.block_len


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
        assert _is_strictly_increasing(self.block_ids)
        assert self.regions
        assert self.block_ids[0] >= 0
        assert tuple(region.region_idx for region in self.regions) == tuple(
            range(len(self.regions))
        )


@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int
    tp_size: int
    block_size: int
    layers: tuple[LayerRemoteLayout, ...]
    imm_id: int | None = None
    fail_imm_id: int | None = None
    abort_imm_id: int | None = None
    expected_imm_count: int = 1
    layers_by_idx: dict[int, LayerRemoteLayout] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layers_by_idx",
            {layer.layer_idx: layer for layer in self.layers},
        )


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
        parsed_block_ids = block_ids
    else:
        parsed_block_ids = tuple(int(block_id) for block_id in raw_block_ids)
    return LayerRemoteLayout(
        layer_name=str(data["layer_name"]),
        layer_idx=int(data["layer_idx"]),
        block_ids=parsed_block_ids,
        regions=tuple(
            TransferRegionLayout(
                region_idx=int(region["region_idx"]),
                base_addr=int(region["base_addr"]),
                block_len=int(region["block_len"]),
                block_stride=(
                    int(region["block_stride"]) if region.get("block_stride") is not None else None
                ),
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
        fail_imm_id=int(data["fail_imm_id"]) if data.get("fail_imm_id") is not None else None,
        abort_imm_id=int(data["abort_imm_id"]) if data.get("abort_imm_id") is not None else None,
        expected_imm_count=int(data.get("expected_imm_count") or 1),
    )


def handshakes_from_dicts(data: Any) -> tuple[PdHandshake, ...]:
    if data is None:
        return ()
    if isinstance(data, str):
        data = _decode_handshake_payload(data)
    iterable = data.values() if isinstance(data, dict) else data
    handshakes = tuple(handshake_from_dict(item) for item in iterable)
    assert all(handshake is not None for handshake in handshakes)
    return handshakes  # type: ignore[return-value]


def encode_handshake_payload(data: Any) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode()
    return base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")


def _decode_handshake_payload(data: str) -> Any:
    raw = zlib.decompress(base64.b64decode(data.encode("ascii")))
    return json.loads(raw)


def layer_layout_to_dict(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "block_ids": list(layer.block_ids),
        "regions": [_region_layout_to_dict(region) for region in layer.regions],
        "mr_desc": layer.mr_desc,
    }


def layer_layout_to_compact_dict(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "regions": [_region_layout_to_dict(region) for region in layer.regions],
        "mr_desc": layer.mr_desc,
    }


def _region_layout_to_dict(region: TransferRegionLayout) -> dict[str, Any]:
    data = {
        "region_idx": region.region_idx,
        "base_addr": region.base_addr,
        "block_len": region.block_len,
    }
    if region.block_stride is not None:
        data["block_stride"] = region.block_stride
    return data


def handshake_to_dict(handshake: PdHandshake) -> dict[str, Any]:
    return {
        "request_id": handshake.request_id,
        "engine_id": handshake.engine_id,
        "tp_rank": handshake.tp_rank,
        "tp_size": handshake.tp_size,
        "block_size": handshake.block_size,
        "layers": [layer_layout_to_dict(layer) for layer in handshake.layers],
        "imm_id": handshake.imm_id,
        "fail_imm_id": handshake.fail_imm_id,
        "abort_imm_id": handshake.abort_imm_id,
        "expected_imm_count": handshake.expected_imm_count,
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
        "fail_imm_id": handshake.fail_imm_id,
        "abort_imm_id": handshake.abort_imm_id,
        "expected_imm_count": handshake.expected_imm_count,
    }


class PdConnectorMetadata(KVConnectorMetadata):
    """Scheduler-to-worker metadata for one vLLM scheduler step."""

    def __init__(
        self,
        reqs_to_wait: dict[str, WaitReqMeta] | None = None,
        reqs_to_push: dict[str, PushReqMeta] | None = None,
        reqs_to_release: set[str] | None = None,
        release_reasons: dict[str, str] | None = None,
        preempted_req_ids: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.reqs_to_wait = reqs_to_wait or {}
        self.reqs_to_push = reqs_to_push or {}
        self.reqs_to_release = reqs_to_release or set()
        self.release_reasons = release_reasons or {}
        self.preempted_req_ids = preempted_req_ids or set()


@dataclass
class PdWorkerMetadata(KVConnectorWorkerMetadata):
    failed_recving: set[str] = field(default_factory=set)

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, PdWorkerMetadata)
        self.failed_recving.update(other.failed_recving)
        return self


@dataclass
class WorkerDoneState:
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    failed_blocks: set[int] = field(default_factory=set)
