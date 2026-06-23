"""Thin RDMA port abstraction used by the P/D connector skeleton."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.layout import (
    BlockRegionSlice,
    LayerBlockSlices,
    block_slices_bytes,
)
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdHandshake,
    handshake_to_dict,
    layer_layout_from_dict,
)

logger = get_connector_logger()
_MISSING = object()


class RdmaPort(Protocol):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def open_request(self, req_id: str, handshake: PdHandshake) -> None: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def wait_for_pushes(self, req_id: str) -> None: ...

    def pull_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def pull_layers(
        self,
        req_id: str,
        layers: list[tuple[int, list[LayerBlockSlices]]],
    ) -> None: ...

    def wait_for_pulls(self, req_id: str) -> None: ...

    def push_done(self, req_id: str) -> None: ...

    def write_stats(self, req_id: str) -> dict[str, Any]: ...

    def read_stats(self, req_id: str) -> dict[str, Any]: ...

    def fail_request(self, req_id: str) -> None: ...

    def abort_request(self, req_id: str) -> None: ...

    def aggregated_link_speed(self) -> int: ...

    def wait_done(self, req_id: str) -> None: ...

    def pop_finished_sending(self) -> set[str]: ...

    def pop_finished_recving(self) -> set[str]: ...

    def close_request(self, req_id: str) -> None: ...


class V1HandshakePort(Protocol):
    def prepare_peer(self, peer_key: str) -> str: ...

    def complete_peer(self, peer_key: str, remote_metadata: str) -> None: ...

    def open_request_with_peer(
        self,
        req_id: str,
        handshake: PdHandshake,
        peer_key: str,
    ) -> None: ...


class MockRdmaPort:
    """A test double that records RDMA calls without touching native verbs."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.remote_handshakes: dict[str, PdHandshake | None] = {}
        self.remote_peer_keys: dict[str, str] = {}
        self.prepared_peers: list[str] = []
        self.completed_peers: list[tuple[str, str]] = []
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = {}
        self.pulled_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = {}
        self.pull_layer_calls: list[tuple[str, int, list[LayerBlockSlices]]] = []
        self.pull_layers_calls: list[
            tuple[str, list[tuple[int, list[LayerBlockSlices]]]]
        ] = []
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def open_request(self, req_id: str, handshake: PdHandshake) -> None:
        self.registered.add(req_id)
        self.remote_handshakes[req_id] = handshake

    def prepare_peer(self, peer_key: str) -> str:
        self.prepared_peers.append(peer_key)
        return f"mock-meta:{peer_key}"

    def complete_peer(self, peer_key: str, remote_metadata: str) -> None:
        self.completed_peers.append((peer_key, remote_metadata))

    def open_request_with_peer(
        self,
        req_id: str,
        handshake: PdHandshake,
        peer_key: str,
    ) -> None:
        self.open_request(req_id, handshake)
        self.remote_peer_keys[req_id] = peer_key

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pushed_layers.setdefault(req_id, [])
        self.pushed_layers[req_id].append((layer_idx, blocks))

    def wait_for_pushes(self, req_id: str) -> None:
        return None

    def pull_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pull_layer_calls.append((req_id, layer_idx, blocks))
        self.pulled_layers.setdefault(req_id, [])
        self.pulled_layers[req_id].append((layer_idx, blocks))

    def pull_layers(
        self,
        req_id: str,
        layers: list[tuple[int, list[LayerBlockSlices]]],
    ) -> None:
        self.pull_layers_calls.append((req_id, layers))
        self.pulled_layers.setdefault(req_id, [])
        self.pulled_layers[req_id].extend(layers)

    def wait_for_pulls(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def push_done(self, req_id: str) -> None:
        self._finished_sending.add(req_id)

    def write_stats(self, req_id: str) -> dict[str, Any]:
        bytes_total = sum(
            block_slices_bytes(blocks) for _, blocks in self.pushed_layers.get(req_id, [])
        )
        return {
            "submitted": len(self.pushed_layers.get(req_id, [])),
            "completed": len(self.pushed_layers.get(req_id, [])),
            "errors": 0,
            "bytes": bytes_total,
            "has_submit": bytes_total > 0,
            "has_complete": bytes_total > 0,
        }

    def read_stats(self, req_id: str) -> dict[str, Any]:
        bytes_total = sum(
            block_slices_bytes(blocks) for _, blocks in self.pulled_layers.get(req_id, [])
        )
        return {
            "submitted": len(self.pulled_layers.get(req_id, [])),
            "completed": len(self.pulled_layers.get(req_id, [])),
            "errors": 0,
            "bytes": bytes_total,
            "has_submit": bytes_total > 0,
            "has_complete": bytes_total > 0,
        }

    def fail_request(self, req_id: str) -> None:
        return None

    def abort_request(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def aggregated_link_speed(self) -> int:
        return 400_000_000_000

    def wait_done(self, req_id: str) -> None:
        return None

    def pop_finished_sending(self) -> set[str]:
        finished = self._finished_sending
        self._finished_sending = set()
        return finished

    def pop_finished_recving(self) -> set[str]:
        finished = self._finished_recving
        self._finished_recving = set()
        return finished

    def close_request(self, req_id: str) -> None:
        self.registered.discard(req_id)
        self.remote_handshakes.pop(req_id, None)
        self.pushed_layers.pop(req_id, None)
        self.pulled_layers.pop(req_id, None)
        self._finished_sending.discard(req_id)
        self._finished_recving.discard(req_id)


def _block_slice_to_native(block: BlockRegionSlice) -> dict[str, int]:
    return {
        "block_id": block.block_id,
        "src_offset_bytes": block.src_offset_bytes,
        "bytes": block.bytes,
    }


def _layer_blocks_to_native(blocks: list[LayerBlockSlices]) -> list[dict[str, Any]]:
    return [
        {
            "regions": [
                {"region_idx": region_idx, **_block_slice_to_native(region)}
                for region_idx, region in enumerate(block.regions)
            ],
        }
        for block in _coalesce_contiguous_blocks(blocks)
    ]


def _coalesce_contiguous_blocks(blocks: list[LayerBlockSlices]) -> list[LayerBlockSlices]:
    if len(blocks) < 2:
        return blocks

    coalesced: list[LayerBlockSlices] = []
    current = blocks[0]
    for block in blocks[1:]:
        if _can_extend_block_range(current, block):
            current = LayerBlockSlices(
                regions=tuple(
                    BlockRegionSlice(
                        block_id=current_region.block_id,
                        src_offset_bytes=current_region.src_offset_bytes,
                        bytes=current_region.bytes + block_region.bytes,
                    )
                    for current_region, block_region in zip(
                        current.regions,
                        block.regions,
                        strict=True,
                    )
                ),
            )
            continue
        coalesced.append(current)
        current = block
    coalesced.append(current)
    return coalesced


def _can_extend_block_range(prev: LayerBlockSlices, nxt: LayerBlockSlices) -> bool:
    if len(prev.regions) != len(nxt.regions):
        return False
    for prev_region, next_region in zip(prev.regions, nxt.regions, strict=True):
        if prev_region.bytes % next_region.bytes != 0:
            return False
        block_count = prev_region.bytes // next_region.bytes
        if prev_region.block_id + block_count != next_region.block_id:
            return False
        if prev_region.src_offset_bytes + prev_region.bytes != next_region.src_offset_bytes:
            return False
    return True


def _layer_to_native(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "block_ids": list(layer.block_ids),
        "regions": [_region_to_native(region) for region in layer.regions],
        "mr_desc": _mr_desc_to_native(layer.mr_desc),
    }


def _region_to_native(region: Any) -> dict[str, int]:
    data = {
        "region_idx": region.region_idx,
        "base_addr": region.base_addr,
        "block_len": region.block_len,
    }
    if region.block_stride is not None:
        data["block_stride"] = region.block_stride
    return data


def _layer_from_native(layer: LayerRemoteLayout | dict[str, Any]) -> LayerRemoteLayout:
    if isinstance(layer, LayerRemoteLayout):
        return layer
    return layer_layout_from_dict(layer)


def _mr_desc_to_native(mr_desc: Any | None) -> Any | None:
    if not isinstance(mr_desc, dict):
        return mr_desc
    addr_rkey_list = mr_desc.get("addr_rkey_list")
    if addr_rkey_list is None:
        return mr_desc
    return {
        **mr_desc,
        "addr_rkey_list": [(str(addr_rkey[0]), int(addr_rkey[1])) for addr_rkey in addr_rkey_list],
    }


class RealRdmaPort:
    """Adapter from connector dataclasses to the native PyO3 RDMA engine.

    The native object is intentionally narrow. It owns v2 TransferEngine state,
    memory registration, peer state, and completion polling. This class only
    converts Python connector metadata to stable dictionaries.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        start = time.perf_counter()
        native_layers = [_layer_to_native(layer) for layer in layers]
        registered = self.engine.register_local_layers(native_layers)
        assert len(registered) == len(layers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[PdConnector] RDMA register_local_layers layers=%d blocks_per_layer=%s regions_per_layer=%s native_ms=%.3f",
            len(layers),
            [len(layer.block_ids) for layer in layers],
            [len(layer.regions) for layer in layers],
            elapsed_ms,
        )
        return tuple(_layer_from_native(layer) for layer in registered)

    def open_request(self, req_id: str, handshake: PdHandshake) -> None:
        start = time.perf_counter()
        # The native engine consumes the sealed wire JSON (pegaflow-pd-wire).
        self.engine.register_remote(req_id, json.dumps(handshake_to_dict(handshake)))
        elapsed_ms = (time.perf_counter() - start) * 1000
        blocks_per_layer = len(handshake.layers[0].block_ids) if handshake.layers else 0
        logger.info(
            "[PdConnector] RDMA open_request req=%s remote_req=%s tp_rank=%d/%d imm_id=%s layers=%d blocks_per_layer=%d block_size=%d native_ms=%.3f",
            req_id,
            handshake.request_id,
            handshake.tp_rank,
            handshake.tp_size,
            handshake.imm_id,
            len(handshake.layers),
            blocks_per_layer,
            handshake.block_size,
            elapsed_ms,
        )

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        native_blocks = _layer_blocks_to_native(blocks)
        start = time.perf_counter()
        self.engine.push_layer(req_id, layer_idx, native_blocks)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[PdConnector] RDMA push_layer req=%s layer=%d input_blocks=%d coalesced_blocks=%d regions=%d bytes=%d native_ms=%.3f",
                req_id,
                layer_idx,
                len(blocks),
                len(native_blocks),
                sum(len(block["regions"]) for block in native_blocks),
                block_slices_bytes(blocks),
                elapsed_ms,
            )

    def wait_for_pushes(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.wait_for_pushes(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA wait_for_pushes req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def pull_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        native_blocks = _layer_blocks_to_native(blocks)
        start = time.perf_counter()
        self.engine.pull_layer(req_id, layer_idx, native_blocks)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[PdConnector] RDMA pull_layer req=%s layer=%d input_blocks=%d coalesced_blocks=%d regions=%d bytes=%d native_ms=%.3f",
                req_id,
                layer_idx,
                len(blocks),
                len(native_blocks),
                sum(len(block["regions"]) for block in native_blocks),
                block_slices_bytes(blocks),
                elapsed_ms,
            )

    def pull_layers(
        self,
        req_id: str,
        layers: list[tuple[int, list[LayerBlockSlices]]],
    ) -> None:
        native_layers = [
            (layer_idx, _layer_blocks_to_native(blocks))
            for layer_idx, blocks in layers
            if blocks
        ]
        if not native_layers:
            return
        start = time.perf_counter()
        self.engine.pull_layers(req_id, native_layers)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[PdConnector] RDMA pull_layers req=%s layers=%d input_blocks=%d coalesced_blocks=%d regions=%d bytes=%d native_ms=%.3f",
                req_id,
                len(native_layers),
                sum(len(blocks) for _, blocks in layers),
                sum(len(blocks) for _, blocks in native_layers),
                sum(len(block["regions"]) for _, blocks in native_layers for block in blocks),
                sum(block_slices_bytes(blocks) for _, blocks in layers),
                elapsed_ms,
            )

    def wait_for_pulls(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.wait_for_pulls(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA wait_for_pulls req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def push_done(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.push_done(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA push_done req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def write_stats(self, req_id: str) -> dict[str, Any]:
        return dict(self.engine.write_stats(req_id))

    def read_stats(self, req_id: str) -> dict[str, Any]:
        return dict(self.engine.read_stats(req_id))

    def fail_request(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.fail_request(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA fail_request req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def abort_request(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.abort_request(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA abort_request req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def aggregated_link_speed(self) -> int:
        return int(self.engine.aggregated_link_speed())

    def wait_done(self, req_id: str) -> None:
        start = time.perf_counter()
        try:
            return self.engine.wait_done(req_id)
        finally:
            logger.info(
                "[PdConnector] RDMA wait_done req=%s native_ms=%.3f",
                req_id,
                (time.perf_counter() - start) * 1000,
            )

    def pop_finished_sending(self) -> set[str]:
        return set(self.engine.pop_finished_sending())

    def pop_finished_recving(self) -> set[str]:
        return set(self.engine.pop_finished_recving())

    def close_request(self, req_id: str) -> None:
        return self.engine.close_request(req_id)


class V1RdmaPort(RealRdmaPort):
    """Adapter for the minimal v1 RDMA data plane used by Pega NIXL.

    The Python-facing engine API is intentionally kept isomorphic to
    ``PdRdmaEngine`` so the connector only swaps the data plane underneath.
    """

    def prepare_peer(self, peer_key: str) -> str:
        return str(self.engine.prepare_peer(peer_key))

    def complete_peer(self, peer_key: str, remote_metadata: str) -> None:
        return self.engine.complete_peer(peer_key, remote_metadata)

    def open_request_with_peer(
        self,
        req_id: str,
        handshake: PdHandshake,
        peer_key: str,
    ) -> None:
        start = time.perf_counter()
        self.engine.register_remote(
            req_id,
            json.dumps(handshake_to_dict(handshake)),
            peer_key,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "[PdConnector] RDMA v1 open_request req=%s remote_req=%s peer=%s layers=%d native_ms=%.3f",
            req_id,
            handshake.request_id,
            peer_key,
            len(handshake.layers),
            elapsed_ms,
        )


def build_rdma_port(
    vllm_config: Any,
    cuda_device: int | None,
    *,
    tp_rank: int | None = None,
    data_plane: str = "v2",
    peer_key: str | None = None,
) -> RdmaPort:
    config = getattr(vllm_config, "kv_transfer_config", None)
    enabled = _extra(config, "pegaflow.pd.rdma.enabled", _MISSING)
    if enabled is not _MISSING and not _as_bool(enabled):
        raise RuntimeError("PdConnector requires native RDMA; pegaflow.pd.rdma.enabled=false")

    _reject_legacy_rank_config(config)
    device = _extra(config, "pegaflow.pd.rdma.device", "cuda")
    resolved_cuda_device = int(cuda_device or 0)
    resolved_tp_rank = _tp_rank(vllm_config) if tp_rank is None else int(tp_rank)
    rank_config = _rank_rdma_config(
        config,
        resolved_tp_rank,
        cuda_device=resolved_cuda_device,
    )
    if data_plane == "v1":
        try:
            from pegaflow.pegaflow import PdRdmaV1Engine
        except ImportError as exc:
            raise RuntimeError(
                "Pega NIXL requires native RDMA extension pegaflow.pegaflow"
            ) from exc
        except AttributeError as exc:
            raise RuntimeError("pegaflow.pegaflow does not expose PdRdmaV1Engine") from exc
        qps_per_peer = int(_extra(config, "pegaflow.pd.rdma.v1.qps_per_peer", 1))
        engine = PdRdmaV1Engine(
            cuda_device=resolved_cuda_device,
            domains=[rank_config.nic],
            qps_per_peer=qps_per_peer,
            local_peer_key=peer_key or f"rank:{rank_config.tp_rank}",
        )
        logger.info(
            "[PdConnector] native RDMA v1 enabled tp_rank=%d cuda=%d nic=%s worker_cpu=%d qps_per_peer=%d link_speed=%s",
            rank_config.tp_rank,
            resolved_cuda_device,
            rank_config.nic,
            rank_config.worker_cpu,
            qps_per_peer,
            engine.aggregated_link_speed(),
        )
        return V1RdmaPort(engine)
    if data_plane != "v2":
        raise ValueError(f"unsupported RDMA data_plane={data_plane!r}")

    try:
        from pegaflow.pegaflow import PdRdmaEngine
    except ImportError as exc:
        raise RuntimeError("PdConnector requires native RDMA extension pegaflow.pegaflow") from exc
    except AttributeError as exc:
        raise RuntimeError("pegaflow.pegaflow does not expose PdRdmaEngine") from exc

    engine = PdRdmaEngine(
        cuda_device=resolved_cuda_device,
        numa_node=None,
        domains=[rank_config.nic],
        device=str(device),
        pin_worker_cpu=rank_config.worker_cpu,
    )
    logger.info(
        "[PdConnector] native RDMA enabled tp_rank=%d cuda=%d nic=%s worker_cpu=%d domains=%d groups=%d link_speed=%s",
        rank_config.tp_rank,
        resolved_cuda_device,
        rank_config.nic,
        rank_config.worker_cpu,
        engine.num_domains(),
        engine.num_groups(),
        engine.aggregated_link_speed(),
    )
    return RealRdmaPort(engine)


def _extra(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get_from_extra_config", None)
    if getter is not None:
        return getter(key, default)
    extra_config = getattr(config, "extra_config", None)
    if isinstance(extra_config, dict):
        return extra_config.get(key, default)
    return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class _RankRdmaConfig:
    tp_rank: int
    nic: str
    worker_cpu: int


def _tp_rank(vllm_config: Any) -> int:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0)


def _reject_legacy_rank_config(config: Any) -> None:
    legacy_keys = (
        "pegaflow.pd.rdma.cuda_device",
        "pegaflow.pd.rdma.numa_node",
        "pegaflow.pd.rdma.domains",
        "pegaflow.pd.rdma.pin_worker_cpu",
    )
    present = [key for key in legacy_keys if _extra(config, key, _MISSING) is not _MISSING]
    if present:
        raise RuntimeError(
            "PdConnector RDMA requires pegaflow.pd.rdma.rank_map; remove legacy keys: "
            + ", ".join(present)
        )


def _rank_rdma_config(
    config: Any,
    tp_rank: int,
    *,
    cuda_device: int | None = None,
) -> _RankRdmaConfig:
    rank_map = _extra(config, "pegaflow.pd.rdma.rank_map", _MISSING)
    if not isinstance(rank_map, dict):
        raise RuntimeError("PdConnector RDMA requires pegaflow.pd.rdma.rank_map")
    _validate_rank_map_cpus(config, rank_map)
    rank_entry = rank_map.get(str(tp_rank))
    selected_rank = tp_rank
    if (
        cuda_device is not None
        and tp_rank == 0
        and cuda_device != 0
        and str(cuda_device) in rank_map
    ):
        rank_entry = rank_map[str(cuda_device)]
        selected_rank = cuda_device
    if not isinstance(rank_entry, dict):
        known = ", ".join(sorted(str(rank) for rank in rank_map))
        raise RuntimeError(
            f"PdConnector RDMA rank_map missing tp_rank={tp_rank}; configured ranks=[{known}]"
        )
    nic = str(rank_entry.get("nic") or "")
    if not nic:
        raise RuntimeError(f"PdConnector RDMA rank_map[{tp_rank}] missing nic")
    worker_cpu = _required_rank_cpu(rank_entry, selected_rank, "worker_cpu")
    return _RankRdmaConfig(
        tp_rank=selected_rank,
        nic=nic,
        worker_cpu=worker_cpu,
    )


def _validate_rank_map_cpus(config: Any, rank_map: dict[Any, Any]) -> None:
    reserved_floor = int(_extra(config, "pegaflow.pd.rdma.reserved_cpu_floor", 16))
    seen: dict[int, str] = {}
    nics: dict[str, list[str]] = {}
    for rank, entry in rank_map.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"PdConnector RDMA rank_map[{rank}] must be an object")
        nic = str(entry.get("nic") or "")
        if nic:
            nics.setdefault(nic, []).append(str(rank))
        cpu = _required_rank_cpu(entry, rank, "worker_cpu")
        if cpu < reserved_floor:
            raise RuntimeError(
                f"PdConnector RDMA rank_map[{rank}] worker_cpu={cpu} is below reserved_cpu_floor={reserved_floor}"
            )
        owner = seen.get(cpu)
        if owner is not None:
            raise RuntimeError(
                f"PdConnector RDMA rank_map CPU {cpu} is reused by {owner} and rank {rank} worker_cpu"
            )
        seen[cpu] = f"rank {rank} worker_cpu"
    for nic, ranks in sorted(nics.items()):
        if len(ranks) > 1:
            logger.warning(
                "[PdConnector] RDMA rank_map shares nic=%s across ranks=%s",
                nic,
                ranks,
            )


def _required_rank_cpu(entry: dict[Any, Any], rank: object, field: str) -> int:
    value = entry.get(field)
    if value is None or value == "":
        raise RuntimeError(f"PdConnector RDMA rank_map[{rank}] missing {field}")
    return int(value)
