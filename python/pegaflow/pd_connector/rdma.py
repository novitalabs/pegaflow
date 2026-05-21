"""Thin RDMA port abstraction used by the P/D connector skeleton."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.layout import BlockSlice, LayerBlockSlices
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdHandshake,
    handshake_to_dict,
)

logger = get_connector_logger()
_MISSING = object()


class RdmaPort(Protocol):
    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]: ...

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None: ...

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None: ...

    def push_done(self, req_id: str) -> None: ...

    def wait_done(self, req_id: str) -> None: ...

    def mark_done(self, req_id: str) -> None: ...

    def pop_finished_sending(self) -> set[str]: ...

    def pop_finished_recving(self) -> set[str]: ...


class NoopRdmaPort:
    """A non-blocking RDMA stub that records calls and completes immediately."""

    def __init__(self) -> None:
        self.local_layers: tuple[LayerRemoteLayout, ...] = ()
        self.registered: set[str] = set()
        self.remote_handshakes: dict[str, PdHandshake | None] = {}
        self.pushed_layers: dict[str, list[tuple[int, list[LayerBlockSlices]]]] = defaultdict(list)
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()

    def register_local_layers(
        self, layers: tuple[LayerRemoteLayout, ...]
    ) -> tuple[LayerRemoteLayout, ...]:
        self.local_layers = layers
        return layers

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None:
        self.registered.add(req_id)
        self.remote_handshakes[req_id] = handshake

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.pushed_layers[req_id].append((layer_idx, blocks))

    def push_done(self, req_id: str) -> None:
        self._finished_sending.add(req_id)

    def wait_done(self, req_id: str) -> None:
        return None

    def mark_done(self, req_id: str) -> None:
        self._finished_recving.add(req_id)

    def pop_finished_sending(self) -> set[str]:
        finished = self._finished_sending
        self._finished_sending = set()
        return finished

    def pop_finished_recving(self) -> set[str]:
        finished = self._finished_recving
        self._finished_recving = set()
        return finished


class MockRdmaPort(NoopRdmaPort):
    """Alias for now; later tests can add stricter copy semantics here."""


def _block_slice_to_native(block: BlockSlice) -> dict[str, int]:
    return {
        "block_id": block.block_id,
        "src_offset_bytes": block.src_offset_bytes,
        "bytes": block.bytes,
    }


def _layer_blocks_to_native(blocks: list[LayerBlockSlices]) -> list[dict[str, Any]]:
    return [
        {
            "k": _block_slice_to_native(block.k),
            "v": _block_slice_to_native(block.v),
        }
        for block in blocks
    ]


def _layer_to_native(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "base_addr": layer.base_addr,
        "block_bytes": layer.block_bytes,
        "block_ids": list(layer.block_ids),
        "k_block_addrs": list(layer.k_block_addrs),
        "v_block_addrs": list(layer.v_block_addrs),
        "mr_desc": layer.mr_desc,
    }


def _layer_from_native(layer: LayerRemoteLayout | dict[str, Any]) -> LayerRemoteLayout:
    if isinstance(layer, LayerRemoteLayout):
        return layer
    block_ids = tuple(int(block_id) for block_id in layer["block_ids"])
    k_block_addrs = tuple(int(addr) for addr in layer["k_block_addrs"])
    v_block_addrs = tuple(int(addr) for addr in layer["v_block_addrs"])
    assert len(block_ids) == len(k_block_addrs) == len(v_block_addrs), (
        "native RDMA layer must preserve a one-to-one block_id/K/V address mapping"
    )
    return LayerRemoteLayout(
        layer_name=str(layer["layer_name"]),
        layer_idx=int(layer["layer_idx"]),
        base_addr=int(layer["base_addr"]),
        block_bytes=int(layer["block_bytes"]),
        block_ids=block_ids,
        k_block_addrs=k_block_addrs,
        v_block_addrs=v_block_addrs,
        mr_desc=layer.get("mr_desc"),
    )


def _handshake_to_native(handshake: PdHandshake | None) -> dict[str, Any] | None:
    return handshake_to_dict(handshake)


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
        native_layers = [_layer_to_native(layer) for layer in layers]
        registered = self.engine.register_local_layers(native_layers)
        return tuple(_layer_from_native(layer) for layer in registered)

    def register_remote(self, req_id: str, handshake: PdHandshake | None = None) -> None:
        self.engine.register_remote(req_id, _handshake_to_native(handshake))

    def push_layer(
        self,
        req_id: str,
        layer_idx: int,
        blocks: list[LayerBlockSlices],
    ) -> None:
        self.engine.push_layer(req_id, layer_idx, _layer_blocks_to_native(blocks))

    def push_done(self, req_id: str) -> None:
        self.engine.push_done(req_id)

    def wait_done(self, req_id: str) -> None:
        wait_done = getattr(self.engine, "wait_done", None)
        if wait_done is None:
            return None
        return wait_done(req_id)

    def mark_done(self, req_id: str) -> None:
        mark_done = getattr(self.engine, "mark_done", None)
        if mark_done is None:
            return None
        return mark_done(req_id)

    def pop_finished_sending(self) -> set[str]:
        return set(self.engine.pop_finished_sending())

    def pop_finished_recving(self) -> set[str]:
        return set(self.engine.pop_finished_recving())


def build_rdma_port(vllm_config: Any, cuda_device: int | None) -> RdmaPort:
    config = getattr(vllm_config, "kv_transfer_config", None)
    enabled = _extra(config, "pegaflow.pd.rdma.enabled", _MISSING)
    if enabled is not _MISSING and not _as_bool(enabled):
        logger.info("[PdConnector] native RDMA disabled by kv_transfer_config")
        return NoopRdmaPort()

    try:
        from pegaflow.pegaflow import PdRdmaEngine
    except ImportError:
        if enabled is not _MISSING:
            raise
        logger.warning("[PdConnector] native RDMA extension is unavailable; using NoopRdmaPort")
        return NoopRdmaPort()
    except AttributeError as exc:
        if enabled is not _MISSING:
            raise RuntimeError("pegaflow.pegaflow does not expose PdRdmaEngine") from exc
        logger.warning("[PdConnector] native PdRdmaEngine is unavailable; using NoopRdmaPort")
        return NoopRdmaPort()

    device = _extra(config, "pegaflow.pd.rdma.device", "cuda")
    configured_cuda_device = _extra(config, "pegaflow.pd.rdma.cuda_device", None)
    numa_node = _extra(config, "pegaflow.pd.rdma.numa_node", None)
    domains = _normalize_domains(_extra(config, "pegaflow.pd.rdma.domains", None))
    pin_worker_cpu = _extra(config, "pegaflow.pd.rdma.pin_worker_cpu", None)
    pin_uvm_cpu = _extra(config, "pegaflow.pd.rdma.pin_uvm_cpu", None)
    resolved_cuda_device = int(
        configured_cuda_device if configured_cuda_device is not None else cuda_device or 0
    )
    engine = PdRdmaEngine(
        cuda_device=resolved_cuda_device,
        numa_node=_optional_int(numa_node),
        domains=domains,
        device=str(device),
        pin_worker_cpu=_optional_int(pin_worker_cpu),
        pin_uvm_cpu=_optional_int(pin_uvm_cpu),
    )
    logger.info(
        "[PdConnector] native RDMA enabled cuda=%d domains=%d groups=%d link_speed=%s",
        resolved_cuda_device,
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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _normalize_domains(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]
