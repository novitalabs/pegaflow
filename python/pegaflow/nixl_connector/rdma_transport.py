# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PegaFlow RDMA transport adapter for the Pega NIXL connector."""

from __future__ import annotations

import logging
from typing import Any

from pegaflow.pd_connector.layout import (
    KvCacheLayout,
    LayerBlockSlices,
    block_ranges_for_remote_write,
    layout_from_tensor,
)
from pegaflow.pd_connector.metadata import (
    BlockIds,
    LayerRemoteLayout,
    PdHandshake,
    flatten_block_ids,
    handshake_from_dict,
    handshake_to_dict,
)
from pegaflow.pd_connector.rdma import RdmaPort, build_rdma_port

logger = logging.getLogger(__name__)


class PegaNixlRdmaTransport:
    """Owns PegaFlow RDMA state used by the Pega NIXL worker.

    This adapter intentionally stays below the existing ``PdDecodeConnector`` /
    ``PdPrefillConnector`` facades. It only reuses the shared RDMA dataclasses
    and the native-engine wrapper.
    """

    def __init__(
        self,
        *,
        vllm_config: Any,
        engine_id: str,
        tp_rank: int,
        tp_size: int,
        rdma: RdmaPort | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.logical_block_size = _logical_block_size(vllm_config)
        self.rdma = rdma
        self.layouts: dict[str, KvCacheLayout] = {}
        self.layer_names: list[str] = []
        self.registered_layers: dict[str, LayerRemoteLayout] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, Any],
        *,
        layer_specs: dict[str, Any] | None = None,
        expected_num_blocks: int | None = None,
    ) -> None:
        layer_specs = layer_specs or {}
        self.layouts = {
            layer_name: layout_from_tensor(
                layer_name,
                tensor,
                layer_spec=_unwrap_uniform_spec(layer_specs.get(layer_name), layer_name),
                logical_block_size=self.logical_block_size,
                expected_num_blocks=expected_num_blocks,
            )
            for layer_name, tensor in kv_caches.items()
        }
        self.layer_names = list(kv_caches)
        if self.rdma is None:
            self.rdma = build_rdma_port(
                self.vllm_config,
                _infer_cuda_device(kv_caches),
                tp_rank=self.tp_rank,
            )
        registered_layers = self.rdma.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        self.registered_layers = {
            layer.layer_name: layer for layer in registered_layers
        }
        logger.info(
            "Pega NIXL RDMA registered %d layers for engine=%s rank=%d/%d",
            len(self.registered_layers),
            self.engine_id,
            self.tp_rank,
            self.tp_size,
        )

    def build_local_handshake(
        self,
        request_id: str,
        block_ids: BlockIds | list[int] | tuple[int, ...],
        *,
        imm_id: int | None = None,
        fail_imm_id: int | None = None,
        abort_imm_id: int | None = None,
        expected_imm_count: int = 1,
    ) -> PdHandshake:
        flat_block_ids = tuple(sorted(_flatten_block_ids(block_ids)))
        if not flat_block_ids:
            raise ValueError(f"request {request_id} has no RDMA blocks to register")
        layers = tuple(
            self._registered_layer_for_blocks(layer_name, flat_block_ids)
            for layer_name in self.layer_names
        )
        return PdHandshake(
            request_id=request_id,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.logical_block_size,
            layers=layers,
            imm_id=imm_id,
            fail_imm_id=fail_imm_id,
            abort_imm_id=abort_imm_id,
            expected_imm_count=expected_imm_count,
        )

    def open_request(self, request_id: str, remote_handshake: PdHandshake) -> None:
        assert self.rdma is not None, "Pega NIXL RDMA transport is not initialized"
        self.rdma.open_request(request_id, remote_handshake)

    def push_blocks(
        self,
        *,
        request_id: str,
        remote_handshake: PdHandshake,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
    ) -> None:
        assert self.rdma is not None, "Pega NIXL RDMA transport is not initialized"
        self.open_request(request_id, remote_handshake)
        remote_map = _remote_block_map(local_block_ids, remote_block_ids)
        if not remote_map:
            logger.warning("Pega NIXL RDMA push has no mapped blocks req=%s", request_id)
            self.rdma.push_done(request_id)
            return

        for layer_idx, layer_name in enumerate(self.layer_names):
            group_idx = 0
            local_group = set(local_block_ids[group_idx]) if local_block_ids else set()
            blocks = block_ranges_for_remote_write(
                self.layouts[layer_name],
                local_group,
                remote_map,
            )
            if blocks:
                self.rdma.push_layer(request_id, layer_idx, blocks)
        self.rdma.wait_for_pushes(request_id)
        self.rdma.push_done(request_id)

    def pop_finished_sending(self) -> set[str]:
        assert self.rdma is not None, "Pega NIXL RDMA transport is not initialized"
        return self.rdma.pop_finished_sending()

    def pop_finished_recving(self) -> set[str]:
        assert self.rdma is not None, "Pega NIXL RDMA transport is not initialized"
        return self.rdma.pop_finished_recving()

    def wait_done(self, request_id: str) -> None:
        assert self.rdma is not None, "Pega NIXL RDMA transport is not initialized"
        self.rdma.wait_done(request_id)

    def close_request(self, request_id: str) -> None:
        if self.rdma is not None:
            self.rdma.close_request(request_id)

    def _registered_layer_for_blocks(
        self,
        layer_name: str,
        block_ids: tuple[int, ...],
    ) -> LayerRemoteLayout:
        registered = self.registered_layers[layer_name]
        return LayerRemoteLayout(
            layer_name=registered.layer_name,
            layer_idx=registered.layer_idx,
            block_ids=block_ids,
            regions=registered.regions,
            mr_desc=registered.mr_desc,
        )


def _logical_block_size(vllm_config: Any) -> int:
    cache_config = getattr(vllm_config, "cache_config", None)
    return int(getattr(cache_config, "block_size", 0) or 16)


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int | None:
    for tensor in kv_caches.values():
        device = getattr(tensor, "device", None)
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
        get_device = getattr(tensor, "get_device", None)
        if get_device is not None:
            try:
                return max(int(get_device()), 0)
            except RuntimeError:
                return None
    return None


def _unwrap_uniform_spec(layer_spec: Any | None, layer_name: str) -> Any | None:
    specs = getattr(layer_spec, "kv_cache_specs", None)
    if isinstance(specs, dict):
        return specs.get(layer_name)
    return layer_spec


def _flatten_block_ids(block_ids: BlockIds | list[int] | tuple[int, ...]) -> set[int]:
    if not block_ids:
        return set()
    if all(isinstance(item, int) for item in block_ids):
        return {int(item) for item in block_ids}
    return flatten_block_ids(block_ids)  # type: ignore[arg-type]


def _remote_block_map(local_block_ids: BlockIds, remote_block_ids: BlockIds) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for local_group, remote_group in zip(local_block_ids, remote_block_ids, strict=False):
        for local_id, remote_id in zip(local_group, remote_group, strict=False):
            mapping[int(local_id)] = int(remote_id)
    return mapping


def layer_block_slices_bytes(blocks: list[LayerBlockSlices]) -> int:
    return sum(region.bytes for block in blocks for region in block.regions)


def handshake_to_wire(handshake: PdHandshake) -> dict[str, Any]:
    return handshake_to_dict(handshake)


def handshake_from_wire(data: dict[str, Any]) -> PdHandshake:
    handshake = handshake_from_dict(data)
    if handshake is None:
        raise ValueError("missing Pega NIXL RDMA handshake")
    return handshake
