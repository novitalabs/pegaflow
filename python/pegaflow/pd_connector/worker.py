"""Worker-side logic for the experimental P/D connector."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.layout import (
    BlockSlice,
    FlashAttnHndLayout,
    LayerBlockSlices,
    unique_blocks_from_slot_mapping,
)
from pegaflow.pd_connector.metadata import (
    LayerRemoteLayout,
    PdConnectorMetadata,
    PdHandshake,
    PdPrefillRequest,
    PdWorkerMetadata,
    PrefillDispatch,
    PushReqMeta,
    WaitReqMeta,
    flatten_block_ids,
    handshake_to_dict,
)
from pegaflow.pd_connector.oob import InMemoryOobPort
from pegaflow.pd_connector.rdma import RdmaPort, build_rdma_port

logger = get_connector_logger()
_SLOT_MAPPING_CACHE_STEPS = 32


class PdWorkerConnector:
    def __init__(
        self,
        vllm_config: Any,
        rdma: RdmaPort | None = None,
        oob: InMemoryOobPort | None = None,
        prefill_sender: Any | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rdma = rdma
        self._rdma_is_injected = rdma is not None
        self.oob = oob or InMemoryOobPort()
        self.engine_id = (
            getattr(getattr(vllm_config, "kv_transfer_config", None), "engine_id", None) or ""
        )
        self.tp_rank, self.tp_size = _tensor_parallel_identity(vllm_config)
        logger.info(
            "[PdConnector] worker initialized engine=%s tp_rank=%d tp_size=%d",
            self.engine_id,
            self.tp_rank,
            self.tp_size,
        )
        self.layouts: dict[str, FlashAttnHndLayout] = {}
        self.layer_names: list[str] = []
        self._registered_layers: dict[str, LayerRemoteLayout] = {}
        self._wait_reqs: dict[str, WaitReqMeta] = {}
        self._push_reqs: dict[str, PushReqMeta] = {}
        self._pending_push_chunks: set[str] = set()
        self._push_chunk_maps: dict[str, tuple[dict[int, int], bool]] = {}
        self._pending_worker_handshakes: dict[str, PdHandshake] = {}
        self._tracker = ChunkTracker()
        self._failed_blocks: set[int] = set()
        self._completed_pushes: set[str] = set()
        self._producer_finished_req_ids: set[str] = set()
        self._remote_block_offsets: dict[str, int] = {}
        self._slot_mapping_block_cache: OrderedDict[tuple[Any, ...], set[int]] = OrderedDict()
        self._forward_step_id = 0
        self._next_imm_id = 1
        self._push_sender = _AsyncLayerPushSender()
        self._push_finalizer = _AsyncPushFinalizer(self._push_sender)
        self._rdma_waiter = _AsyncRdmaDoneWaiter(self.rdma) if self.rdma is not None else None
        self._prefill_sender = prefill_sender or _AsyncPrefillSender()

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.layouts = {
            layer_name: FlashAttnHndLayout.from_tensor(layer_name, tensor)
            for layer_name, tensor in kv_caches.items()
        }
        self.layer_names = list(kv_caches.keys())
        if not self._rdma_is_injected:
            self.rdma = build_rdma_port(
                self.vllm_config,
                _infer_cuda_device(kv_caches),
                tp_rank=self.tp_rank,
            )
            self._rdma_waiter = _AsyncRdmaDoneWaiter(self.rdma)
        assert self.rdma is not None
        registered_layers = self.rdma.register_local_layers(
            tuple(
                self.layouts[layer_name].remote_layout(layer_idx)
                for layer_idx, layer_name in enumerate(self.layer_names)
            )
        )
        self._registered_layers = {layer.layer_name: layer for layer in registered_layers}
        logger.info(
            "[PdConnector] registered %d FlashAttention HND KV cache layers",
            len(self.layouts),
        )

    def start_load_kv(
        self,
        metadata: PdConnectorMetadata,
        forward_context: Any,
        **kwargs: Any,
    ) -> None:
        self._forward_step_id += 1
        logger.debug(
            "[PdConnector] worker start_load_kv metadata=%s wait_reqs=%s push_reqs=%s release=%s known_wait=%s known_push=%s",
            metadata,
            sorted(metadata.reqs_to_wait),
            sorted(metadata.reqs_to_push),
            sorted(metadata.reqs_to_release),
            sorted(self._wait_reqs),
            sorted(self._push_reqs),
        )
        assert self.rdma is not None, "PdConnector RDMA port is not initialized"
        assert self._rdma_waiter is not None, "PdConnector RDMA waiter is not initialized"
        for req_id, req in metadata.reqs_to_wait.items():
            if req_id in self._wait_reqs:
                logger.info("[PdConnector] D wait req=%s already registered", req_id)
                continue
            self._wait_reqs[req_id] = req
            handshake = self._build_handshake(
                req.done_request_id,
                flatten_block_ids(req.local_block_ids),
            )
            self.oob.publish_prefill_request(
                PdPrefillRequest(
                    request_id=req.remote_request_id,
                    prompt_token_ids=req.prompt_token_ids,
                    producer_kv_transfer_params=_producer_params(
                        target_engine_id=self.engine_id,
                        target_request_id=req.done_request_id,
                        handshake=handshake,
                    ),
                    handshake=handshake,
                )
            )
            self.rdma.register_remote(req_id, handshake)
            self._rdma_waiter.submit(req_id)
            if req.prefill_url:
                self._pending_worker_handshakes[req_id] = handshake
            logger.info(
                "[PdConnector] D queued async wait req=%s remote_req=%s prefill_url=%s",
                req_id,
                req.remote_request_id,
                req.prefill_url or "<oob>",
            )

        for req_id, dispatch in metadata.prefill_dispatches.items():
            if self.tp_rank != 0:
                continue
            self._prefill_sender.submit(_prefill_task_from_dispatch(dispatch))
            logger.info(
                "[PdConnector] D rank0 submitted prefill dispatch req=%s remote_req=%s ranks=%s",
                req_id,
                dispatch.request_id,
                [handshake.tp_rank for handshake in dispatch.handshakes],
            )

        for req_id, req in metadata.reqs_to_push.items():
            self._tracker.add_request(req_id)
            prefill_request = self.oob.get_prefill_request(req.target_request_id)
            handshake = self._select_push_handshake(req)
            if handshake is None and prefill_request is not None:
                handshake = prefill_request.handshake
                req = replace(req, handshake=handshake)
            self._push_reqs[req_id] = req
            self._pending_push_chunks.add(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self.rdma.register_remote(req_id, handshake)
            logger.info(
                "[PdConnector] P queued async push req=%s target_req=%s",
                req_id,
                req.target_request_id,
            )

        for req_id in metadata.reqs_to_release:
            logger.debug("[PdConnector] worker release req=%s", req_id)
            self._wait_reqs.pop(req_id, None)
            self._push_reqs.pop(req_id, None)
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self._pending_worker_handshakes.pop(req_id, None)
            self._tracker.remove(req_id)
            close_request = getattr(self.rdma, "close_request", None)
            if close_request is not None:
                close_request(req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        assert layer_name in self.layouts, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        return None

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        layout = self.layouts.get(layer_name)
        assert layout is not None, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self.layouts)}"
        )
        # Re-assert the runtime tensor. CUDA graph or backend changes must not
        # silently swap in a different layout.
        runtime_layout = FlashAttnHndLayout.from_tensor(layer_name, kv_layer)
        assert runtime_layout.shape == layout.shape, (
            f"PdConnector KV shape changed for {layer_name}: "
            f"registered={layout.shape} runtime={runtime_layout.shape}"
        )
        if not self._push_reqs:
            return
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            logger.warning(
                "[PdConnector] P save skipped layer=%s because attn_metadata has no slot_mapping",
                layer_name,
            )
            return
        touched_blocks = self._cached_unique_blocks_from_slot_mapping(
            slot_mapping,
            layout.block_size,
        )
        if not touched_blocks:
            return
        self._push_touched_blocks(layer_name, touched_blocks)

    def wait_for_save(self) -> None:
        logger.debug(
            "[PdConnector] worker wait_for_save push_reqs=%s pending_chunks=%s chunk_maps=%s tracker=%s",
            sorted(self._push_reqs),
            sorted(self._pending_push_chunks),
            sorted(self._push_chunk_maps),
            sorted(self._tracker._requests),
        )
        return None

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        logger.debug(
            "[PdConnector] worker get_finished enter finished_req_ids=%s wait_reqs=%s push_reqs=%s",
            sorted(finished_req_ids),
            sorted(self._wait_reqs),
            sorted(self._push_reqs),
        )
        self._producer_finished_req_ids.update(finished_req_ids)
        finished_sending = self.rdma.pop_finished_sending()
        self._completed_pushes.update(finished_sending)
        finished_recving = self.rdma.pop_finished_recving()
        releasable_sending = self._completed_pushes & self._producer_finished_req_ids
        for req_id in releasable_sending:
            self._completed_pushes.discard(req_id)
            self._producer_finished_req_ids.discard(req_id)
            self._push_reqs.pop(req_id, None)
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self._tracker.remove(req_id)
            self._remote_block_offsets = {
                key: value for key, value in self._remote_block_offsets.items() if key != req_id
            }
        for req_id in finished_recving:
            self._wait_reqs.pop(req_id, None)
            close_request = getattr(self.rdma, "close_request", None)
            if close_request is not None:
                close_request(req_id)
        logger.debug(
            "[PdConnector] worker get_finished exit sending=%s recving=%s remaining_wait=%s remaining_push=%s",
            sorted(releasable_sending),
            sorted(finished_recving),
            sorted(self._wait_reqs),
            sorted(self._push_reqs),
        )
        return releasable_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = self._failed_blocks
        self._failed_blocks = set()
        return failed

    def build_connector_worker_meta(self) -> PdWorkerMetadata | None:
        if not self._pending_worker_handshakes:
            return None
        handshakes = {
            req_id: {handshake.tp_rank: handshake}
            for req_id, handshake in self._pending_worker_handshakes.items()
        }
        self._pending_worker_handshakes = {}
        return PdWorkerMetadata(handshakes=handshakes)

    def shutdown(self) -> None:
        self._wait_reqs.clear()
        self._push_reqs.clear()
        self._pending_push_chunks.clear()
        self._push_chunk_maps.clear()
        self._slot_mapping_block_cache.clear()
        self._completed_pushes.clear()
        self._producer_finished_req_ids.clear()
        self._pending_worker_handshakes.clear()
        self._remote_block_offsets.clear()
        self._push_finalizer.close()
        self._push_sender.close()
        if self._rdma_waiter is not None:
            self._rdma_waiter.close()
        close = getattr(self._prefill_sender, "close", None)
        if close is not None:
            close()

    def _rdma(self) -> RdmaPort:
        assert self.rdma is not None, "PdConnector RDMA port is not initialized"
        return self.rdma

    def _layer_idx(self, layer_name: str) -> int:
        try:
            return self.layer_names.index(layer_name)
        except ValueError as exc:
            raise AssertionError(f"unknown layer {layer_name}") from exc

    def _cached_unique_blocks_from_slot_mapping(
        self,
        slot_mapping: Any,
        block_size: int,
    ) -> set[int]:
        key = _slot_mapping_cache_key(slot_mapping, block_size)
        key = (self._forward_step_id, *key)
        cached = self._slot_mapping_block_cache.get(key)
        if cached is not None:
            self._slot_mapping_block_cache.move_to_end(key)
            return cached
        start = time.perf_counter()
        blocks = unique_blocks_from_slot_mapping(slot_mapping, block_size)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._slot_mapping_block_cache[key] = blocks
        self._slot_mapping_block_cache.move_to_end(key)
        while len(self._slot_mapping_block_cache) > _SLOT_MAPPING_CACHE_STEPS:
            self._slot_mapping_block_cache.popitem(last=False)
        logger.info(
            "[PdConnector] P extracted slot_mapping blocks blocks=%d latency_ms=%.3f",
            len(blocks),
            elapsed_ms,
        )
        return blocks

    def _build_handshake(self, req_id: str, block_ids: set[int]) -> PdHandshake:
        imm_id = self._alloc_imm_id()
        return PdHandshake(
            request_id=req_id,
            engine_id=self.engine_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=next(iter(self.layouts.values())).block_size,
            kv_layout="HND",
            layers=tuple(
                self._remote_layout_with_mr_desc(layer_name, layer_idx, block_ids)
                for layer_idx, layer_name in enumerate(self.layer_names)
            ),
            imm_id=imm_id,
        )

    def _alloc_imm_id(self) -> int:
        imm_id = self._next_imm_id
        self._next_imm_id += 1
        if self._next_imm_id > 0xFFFF_FFFF:
            self._next_imm_id = 1
        return imm_id

    def _push_touched_blocks(self, layer_name: str, touched_blocks: set[int]) -> None:
        layer_idx = self._layer_idx(layer_name)
        layout = self.layouts[layer_name]
        for req_id, req in list(self._push_reqs.items()):
            req = self._push_reqs.get(req_id)
            if req is None:
                continue
            if self._tracker.is_done(req_id):
                continue
            req_blocks = flatten_block_ids(req.local_block_ids)
            if not req_blocks:
                continue
            selected_blocks = req_blocks & touched_blocks
            if not selected_blocks:
                continue
            remote_block_ids, all_chunks_seen = self._push_chunk_maps.get(req_id, ({}, False))
            if not remote_block_ids:
                remote_block_ids, all_chunks_seen = self._remote_block_id_map(
                    req_id,
                    req,
                    selected_blocks,
                )
                self._push_chunk_maps[req_id] = (remote_block_ids, all_chunks_seen)
            assert selected_blocks.issubset(remote_block_ids), (
                "PdConnector selected blocks must match the current registered push chunk; "
                f"req={req_id} selected={sorted(selected_blocks)} "
                f"mapped={sorted(remote_block_ids)}"
            )
            block_slices = self._block_ranges_for_remote_write(
                layout,
                selected_blocks,
                remote_block_ids,
            )
            self._push_layer_async(
                req_id,
                req,
                layer_idx,
                block_slices,
                num_blocks=len(selected_blocks),
            )
            self._tracker.mark_blocks_pushed(req_id, layer_idx, selected_blocks)
            if layer_idx != len(self.layer_names) - 1:
                continue
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            chunk_complete = self._tracker.has_pushed_all_blocks(
                req_id,
                selected_blocks,
                num_layers=len(self.layer_names),
            )
            if not (chunk_complete and all_chunks_seen):
                logger.info(
                    "[PdConnector] P pushed RDMA chunk req=%s target_req=%s layers=%d blocks=%d",
                    req_id,
                    req.target_request_id,
                    len(self.layer_names),
                    len(selected_blocks),
                )
                continue
            self._tracker.mark_done(req_id)
            self._push_finalizer.submit(
                _PushFinalizeTask(
                    rdma=self.rdma,
                    req_id=req_id,
                    target_request_id=req.target_request_id,
                    num_layers=len(self.layer_names),
                    num_blocks=len(selected_blocks),
                )
            )
            logger.info(
                "[PdConnector] P queued RDMA finalize req=%s target_req=%s layers=%d blocks=%d source=save_kv_layer",
                req_id,
                req.target_request_id,
                len(self.layer_names),
                len(selected_blocks),
            )

    def _push_layer_async(
        self,
        req_id: str,
        req: PushReqMeta,
        layer_idx: int,
        block_slices: list[LayerBlockSlices],
        *,
        num_blocks: int,
    ) -> None:
        assert self.rdma is not None, "PdConnector RDMA port is not initialized"
        self._push_sender.submit(
            _LayerPushTask(
                rdma=self.rdma,
                req_id=req_id,
                target_request_id=req.target_request_id,
                layer_idx=layer_idx,
                block_slices=block_slices,
                num_blocks=num_blocks,
            )
        )

    def _select_push_handshake(self, req: PushReqMeta) -> PdHandshake | None:
        if req.handshake is not None:
            return req.handshake
        if not req.handshakes:
            return None
        for handshake in req.handshakes:
            if handshake.tp_rank == self.tp_rank:
                return handshake
        raise AssertionError(
            f"PdConnector missing handshake for tp_rank={self.tp_rank}; "
            f"available={[handshake.tp_rank for handshake in req.handshakes]}"
        )

    def _remote_block_id_map(
        self,
        req_id: str,
        req: PushReqMeta,
        local_block_ids: set[int],
    ) -> tuple[dict[int, int], bool]:
        handshake = self._select_push_handshake(req)
        if handshake is None or not handshake.layers:
            return {block_id: block_id for block_id in local_block_ids}, True
        remote_block_ids = handshake.layers[0].block_ids
        for layer in handshake.layers[1:]:
            assert layer.block_ids == remote_block_ids, (
                "PdConnector expects one decode block-id layout shared by all layers; "
                f"layer=0 blocks={list(remote_block_ids)} layer={layer.layer_idx} "
                f"blocks={list(layer.block_ids)}"
            )
        ordered_local = sorted(local_block_ids)
        if len(ordered_local) == len(remote_block_ids):
            self._remote_block_offsets[req_id] = len(remote_block_ids)
            return dict(zip(ordered_local, remote_block_ids, strict=True)), True

        offset = self._remote_block_offsets.get(req_id, 0)
        next_offset = offset + len(ordered_local)
        assert next_offset <= len(remote_block_ids), (
            "PdConnector P/D block count mismatch "
            f"offset={offset} local_blocks={ordered_local} remote_blocks={list(remote_block_ids)}"
        )
        remote_chunk = remote_block_ids[offset:next_offset]
        self._remote_block_offsets[req_id] = next_offset
        return dict(zip(ordered_local, remote_chunk, strict=True)), next_offset == len(
            remote_block_ids
        )

    def _has_pushed_all_remote_blocks(
        self,
        req_id: str,
        req: PushReqMeta,
        local_block_ids: set[int],
    ) -> bool:
        handshake = self._select_push_handshake(req)
        if handshake is None or not handshake.layers:
            return self._tracker.has_pushed_all_blocks(
                req_id,
                local_block_ids,
                num_layers=len(self.layer_names),
            )
        return self._remote_block_offsets.get(req_id, 0) >= len(handshake.layers[0].block_ids)

    @staticmethod
    def _block_slices_for_remote_write(
        layout: FlashAttnHndLayout,
        *,
        local_block_id: int,
        remote_block_id: int,
    ) -> LayerBlockSlices:
        local = layout.block_slices(local_block_id)
        return LayerBlockSlices(
            k=replace(local.k, block_id=remote_block_id),
            v=replace(local.v, block_id=remote_block_id),
        )

    @staticmethod
    def _block_ranges_for_remote_write(
        layout: FlashAttnHndLayout,
        local_block_ids: set[int],
        remote_block_ids: dict[int, int],
    ) -> list[LayerBlockSlices]:
        sorted_local_blocks = sorted(local_block_ids)
        if not sorted_local_blocks:
            return []

        ranges: list[LayerBlockSlices] = []
        start_local = sorted_local_blocks[0]
        prev_local = start_local
        start_remote = remote_block_ids[start_local]
        prev_remote = start_remote
        count = 1

        for local_block_id in sorted_local_blocks[1:]:
            remote_block_id = remote_block_ids[local_block_id]
            if local_block_id == prev_local + 1 and remote_block_id == prev_remote + 1:
                prev_local = local_block_id
                prev_remote = remote_block_id
                count += 1
                continue
            ranges.append(
                PdWorkerConnector._block_range_for_remote_write(
                    layout,
                    local_block_id=start_local,
                    remote_block_id=start_remote,
                    count=count,
                )
            )
            start_local = prev_local = local_block_id
            start_remote = prev_remote = remote_block_id
            count = 1

        ranges.append(
            PdWorkerConnector._block_range_for_remote_write(
                layout,
                local_block_id=start_local,
                remote_block_id=start_remote,
                count=count,
            )
        )
        return ranges

    @staticmethod
    def _block_range_for_remote_write(
        layout: FlashAttnHndLayout,
        *,
        local_block_id: int,
        remote_block_id: int,
        count: int,
    ) -> LayerBlockSlices:
        local = layout.block_slices(local_block_id)
        bytes_total = layout.block_bytes * count
        return LayerBlockSlices(
            k=BlockSlice(
                block_id=remote_block_id,
                src_offset_bytes=local.k.src_offset_bytes,
                bytes=bytes_total,
            ),
            v=BlockSlice(
                block_id=remote_block_id,
                src_offset_bytes=local.v.src_offset_bytes,
                bytes=bytes_total,
            ),
        )

    def _remote_layout_with_mr_desc(
        self,
        layer_name: str,
        layer_idx: int,
        block_ids: set[int],
    ) -> LayerRemoteLayout:
        layout = self.layouts[layer_name].remote_layout(layer_idx, block_ids)
        registered = self._registered_layers.get(layer_name)
        if registered is None:
            return layout
        return replace(layout, mr_desc=registered.mr_desc)


def _slot_mapping_cache_key(slot_mapping: Any, block_size: int) -> tuple[Any, ...]:
    data_ptr = getattr(slot_mapping, "data_ptr", None)
    if callable(data_ptr):
        numel = getattr(slot_mapping, "numel", None)
        shape = getattr(slot_mapping, "shape", ())
        return (
            "tensor",
            int(data_ptr()),
            int(numel()) if callable(numel) else None,
            tuple(int(dim) for dim in shape),
            block_size,
        )
    return ("object", id(slot_mapping), block_size)


@dataclass(frozen=True)
class _LayerPushTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    layer_idx: int
    block_slices: list[LayerBlockSlices]
    num_blocks: int


@dataclass(frozen=True)
class _PushFinalizeTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    num_layers: int
    num_blocks: int


class _AsyncLayerPushSender:
    def __init__(self) -> None:
        self._queue: queue.Queue[_LayerPushTask | None] = queue.Queue()
        self._condition = threading.Condition()
        self._inflight = 0
        self._inflight_by_req: dict[str, int] = {}
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="pd-rdma-push-sender", daemon=True)
        self._thread.start()

    def submit(self, task: _LayerPushTask) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            self._inflight += 1
            self._inflight_by_req[task.req_id] = self._inflight_by_req.get(task.req_id, 0) + 1
        self._queue.put(task)

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def wait_req(self, req_id: str) -> None:
        with self._condition:
            while self._inflight_by_req.get(req_id, 0) > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                _run_layer_push(task)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            finally:
                if task is not None:
                    with self._condition:
                        self._inflight -= 1
                        remaining = self._inflight_by_req.get(task.req_id, 0) - 1
                        if remaining > 0:
                            self._inflight_by_req[task.req_id] = remaining
                        else:
                            self._inflight_by_req.pop(task.req_id, None)
                        self._condition.notify_all()
                self._queue.task_done()


def _run_layer_push(task: _LayerPushTask) -> None:
    bytes_total = sum(block.k.bytes + block.v.bytes for block in task.block_slices)
    start = time.perf_counter()
    task.rdma.push_layer(task.req_id, task.layer_idx, task.block_slices)
    elapsed_s = time.perf_counter() - start
    bandwidth_gbps = (bytes_total * 8 / elapsed_s / 1e9) if elapsed_s > 0 else 0.0
    logger.info(
        "[PdConnector] P RDMA push req=%s target_req=%s layer=%d blocks=%d ranges=%d bytes=%d latency_ms=%.3f bandwidth_gbps=%.2f",
        task.req_id,
        task.target_request_id,
        task.layer_idx,
        task.num_blocks,
        len(task.block_slices),
        bytes_total,
        elapsed_s * 1000,
        bandwidth_gbps,
    )


class _AsyncPushFinalizer:
    def __init__(self, push_sender: _AsyncLayerPushSender) -> None:
        self._push_sender = push_sender
        self._queue: queue.Queue[_PushFinalizeTask | None] = queue.Queue()
        self._condition = threading.Condition()
        self._submitted: set[str] = set()
        self._inflight = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="pd-rdma-push-finalizer", daemon=True
        )
        self._thread.start()

    def submit(self, task: _PushFinalizeTask) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            if task.req_id in self._submitted:
                return
            self._submitted.add(task.req_id)
            self._inflight += 1
        self._queue.put(task)

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                start = time.perf_counter()
                self._push_sender.wait_req(task.req_id)
                task.rdma.wait_for_pushes(task.req_id)
                task.rdma.push_done(task.req_id)
                elapsed_s = time.perf_counter() - start
                logger.info(
                    "[PdConnector] P finished RDMA push req=%s target_req=%s layers=%d blocks=%d finalize_ms=%.3f source=finalizer",
                    task.req_id,
                    task.target_request_id,
                    task.num_layers,
                    task.num_blocks,
                    elapsed_s * 1000,
                )
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
            finally:
                if task is not None:
                    with self._condition:
                        self._submitted.discard(task.req_id)
                        self._inflight -= 1
                        self._condition.notify_all()
                self._queue.task_done()


class _AsyncRdmaDoneWaiter:
    def __init__(self, rdma: RdmaPort) -> None:
        self.rdma = rdma
        self._queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        self._submitted: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="pd-rdma-done-waiter", daemon=True)
        self._thread.start()

    def submit(self, req_id: str) -> None:
        if req_id in self._submitted:
            return
        self._submitted.add(req_id)
        self._queue.put((req_id, time.time_ns()))

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            req_id, submit_ts_ns = item
            try:
                self.rdma.wait_done(req_id)
                done_ts_ns = time.time_ns()
                logger.info(
                    "[PdConnector] D received RDMA done req=%s wait_ms=%.3f ts_ns=%d",
                    req_id,
                    (done_ts_ns - submit_ts_ns) / 1_000_000,
                    done_ts_ns,
                )
            except Exception:
                logger.exception("[PdConnector] D RDMA done wait failed req=%s", req_id)


@dataclass(frozen=True)
class _PrefillHttpTask:
    request_id: str
    prefill_url: str
    model: str
    prompt_token_ids: tuple[int, ...]
    max_tokens: int
    kv_transfer_params: dict[str, Any]


class _AsyncPrefillSender:
    def __init__(self) -> None:
        self._queue: queue.Queue[_PrefillHttpTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="pd-prefill-sender", daemon=True)
        self._thread.start()

    def submit(self, task: _PrefillHttpTask) -> None:
        self._queue.put(task)

    def close(self) -> None:
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                _post_prefill_request(task)
            finally:
                self._queue.task_done()


def _post_prefill_request(task: _PrefillHttpTask) -> None:
    url = task.prefill_url.rstrip("/") + "/v1/completions"
    start_ts_ns = time.time_ns()
    body = {
        "model": task.model,
        "prompt": list(task.prompt_token_ids),
        "max_tokens": task.max_tokens,
        "temperature": 0,
        "stream": False,
        "request_id": task.request_id,
        "kv_transfer_params": task.kv_transfer_params,
    }
    payload = json.dumps(body).encode()
    logger.info(
        "[PdConnector] D -> P prefill request req=%s url=%s tokens=%d payload_bytes=%d target_req=%s ts_ns=%d",
        task.request_id,
        url,
        len(task.prompt_token_ids),
        len(payload),
        task.kv_transfer_params.get("target_request_id"),
        start_ts_ns,
    )
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            response_body = response.read()
            end_ts_ns = time.time_ns()
            logger.info(
                "[PdConnector] D -> P prefill completed req=%s status=%s bytes=%d latency_ms=%.3f ts_ns=%d",
                task.request_id,
                response.status,
                len(response_body),
                (end_ts_ns - start_ts_ns) / 1_000_000,
                end_ts_ns,
            )
    except HTTPError as exc:
        response_body = exc.read()
        logger.error(
            "[PdConnector] D -> P prefill failed req=%s status=%s body=%s",
            task.request_id,
            exc.code,
            response_body[:512],
        )
    except URLError:
        logger.exception(
            "[PdConnector] D -> P prefill connection failed req=%s url=%s",
            task.request_id,
            url,
        )


def _prefill_task_from_dispatch(dispatch: PrefillDispatch) -> _PrefillHttpTask:
    params = _producer_params(
        dispatch.target_engine_id,
        dispatch.target_request_id,
    )
    params["pd_handshakes"] = [handshake_to_dict(handshake) for handshake in dispatch.handshakes]
    return _PrefillHttpTask(
        request_id=dispatch.request_id,
        prefill_url=dispatch.prefill_url,
        model=dispatch.model,
        prompt_token_ids=dispatch.prompt_token_ids,
        max_tokens=dispatch.max_tokens,
        kv_transfer_params=params,
    )


def _producer_params(
    target_engine_id: str,
    target_request_id: str,
    handshake: PdHandshake | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "do_remote_prefill_sender": True,
        "target_engine_id": target_engine_id,
        "target_request_id": target_request_id,
    }
    if handshake is not None:
        params["pd_handshake"] = handshake_to_dict(handshake)
    return params


def _infer_cuda_device(kv_caches: dict[str, Any]) -> int | None:
    for tensor in kv_caches.values():
        device = getattr(tensor, "device", None)
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
    return None


def _tensor_parallel_identity(vllm_config: Any) -> tuple[int, int]:
    try:
        return (
            int(get_tensor_model_parallel_rank()),
            int(get_tensor_model_parallel_world_size()),
        )
    except Exception:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        return (
            int(getattr(parallel_config, "tensor_parallel_rank", 0) or 0),
            int(getattr(parallel_config, "tensor_parallel_size", 1) or 1),
        )
