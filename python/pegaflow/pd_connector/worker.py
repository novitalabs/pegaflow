"""Worker-side logic for the experimental P/D connector."""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any

from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.kv_params import ProducerKvParams
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
    PdWorkerMetadata,
    PushReqMeta,
    WaitReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.prefill import AsyncPrefillSender, PrefillHttpTask
from pegaflow.pd_connector.rdma import RdmaPort, build_rdma_port

logger = get_connector_logger()
_SLOT_MAPPING_CACHE_STEPS = 32


class PdWorkerConnector:
    def __init__(
        self,
        vllm_config: Any,
        rdma: RdmaPort | None = None,
        prefill_sender: Any | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.rdma = rdma
        self._rdma_is_injected = rdma is not None
        self.engine_id = getattr(vllm_config.kv_transfer_config, "engine_id", None) or ""
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
        self._peer_layouts: dict[int, dict[str, FlashAttnHndLayout]] = {}
        self._peer_mr_descs: dict[int, dict[str, Any]] = {}
        self._tracker = ChunkTracker()
        self._failed_blocks: set[int] = set()
        self._completed_pushes: set[str] = set()
        self._producer_finished_req_ids: set[str] = set()
        self._remote_block_offsets: dict[str, int] = {}
        self._push_traces: dict[str, _PushTrace] = {}
        self._slot_mapping_block_cache: OrderedDict[tuple[Any, ...], set[int]] = OrderedDict()
        self._forward_step_id = 0
        self._next_imm_id = 1
        self._push_sender = _AsyncLayerPushSender()
        self._push_finalizer = _AsyncPushFinalizer(self._push_sender)
        self._rdma_waiter = _AsyncRdmaDoneWaiter(self.rdma) if self.rdma is not None else None
        self._prefill_sender = prefill_sender or AsyncPrefillSender()

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
        self._gather_peer_info()
        logger.info(
            "[PdConnector] registered %d FlashAttention HND KV cache layers, gathered %d peer ranks",
            len(self.layouts),
            len(self._peer_layouts),
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
            self.rdma.open_request(req_id, handshake)
            self._rdma_waiter.submit(req_id)
            logger.info(
                "[PdConnector] D queued async wait req=%s remote_req=%s prefill_url=%s rank=%d blocks=%d ts_ns=%d",
                req_id,
                req.remote_request_id,
                req.prefill_url or "<oob>",
                self.tp_rank,
                len(flatten_block_ids(req.local_block_ids)),
                time.time_ns(),
            )
            if req.prefill_url and self.tp_rank == 0:
                self._dispatch_prefill(req, handshake.imm_id)

        for req_id, req in metadata.reqs_to_push.items():
            self._tracker.add_request(req_id)
            handshake = self._select_push_handshake(req)
            self._push_reqs[req_id] = req
            trace = self._push_traces.setdefault(req_id, _PushTrace(queued_ts_ns=time.time_ns()))
            queued_ts_ns = time.time_ns()
            trace.chunk_queued_ts_ns = queued_ts_ns
            self._pending_push_chunks.add(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self.rdma.open_request(req_id, handshake)
            logger.info(
                "[PdConnector] P queued async push req=%s target_req=%s rank=%d blocks=%d layers=%d ts_ns=%d",
                req_id,
                req.target_request_id,
                self.tp_rank,
                len(flatten_block_ids(req.local_block_ids)),
                len(req.handshakes[0].layers)
                if req.handshakes
                else (len(handshake.layers) if handshake else 0),
                queued_ts_ns,
            )

        for req_id in metadata.reqs_to_release:
            logger.debug("[PdConnector] worker release req=%s", req_id)
            self._wait_reqs.pop(req_id, None)
            self._push_reqs.pop(req_id, None)
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            self._push_traces.pop(req_id, None)
            self._tracker.remove(req_id)
            self.rdma.close_request(req_id)

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
        slot_mapping = attn_metadata.slot_mapping
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
            self._push_traces.pop(req_id, None)
            self._tracker.remove(req_id)
            self._remote_block_offsets = {
                key: value for key, value in self._remote_block_offsets.items() if key != req_id
            }
        for req_id in finished_recving:
            self._wait_reqs.pop(req_id, None)
            self.rdma.close_request(req_id)
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
        return None

    def shutdown(self) -> None:
        self._wait_reqs.clear()
        self._push_reqs.clear()
        self._pending_push_chunks.clear()
        self._push_chunk_maps.clear()
        self._slot_mapping_block_cache.clear()
        self._completed_pushes.clear()
        self._producer_finished_req_ids.clear()
        self._remote_block_offsets.clear()
        self._push_traces.clear()
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
            layers=tuple(
                self._remote_layout_with_mr_desc(layer_name, layer_idx, block_ids)
                for layer_idx, layer_name in enumerate(self.layer_names)
            ),
            imm_id=imm_id,
        )

    def _gather_peer_info(self) -> None:
        mr_descs = {name: layer.mr_desc for name, layer in self._registered_layers.items()}
        if self.tp_size <= 1:
            self._peer_layouts = {0: self.layouts}
            self._peer_mr_descs = {0: mr_descs}
            return
        try:
            gathered = _all_gather_peer_info(self.layouts, mr_descs, self.tp_size)
        except Exception:
            logger.warning(
                "[PdConnector] all_gather_object unavailable, rank 0 dispatch limited to local rank",
            )
            self._peer_layouts = {self.tp_rank: self.layouts}
            self._peer_mr_descs = {self.tp_rank: mr_descs}
            return
        for rank, (layouts, descs) in enumerate(gathered):
            self._peer_layouts[rank] = layouts
            self._peer_mr_descs[rank] = descs

    def _dispatch_prefill(self, req: WaitReqMeta, imm_id: int) -> None:
        block_ids = flatten_block_ids(req.local_block_ids)
        all_handshakes = self._build_all_rank_handshakes(
            req.done_request_id,
            block_ids,
            imm_id,
        )
        params = ProducerKvParams(
            target_engine_id=self.engine_id,
            target_request_id=req.done_request_id,
            handshakes=all_handshakes,
        )
        task = PrefillHttpTask(
            request_id=req.remote_request_id,
            prefill_url=req.prefill_url,
            model=req.model,
            prompt_token_ids=req.prompt_token_ids,
            max_tokens=req.prefill_max_tokens,
            kv_transfer_params=params.to_dict(),
        )
        self._prefill_sender.submit(task)
        logger.info(
            "[PdConnector] D rank0 dispatched prefill req=%s remote_req=%s ranks=%d ts_ns=%d",
            req.remote_request_id,
            req.done_request_id,
            len(all_handshakes),
            time.time_ns(),
        )

    def _build_all_rank_handshakes(
        self,
        req_id: str,
        block_ids: set[int],
        imm_id: int,
    ) -> tuple[PdHandshake, ...]:
        result = []
        block_size = next(iter(self.layouts.values())).block_size
        for rank in range(self.tp_size):
            peer_layouts = self._peer_layouts[rank]
            peer_mr_descs = self._peer_mr_descs[rank]
            layers = tuple(
                replace(
                    peer_layouts[name].remote_layout(layer_idx, block_ids),
                    mr_desc=peer_mr_descs.get(name),
                )
                for layer_idx, name in enumerate(self.layer_names)
            )
            result.append(
                PdHandshake(
                    request_id=req_id,
                    engine_id=self.engine_id,
                    tp_rank=rank,
                    tp_size=self.tp_size,
                    block_size=block_size,
                    layers=layers,
                    imm_id=imm_id,
                )
            )
        return tuple(result)

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
            trace = self._push_traces.setdefault(
                req_id,
                _PushTrace(queued_ts_ns=time.time_ns()),
            )
            save_ts_ns = time.time_ns()
            if trace.first_save_ts_ns is None:
                trace.first_save_ts_ns = save_ts_ns
            if trace.chunk_first_save_ts_ns is None:
                trace.chunk_first_save_ts_ns = save_ts_ns
                logger.info(
                    "[PdConnector] P first save_kv_layer chunk req=%s target_req=%s chunk=%d layer=%d blocks=%d schedule_to_chunk_first_save_ms=%.3f schedule_to_request_first_save_ms=%.3f ts_ns=%d",
                    req_id,
                    req.target_request_id,
                    trace.chunk_count + 1,
                    layer_idx,
                    len(selected_blocks),
                    _elapsed_ms(trace.chunk_queued_ts_ns, save_ts_ns),
                    _elapsed_ms(trace.queued_ts_ns, trace.first_save_ts_ns),
                    save_ts_ns,
                )
            remote_block_ids, all_chunks_seen = self._push_chunk_maps.get(req_id, ({}, False))
            submit_start_ns = time.perf_counter_ns()
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
            rdma_bytes = _block_slices_bytes(block_slices)
            self._push_layer_async(
                req_id,
                req,
                layer_idx,
                block_slices,
                num_blocks=len(selected_blocks),
            )
            submit_ns = time.perf_counter_ns() - submit_start_ns
            trace.kv_submit_ns += submit_ns
            trace.chunk_kv_submit_ns += submit_ns
            trace.rdma_bytes += rdma_bytes
            trace.chunk_rdma_bytes += rdma_bytes
            trace.layer_submit_count += 1
            trace.chunk_layer_submit_count += 1
            self._tracker.mark_blocks_pushed(req_id, layer_idx, selected_blocks)
            if layer_idx != len(self.layer_names) - 1:
                continue
            trace.chunk_count += 1
            trace.last_save_ts_ns = time.time_ns()
            self._pending_push_chunks.discard(req_id)
            self._push_chunk_maps.pop(req_id, None)
            chunk_complete = self._tracker.has_pushed_all_blocks(
                req_id,
                selected_blocks,
                num_layers=len(self.layer_names),
            )
            if not (chunk_complete and all_chunks_seen):
                logger.info(
                    "[PdConnector] P observed prefill chunk req=%s target_req=%s chunk=%d layers=%d blocks=%d all_chunks_seen=%s chunk_forward_ms=%.3f request_forward_ms=%.3f chunk_kv_submit_ms=%.3f request_kv_submit_ms=%.3f chunk_observed_gbps=%.2f request_observed_gbps=%.2f chunk_layer_submits=%d request_layer_submits=%d ts_ns=%d",
                    req_id,
                    req.target_request_id,
                    trace.chunk_count,
                    len(self.layer_names),
                    len(selected_blocks),
                    all_chunks_seen,
                    _elapsed_ms(trace.chunk_first_save_ts_ns, trace.last_save_ts_ns),
                    _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                    trace.chunk_kv_submit_ns / 1_000_000,
                    trace.kv_submit_ns / 1_000_000,
                    _gbps(
                        trace.chunk_rdma_bytes, trace.chunk_first_save_ts_ns, trace.last_save_ts_ns
                    ),
                    _gbps(trace.rdma_bytes, trace.first_save_ts_ns, trace.last_save_ts_ns),
                    trace.chunk_layer_submit_count,
                    trace.layer_submit_count,
                    trace.last_save_ts_ns,
                )
                trace.reset_chunk()
                continue
            self._tracker.mark_done(req_id)
            finalize_ts_ns = time.time_ns()
            logger.info(
                "[PdConnector] P observed prefill final chunk req=%s target_req=%s chunks=%d layers=%d blocks=%d schedule_to_last_save_ms=%.3f chunk_forward_ms=%.3f request_forward_ms=%.3f chunk_kv_submit_ms=%.3f request_kv_submit_ms=%.3f chunk_observed_gbps=%.2f request_observed_gbps=%.2f chunk_layer_submits=%d request_layer_submits=%d ts_ns=%d",
                req_id,
                req.target_request_id,
                trace.chunk_count,
                len(self.layer_names),
                len(selected_blocks),
                (finalize_ts_ns - trace.queued_ts_ns) / 1_000_000,
                _elapsed_ms(trace.chunk_first_save_ts_ns, trace.last_save_ts_ns),
                _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                trace.chunk_kv_submit_ns / 1_000_000,
                trace.kv_submit_ns / 1_000_000,
                _gbps(trace.chunk_rdma_bytes, trace.chunk_first_save_ts_ns, trace.last_save_ts_ns),
                _gbps(trace.rdma_bytes, trace.first_save_ts_ns, trace.last_save_ts_ns),
                trace.chunk_layer_submit_count,
                trace.layer_submit_count,
                finalize_ts_ns,
            )
            self._push_finalizer.submit(
                _PushFinalizeTask(
                    rdma=self.rdma,
                    req_id=req_id,
                    target_request_id=req.target_request_id,
                    num_layers=len(self.layer_names),
                    num_blocks=len(selected_blocks),
                    chunk_count=trace.chunk_count,
                    first_save_ts_ns=trace.first_save_ts_ns,
                    last_save_ts_ns=trace.last_save_ts_ns,
                    finalize_queued_ts_ns=finalize_ts_ns,
                    kv_submit_ns=trace.kv_submit_ns,
                    rdma_bytes=trace.rdma_bytes,
                    layer_submit_count=trace.layer_submit_count,
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
                queued_ts_ns=time.time_ns(),
            )
        )

    def _select_push_handshake(self, req: PushReqMeta) -> PdHandshake:
        assert req.handshakes, (
            f"PdConnector push request has no handshakes; target_req={req.target_request_id}"
        )
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
        if not handshake.layers:
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
    return (
        int(slot_mapping.data_ptr()),
        int(slot_mapping.numel()),
        tuple(int(dim) for dim in slot_mapping.shape),
        block_size,
    )


def _elapsed_ms(start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if start_ts_ns is None or end_ts_ns is None:
        return 0.0
    return (end_ts_ns - start_ts_ns) / 1_000_000


def _gbps(bytes_total: int, start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if bytes_total <= 0 or start_ts_ns is None or end_ts_ns is None or end_ts_ns <= start_ts_ns:
        return 0.0
    return bytes_total * 8 / ((end_ts_ns - start_ts_ns) / 1_000_000_000) / 1e9


def _block_slices_bytes(block_slices: list[LayerBlockSlices]) -> int:
    return sum(block.k.bytes + block.v.bytes for block in block_slices)


@dataclass(frozen=True)
class _LayerPushTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    layer_idx: int
    block_slices: list[LayerBlockSlices]
    num_blocks: int
    queued_ts_ns: int


@dataclass
class _PushTrace:
    queued_ts_ns: int
    chunk_queued_ts_ns: int | None = None
    first_save_ts_ns: int | None = None
    chunk_first_save_ts_ns: int | None = None
    last_save_ts_ns: int | None = None
    kv_submit_ns: int = 0
    chunk_kv_submit_ns: int = 0
    rdma_bytes: int = 0
    chunk_rdma_bytes: int = 0
    layer_submit_count: int = 0
    chunk_layer_submit_count: int = 0
    chunk_count: int = 0

    def reset_chunk(self) -> None:
        self.chunk_queued_ts_ns = None
        self.chunk_first_save_ts_ns = None
        self.chunk_kv_submit_ns = 0
        self.chunk_rdma_bytes = 0
        self.chunk_layer_submit_count = 0


@dataclass(frozen=True)
class _PushFinalizeTask:
    rdma: RdmaPort
    req_id: str
    target_request_id: str
    num_layers: int
    num_blocks: int
    chunk_count: int
    first_save_ts_ns: int | None
    last_save_ts_ns: int | None
    finalize_queued_ts_ns: int
    kv_submit_ns: int
    rdma_bytes: int
    layer_submit_count: int


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
    bytes_total = _block_slices_bytes(task.block_slices)
    start_ts_ns = time.time_ns()
    start = time.perf_counter()
    task.rdma.push_layer(task.req_id, task.layer_idx, task.block_slices)
    elapsed_s = time.perf_counter() - start
    done_ts_ns = time.time_ns()
    bandwidth_gbps = (bytes_total * 8 / elapsed_s / 1e9) if elapsed_s > 0 else 0.0
    logger.info(
        "[PdConnector] P RDMA push req=%s target_req=%s layer=%d blocks=%d ranges=%d bytes=%d queue_wait_ms=%.3f latency_ms=%.3f submit_to_done_ms=%.3f bandwidth_gbps=%.2f ts_ns=%d",
        task.req_id,
        task.target_request_id,
        task.layer_idx,
        task.num_blocks,
        len(task.block_slices),
        bytes_total,
        _elapsed_ms(task.queued_ts_ns, start_ts_ns),
        elapsed_s * 1000,
        _elapsed_ms(task.queued_ts_ns, done_ts_ns),
        bandwidth_gbps,
        done_ts_ns,
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
                done_ts_ns = time.time_ns()
                elapsed_s = time.perf_counter() - start
                logger.info(
                    "[PdConnector] P finished RDMA push req=%s target_req=%s chunks=%d layers=%d blocks=%d rdma_bytes=%d finalize_ms=%.3f finalize_queued_to_imm_ms=%.3f observed_forward_ms=%.3f kv_submit_ms=%.3f last_save_to_imm_ms=%.3f first_save_to_imm_ms=%.3f last_save_to_imm_gbps=%.2f first_save_to_imm_gbps=%.2f layer_submits=%d source=finalizer ts_ns=%d",
                    task.req_id,
                    task.target_request_id,
                    task.chunk_count,
                    task.num_layers,
                    task.num_blocks,
                    task.rdma_bytes,
                    elapsed_s * 1000,
                    _elapsed_ms(task.finalize_queued_ts_ns, done_ts_ns),
                    _elapsed_ms(task.first_save_ts_ns, task.last_save_ts_ns),
                    task.kv_submit_ns / 1_000_000,
                    _elapsed_ms(task.last_save_ts_ns, done_ts_ns),
                    _elapsed_ms(task.first_save_ts_ns, done_ts_ns),
                    _gbps(task.rdma_bytes, task.last_save_ts_ns, done_ts_ns),
                    _gbps(task.rdma_bytes, task.first_save_ts_ns, done_ts_ns),
                    task.layer_submit_count,
                    done_ts_ns,
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


def _all_gather_peer_info(
    layouts: dict[str, FlashAttnHndLayout],
    mr_descs: dict[str, Any],
    tp_size: int,
) -> list[tuple[dict[str, FlashAttnHndLayout], dict[str, Any]]]:
    import torch.distributed as dist

    gathered: list[tuple[dict[str, FlashAttnHndLayout], dict[str, Any]] | None] = [None] * tp_size
    dist.all_gather_object(gathered, (layouts, mr_descs))
    return gathered  # type: ignore[return-value]
