"""P-side (prefill) worker logic — pushes KV via RDMA."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pd_connector.chunk_tracker import ChunkTracker
from pegaflow.pd_connector.layout import (
    BlockRegionSlice,
    FlashAttnHndLayout,
    LayerBlockSlices,
    block_ranges_for_remote_write,
    block_slices_bytes,
    layout_from_tensor,
)
from pegaflow.pd_connector.layout_mapping import (
    HeadSlice,
    PushLayoutPlan,
    PushTargetPlan,
    build_push_layout_plan,
)
from pegaflow.pd_connector.metadata import (
    RELEASE_CONSUMER_ABORT,
    RELEASE_PRODUCER_ABORT,
    RELEASE_PRODUCER_PREEMPTED,
    LayerRemoteLayout,
    PdHandshake,
    PushReqMeta,
    flatten_block_ids,
)
from pegaflow.pd_connector.rdma import RdmaPort

if TYPE_CHECKING:
    from pegaflow.pd_connector.worker import PdWorkerBase

logger = get_connector_logger()


class PrefillHandler:
    """Handles P-side (prefill) requests: KV push via RDMA."""

    def __init__(self, worker: PdWorkerBase) -> None:
        self._w = worker
        self._push_reqs: dict[str, PushReqMeta] = {}
        self._pending_push_chunks: set[str] = set()
        self._push_chunk_maps: dict[tuple[str, int], tuple[dict[str, dict[int, int]], bool]] = {}
        self._tracker = ChunkTracker()
        self._completed_pushes: set[str] = set()
        self._completed_physical_pushes: set[str] = set()
        self._producer_finished_req_ids: set[str] = set()
        self._remote_block_offsets: dict[str, int] = {}
        self._push_plans: dict[str, PushLayoutPlan] = {}
        self._physical_to_logical: dict[str, str] = {}
        self._logical_to_physical: dict[str, tuple[str, ...]] = {}
        self._push_traces: dict[str, _PushTrace] = {}
        self._skipped_pushes = 0
        self._push_sender = _AsyncLayerPushSender(metrics=worker.metrics)
        self._push_finalizer = _AsyncPushFinalizer(self._push_sender, metrics=worker.metrics)

    def process_push_reqs(self, reqs_to_push: dict[str, PushReqMeta]) -> None:
        for req_id, req in reqs_to_push.items():
            self._tracker.add_request(req_id)
            plan = self._build_push_layout_plan(req)
            if plan.should_skip:
                self._skipped_pushes += 1
                self._w.metrics.record_prefill_skipped_push()
                self._completed_pushes.add(req_id)
                logger.info(
                    "[PdConnector] P skipped push req=%s target_req=%s rank=%d skipped_total=%d",
                    req_id,
                    req.target_request_id,
                    self._w.tp_rank,
                    self._skipped_pushes,
                )
                continue
            self._push_reqs[req_id] = req
            self._w.metrics.set_prefill_active_pushes(len(self._push_reqs))
            self._push_plans[req_id] = plan
            self._push_traces.setdefault(req_id, _PushTrace(queued_ts_ns=time.time_ns()))
            self._pending_push_chunks.add(req_id)
            self._clear_push_chunk_maps(req_id)
            physical_req_ids = self._physical_req_ids(req_id, plan)
            self._logical_to_physical[req_id] = physical_req_ids
            for physical_req_id, target in zip(physical_req_ids, plan.targets, strict=True):
                self._physical_to_logical[physical_req_id] = req_id
                local_layout = next(iter(self._w.layouts.values()), None)
                self._w.rdma.open_request(
                    physical_req_id,
                    _target_handshake_for_local_layout(
                        target,
                        local_layout,
                    ),
                )
            logger.info(
                "[PdConnector] P queued push req=%s target_req=%s rank=%d physical_reqs=%d blocks=%d",
                req_id,
                req.target_request_id,
                self._w.tp_rank,
                len(physical_req_ids),
                len(flatten_block_ids(req.local_block_ids)),
            )

    def release(self, req_id: str, reason: str = RELEASE_CONSUMER_ABORT) -> tuple[str, ...]:
        had_push_state = (
            req_id in self._push_reqs
            or req_id in self._pending_push_chunks
            or any(key[0] == req_id for key in self._push_chunk_maps)
            or req_id in self._push_plans
            or req_id in self._logical_to_physical
            or req_id in self._push_traces
            or self._tracker.has_request(req_id)
        )
        self._push_reqs.pop(req_id, None)
        self._w.metrics.set_prefill_active_pushes(len(self._push_reqs))
        if had_push_state:
            self._w.metrics.record_prefill_release()
        self._pending_push_chunks.discard(req_id)
        self._clear_push_chunk_maps(req_id)
        self._push_plans.pop(req_id, None)
        physical_req_ids = self._logical_to_physical.pop(req_id, ())
        if physical_req_ids:
            self._push_sender.cancel_many(physical_req_ids)
            self._push_finalizer.cancel_many(physical_req_ids)
            if reason == RELEASE_CONSUMER_ABORT:
                self._abort_physical_requests(physical_req_ids)
            elif reason in (RELEASE_PRODUCER_ABORT, RELEASE_PRODUCER_PREEMPTED):
                self._fail_physical_requests(physical_req_ids)
        for physical_req_id in physical_req_ids:
            self._physical_to_logical.pop(physical_req_id, None)
            self._completed_physical_pushes.discard(physical_req_id)
        self._push_traces.pop(req_id, None)
        self._tracker.remove(req_id)
        return physical_req_ids

    def _abort_physical_requests(self, physical_req_ids: tuple[str, ...]) -> None:
        for physical_req_id in physical_req_ids:
            try:
                self._drain_physical_request(physical_req_id)
                self._w.rdma.abort_request(physical_req_id)
            except Exception:
                logger.exception(
                    "[PdConnector] P failed to notify decode abort ack req=%s",
                    physical_req_id,
                )

    def _fail_physical_requests(self, physical_req_ids: tuple[str, ...]) -> None:
        for physical_req_id in physical_req_ids:
            try:
                self._drain_physical_request(physical_req_id)
                self._w.rdma.fail_request(physical_req_id)
            except Exception:
                logger.exception(
                    "[PdConnector] P failed to notify decode abort req=%s",
                    physical_req_id,
                )

    def _drain_physical_request(self, physical_req_id: str) -> None:
        self._push_sender.wait_req(physical_req_id)
        self._w.rdma.wait_for_pushes(physical_req_id)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if not self._push_reqs:
            return
        layout = self._w.layouts.get(layer_name)
        assert layout is not None, (
            f"PdConnector saw unknown layer {layer_name}; registered={list(self._w.layouts)}"
        )
        # Re-assert the runtime tensor. CUDA graph or backend changes must not
        # silently swap in a different layout.
        runtime_layout = layout_from_tensor(
            layer_name,
            kv_layer,
            layer_spec=self._w._layer_spec(layer_name),
            logical_block_size=self._w.logical_block_size,
            expected_num_blocks=layout.num_blocks,
        )
        assert type(runtime_layout) is type(layout), (
            f"PdConnector KV layout type changed for {layer_name}: "
            f"registered={type(layout).__name__} runtime={type(runtime_layout).__name__}"
        )
        assert runtime_layout.shape == layout.shape, (
            f"PdConnector KV shape changed for {layer_name}: "
            f"registered={layout.shape} runtime={runtime_layout.shape}"
        )
        import torch

        if torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            event = None
        self._push_pending_blocks(layer_name, event)

    def wait_for_save(self) -> None:
        logger.debug(
            "[PdConnector] worker wait_for_save push_reqs=%s pending_chunks=%s chunk_maps=%s tracker=%s",
            sorted(self._push_reqs),
            sorted(self._pending_push_chunks),
            sorted(self._push_chunk_maps),
            sorted(self._tracker._requests),
        )

    def get_finished_sending(self, finished_req_ids: set[str]) -> set[str]:
        """Return req_ids that are done sending and also finished by the producer."""
        self._producer_finished_req_ids.update(finished_req_ids)
        finished_sending = self._w.rdma.pop_finished_sending()
        self._record_finished_physical_pushes(finished_sending)
        releasable_sending = self._completed_pushes & self._producer_finished_req_ids
        for req_id in releasable_sending:
            self._completed_pushes.discard(req_id)
            self._producer_finished_req_ids.discard(req_id)
            self._push_reqs.pop(req_id, None)
            self._w.metrics.set_prefill_active_pushes(len(self._push_reqs))
            self._pending_push_chunks.discard(req_id)
            self._clear_push_chunk_maps(req_id)
            self._push_plans.pop(req_id, None)
            physical_req_ids = self._logical_to_physical.pop(req_id, ())
            for physical_req_id in dict.fromkeys(physical_req_ids):
                self._physical_to_logical.pop(physical_req_id, None)
                self._completed_physical_pushes.discard(physical_req_id)
                self._w.rdma.close_request(physical_req_id)
            self._push_traces.pop(req_id, None)
            self._tracker.remove(req_id)
            self._remote_block_offsets = {
                key: value
                for key, value in self._remote_block_offsets.items()
                if not (key == req_id or key.startswith(f"{req_id}#"))
            }
        return releasable_sending

    def shutdown(self) -> None:
        self._push_reqs.clear()
        self._w.metrics.set_prefill_active_pushes(0)
        self._pending_push_chunks.clear()
        self._push_chunk_maps.clear()
        self._completed_pushes.clear()
        self._completed_physical_pushes.clear()
        self._producer_finished_req_ids.clear()
        self._remote_block_offsets.clear()
        self._push_plans.clear()
        self._physical_to_logical.clear()
        self._logical_to_physical.clear()
        self._push_traces.clear()
        self._push_finalizer.close()
        self._push_sender.close()

    @property
    def push_reqs(self) -> dict[str, PushReqMeta]:
        return self._push_reqs

    def has_state(self) -> bool:
        return any(
            (
                self._push_reqs,
                self._pending_push_chunks,
                self._completed_pushes,
                self._completed_physical_pushes,
                self._producer_finished_req_ids,
            )
        ) or not (self._push_sender.is_idle() and self._push_finalizer.is_idle())

    def _push_pending_blocks(self, layer_name: str, event: Any) -> None:
        layer_idx = self._w._layer_idx(layer_name)
        layout = self._w.layouts[layer_name]
        for req_id, req in list(self._push_reqs.items()):
            req = self._push_reqs.get(req_id)
            if req is None:
                continue
            if self._tracker.is_done(req_id):
                continue
            req_blocks = self._w.block_ids_for_layer(req.local_block_ids, layer_name)
            if not req_blocks:
                continue
            trace = self._push_traces.setdefault(
                req_id,
                _PushTrace(queued_ts_ns=time.time_ns()),
            )
            if trace.first_save_ts_ns is None:
                trace.first_save_ts_ns = time.time_ns()
            plan = self._push_plans[req_id]
            chunk_key = (req_id, layer_idx)
            target_maps, all_chunks_seen = self._push_chunk_maps.get(chunk_key, ({}, False))
            if not target_maps:
                target_maps, all_chunks_seen = self._remote_block_id_maps(
                    req_id,
                    plan,
                    req_blocks,
                    layer_idx,
                )
                self._push_chunk_maps[chunk_key] = (target_maps, all_chunks_seen)
            assert self._w.rdma is not None
            rdma_bytes = 0
            pushed_req_blocks: set[int] = set()
            physical_req_ids = self._logical_to_physical[req_id]
            for physical_req_id, target in zip(physical_req_ids, plan.targets, strict=True):
                remote_block_ids = target_maps[physical_req_id]
                if not remote_block_ids:
                    continue
                target_req_blocks = set(remote_block_ids)
                pushed_req_blocks.update(target_req_blocks)
                block_slices = _target_block_ranges_for_remote_write(
                    layout,
                    target,
                    target_req_blocks,
                    remote_block_ids,
                )
                rdma_bytes += block_slices_bytes(block_slices)
                self._push_sender.submit(
                    _LayerPushTask(
                        rdma=self._w.rdma,
                        req_id=physical_req_id,
                        layer_idx=layer_idx,
                        block_slices=block_slices,
                        event=event,
                    )
                )
            trace.rdma_bytes += rdma_bytes
            self._tracker.mark_blocks_pushed(req_id, layer_idx, pushed_req_blocks)
            if layer_idx != len(self._w.layer_names) - 1:
                continue
            trace.chunk_count += 1
            trace.last_save_ts_ns = time.time_ns()
            self._pending_push_chunks.discard(req_id)
            all_layer_chunks_seen = self._all_layer_chunks_seen(
                req_id,
                current_layer_idx=layer_idx,
                current_all_chunks_seen=all_chunks_seen,
            )
            self._clear_push_chunk_maps(req_id)
            chunk_complete = self._tracker.has_pushed_all_blocks(
                req_id,
                self._block_ids_by_layer(req.local_block_ids),
                num_layers=len(self._w.layer_names),
            ) or all_layer_chunks_seen
            if not (chunk_complete and all_layer_chunks_seen):
                logger.info(
                    "[PdConnector] P chunk req=%s target_req=%s chunk=%d blocks=%d forward_ms=%.3f",
                    req_id,
                    req.target_request_id,
                    trace.chunk_count,
                    len(req_blocks),
                    _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                )
                continue
            self._tracker.mark_done(req_id)
            finalize_ts_ns = time.time_ns()
            ready_gbps = _gbps(trace.rdma_bytes, trace.first_save_ts_ns, trace.last_save_ts_ns)
            link_gbps = _rdma_link_gbps(self._w.rdma)
            logger.info(
                "[PdConnector] P all chunks done req=%s target_req=%s chunks=%d blocks=%d rdma_bytes=%d schedule_to_save_ms=%.3f forward_ms=%.3f gbps=%.2f link_gbps=%.2f ready_link_util_pct=%.2f ts_ns=%d",
                req_id,
                req.target_request_id,
                trace.chunk_count,
                len(req_blocks),
                trace.rdma_bytes,
                (finalize_ts_ns - trace.queued_ts_ns) / 1_000_000,
                _elapsed_ms(trace.first_save_ts_ns, trace.last_save_ts_ns),
                ready_gbps,
                link_gbps,
                _pct(ready_gbps, link_gbps),
                finalize_ts_ns,
            )
            self._push_finalizer.submit(
                _PushFinalizeTask(
                    rdma=self._w.rdma,
                    req_ids=self._logical_to_physical[req_id],
                    target_request_id=req.target_request_id,
                    num_blocks=len(req_blocks),
                    chunk_count=trace.chunk_count,
                    first_save_ts_ns=trace.first_save_ts_ns,
                    finalize_queued_ts_ns=finalize_ts_ns,
                    schedule_queued_ts_ns=trace.queued_ts_ns,
                    rdma_bytes=trace.rdma_bytes,
                )
            )

    def _build_push_layout_plan(self, req: PushReqMeta) -> PushLayoutPlan:
        assert req.handshakes, (
            f"PdConnector push request has no handshakes; target_req={req.target_request_id}"
        )
        if not self._w.layouts:
            assert self._w.use_mla, "PdConnector push requires registered KV cache layouts"
            try:
                handshake = self._select_push_handshake(req)
            except _SkipPushRank:
                return PushLayoutPlan(targets=())
            return PushLayoutPlan(targets=(PushTargetPlan(handshake=handshake, head_slices=()),))
        first_layout = next(iter(self._w.layouts.values()))
        local_heads = int(getattr(first_layout, "num_kv_heads", 1))
        first_handshake = req.handshakes[0]
        remote_heads = _remote_num_kv_heads(first_handshake, first_layout, local_heads)
        total_heads = _total_num_kv_heads(
            local_heads=local_heads,
            local_tp_size=self._w.tp_size,
            remote_heads=remote_heads,
            remote_tp_size=first_handshake.tp_size,
            use_mla=self._w.use_mla,
        )
        return build_push_layout_plan(
            prefill_tp_rank=self._w.tp_rank,
            prefill_tp_size=self._w.tp_size,
            decode_handshakes=req.handshakes,
            local_num_kv_heads=local_heads,
            remote_num_kv_heads=remote_heads,
            total_num_kv_heads=total_heads,
            use_mla=self._w.use_mla,
        )

    def _physical_req_ids(self, req_id: str, plan: PushLayoutPlan) -> tuple[str, ...]:
        if len(plan.targets) == 1:
            return (req_id,)
        return tuple(f"{req_id}#d{target.handshake.tp_rank}" for target in plan.targets)

    def _record_finished_physical_pushes(self, physical_req_ids: set[str]) -> None:
        self._completed_physical_pushes.update(physical_req_ids)
        for physical_req_id in physical_req_ids:
            logical_req_id = self._physical_to_logical.get(physical_req_id, physical_req_id)
            physical_for_logical = self._logical_to_physical.get(logical_req_id, (logical_req_id,))
            if set(physical_for_logical).issubset(self._completed_physical_pushes):
                self._completed_pushes.add(logical_req_id)

    def _select_push_handshake(self, req: PushReqMeta) -> PdHandshake:
        assert req.handshakes, (
            f"PdConnector push request has no handshakes; target_req={req.target_request_id}"
        )
        _assert_handshake_tp_consistency(req.handshakes)
        decode_tp_size = req.handshakes[0].tp_size
        if self._w.use_mla:
            assert self._w.tp_size >= decode_tp_size, (
                "PdConnector MLA heterogeneous TP requires prefill TP >= decode TP; "
                f"prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
            )
            assert self._w.tp_size % decode_tp_size == 0, (
                "PdConnector MLA heterogeneous TP requires prefill TP to be a "
                f"multiple of decode TP; prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
            )
            ratio = self._w.tp_size // decode_tp_size
            if self._w.tp_rank % ratio != 0:
                raise _SkipPushRank
            target_rank = self._w.tp_rank // ratio
            for handshake in req.handshakes:
                if handshake.tp_rank == target_rank:
                    return handshake
            raise AssertionError(
                f"PdConnector missing MLA target handshake for prefill_tp_rank={self._w.tp_rank} "
                f"decode_tp_rank={target_rank}; "
                f"available={[handshake.tp_rank for handshake in req.handshakes]}"
            )

        assert self._w.tp_size == decode_tp_size, (
            "PdConnector non-MLA requires equal P/D TP sizes; "
            f"prefill_tp={self._w.tp_size} decode_tp={decode_tp_size}"
        )
        for handshake in req.handshakes:
            if handshake.tp_rank == self._w.tp_rank:
                return handshake
        raise AssertionError(
            f"PdConnector missing handshake for tp_rank={self._w.tp_rank}; "
            f"available={[handshake.tp_rank for handshake in req.handshakes]}"
        )

    def _remote_block_id_maps(
        self,
        req_id: str,
        plan: PushLayoutPlan,
        local_block_ids: set[int],
        layer_idx: int,
    ) -> tuple[dict[str, dict[int, int]], bool]:
        result = {}
        all_chunks_seen = True
        for physical_req_id, target in zip(
            self._logical_to_physical[req_id],
            plan.targets,
            strict=True,
        ):
            remote_block_ids, target_all_chunks_seen = _remote_block_id_map_for_handshake(
                offset_key=f"{physical_req_id}#l{layer_idx}",
                handshake=target.handshake,
                local_block_ids=local_block_ids,
                layer_idx=layer_idx,
                remote_block_offsets=self._remote_block_offsets,
            )
            result[physical_req_id] = remote_block_ids
            all_chunks_seen = all_chunks_seen and target_all_chunks_seen
        return result, all_chunks_seen

    def _clear_push_chunk_maps(self, req_id: str) -> None:
        self._push_chunk_maps = {
            key: value for key, value in self._push_chunk_maps.items() if key[0] != req_id
        }

    def _block_ids_by_layer(self, block_ids: Any) -> dict[int, set[int]]:
        return {
            layer_idx: self._w.block_ids_for_layer(block_ids, layer_name)
            for layer_idx, layer_name in enumerate(self._w.layer_names)
        }

    def _all_layer_chunks_seen(
        self,
        req_id: str,
        *,
        current_layer_idx: int,
        current_all_chunks_seen: bool,
    ) -> bool:
        seen_by_layer = {
            layer_idx: all_chunks_seen
            for (chunk_req_id, layer_idx), (
                _target_maps,
                all_chunks_seen,
            ) in self._push_chunk_maps.items()
            if chunk_req_id == req_id
        }
        seen_by_layer[current_layer_idx] = current_all_chunks_seen
        return all(
            seen_by_layer.get(layer_idx, False) for layer_idx in range(len(self._w.layer_names))
        )


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _elapsed_ms(start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if start_ts_ns is None or end_ts_ns is None:
        return 0.0
    return (end_ts_ns - start_ts_ns) / 1_000_000


def _gbps(bytes_total: int, start_ts_ns: int | None, end_ts_ns: int | None) -> float:
    if bytes_total <= 0 or start_ts_ns is None or end_ts_ns is None or end_ts_ns <= start_ts_ns:
        return 0.0
    return bytes_total * 8 / ((end_ts_ns - start_ts_ns) / 1_000_000_000) / 1e9


def _pct(value: float, total: float) -> float:
    if value <= 0 or total <= 0:
        return 0.0
    return value / total * 100


def _rdma_link_gbps(rdma: RdmaPort | None) -> float:
    if rdma is None:
        return 0.0
    try:
        return rdma.aggregated_link_speed() / 1e9
    except Exception:
        logger.exception("[PdConnector] failed to read RDMA link speed")
        return 0.0


def _target_block_ranges_for_remote_write(
    layout: Any,
    target: PushTargetPlan,
    local_block_ids: set[int],
    remote_block_ids: dict[int, int],
) -> list[LayerBlockSlices]:
    if not target.head_slices:
        return block_ranges_for_remote_write(layout, local_block_ids, remote_block_ids)
    assert isinstance(layout, FlashAttnHndLayout), (
        "PdConnector head-sliced layout mapping requires FlashAttention HND layout; "
        f"layout={type(layout).__name__}"
    )
    ranges = []
    for local_block_id in sorted(local_block_ids):
        remote_block_id = remote_block_ids[local_block_id]
        regions = []
        for head_slice in target.head_slices:
            block_slice = layout.block_head_slices(
                local_block_id,
                head_slice.local_start,
                head_slice.local_end,
            )
            regions.extend(
                BlockRegionSlice(
                    block_id=remote_block_id,
                    src_offset_bytes=region.src_offset_bytes,
                    bytes=region.bytes,
                )
                for region in block_slice.regions
            )
        ranges.append(LayerBlockSlices(regions=tuple(regions)))
    return ranges


def _target_handshake_for_local_layout(
    target: PushTargetPlan,
    layout: Any,
) -> PdHandshake:
    if not target.head_slices or not target.handshake.layers:
        return target.handshake
    assert isinstance(layout, FlashAttnHndLayout), (
        "PdConnector head-sliced layout mapping requires FlashAttention HND layout; "
        f"layout={type(layout).__name__}"
    )
    head_slice = _single_remote_head_slice(target.head_slices)
    return replace(
        target.handshake,
        layers=tuple(
            _remote_head_layer(layer, head_slice, layout.head_bytes)
            for layer in target.handshake.layers
        ),
    )


def _remote_head_layer(
    layer: LayerRemoteLayout,
    head_slice: HeadSlice,
    head_bytes: int,
) -> LayerRemoteLayout:
    return replace(
        layer,
        regions=tuple(
            replace(
                region,
                base_addr=region.base_addr + head_slice.remote_start * head_bytes,
                block_len=(head_slice.remote_end - head_slice.remote_start) * head_bytes,
                block_stride=region.block_stride or region.block_len,
            )
            for region in layer.regions
        ),
    )


def _single_remote_head_slice(head_slices: tuple[HeadSlice, ...]) -> HeadSlice:
    assert head_slices, "PdConnector target has no head slices"
    assert len(head_slices) == 1, (
        "PdConnector first head-sliced version expects one contiguous head slice per target; "
        f"head_slices={head_slices}"
    )
    return head_slices[0]


def _remote_block_id_map_for_handshake(
    *,
    offset_key: str,
    handshake: PdHandshake,
    local_block_ids: set[int],
    layer_idx: int,
    remote_block_offsets: dict[str, int],
) -> tuple[dict[int, int], bool]:
    if not handshake.layers:
        return {block_id: block_id for block_id in local_block_ids}, True
    remote_layer = next(
        (layer for layer in handshake.layers if layer.layer_idx == layer_idx),
        None,
    )
    assert remote_layer is not None, (
        "PdConnector missing decode layer layout for push "
        f"layer_idx={layer_idx} available={[layer.layer_idx for layer in handshake.layers]}"
    )
    remote_block_ids = remote_layer.block_ids
    ordered_local = sorted(local_block_ids)
    if len(ordered_local) == len(remote_block_ids):
        remote_block_offsets[offset_key] = len(remote_block_ids)
        return dict(zip(ordered_local, remote_block_ids, strict=True)), True

    offset = remote_block_offsets.get(offset_key, 0)
    if offset >= len(remote_block_ids):
        return {}, True
    next_offset = min(offset + len(ordered_local), len(remote_block_ids))
    ordered_local = ordered_local[: next_offset - offset]
    remote_chunk = remote_block_ids[offset:next_offset]
    remote_block_offsets[offset_key] = next_offset
    return dict(zip(ordered_local, remote_chunk, strict=True)), next_offset == len(remote_block_ids)


def _remote_num_kv_heads(
    handshake: PdHandshake,
    local_layout: Any,
    fallback: int,
) -> int:
    if not handshake.layers or not isinstance(local_layout, FlashAttnHndLayout):
        return fallback
    layer = handshake.layers[0]
    if len(layer.regions) < 1:
        return fallback
    head_bytes = local_layout.head_bytes
    block_len = layer.regions[0].block_len
    if block_len % head_bytes != 0:
        return fallback
    return max(1, block_len // head_bytes)


def _total_num_kv_heads(
    *,
    local_heads: int,
    local_tp_size: int,
    remote_heads: int,
    remote_tp_size: int,
    use_mla: bool,
) -> int:
    if use_mla:
        return local_heads
    return min(local_heads * local_tp_size, remote_heads * remote_tp_size)


def _assert_handshake_tp_consistency(handshakes: tuple[PdHandshake, ...]) -> None:
    tp_size = handshakes[0].tp_size
    assert all(handshake.tp_size == tp_size for handshake in handshakes), (
        f"PdConnector handshakes disagree on decode TP size: "
        f"{[handshake.tp_size for handshake in handshakes]}"
    )


class _SkipPushRank(Exception):
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LayerPushTask:
    rdma: RdmaPort
    req_id: str
    layer_idx: int
    block_slices: list[LayerBlockSlices]
    event: Any = None


@dataclass
class _PushTrace:
    queued_ts_ns: int
    first_save_ts_ns: int | None = None
    last_save_ts_ns: int | None = None
    rdma_bytes: int = 0
    chunk_count: int = 0


@dataclass(frozen=True)
class _PushFinalizeTask:
    rdma: RdmaPort
    req_ids: tuple[str, ...]
    target_request_id: str
    num_blocks: int
    chunk_count: int
    first_save_ts_ns: int | None
    finalize_queued_ts_ns: int
    schedule_queued_ts_ns: int
    rdma_bytes: int


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------


class _AsyncLayerPushSender:
    def __init__(self, metrics: Any | None = None, max_workers: int = 16) -> None:
        self._metrics = metrics
        self._condition = threading.Condition()
        self._inflight = 0
        self._inflight_by_req: dict[str, int] = {}
        self._cancelled: set[str] = set()
        self._error: BaseException | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="pd-rdma-push",
        )

    def submit(self, task: _LayerPushTask) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("PdConnector RDMA push sender is closed")
            if self._error is not None:
                raise self._error
            self._inflight += 1
            self._inflight_by_req[task.req_id] = self._inflight_by_req.get(task.req_id, 0) + 1
            self._set_inflight_metric_locked()
        try:
            self._executor.submit(self._run_task, task)
        except BaseException:
            with self._condition:
                self._finish_task(task)
            raise

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

    def cancel(self, req_id: str) -> None:
        self.cancel_many((req_id,))

    def cancel_many(self, req_ids: tuple[str, ...]) -> None:
        with self._condition:
            self._cancelled.update(
                req_id for req_id in req_ids if self._inflight_by_req.get(req_id, 0) > 0
            )
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def is_idle(self) -> bool:
        with self._condition:
            return self._inflight == 0 and self._error is None

    def _run_task(self, task: _LayerPushTask) -> None:
        try:
            with self._condition:
                cancelled = task.req_id in self._cancelled
            if not cancelled:
                _run_layer_push(task)
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            with self._condition:
                self._finish_task(task)

    def _finish_task(self, task: _LayerPushTask) -> None:
        self._inflight -= 1
        remaining = self._inflight_by_req.get(task.req_id, 0) - 1
        if remaining > 0:
            self._inflight_by_req[task.req_id] = remaining
        else:
            self._inflight_by_req.pop(task.req_id, None)
            self._cancelled.discard(task.req_id)
        self._set_inflight_metric_locked()
        self._condition.notify_all()

    def _set_inflight_metric_locked(self) -> None:
        if self._metrics is not None:
            self._metrics.set_prefill_inflight_push_tasks(self._inflight)


def _run_layer_push(task: _LayerPushTask) -> None:
    if task.event is not None:
        task.event.synchronize()
    task.rdma.push_layer(task.req_id, task.layer_idx, task.block_slices)


class _AsyncPushFinalizer:
    def __init__(
        self,
        push_sender: _AsyncLayerPushSender,
        metrics: Any | None = None,
        max_workers: int = 16,
    ) -> None:
        self._push_sender = push_sender
        self._metrics = metrics
        self._condition = threading.Condition()
        self._submitted: set[tuple[str, ...]] = set()
        self._cancelled: set[str] = set()
        self._inflight = 0
        self._error: BaseException | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="pd-rdma-finalize",
        )

    def submit(self, task: _PushFinalizeTask) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("PdConnector RDMA push finalizer is closed")
            if self._error is not None:
                raise self._error
            if task.req_ids in self._submitted:
                return
            self._submitted.add(task.req_ids)
            self._inflight += 1
            self._set_inflight_metric_locked()
        try:
            self._executor.submit(self._run_task, task)
        except BaseException:
            with self._condition:
                self._finish_task(task)
            raise

    def cancel_many(self, req_ids: tuple[str, ...]) -> None:
        with self._condition:
            submitted_req_ids = {
                submitted_req_id for submitted in self._submitted for submitted_req_id in submitted
            }
            self._cancelled.update(req_id for req_id in req_ids if req_id in submitted_req_ids)
            self._condition.notify_all()

    def wait_all(self) -> None:
        with self._condition:
            while self._inflight > 0 and self._error is None:
                self._condition.wait()
            if self._error is not None:
                error = self._error
                self._error = None
                raise error

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def is_idle(self) -> bool:
        with self._condition:
            return self._inflight == 0 and self._error is None

    def _run_task(self, task: _PushFinalizeTask) -> None:
        try:
            wait_for_pushes_s = 0.0
            completed = False
            for req_id in task.req_ids:
                if self._is_cancelled(req_id):
                    continue
                self._push_sender.wait_req(req_id)
                if self._is_cancelled(req_id):
                    continue
                per_req_wait_start_ts_ns = time.time_ns()
                task.rdma.wait_for_pushes(req_id)
                wait_for_pushes_s += (time.time_ns() - per_req_wait_start_ts_ns) / 1_000_000_000
                if self._is_cancelled(req_id):
                    continue
                task.rdma.push_done(req_id)
                completed = True
            done_ts_ns = time.time_ns()
            if completed and self._metrics is not None:
                self._metrics.record_prefill_push(
                    duration_s=(done_ts_ns - task.schedule_queued_ts_ns) / 1_000_000_000,
                    first_save_to_done_s=(
                        (done_ts_ns - task.first_save_ts_ns) / 1_000_000_000
                        if task.first_save_ts_ns is not None
                        else None
                    ),
                    wait_for_pushes_s=wait_for_pushes_s,
                    blocks=task.num_blocks,
                    bytes_total=task.rdma_bytes,
                    success=True,
                )
            save_gbps = _gbps(task.rdma_bytes, task.first_save_ts_ns, done_ts_ns)
            link_gbps = _rdma_link_gbps(task.rdma)
            logger.info(
                "[PdConnector] P RDMA done reqs=%s target_req=%s chunks=%d blocks=%d "
                "rdma_bytes=%d save_to_imm_ms=%.3f schedule_to_imm_ms=%.3f "
                "save_gbps=%.2f tail_gbps=%.2f link_gbps=%.2f link_util_pct=%.2f",
                list(task.req_ids),
                task.target_request_id,
                task.chunk_count,
                task.num_blocks,
                task.rdma_bytes,
                _elapsed_ms(task.first_save_ts_ns, done_ts_ns),
                _elapsed_ms(task.schedule_queued_ts_ns, done_ts_ns),
                save_gbps,
                _gbps(task.rdma_bytes, task.finalize_queued_ts_ns, done_ts_ns),
                link_gbps,
                _pct(save_gbps, link_gbps),
            )
        except BaseException as exc:
            logger.exception(
                "[PdConnector] P finalize failed reqs=%s target_req=%s chunks=%d blocks=%d bytes=%d",
                list(task.req_ids),
                task.target_request_id,
                task.chunk_count,
                task.num_blocks,
                task.rdma_bytes,
            )
            with self._condition:
                self._error = exc
                self._condition.notify_all()
            if self._metrics is not None:
                self._metrics.record_prefill_push(
                    duration_s=(time.time_ns() - task.schedule_queued_ts_ns) / 1_000_000_000,
                    first_save_to_done_s=None,
                    wait_for_pushes_s=None,
                    blocks=task.num_blocks,
                    bytes_total=task.rdma_bytes,
                    success=False,
                )
        finally:
            with self._condition:
                self._finish_task(task)

    def _is_cancelled(self, req_id: str) -> bool:
        with self._condition:
            return req_id in self._cancelled

    def _finish_task(self, task: _PushFinalizeTask) -> None:
        self._submitted.discard(task.req_ids)
        for req_id in task.req_ids:
            self._cancelled.discard(req_id)
        self._inflight -= 1
        self._set_inflight_metric_locked()
        self._condition.notify_all()

    def _set_inflight_metric_locked(self) -> None:
        if self._metrics is not None:
            self._metrics.set_prefill_inflight_finalize_tasks(self._inflight)
