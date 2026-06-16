"""Data carriers for the P-side (prefill) push pipeline.

These are pure value types shared between ``PrefillHandler`` and the async
push/finalize executors. Kept in their own module so the handler and executor
files stay focused on behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pegaflow.pd_connector.layout import LayerBlockSlices
from pegaflow.pd_connector.rdma import RdmaPort


class _SkipPushRank(Exception):
    pass


@dataclass(frozen=True)
class _LayerPushTask:
    rdma: RdmaPort
    req_id: str
    layer_idx: int
    block_slices: list[LayerBlockSlices]
    event: Any = None


@dataclass(frozen=True)
class _PreparedTargetPush:
    physical_req_id: str
    block_slices: list[LayerBlockSlices]


@dataclass(frozen=True)
class _PreparedLayerPush:
    req_blocks: frozenset[int]
    pushed_req_blocks: frozenset[int]
    target_pushes: tuple[_PreparedTargetPush, ...]
    rdma_bytes: int
    all_chunks_seen: bool


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


__all__ = [
    "_SkipPushRank",
    "_LayerPushTask",
    "_PreparedTargetPush",
    "_PreparedLayerPush",
    "_PushTrace",
    "_PushFinalizeTask",
]
