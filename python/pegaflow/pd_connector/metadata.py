"""Metadata exchanged by the experimental P/D RDMA push connector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

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
class RemoteEndpoint:
    engine_id: str
    host: str | None = None
    port: int | None = None
    tp_size: int = 1
    done_endpoint: str | None = None


@dataclass(frozen=True)
class WaitReqMeta:
    local_block_ids: BlockIds
    remote: RemoteEndpoint
    remote_request_id: str
    num_prompt_tokens: int
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PushReqMeta:
    local_block_ids: BlockIds
    target: RemoteEndpoint
    target_request_id: str
    num_prompt_tokens: int


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    base_addr: int
    block_bytes: int
    k_block_addrs: tuple[int, ...]
    v_block_addrs: tuple[int, ...]
    mr_desc: Any | None = None


@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int
    tp_size: int
    block_size: int
    kv_layout: str
    layers: tuple[LayerRemoteLayout, ...]


@dataclass(frozen=True)
class PdPrefillRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    producer_kv_transfer_params: dict[str, Any]
    handshake: PdHandshake


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

    def __repr__(self) -> str:
        return (
            "PdConnectorMetadata("
            f"wait={len(self.reqs_to_wait)}, "
            f"push={len(self.reqs_to_push)}, "
            f"release={len(self.reqs_to_release)})"
        )


@dataclass
class WorkerDoneState:
    finished_sending: set[str] = field(default_factory=set)
    finished_recving: set[str] = field(default_factory=set)
    failed_blocks: set[int] = field(default_factory=set)
