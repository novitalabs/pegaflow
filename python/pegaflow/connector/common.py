"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import enum
import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.logging_utils import get_connector_logger
from pegaflow.pegaflow import EngineRpcClient

if TYPE_CHECKING:
    from pegaflow.connector.state_manager import ServiceStateManager

logger = get_connector_logger()


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""

    instance_id: str
    namespace: str
    block_size: int
    num_layers: int
    tp_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient
    state_manager: "ServiceStateManager"


class RequestPhase(enum.Enum):
    """Lifecycle phase of a request in the KV connector."""

    LOOKUP = "lookup"  # Waiting for lookup result from external storage
    LOADING = "loading"  # Need to load KV from external storage
    ACTIVE = "active"  # Actively generating (may be saving concurrently)
    DRAINING = "draining"  # Generation done, waiting for async save to complete
    DONE = "done"  # Fully completed


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    num_tokens: int


@dataclass(slots=True)
class LoadState:
    """
    Mutable state for an in-progress load operation.

    Lifecycle:
    - Created by on_lookup() when cache hit is detected
    - Updated by on_alloc() with allocated block IDs
    - Consumed by consume_load_intent() which returns LoadIntent and clears state
    """

    hit_blocks: int
    computed_blocks: int
    allocated_blocks: list[int] = field(default_factory=list)
    external_tokens: int = 0

    def to_intent(
        self,
        block_hashes: tuple[bytes, ...],
        block_size: int,
    ) -> LoadIntent | None:
        """
        Convert to LoadIntent if conditions are met.

        Returns None if:
        - No external tokens to load
        - All hits are already computed (prefix cache)
        - No blocks to load after accounting for computed blocks
        """
        if self.external_tokens <= 0 or self.hit_blocks <= self.computed_blocks:
            return None

        num_blocks = min(self.hit_blocks, len(self.allocated_blocks), len(block_hashes))
        load_blocks = num_blocks - self.computed_blocks
        if load_blocks <= 0:
            return None

        start = self.computed_blocks
        return LoadIntent(
            block_ids=tuple(self.allocated_blocks[start : start + load_blocks]),
            block_hashes=block_hashes[start : start + load_blocks],
            num_tokens=load_blocks * block_size,
        )


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


class RequestTracker:
    """
    Tracks the KV cache state for a single request.

    Load state lifecycle:
    - on_lookup() creates LoadState (or None if no hit)
    - on_alloc() updates LoadState with allocated blocks
    - consume_load_intent() returns LoadIntent and clears LoadState
    - Preemption: next on_lookup() replaces stale LoadState
    """

    __slots__ = (
        "request_id",
        "_block_hashes",
        "_block_size",
        "_load",
        "_allocated_blocks",
        "_scheduled_tokens",
        "_stored_blocks",
        "_total_layers",
        "_saved_layers",
        "_finished",
    )

    def __init__(
        self,
        request_id: str,
        block_hashes: list[bytes],
        block_size: int,
        num_layers: int,
    ):
        self.request_id = request_id
        self._block_hashes = tuple(block_hashes)
        self._block_size = block_size
        self._load: LoadState | None = None
        self._allocated_blocks: list[int] = []
        self._scheduled_tokens: int = 0
        self._stored_blocks: int = 0
        self._total_layers = num_layers
        self._saved_layers: int = 0
        self._finished: bool = False

    @property
    def phase(self) -> RequestPhase:
        if self._load is not None:
            return RequestPhase.LOADING
        if not self._finished:
            return RequestPhase.ACTIVE
        if self._saved_layers < self._total_layers:
            return RequestPhase.DRAINING
        return RequestPhase.DONE

    @property
    def num_blocks(self) -> int:
        return len(self._block_hashes)

    def on_lookup(self, hit_blocks: int, computed_blocks: int) -> None:
        """New lookup = fresh load state. Handles preemption implicitly."""
        self._load = (
            LoadState(hit_blocks=hit_blocks, computed_blocks=computed_blocks)
            if hit_blocks > computed_blocks
            else None
        )
        self._allocated_blocks = []

    def on_alloc(self, block_ids: list[int], num_external_tokens: int) -> None:
        self._allocated_blocks.extend(block_ids)
        if self._load is not None:
            self._load.allocated_blocks.extend(block_ids)
            if num_external_tokens > 0:
                self._load.external_tokens = num_external_tokens

    def consume_load_intent(self) -> LoadIntent | None:
        load, self._load = self._load, None
        if load is None:
            return None
        return load.to_intent(self._block_hashes, self._block_size)

    def on_scheduled(self, num_tokens: int) -> None:
        self._scheduled_tokens += num_tokens

    def on_layer_saved(self) -> None:
        self._saved_layers += 1

    def on_finished(self) -> None:
        self._finished = True

    def consume_save_intent(self) -> SaveIntent | None:
        saveable = min(
            len(self._block_hashes),
            len(self._allocated_blocks),
            self._scheduled_tokens // self._block_size,
        )
        new_blocks = saveable - self._stored_blocks
        if new_blocks <= 0:
            return None

        start = self._stored_blocks
        self._stored_blocks = start + new_blocks
        return SaveIntent(
            block_ids=tuple(self._allocated_blocks[start : self._stored_blocks]),
            block_hashes=self._block_hashes[start : self._stored_blocks],
        )

    def should_hold_blocks(self) -> bool:
        return (
            self._finished and self._stored_blocks > 0 and self._saved_layers < self._total_layers
        )

    def is_done(self) -> bool:
        return self.phase == RequestPhase.DONE

    def __repr__(self) -> str:
        return (
            f"RequestTracker({self.request_id}, {self.phase.value}, "
            f"load={self._load}, alloc={len(self._allocated_blocks)}, "
            f"stored={self._stored_blocks}, saved={self._saved_layers}/{self._total_layers})"
        )


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, saves={len(self.save_intents)})"
        )


def resolve_instance_id(vllm_config, dp_rank_suffix: bool = True) -> str:
    """Resolve or generate connector instance_id with optional DP rank suffix."""
    instance_id = vllm_config.kv_transfer_config.engine_id
    if instance_id:
        logger.info("[PegaKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get("PEGAFLOW_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.info(
            "[PegaKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id,
        )

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.info(
                    "[PegaKVConnector] Appended DP rank to instance_id: %s (dp_size=%d, local_dp_rank=%d)",
                    instance_id,
                    parallel_config.data_parallel_size,
                    local_dp_rank,
                )

    return instance_id


def derive_namespace(vllm_config, tp_size: int) -> str:
    """
    Derive namespace for storage isolation.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config

    factors = {
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "tp_size": tp_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


__all__ = [
    "ConnectorContext",
    "LoadIntent",
    "LoadState",
    "PegaConnectorMetadata",
    "RequestPhase",
    "RequestTracker",
    "SaveIntent",
    "derive_namespace",
    "logger",
    "resolve_instance_id",
]
