"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.connector.connector_metrics import PegaKVConnectorStats, PegaPromMetrics
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
    world_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient
    state_manager: "ServiceStateManager"
    is_mla: bool = False
    dcp_world_size: int = 1
    pcp_world_size: int = 1
    dcp_rank: int = 0

    @property
    def virtual_block_size(self) -> int:
        """Block size as seen by the scheduler.

        vLLM computes scheduler_block_size = block_size * dcp * pcp.
        request.block_hashes has one hash per scheduler_block_size tokens,
        so all scheduler-side arithmetic must use this value.
        """
        return self.block_size * self.dcp_world_size * self.pcp_world_size

    @property
    def effective_tp_rank(self) -> int:
        """TP rank for PegaFlow server calls.

        - MLA without DCP: 0 (data identical across TP ranks).
        - MLA with DCP: dcp_rank (each DCP rank stores different interleaved tokens).
        - Non-MLA: tp_rank (each TP rank has different KV heads, already unique).
        """
        if self.is_mla:
            return self.dcp_rank
        return self.tp_rank or 0

    @property
    def effective_tp_size(self) -> int:
        """TP size for PegaFlow server calls.

        - MLA without DCP: 1.
        - MLA with DCP: dcp_world_size.
        - Non-MLA: tp_size (unique per TP rank regardless of DCP).
        """
        if self.is_mla:
            return max(1, self.dcp_world_size)
        return self.tp_size


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]
    num_tokens: int


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


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


def parse_env_int(name: str, default: int) -> int:
    """Parse an integer from environment variable with fallback to default.

    Note: This function is typically called at module import time for class-level
    configuration. Changing the environment variable after module import will not
    affect values that were already read.

    Args:
        name: Environment variable name.
        default: Default value if env var is not set or invalid.

    Returns:
        Parsed integer value or default.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', using default %d", name, value, default)
        return default


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


def derive_namespace(
    vllm_config,
    tp_size: int,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
) -> str:
    """
    Derive namespace for storage isolation.

    Different DCP/PCP configurations produce incompatible KV data, so both
    are included as factors to prevent cross-contamination.
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
        "dcp_world_size": dcp_world_size,
        "pcp_world_size": pcp_world_size,
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


def detect_mla(vllm_config) -> bool:
    """Detect if the model uses Multi-head Latent Attention (e.g. DeepSeek V2/V3)."""
    hf_config = vllm_config.model_config.hf_text_config
    return getattr(hf_config, "kv_lora_rank", None) is not None


__all__ = [
    "ConnectorContext",
    "LoadIntent",
    "PegaConnectorMetadata",
    "PegaKVConnectorStats",
    "PegaPromMetrics",
    "SaveIntent",
    "derive_namespace",
    "detect_mla",
    "logger",
    "parse_env_int",
    "resolve_instance_id",
]
