"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from pegaflow.connector.connector_metrics import PegaKVConnectorStats, PegaPromMetrics
from pegaflow.logging_utils import get_connector_logger
from pegaflow.pegaflow import EngineRpcClient

if TYPE_CHECKING:
    from pegaflow.connector.state_manager import ServiceStateManager

logger = get_connector_logger()


class PegaConnectorMode(str, Enum):
    """Read/write behavior for the PegaFlow connector."""

    READ_WRITE = "read_write"
    SAVE_ONLY = "save_only"

    @classmethod
    def from_config(cls, value: object) -> "PegaConnectorMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for mode in cls:
                if normalized == mode.value:
                    return mode
        allowed = ", ".join(mode.value for mode in cls)
        raise ValueError(f"Unsupported pegaflow.mode {value!r}; expected one of: {allowed}")


@dataclass(frozen=True)
class ConnectorContext:
    """Shared configuration for scheduler/worker connectors."""

    instance_id: str
    namespace: str
    block_size: int
    tp_size: int
    world_size: int
    tp_rank: int | None
    device_id: int | None
    engine_client: EngineRpcClient
    state_manager: "ServiceStateManager"
    is_mla: bool = False
    transfer_backend: str = "direct"
    dcp_world_size: int = 1
    pcp_world_size: int = 1
    dcp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    mode: PegaConnectorMode = PegaConnectorMode.READ_WRITE
    wait_for_full_prefix: bool = False

    @property
    def read_enabled(self) -> bool:
        return self.mode is PegaConnectorMode.READ_WRITE

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

    block_ids_by_group: tuple[tuple[int, ...], ...]
    lease: bytes
    num_tokens: int


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[bytes, ...]


@dataclass(frozen=True)
class CacheGroupLayout:
    """Stable vLLM cache-group order shared by scheduler and worker."""

    layer_names: tuple[tuple[str, ...], ...]
    hash_group_index: int
    has_recurrent_state: bool

    @classmethod
    def from_config(cls, kv_cache_config) -> "CacheGroupLayout":
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if not groups:
            return cls(layer_names=((),), hash_group_index=0, has_recurrent_state=False)

        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

        block_sizes = {group.kv_cache_spec.block_size for group in groups}
        if len(groups) > 1 and len(block_sizes) != 1:
            raise RuntimeError(
                "PegaFlow HMA requires cache groups with identical logical block sizes"
            )

        hash_group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if isinstance(group.kv_cache_spec, FullAttentionSpec)
            ),
            None,
        )
        if hash_group_index is None and len(groups) > 1:
            raise RuntimeError(
                "PegaFlow requires a dense FullAttention cache group for block hashes"
            )
        hash_group_index = hash_group_index or 0

        return cls(
            layer_names=tuple(tuple(group.layer_names) for group in groups),
            hash_group_index=hash_group_index,
            has_recurrent_state=any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups),
        )

    @property
    def group_count(self) -> int:
        return len(self.layer_names)

    def layer_to_group(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group_index, names in enumerate(self.layer_names):
            for name in names:
                if name in result:
                    raise RuntimeError(f"KV cache layer belongs to multiple groups: {name}")
                result[name] = group_index
        return result


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
        preempted_req_ids: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        self.preempted_req_ids: set[str] = preempted_req_ids or set()

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
        logger.debug("[PegaKVConnector] Using kv_transfer_config.engine_id: %s", instance_id)
        return instance_id

    instance_id = vllm_config.instance_id or os.environ.get("PEGAFLOW_INSTANCE_ID", "")
    if not instance_id:
        instance_id = uuid.uuid4().hex
        logger.debug(
            "[PegaKVConnector] No instance_id from vLLM; generated fallback %s",
            instance_id,
        )

    if dp_rank_suffix:
        parallel_config = vllm_config.parallel_config
        if parallel_config.data_parallel_size > 1:
            local_dp_rank = parallel_config.data_parallel_rank_local
            if local_dp_rank is not None:
                instance_id = f"{instance_id}_dp{local_dp_rank}"
                logger.debug(
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
    cross_layer_blocks: bool = False,
) -> str:
    """
    Derive namespace for storage isolation.

    Every factor that changes the on-storage KV block layout must be included,
    otherwise two incompatible layouts share one namespace and a load hits the
    server-side slot-count guard (`stored block has N slots but instance
    expects M`). Beyond DCP/PCP and cross-layer, this covers:

    - `pp_size`: the pipeline-parallel degree decides how the model's layers
      are split across stages, so a given server registers a different layer
      subset (and slot count) per degree.
    - `mla_layer_split_kv_cache`: MLA layer-split registration shards each
      block's slots across ranks, a different per-block layout than the
      default full-slot registration.
    """
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    additional_config = getattr(vllm_config, "additional_config", None) or {}

    factors = {
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "tp_size": tp_size,
        "pp_size": vllm_config.parallel_config.pipeline_parallel_size,
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
        "dcp_world_size": dcp_world_size,
        "pcp_world_size": pcp_world_size,
        "cross_layer_blocks": cross_layer_blocks,
        "mla_layer_split_kv_cache": bool(additional_config.get("mla_layer_split_kv_cache", False)),
    }

    factor_str = str(sorted(factors.items()))
    hash_suffix = hashlib.sha256(factor_str.encode()).hexdigest()[:8]
    return f"{hash_suffix}"


def detect_mla(vllm_config) -> bool:
    """Detect if the model uses Multi-head Latent Attention (e.g. DeepSeek V2/V3)."""
    hf_config = vllm_config.model_config.hf_text_config
    return getattr(hf_config, "kv_lora_rank", None) is not None


_TRANSFER_BACKENDS = ("direct", "kernel")


def resolve_transfer_backend(is_mla: bool, override: str | None) -> str:
    """Pick the engine's H2D/D2H backend for this model.

    MLA models save/load many small, highly fragmented slots where the kernel
    backend's single launch beats one cuMemcpyAsync per slot; everything else
    defaults to direct (best bandwidth for few/large transfers). A non-empty
    `override` (from `pegaflow.transfer_backend`) wins, and an unknown value is
    rejected rather than silently falling back.
    """
    if override is None:
        return "kernel" if is_mla else "direct"
    normalized = override.strip().lower()
    if normalized not in _TRANSFER_BACKENDS:
        allowed = ", ".join(_TRANSFER_BACKENDS)
        raise ValueError(
            f"Unsupported pegaflow.transfer_backend {override!r}; expected one of: {allowed}"
        )
    return normalized


__all__ = [
    "ConnectorContext",
    "LoadIntent",
    "PegaConnectorMode",
    "PegaConnectorMetadata",
    "PegaKVConnectorStats",
    "PegaPromMetrics",
    "SaveIntent",
    "derive_namespace",
    "detect_mla",
    "logger",
    "parse_env_int",
    "resolve_instance_id",
    "resolve_transfer_backend",
]
