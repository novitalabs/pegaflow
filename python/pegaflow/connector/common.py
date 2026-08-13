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
class TpShardTopology:
    """Equal contiguous TP shards served by node-local PegaFlow instances."""

    endpoints: tuple[str, ...]
    global_tp_size: int
    global_world_size: int

    @classmethod
    def from_config(
        cls,
        default_endpoint: str,
        configured_endpoints: object,
        global_tp_size: int,
        global_world_size: int,
    ) -> "TpShardTopology":
        if configured_endpoints is None:
            endpoints = (default_endpoint,)
        elif not isinstance(configured_endpoints, (list, tuple)):
            raise ValueError("pegaflow.tp_shard_endpoints must be a list of endpoints")
        else:
            endpoints = tuple(configured_endpoints)

        if not endpoints or any(
            not isinstance(endpoint, str) or not endpoint for endpoint in endpoints
        ):
            raise ValueError("pegaflow.tp_shard_endpoints must contain non-empty strings")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("pegaflow.tp_shard_endpoints must not contain duplicates")
        if global_tp_size <= 0 or global_tp_size % len(endpoints) != 0:
            raise ValueError(
                f"tensor_parallel_size={global_tp_size} must be divisible by "
                f"the {len(endpoints)} PegaFlow TP shards"
            )
        if global_world_size <= 0 or global_world_size % len(endpoints) != 0:
            raise ValueError(
                f"world_size={global_world_size} must be divisible by "
                f"the {len(endpoints)} PegaFlow TP shards"
            )
        return cls(
            endpoints=endpoints,
            global_tp_size=global_tp_size,
            global_world_size=global_world_size,
        )

    @property
    def shard_count(self) -> int:
        return len(self.endpoints)

    @property
    def local_tp_size(self) -> int:
        return self.global_tp_size // self.shard_count

    @property
    def local_world_size(self) -> int:
        return self.global_world_size // self.shard_count

    def shard_index(self, tp_rank: int) -> int:
        if tp_rank < 0 or tp_rank >= self.global_tp_size:
            raise ValueError(
                f"tp_rank={tp_rank} is outside tensor_parallel_size={self.global_tp_size}"
            )
        return tp_rank // self.local_tp_size

    def local_tp_rank(self, tp_rank: int) -> int:
        return tp_rank % self.local_tp_size

    def namespace(self, base_namespace: str, shard_index: int) -> str:
        if self.shard_count == 1:
            return base_namespace
        if shard_index < 0 or shard_index >= self.shard_count:
            raise ValueError(
                f"TP shard index {shard_index} is outside shard_count={self.shard_count}"
            )
        return f"{base_namespace}:tp-shard-{shard_index}-of-{self.shard_count}"


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
    collapse_mla_tp: bool = True
    transfer_backend: str = "direct"
    dcp_world_size: int = 1
    pcp_world_size: int = 1
    dcp_rank: int = 0
    pp_rank: int = 0
    pp_size: int = 1
    mode: PegaConnectorMode = PegaConnectorMode.READ_WRITE
    wait_for_full_prefix: bool = False
    tp_shards: TpShardTopology | None = None

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
        - Hybrid MLA: tp_rank (non-MLA cache groups differ across TP ranks).
        - Non-MLA: tp_rank (each TP rank has different KV heads, already unique).
        """
        if self.is_mla and self.collapse_mla_tp:
            return self.dcp_rank
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def effective_tp_size(self) -> int:
        """TP size for PegaFlow server calls.

        - MLA without DCP: 1.
        - MLA with DCP: dcp_world_size.
        - Hybrid MLA: tp_size.
        - Non-MLA: tp_size (unique per TP rank regardless of DCP).
        """
        if self.is_mla and self.collapse_mla_tp:
            return max(1, self.dcp_world_size)
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def effective_world_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_world_size
        return self.world_size

    @property
    def local_physical_tp_rank(self) -> int:
        tp_rank = self.tp_rank or 0
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_rank(tp_rank)
        return tp_rank

    @property
    def local_physical_tp_size(self) -> int:
        if self.tp_shards is not None:
            return self.tp_shards.local_tp_size
        return self.tp_size

    @property
    def tp_shard_index(self) -> int:
        if self.tp_shards is None or self.tp_rank is None:
            return 0
        return self.tp_shards.shard_index(self.tp_rank)

    @property
    def tp_shard_count(self) -> int:
        return self.tp_shards.shard_count if self.tp_shards is not None else 1


@dataclass(frozen=True)
class LoadIntent:
    """Intent for a KV load operation."""

    block_ids_by_group: tuple[tuple[int | None, ...], ...]
    leases: tuple[bytes, ...]
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
    recurrent_group_indices: frozenset[int]
    recurrent_layer_names: frozenset[str]

    @classmethod
    def from_config(cls, kv_cache_config) -> "CacheGroupLayout":
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if not groups:
            return cls(
                layer_names=((),),
                hash_group_index=0,
                has_recurrent_state=False,
                recurrent_group_indices=frozenset(),
                recurrent_layer_names=frozenset(),
            )

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
        )

        specs = tuple(group.kv_cache_spec for group in groups)
        if len(specs) == 1:
            spec = specs[0]
            is_uniform_mla = (
                type(spec) is UniformTypeKVCacheSpecs
                and bool(spec.kv_cache_specs)
                and all(
                    type(layer_spec) is MLAAttentionSpec
                    for layer_spec in spec.kv_cache_specs.values()
                )
            )
            if type(spec) not in (FullAttentionSpec, MLAAttentionSpec) and not is_uniform_mla:
                raise RuntimeError(
                    "PegaFlow supports a single cache group only for FullAttention, MLA, "
                    "or uniformly grouped MLA layers"
                )
        else:
            if any(not isinstance(spec, (FullAttentionSpec, MambaSpec)) for spec in specs):
                raise RuntimeError(
                    "PegaFlow HMA supports only FullAttention and Mamba cache groups"
                )

            has_full_attention = any(isinstance(spec, FullAttentionSpec) for spec in specs)
            has_mamba = any(isinstance(spec, MambaSpec) for spec in specs)
            if not has_full_attention:
                raise RuntimeError(
                    "PegaFlow requires a dense FullAttention cache group for block hashes"
                )
            if not has_mamba:
                raise RuntimeError(
                    "PegaFlow HMA requires both FullAttention and Mamba cache groups"
                )
            if any(
                isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align" for spec in specs
            ):
                raise RuntimeError("PegaFlow HMA requires mamba_cache_mode='align'")

        block_sizes = {group.kv_cache_spec.block_size for group in groups}
        if len(groups) > 1 and len(block_sizes) != 1:
            raise RuntimeError(
                "PegaFlow HMA requires cache groups with identical logical block sizes"
            )

        hash_group_index = (
            0
            if len(groups) == 1
            else next(
                (
                    index
                    for index, group in enumerate(groups)
                    if isinstance(group.kv_cache_spec, FullAttentionSpec)
                ),
                None,
            )
        )
        if hash_group_index is None:
            raise RuntimeError(
                "PegaFlow requires a dense FullAttention cache group for block hashes"
            )

        return cls(
            layer_names=tuple(tuple(group.layer_names) for group in groups),
            hash_group_index=hash_group_index,
            has_recurrent_state=any(isinstance(group.kv_cache_spec, MambaSpec) for group in groups),
            recurrent_group_indices=frozenset(
                index
                for index, group in enumerate(groups)
                if isinstance(group.kv_cache_spec, MambaSpec)
            ),
            recurrent_layer_names=frozenset(
                layer_name
                for group in groups
                if isinstance(group.kv_cache_spec, MambaSpec)
                for layer_name in group.layer_names
            ),
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
        ready_save_intents: dict[str, SaveIntent] | None = None,
        preempted_req_ids: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        self.ready_save_intents: dict[str, SaveIntent] = ready_save_intents or {}
        self.preempted_req_ids: set[str] = preempted_req_ids or set()

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, "
            f"saves={len(self.save_intents)}, ready_saves={len(self.ready_save_intents)})"
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
    - `is_hma_enabled`: vLLM's hybrid cache manager changes whether hybrid
      cache layouts can share one logical block namespace.
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
        "is_hma_enabled": not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager,
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
    "TpShardTopology",
    "derive_namespace",
    "detect_mla",
    "logger",
    "parse_env_int",
    "resolve_instance_id",
    "resolve_transfer_backend",
]
