"""
Shared types and helpers for the PegaFlow vLLM connector.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

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
    # Token span of one `Request.block_hashes` entry; `None` means one per
    # scheduler block.
    hash_block_size: int | None = None

    @property
    def read_enabled(self) -> bool:
        return self.mode is PegaConnectorMode.READ_WRITE

    @property
    def virtual_block_size(self) -> int:
        """Block size as seen by the scheduler.

        vLLM's scheduler block size is ``block_size * dcp``. PCP changes
        which ranks process a request, but does not change the token
        granularity of scheduler block hashes.
        """
        return self.block_size * self.dcp_world_size

    @property
    def hash_scale(self) -> int:
        """`Request.block_hashes` entries per scheduler block.

        vLLM hashes every `hash_block_size` tokens, which is finer than the
        scheduler block for hybrid models (GCD of the group block sizes, or
        `--prefix-match-unit`). Each hash chains over its whole prefix, so the
        last one inside a block is that block's key.
        """
        return self.virtual_block_size // (self.hash_block_size or self.virtual_block_size)

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
    # Optional per-cache-group leases.  The legacy ``leases`` field remains
    # the group-0 compatibility path; heterogeneous layouts use one lease
    # vector per cache group so block counts can differ safely. Groups with
    # identical storage queries share a lease; workers combine their targets
    # into one backend load and release that lease exactly once.
    leases_by_group: tuple[tuple[bytes, ...], ...] | None = None
    # Hybrid-cache loads carry one membership lease per recurrent storage
    # group (pinned checkpoints in hit-positions order) on top of the
    # attention prefix leases. See RecurrentLoadHold.
    recurrent_hold: "RecurrentLoadHold | None" = None


@dataclass(frozen=True)
class RecurrentLoadHold:
    """Pinned recurrent checkpoints for one hybrid external load.

    Indexed by ``sorted(recurrent_group_indices)`` on the outside and TP
    shard on the inside: ``leases[g][shard]`` is the membership lease over
    group ``g``'s hit blocks; ``hit_positions[g][shard]`` lists each leased
    block's position in the scheduler's query hash list (lease order).
    ``checkpoint`` is the chosen query position — the mamba state stored
    there covers all tokens through the end of that block (vLLM convention:
    state block ``i`` ends at token ``(i + 1) * block_size``), so the
    resumable prefix is ``checkpoint + 1`` blocks.
    """

    leases: tuple[tuple[bytes, ...], ...]
    hit_positions: tuple[tuple[tuple[int, ...], ...], ...]
    checkpoint: int


def reconcile_hybrid_hit(
    attention_hit_blocks: int,
    recurrent_hits: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[int, int | None, frozenset[int]]:
    """Combine per-group query results into one hybrid hit.

    ``attention_hit_blocks`` is the (already shard-minimized) attention prefix
    length in blocks. ``recurrent_hits[g][s]`` lists the query positions whose
    checkpoint block is cached in recurrent group ``g`` on shard ``s``.

    HMA needs the whole prefix resumable: every recurrent group must hold a
    checkpoint state inside the attention prefix (attention KV alone cannot
    skip mamba's sequential prefill), and that state must exist on every TP
    shard. Returns ``(hit_blocks, checkpoint, usable)`` where ``hit_blocks``
    is ``checkpoint + 1`` — the checkpoint covers tokens through the end of
    its own block — and ``usable`` is every legal boundary position (for
    re-derivation when the token budget later shrinks the hit). ``(0, None,
    frozenset())`` means no usable boundary: recompute from scratch.
    """
    if attention_hit_blocks <= 0 or not recurrent_hits:
        return 0, None, frozenset()
    # A checkpoint position is usable only inside the attention prefix AND
    # present in every recurrent group on every shard.
    usable: set[int] | None = None
    for group_hits in recurrent_hits:
        for shard_hits in group_hits:
            in_prefix = {p for p in shard_hits if p < attention_hit_blocks}
            usable = in_prefix if usable is None else usable & in_prefix
            if not usable:
                return 0, None, frozenset()
    if not usable:
        return 0, None, frozenset()
    checkpoint = max(usable)
    return checkpoint + 1, checkpoint, frozenset(usable)


@dataclass(frozen=True)
class SaveIntent:
    """Intent for a KV save operation."""

    block_ids_by_group: tuple[tuple[int, ...], ...]
    block_hashes: tuple[bytes, ...]
    # Optional per-group hash vectors.  When absent, ``block_hashes`` is used
    # for every group for backwards compatibility with uniform layouts.
    block_hashes_by_group: tuple[tuple[bytes, ...], ...] | None = None
    # Sliding source blocks can leave the window while the request runs.
    # This job holds their GPU references until every worker finishes saving.
    gpu_pin_job_id: int | None = None


@dataclass(frozen=True)
class CacheGroupLayout:
    """Stable vLLM cache-group order shared by scheduler and worker.

    `storage_group_ids` maps connector groups onto engine storage groups:
    dense full attention uses group 0 (prefix cadence), sliding-window
    attention uses a secondary membership group, and recurrent groups get
    subsequent ids (group-encoded keys).

    Scratch groups (KpoolTailSpec) hold per-request circular state and are
    never registered, saved, or loaded. They remain in the group order so
    connector indices match vLLM's block tables.
    """

    layer_names: tuple[tuple[str, ...], ...]
    hash_group_index: int
    has_recurrent_state: bool
    recurrent_group_indices: frozenset[int]
    recurrent_layer_names: frozenset[str]
    sliding_window_group_indices: frozenset[int] = frozenset()
    group_sliding_windows: tuple[int | None, ...] = ()
    storage_group_ids: tuple[int, ...] = (0,)
    group_block_sizes: tuple[int, ...] = ()
    scratch_group_indices: frozenset[int] = frozenset()
    scratch_layer_names: frozenset[str] = frozenset()

    @classmethod
    def from_config(
        cls,
        kv_cache_config,
        *,
        allow_sliding_window: bool = True,
        hash_block_size: int | None = None,
    ) -> "CacheGroupLayout":
        groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
        if not groups:
            return cls(
                layer_names=((),),
                hash_group_index=0,
                has_recurrent_state=False,
                recurrent_group_indices=frozenset(),
                recurrent_layer_names=frozenset(),
                sliding_window_group_indices=frozenset(),
                group_sliding_windows=(None,),
                group_block_sizes=(0,),
            )

        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MLAAttentionSpec,
            SlidingWindowSpec,
            UniformTypeKVCacheSpecs,
        )

        try:
            from vllm.v1.kv_cache_interface import KpoolTailSpec
        except ImportError:
            KpoolTailSpec = None

        specs = tuple(group.kv_cache_spec for group in groups)
        layer_specs = tuple(
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
            for spec in specs
        )
        scratch_group_indices = frozenset(
            index
            for index, layers in enumerate(layer_specs)
            if KpoolTailSpec is not None
            and layers
            and all(type(layer) is KpoolTailSpec for layer in layers)
        )
        uniform_attention = tuple(
            isinstance(spec, UniformTypeKVCacheSpecs)
            and bool(layers)
            and (
                all(isinstance(layer, MLAAttentionSpec) for layer in layers)
                or all(
                    isinstance(layer, (FullAttentionSpec, SlidingWindowSpec))
                    and not isinstance(layer, MLAAttentionSpec)
                    for layer in layers
                )
            )
            for spec, layers in zip(specs, layer_specs, strict=True)
        )
        recurrent_group_indices = frozenset(
            index for index, spec in enumerate(specs) if isinstance(spec, MambaSpec)
        )
        sliding_window_group_indices = frozenset(
            index
            for index, layers in enumerate(layer_specs)
            if index not in scratch_group_indices
            and any(isinstance(layer, SlidingWindowSpec) for layer in layers)
        )
        if sliding_window_group_indices and not allow_sliding_window:
            raise RuntimeError(
                "PegaFlow SlidingWindowSpec requires the vLLM hybrid KV cache manager"
            )

        if len(specs) == 1:
            if isinstance(specs[0], SlidingWindowSpec):
                raise RuntimeError(
                    "PegaFlow requires a dense FullAttention cache group alongside SlidingWindowSpec"
                )
            if uniform_attention[0] and sliding_window_group_indices:
                raise RuntimeError(
                    "PegaFlow requires SlidingWindowSpec layers to use a separate "
                    "cache group from FullAttention layers"
                )
            if (
                type(specs[0]) not in (FullAttentionSpec, MLAAttentionSpec)
                and not uniform_attention[0]
            ):
                raise RuntimeError(
                    "PegaFlow supports a single cache group only for FullAttention, MLA, "
                    "or uniformly grouped attention layers"
                )
            hash_group_index = 0
        else:
            if any(
                index not in scratch_group_indices
                and not isinstance(
                    spec, (FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec, MambaSpec)
                )
                and not uniform
                for index, (spec, uniform) in enumerate(zip(specs, uniform_attention, strict=True))
            ):
                raise RuntimeError(
                    "PegaFlow HMA supports only attention, Mamba, and kpool-tail scratch cache groups"
                )
            hash_group_index = next(
                (
                    index
                    for index, layers in enumerate(layer_specs)
                    if index not in scratch_group_indices
                    and any(
                        isinstance(layer, (FullAttentionSpec, MLAAttentionSpec)) for layer in layers
                    )
                ),
                None,
            )
            if hash_group_index is None:
                raise RuntimeError(
                    "PegaFlow requires a dense FullAttention cache group for block hashes"
                )
            if sliding_window_group_indices and recurrent_group_indices:
                raise RuntimeError(
                    "PegaFlow does not support combining SlidingWindowSpec with Mamba cache groups"
                )
            if not sliding_window_group_indices and not recurrent_group_indices:
                raise RuntimeError(
                    "PegaFlow HMA requires FullAttention with SlidingWindow or Mamba cache groups"
                )
            if any(specs[index].mamba_cache_mode != "align" for index in recurrent_group_indices):
                raise RuntimeError("PegaFlow HMA requires mamba_cache_mode='align'")

        group_block_sizes = tuple(int(spec.block_size) for spec in specs)
        if any(size <= 0 for size in group_block_sizes):
            raise RuntimeError("PegaFlow cache groups require positive logical block sizes")
        if any(
            index not in scratch_group_indices and int(layer.block_size) != size
            for index, (size, layers) in enumerate(zip(group_block_sizes, layer_specs, strict=True))
            for layer in layers
        ):
            raise RuntimeError(
                "PegaFlow requires layers within each cache group to share its logical block size"
            )
        if hash_block_size is not None:
            if hash_block_size <= 0:
                raise ValueError(f"hash block size must be > 0, got {hash_block_size}")
            # Single-group DCP hashes span multiple physical blocks.
            if len(groups) > 1 and any(
                size % hash_block_size
                for index, size in enumerate(group_block_sizes)
                if index not in scratch_group_indices
            ):
                raise RuntimeError(
                    "PegaFlow cache group block sizes must be integer multiples of "
                    f"hash block size {hash_block_size}; got {group_block_sizes}"
                )
        dense_block_size = group_block_sizes[hash_group_index]
        if any(
            size != dense_block_size
            for index, size in enumerate(group_block_sizes)
            if index not in recurrent_group_indices
            and index not in sliding_window_group_indices
            and index not in scratch_group_indices
        ):
            raise RuntimeError(
                "PegaFlow requires dense attention groups to share a logical block size"
            )
        if any(group_block_sizes[index] != dense_block_size for index in recurrent_group_indices):
            raise RuntimeError(
                "PegaFlow requires recurrent cache groups to use the dense attention block size"
            )
        if any(
            dense_block_size % group_block_sizes[index] for index in sliding_window_group_indices
        ):
            raise RuntimeError(
                "PegaFlow requires SlidingWindow block sizes to divide the dense attention "
                "block size so dense KV blocks match the scheduler alignment"
            )

        group_sliding_windows = tuple(
            next(
                (
                    int(layer.sliding_window)
                    for layer in layers
                    if isinstance(layer, SlidingWindowSpec)
                    and not (KpoolTailSpec is not None and type(layer) is KpoolTailSpec)
                ),
                None,
            )
            for layers in layer_specs
        )
        # Dense groups share the prefix; sliding groups share membership keys.
        # Sliding and recurrent layouts are mutually exclusive.
        storage_group_ids = tuple(
            1 + sum(other < index for other in recurrent_group_indices)
            if index in recurrent_group_indices
            else int(index in sliding_window_group_indices)
            for index in range(len(groups))
        )

        return cls(
            layer_names=tuple(tuple(group.layer_names) for group in groups),
            hash_group_index=hash_group_index,
            has_recurrent_state=bool(recurrent_group_indices),
            recurrent_group_indices=recurrent_group_indices,
            recurrent_layer_names=frozenset(
                layer_name
                for group in groups
                if isinstance(group.kv_cache_spec, MambaSpec)
                for layer_name in group.layer_names
            ),
            sliding_window_group_indices=sliding_window_group_indices,
            group_sliding_windows=group_sliding_windows,
            storage_group_ids=storage_group_ids,
            group_block_sizes=group_block_sizes,
            scratch_group_indices=scratch_group_indices,
            scratch_layer_names=frozenset(
                layer_name
                for index in scratch_group_indices
                for layer_name in groups[index].layer_names
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

    def storage_group_of(self, group_index: int) -> int:
        """Engine storage group id for a connector cache group index."""
        return self.storage_group_ids[group_index]

    def block_size_of(self, group_index: int) -> int:
        """Return the logical token block size for a cache group."""
        return self.group_block_sizes[group_index]

    def sliding_window_of(self, group_index: int) -> int | None:
        """Return the token window for a sliding group, if one is declared."""
        return self.group_sliding_windows[group_index]

    @property
    def requires_group_specific_block_mapping(self) -> bool:
        """Whether groups need different numbers of hashes per scheduler block."""
        return any(
            self.group_block_sizes[index] != self.group_block_sizes[self.hash_group_index]
            for index in self.sliding_window_group_indices
        )


class PegaConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to worker for KV cache operations."""

    def __init__(
        self,
        load_intents: dict[str, LoadIntent] | None = None,
        save_intents: dict[str, SaveIntent] | None = None,
        boundary_save_intents: dict[int, SaveIntent] | None = None,
        preempted_req_ids: set[str] | None = None,
    ):
        super().__init__()
        # Maps request_id -> intent
        self.load_intents: dict[str, LoadIntent] = load_intents or {}
        self.save_intents: dict[str, SaveIntent] = save_intents or {}
        # HMA: recurrent boundary states handed off by vLLM this step, keyed
        # by a scheduler-issued job id. Their blocks are pinned by the
        # scheduler until every worker reports the job through
        # PegaWorkerMetadata, so they are decoupled from request lifetimes.
        self.boundary_save_intents: dict[int, SaveIntent] = boundary_save_intents or {}
        self.preempted_req_ids: set[str] = preempted_req_ids or set()

    def __repr__(self) -> str:
        return (
            f"PegaConnectorMetadata(loads={len(self.load_intents)}, "
            f"saves={len(self.save_intents)}, "
            f"boundary_saves={len(self.boundary_save_intents)})"
        )


@dataclass
class PegaWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler completion report for pinned GPU save jobs.

    ``completed_boundary_jobs`` maps a job id to the number of workers that
    finished it (successfully or not). vLLM aggregates one instance per
    worker before the scheduler sees it. Sliding saves use the same job
    lifecycle as recurrent boundary-state saves.
    """

    completed_boundary_jobs: dict[int, int]

    def aggregate(self, other: "KVConnectorWorkerMetadata") -> "PegaWorkerMetadata":
        if not isinstance(other, PegaWorkerMetadata):
            raise TypeError(f"cannot aggregate {type(other).__name__} into PegaWorkerMetadata")
        for job_id, count in other.completed_boundary_jobs.items():
            self.completed_boundary_jobs[job_id] = (
                self.completed_boundary_jobs.get(job_id, 0) + count
            )
        return self


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
    hash_block_size: int | None = None,
    cache_group_layout: CacheGroupLayout | None = None,
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
    - `hash_block_size` / `block_size`: decide which chained hash keys a block
      and how many tokens it spans; `mamba_*`: recurrent state layout.
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
        "hash_block_size": hash_block_size,
        "block_size": getattr(cache_config, "block_size", None),
        "mamba_cache_mode": getattr(cache_config, "mamba_cache_mode", None),
        "mamba_ssm_cache_dtype": getattr(cache_config, "mamba_ssm_cache_dtype", None),
    }
    if cache_group_layout is not None and cache_group_layout.sliding_window_group_indices:
        # Group specs are shared across PP stages; layer names are worker-local.
        # Keep the existing namespace for layouts without sliding attention.
        factors["cache_group_block_sizes"] = cache_group_layout.group_block_sizes

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
    "PegaWorkerMetadata",
    "RecurrentLoadHold",
    "SaveIntent",
    "TpShardTopology",
    "derive_namespace",
    "detect_mla",
    "logger",
    "parse_env_int",
    "reconcile_hybrid_hit",
    "resolve_instance_id",
    "resolve_transfer_backend",
]
