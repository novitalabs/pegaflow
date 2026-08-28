"""Scoped vLLM adapter for coherent HMA local prefix hits."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

_lock = Lock()
_bound_pools: dict[int, tuple[object, int]] = {}
_coordinator_type: type | None = None
_original_lookup: Callable[..., Any] | None = None
_adapted_lookup: Callable[..., Any] | None = None


def enable_reconciled_hma_local_hits(block_pool: object) -> None:
    """Make vLLM report only locally resumable hybrid prefixes for this pool."""
    global _adapted_lookup, _coordinator_type, _original_lookup

    with _lock:
        pool_id = id(block_pool)
        existing = _bound_pools.get(pool_id)
        if existing is not None:
            if existing[0] is not block_pool:
                raise RuntimeError("HMA local-cache block-pool identity collision")
            _bound_pools[pool_id] = (block_pool, existing[1] + 1)
            return

        if _coordinator_type is None:
            from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

            coordinator_type = HybridKVCacheCoordinator
            original_lookup = coordinator_type.find_longest_cache_hit_per_group

            def find_reconciled_cache_hit_per_group(
                coordinator,
                block_hashes,
                max_cache_hit_length,
            ):
                with _lock:
                    entry = _bound_pools.get(id(coordinator.block_pool))
                    is_bound = entry is not None and entry[0] is coordinator.block_pool

                if not is_bound:
                    return original_lookup(
                        coordinator,
                        block_hashes,
                        max_cache_hit_length,
                    )

                blocks, hit_length, _ = coordinator.find_longest_cache_hit(
                    block_hashes,
                    max_cache_hit_length,
                )
                group_count = len(coordinator.kv_cache_config.kv_cache_groups)
                return blocks, (hit_length,) * group_count

            coordinator_type.find_longest_cache_hit_per_group = find_reconciled_cache_hit_per_group
            _coordinator_type = coordinator_type
            _original_lookup = original_lookup
            _adapted_lookup = find_reconciled_cache_hit_per_group
        elif (
            _adapted_lookup is None
            or _coordinator_type.find_longest_cache_hit_per_group is not _adapted_lookup
        ):
            raise RuntimeError("vLLM HMA cache lookup changed while PegaFlow adapter was active")

        _bound_pools[pool_id] = (block_pool, 1)


def disable_reconciled_hma_local_hits(block_pool: object) -> None:
    """Release one binding and restore vLLM after the final binding closes."""
    global _adapted_lookup, _coordinator_type, _original_lookup

    with _lock:
        pool_id = id(block_pool)
        existing = _bound_pools.get(pool_id)
        if existing is None or existing[0] is not block_pool:
            raise RuntimeError("HMA local-cache block pool is not bound")
        if existing[1] > 1:
            _bound_pools[pool_id] = (block_pool, existing[1] - 1)
            return
        del _bound_pools[pool_id]

        if _bound_pools:
            return
        if _coordinator_type is None or _original_lookup is None or _adapted_lookup is None:
            raise RuntimeError("HMA local-cache adapter lost its installation state")
        if _coordinator_type.find_longest_cache_hit_per_group is not _adapted_lookup:
            raise RuntimeError("vLLM HMA cache lookup changed while PegaFlow adapter was active")

        _coordinator_type.find_longest_cache_hit_per_group = _original_lookup
        _coordinator_type = None
        _original_lookup = None
        _adapted_lookup = None


__all__ = [
    "disable_reconciled_hma_local_hits",
    "enable_reconciled_hma_local_hits",
]
