from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
import uuid
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.hicache_storage import get_hash_str
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

from pegaflow import EngineRpcClient, PyLoadState
from pegaflow.ipc_wrapper import CudaIPCWrapper

logger = logging.getLogger(__name__)

# Engine server endpoint (gRPC URL)
ENGINE_ENDPOINT = os.environ.get("PEGAFLOW_ENGINE_ENDPOINT", "http://127.0.0.1:50055")

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


def _default_instance_id() -> str:
    return os.environ.get("PEGAFLOW_INSTANCE_ID", uuid.uuid4().hex)


def _default_namespace(model_config: ModelConfig | None, tp_size: int = 1) -> str:
    factors = {
        "model": getattr(model_config, "model_path", None),
        "dtype": str(getattr(model_config, "dtype", "")),
        "num_hidden_layers": getattr(model_config, "num_hidden_layers", None),
        "num_kv_heads": getattr(model_config, "num_key_value_heads", None),
        "head_dim": getattr(model_config, "head_dim", None),
        "tp_size": tp_size,
    }

    # Filter out None values for cleaner hashing
    factors = {k: v for k, v in factors.items() if v is not None}

    payload = str(sorted(factors.items())).encode()
    return hashlib.sha256(payload).hexdigest()[:8]


def _seed_from_extra_key(extra_key: str | None) -> str | None:
    if extra_key is None:
        return None
    return hashlib.sha256(str(extra_key).encode()).hexdigest()


def _resolve_device_id() -> int:
    """
    Return the global CUDA device id even when CUDA_VISIBLE_DEVICES masks GPUs.

    torch.cuda.current_device() returns the local index within the visible set,
    but we need the actual global device ID for operations like CUDA IPC.
    """
    local_id = torch.cuda.current_device()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return local_id

    slots = [slot.strip() for slot in visible.split(",") if slot.strip()]
    try:
        return int(slots[local_id])
    except (IndexError, ValueError):
        return local_id


def _hash_pages(
    token_ids: Sequence[int],
    page_size: int,
    prior_hash: str | None = None,
    extra_key: str | None = None,
) -> tuple[list[bytes], str | None]:
    """Compute chained SHA256 hashes per page."""
    current = prior_hash or _seed_from_extra_key(extra_key)
    hashes: list[bytes] = []

    for start in range(0, len(token_ids), page_size):
        page = token_ids[start : start + page_size]
        current = get_hash_str(page, prior_hash=current)
        hashes.append(bytes.fromhex(current))

    return hashes, current


class PeagflowRadixCache(RadixCache):
    """RadixCache + PegaFlow RPC backend."""

    def __init__(
        self,
        params: CacheInitParams,
        model_config: ModelConfig | None = None,
        tp_size: int = 1,
        rank: int = 0,
        instance_id: str | None = None,
        namespace: str | None = None,
        engine_endpoint: str | None = None,
    ):
        super().__init__(params)

        device_pool = self.token_to_kv_pool_allocator.get_kvcache()
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        self.is_mla = isinstance(device_pool, MLATokenToKVPool)

        if EngineRpcClient is None:
            raise RuntimeError("pegaflow extension is not available (EngineRpcClient missing)")

        self.tp_rank = rank
        self.tp_size = tp_size
        self.pp_rank = params.pp_rank
        self.pp_size = params.pp_size
        self.world_size = 1  # TODO: hardcode

        # Get PP group for synchronization (None if pp_size <= 1)
        self._pp_group = self._get_pp_group()
        # Store TP cache group for hit_blocks sync
        self._tp_cache_group = params.tp_cache_group

        # Synchronize instance_id across TP ranks
        self.instance_id = self._sync_instance_id(instance_id, self._tp_cache_group)
        self.namespace = namespace or _default_namespace(
            model_config, 1 if self.is_mla else tp_size
        )

        # Device id inferred from KV cache buffers; fallback to resolved global device
        self.device_id: int = _resolve_device_id()

        self.engine_client = EngineRpcClient(engine_endpoint or ENGINE_ENDPOINT)

        self._layer_names: list[str] = []

        self._register_kv_caches()

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        # Cleanup on GC or interpreter shutdown (handles Ctrl+C gracefully)
        self._finalizer = weakref.finalize(
            self,
            PeagflowRadixCache._unregister_context,
            self.instance_id,
            self.engine_client,
            self.tp_rank,
        )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _get_pp_group(self):
        """Get the PP group coordinator if pp_size > 1."""
        if self.pp_size <= 1:
            return None
        try:
            from sglang.srt.distributed import get_pp_group

            return get_pp_group()
        except (ImportError, AssertionError) as e:
            logger.debug(f"[PeagflowRadixCache] PP group not available: {e}")
            return None

    def _sync_instance_id(
        self, instance_id: str | None, tp_cache_group: torch.distributed.ProcessGroup | None
    ) -> str:
        """Synchronize instance_id across TP ranks using gloo broadcast.

        Rank 0 generates the instance_id and broadcasts it to all other ranks.
        """
        if self.tp_size <= 1 or tp_cache_group is None:
            # Single rank, no sync needed
            return instance_id or _default_instance_id()

        # tp_cache_group is a standard PyTorch ProcessGroup (gloo-backed)
        group_ranks = torch.distributed.get_process_group_ranks(tp_cache_group)

        # Rank 0 generates the instance_id
        resolved_id = (instance_id or _default_instance_id()) if self.tp_rank == 0 else ""

        # Broadcast the instance_id from rank 0 to all others
        id_list = [resolved_id]
        torch.distributed.broadcast_object_list(id_list, src=group_ranks[0], group=tp_cache_group)
        resolved_id = id_list[0]

        logger.debug(
            f"[PeagflowRadixCache] Synced instance_id={resolved_id} (tp_rank={self.tp_rank})"
        )

        return resolved_id

    def _sync_hit_blocks(self, hit_blocks: int) -> int:
        """Sync hit_blocks across TP and PP ranks using AllReduce MIN.

        Different TP ranks may have different cache hits due to timing.
        Different PP stages have their own PegaFlow servers with different blocks.
        We take the minimum hit_blocks across all ranks to ensure consistency.

        Sync order: TP first, then PP.
        """
        original_hit_blocks = hit_blocks

        # Step 1: TP AllReduce MIN (within each PP stage)
        if self.tp_size > 1 and self._tp_cache_group is not None:
            hit_tensor = torch.tensor([hit_blocks], dtype=torch.int64, device="cpu")
            torch.distributed.all_reduce(
                hit_tensor, op=torch.distributed.ReduceOp.MIN, group=self._tp_cache_group
            )
            hit_blocks = int(hit_tensor.item())

        # Step 2: PP AllReduce MIN (across PP stages)
        if self.pp_size > 1 and self._pp_group is not None:
            hit_tensor = torch.tensor([hit_blocks], dtype=torch.int64, device="cpu")
            torch.distributed.all_reduce(
                hit_tensor, op=torch.distributed.ReduceOp.MIN, group=self._pp_group.cpu_group
            )
            hit_blocks = int(hit_tensor.item())
        if hit_blocks != original_hit_blocks:
            logger.debug(
                f"[PeagflowRadixCache] AllReduce MIN: local={original_hit_blocks} -> min={hit_blocks} "
                f"(tp_rank={self.tp_rank}, pp_rank={self.pp_rank})"
            )

        return hit_blocks

    @staticmethod
    def _unregister_context(instance_id: str, client: EngineRpcClient, tp_rank: int) -> None:
        if tp_rank != 0:
            return
        try:
            logger.info(f"[PeagflowRadixCache] Unregistering instance={instance_id}")
            ok, msg = client.unregister_context(instance_id)
            if not ok:
                logger.warning(f"[PeagflowRadixCache] Unregister failed: {msg}")
        except Exception as e:
            logger.warning(f"[PeagflowRadixCache] Unregister exception: {e}")

    def _register_kv_caches(self) -> None:
        device_pool = self.token_to_kv_pool_allocator.get_kvcache()
        if self.is_mla:
            k_pool = getattr(
                device_pool,
                "kv_buffer",
                getattr(self.token_to_kv_pool_allocator, "kv_buffer", None),
            )
            if k_pool is None:
                raise RuntimeError("Unable to locate KV buffers from token_to_kv_pool_allocator")
        else:
            k_pool = getattr(
                device_pool,
                "k_buffer",
                getattr(self.token_to_kv_pool_allocator, "k_buffer", None),
            )
            v_pool = getattr(
                device_pool,
                "v_buffer",
                getattr(self.token_to_kv_pool_allocator, "v_buffer", None),
            )

            if k_pool is None or v_pool is None:
                raise RuntimeError("Unable to locate KV buffers from token_to_kv_pool_allocator")

        num_layers = self.end_layer - self.start_layer
        layer_names: list[str] = [f"layer_{idx}" for idx in range(self.start_layer, self.end_layer)]
        if not self.is_mla:
            num_layers *= 2

        registered_layer_names: list[str] = []

        if self.is_mla:
            for layer_name, k_tensor in zip(layer_names, k_pool, strict=True):
                self._register_single_layer(
                    tensor=k_tensor,
                    layer_name=layer_name,
                    num_layers=num_layers,
                )
                registered_layer_names.append(layer_name)

        else:
            for layer_name, (k_tensor, v_tensor) in zip(
                layer_names, zip(k_pool, v_pool, strict=True), strict=True
            ):
                k_name = f"{layer_name}_k"
                v_name = f"{layer_name}_v"
                self._register_single_layer(
                    tensor=k_tensor,
                    layer_name=k_name,
                    num_layers=num_layers,
                )
                self._register_single_layer(
                    tensor=v_tensor,
                    layer_name=v_name,
                    num_layers=num_layers,
                )
                registered_layer_names.append(k_name)
                registered_layer_names.append(v_name)

        self._layer_names = registered_layer_names

    def _register_single_layer(
        self,
        tensor: torch.Tensor,
        layer_name: str,
        num_layers: int,
    ) -> str:
        wrapper = CudaIPCWrapper(tensor)
        wrapper_bytes = pickle.dumps(wrapper)

        shape = tuple(tensor.shape)
        stride = tuple(tensor.stride())
        element_size = tensor.element_size()

        slots = shape[0]
        stride0_bytes = stride[0] * element_size

        assert self.page_size > 0, "page_size must be > 0 for PegaFlow radix cache"
        assert (
            slots % self.page_size == 0
        ), f"KV slots ({slots}) must be divisible by page_size ({self.page_size})"

        num_blocks = slots // self.page_size
        bytes_per_block = stride0_bytes * self.page_size

        ok, message = self.engine_client.register_context(
            self.instance_id,
            self.namespace,
            0 if self.is_mla else self.tp_rank,
            1 if self.is_mla else self.tp_size,
            self.world_size,
            self.device_id,
            num_layers,
            layer_name,
            wrapper_bytes,
            num_blocks,
            bytes_per_block,
            0,  # kv_stride_bytes
            1,  # segments
        )

        if not ok:
            raise RuntimeError(f"Register context failed for {layer_name}: {message}")

        logger.info(
            f"[PeagflowRadixCache] Registered {layer_name} (num_blocks={num_blocks}, bytes_per_block={bytes_per_block}, page_size={self.page_size})"
        )

        return layer_name

    def _unpin_blocks(self, block_hashes: list[bytes]) -> None:
        """Unpin blocks that were pinned during query but not consumed by load."""
        if not block_hashes:
            return
        try:
            ok, message = self.engine_client.unpin(self.instance_id, block_hashes)
            if not ok:
                logger.warning(f"[PeagflowRadixCache] unpin failed: {message}")
        except Exception as e:
            logger.warning(f"[PeagflowRadixCache] unpin failed: {e}")

    def _split_blocks(
        self, token_slots: torch.Tensor, start: int, length: int
    ) -> tuple[list[int], list[int]]:
        """Return block_ids and token indices grouped by page/block."""
        block_size = self.page_size
        block_ids: list[int] = []
        token_indices: list[int] = []

        for i in range(0, length, block_size):
            block_token_start = start + i
            token_indices.append(block_token_start)
            slot_val = int(token_slots[i].item())
            block_ids.append(slot_val // block_size)

        return block_ids, token_indices

    def _await_load(self, load_state: PyLoadState, timeout_s: float = 5.0) -> bool:
        start = time.time()
        while not load_state.is_ready():
            if (time.time() - start) > timeout_s:
                return False
            time.sleep(0.001)
        return load_state.get_state() > 0

    # --------------------------------------------------------------------- #
    # Radix overrides
    # --------------------------------------------------------------------- #
    def reset(self):  # type: ignore[override]
        super().reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:  # type: ignore[override]
        key: RadixKey = params.key
        if self.disable or not key:
            return super().match_prefix(params)

        base_res = super().match_prefix(params)
        value: torch.Tensor = base_res.device_indices
        last_node: TreeNode = base_res.last_device_node

        logger.debug(
            f"[PeagflowRadixCache] match_prefix base hit_len={value.numel()} key_len={len(key)}"
        )

        if len(key) == 0:
            logger.debug("[PeagflowRadixCache] key aligned to zero, skip cache load")
            return base_res

        if value.numel() == len(key):
            return base_res

        uncached_len = len(key) - value.numel()
        uncached_len = uncached_len // self.page_size * self.page_size

        # Track local hit_blocks and allocated slots for sync
        hit_blocks = 0
        token_slots = None
        block_size = self.page_size
        start_idx = value.numel()

        if uncached_len > 0:
            if self.token_to_kv_pool_allocator.available_size() < uncached_len:
                self.evict(uncached_len)

            token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
            if token_slots is None or len(token_slots) == 0:
                logger.info(f"[PeagflowRadixCache] alloc returned empty (need={uncached_len})")
                # Don't return early - still need to sync with hit_blocks=0
                token_slots = None
            else:
                block_ids, _ = self._split_blocks(token_slots, start_idx, uncached_len)

                prefix_tokens = key.token_ids[:start_idx]
                prior_hash = None
                if prefix_tokens:
                    _, prior_hash = _hash_pages(prefix_tokens, block_size, extra_key=key.extra_key)

                missing_tokens = key.token_ids[start_idx : start_idx + uncached_len]
                block_hashes, _ = _hash_pages(
                    missing_tokens, block_size, prior_hash=prior_hash, extra_key=key.extra_key
                )

                # Query availability before issuing load
                try:
                    query_res = self.engine_client.query(self.instance_id, block_hashes)
                    logger.debug(f"[PeagflowRadixCache] query hash: {block_hashes[0]}")
                    hit_blocks = (
                        query_res.get("hit_blocks", 0) if isinstance(query_res, dict) else 0
                    )
                except Exception as e:
                    logger.warning(f"[PeagflowRadixCache] query failed: {e}")
                    hit_blocks = 0

                logger.debug(
                    f"[PeagflowRadixCache] query result: hit_blocks={hit_blocks} requested={len(block_ids)}"
                )

                hit_blocks = min(hit_blocks, len(block_ids))

        if hit_blocks > 0 and token_slots is not None:
            # Load hit_blocks from local PegaFlow server (what this PP stage has)
            load_block_ids = block_ids[:hit_blocks]
            load_block_hashes = block_hashes[:hit_blocks]

            load_state = PyLoadState()
            ok, message = self.engine_client.load(
                self.instance_id,
                0 if self.is_mla else self.tp_rank,
                self.device_id,
                load_state.shm_name(),
                self._layer_names,
                load_block_ids,
                load_block_hashes,
            )

            if not ok:
                logger.warning(f"[PeagflowRadixCache] load failed: {message}")
                # Don't return early - set hit_blocks=0 and proceed to sync
                self.token_to_kv_pool_allocator.free(token_slots)
                token_slots = None
                hit_blocks = 0
            elif not self._await_load(load_state):
                logger.warning("[PeagflowRadixCache] load timed out or failed")
                # Don't return early - set hit_blocks=0 and proceed to sync
                self.token_to_kv_pool_allocator.free(token_slots)
                token_slots = None
                hit_blocks = 0
            else:
                logger.debug(f"[PeagflowRadixCache] loaded blocks={hit_blocks} from local server")

        # CRITICAL: Sync hit_blocks across TP and PP ranks using AllReduce MIN.
        # TP ranks may have different cache hits; PP stages have different servers.
        # Take the minimum to ensure all ranks use the same prefix length.
        real_hit_blocks = self._sync_hit_blocks(hit_blocks)

        if real_hit_blocks == 0 or token_slots is None:
            # No common blocks across PP stages, or local load failed - discard everything
            if token_slots is not None:
                self.token_to_kv_pool_allocator.free(token_slots)
            return base_res

        # Use only real_hit_blocks worth of data (common across all PP stages)
        fetched_tokens = real_hit_blocks * block_size

        # Trim unused slots: discard tokens beyond what all PP stages have
        if fetched_tokens < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[fetched_tokens:])
            token_slots = token_slots[:fetched_tokens]

        logger.debug(
            f"[PeagflowRadixCache] using blocks={real_hit_blocks} tokens={fetched_tokens} "
            f"(local had {hit_blocks}, discarded {hit_blocks - real_hit_blocks})"
        )

        new_node = TreeNode(priority=last_node.priority)
        start = value.numel()
        end = start + fetched_tokens
        new_node.key = key[start:end]
        new_node.value = token_slots
        new_node.parent = last_node
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        last_node = new_node

        value = torch.cat([value, token_slots])
        self.evictable_size_ += fetched_tokens

        self._record_store_event(new_node.parent)
        self._record_store_event(new_node)

        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:  # type: ignore[override]
        super().cache_finished_req(req, is_insert=is_insert)
        if not is_insert:
            return

        from sglang.srt.server_args import get_global_server_args

        global_server_args = get_global_server_args()
        topk = global_server_args.speculative_eagle_topk
        enable_kv_committed_len = topk is None or topk == 1
        if enable_kv_committed_len:
            kv_committed_len = req.kv_committed_len
        else:
            kv_committed_len = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, :kv_committed_len]

        block_size = self.page_size

        block_hashes, _ = _hash_pages(token_ids, block_size, extra_key=req.extra_key)
        block_ids: list[int] = []
        for i in range(0, kv_committed_len, block_size):
            block_ids.append(int(kv_indices[i].item()) // block_size)

        # Ensure GPU writes are done before save
        torch.cuda.synchronize()

        saves = list(
            zip(
                self._layer_names,
                [block_ids] * len(self._layer_names),
                [block_hashes] * len(self._layer_names),
                strict=True,
            )
        )

        logger.debug(
            f"[PeagflowRadixCache] save req={getattr(req, 'request_id', None)} blocks={len(block_ids)} layers={len(saves)}"
        )

        try:
            ok, message = self.engine_client.save(
                self.instance_id,
                0 if self.is_mla else self.tp_rank,
                self.device_id,
                saves,
            )
            if not ok:
                logger.warning(f"[PeagflowRadixCache] save failed: {message}")
        except Exception as e:
            logger.warning(f"[PeagflowRadixCache] save exception: {e}")

    def evict(self, num_tokens: int) -> None:  # type: ignore[override]
        if self.disable:
            return
        self.store_stream.synchronize()
        super().evict(num_tokens)

    def unregister_context(self) -> None:
        PeagflowRadixCache._unregister_context(self.instance_id, self.engine_client, self.tp_rank)


__all__ = ["PeagflowRadixCache"]
