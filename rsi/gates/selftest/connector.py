"""Gate self-test arms (M8).

``SelftestConnector`` is a minimal in-process KV store with sabotage modes. It
is not PegaFlow and not a reference for onboarding agents: it exists so every
gate run on a new model can prove the gate catches a wrong connector.

``kv_connector_extra_config.mode``:

- ``honest``: at finish, copies every complete computed prompt block (all
  layers, through the layer views) to CPU; serves the longest saved prefix.
  Positive control: must pass the G2 digest and output gates.
- ``liar``: reports every complete prompt block as a hit and loads nothing.
- ``bitflip``: loads correctly, then flips one exponent bit in one element of
  layer 0 of the first loaded block of every group.
- ``offbyone``: loads block ``i + 1``'s bytes into slot ``i``.

The negative modes must fail ``G2.digest_self``, ``G2.digest_vs_baseline``
and ``G2.gsm8k_outputs``.

Full-attention groups only (incl. MLA); hybrid models need an extension here
before their self-test can run.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, group_kernel_blocks

from rsi.gates.rsi_probe import _parts, prefix_keys

MODES = ("honest", "liar", "bitflip", "offbyone")


@dataclass
class _Job:
    req_id: str
    # Per group: (prefix key at the block end, block id).
    blocks: list[list[tuple[str, int]]]


@dataclass
class SelftestMetadata(KVConnectorMetadata):
    saves: list[_Job] = field(default_factory=list)
    loads: list[_Job] = field(default_factory=list)


class SelftestConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self._mode = str(extra.get("mode", "honest"))
        if self._mode not in MODES:
            raise ValueError(f"SelftestConnector mode must be one of {MODES}, got {self._mode!r}")
        self._groups = kv_cache_config.kv_cache_groups
        for gid, g in enumerate(self._groups):
            spec = g.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or getattr(spec, "sliding_window", None):
                raise NotImplementedError(
                    f"SelftestConnector: group {gid} is {type(spec).__name__}; "
                    "only full-attention groups are supported"
                )
        self._bs = [g.kv_cache_spec.block_size for g in self._groups]
        self._unit = math.lcm(*self._bs)
        if role == KVConnectorRole.SCHEDULER:
            self._saved: set[str] = set()  # prefix keys at unit ends, all groups saved
            self._local: dict[str, int] = {}
            self._saves: list[_Job] = []
            self._loads: list[_Job] = []
        else:
            self._views: list[list[torch.Tensor]] = []
            self._store: dict[tuple[int, str], list[torch.Tensor]] = {}
            self._sending: set[str] = set()
            self._lock = threading.Lock()

    # ------------------------------------------------------- scheduler side
    def _job(self, req_id: str, tokens: list[int], lo: int, hi: int, block_ids) -> _Job:
        per_group = []
        for gid, bs in enumerate(self._bs):
            idx = list(range(lo // bs, hi // bs))
            keys = prefix_keys(tokens, [(i + 1) * bs for i in idx])
            per_group.append([(k, block_ids[gid][i]) for k, i in zip(keys, idx, strict=True)])
        return _Job(req_id, per_group)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._local[request.request_id] = num_computed_tokens
        units = (request.num_prompt_tokens - 1) // self._unit
        if self._mode == "liar":
            n = units
        else:
            tokens = list(request.prompt_token_ids or [])
            keys = prefix_keys(tokens, [self._unit * (k + 1) for k in range(units)])
            n = next((k for k, key in enumerate(keys) if key not in self._saved), units)
        return max(0, n * self._unit - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens <= 0 or self._mode == "liar":
            return
        local = self._local.get(request.request_id, 0)
        tokens = list(request.prompt_token_ids or [])
        self._loads.append(
            self._job(
                request.request_id,
                tokens,
                local,
                local + num_external_tokens,
                blocks.get_block_ids(),
            )
        )

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = SelftestMetadata(saves=self._saves, loads=self._loads)
        self._saves, self._loads = [], []
        return meta

    def request_finished(self, request, block_ids):
        return self._finish(request, (block_ids,))

    def request_finished_all_groups(self, request, block_ids):
        return self._finish(request, block_ids)

    def _finish(self, request, block_ids):
        self._local.pop(request.request_id, None)
        if self._mode == "liar":
            return False, None
        end = min(request.num_computed_tokens, request.num_prompt_tokens)
        end = end // self._unit * self._unit
        tokens = list(request.prompt_token_ids or [])
        keys = prefix_keys(tokens, list(range(self._unit, end + 1, self._unit)))
        # Saves always cover a whole prefix, so the unsaved units are a suffix.
        first = next((k for k, key in enumerate(keys) if key not in self._saved), len(keys))
        if first == len(keys):
            return False, None
        self._saved.update(keys[first:])
        self._saves.append(
            self._job(request.request_id, tokens, first * self._unit, end, block_ids)
        )
        # Keep the blocks until the worker has copied them (next step).
        return True, None

    # ---------------------------------------------------------- worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        num_blocks = {}
        for t in self._kv_cache_config.kv_cache_tensors:
            for name in t.layers:
                num_blocks[name] = self._kv_cache_config.num_blocks_of(t)
        self._views = [
            [
                group_kernel_blocks(x, num_blocks[name])
                for name in g.layer_names
                for x in _parts(kv_caches[name])
            ]
            for g in self._groups
        ]

    def start_load_kv(self, forward_context, **kwargs) -> None:
        meta = self._get_connector_metadata()
        if not isinstance(meta, SelftestMetadata):
            return
        # Saves first: a load in the same step may read what was just saved.
        for job in meta.saves:
            for gid, blocks in enumerate(job.blocks):
                if not blocks:
                    continue
                views = self._views[gid]
                idx = torch.tensor([b for _, b in blocks], device=views[0].device)
                copies = [v.index_select(0, idx).cpu() for v in views]
                for j, (key, _) in enumerate(blocks):
                    self._store[(gid, key)] = [c[j] for c in copies]
            with self._lock:
                self._sending.add(job.req_id)
        for job in meta.loads:
            for gid, blocks in enumerate(job.blocks):
                if not blocks:
                    continue
                keys = [k for k, _ in blocks]
                if self._mode == "offbyone" and len(keys) > 1:
                    keys = keys[1:] + keys[-1:]
                views = self._views[gid]
                idx = torch.tensor([b for _, b in blocks], device=views[0].device)
                for li, v in enumerate(views):
                    src = torch.stack([self._store[(gid, k)][li] for k in keys])
                    if self._mode == "bitflip" and li == 0:
                        raw = src[0].reshape(-1).view(torch.uint8)
                        # Highest byte of element 0 (little endian): an exponent bit.
                        raw[src.element_size() - 1] ^= 0x40
                    v.index_copy_(0, idx, src.to(v.device, non_blocking=False))

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return

    def get_finished(self, finished_req_ids: set[str]):
        with self._lock:
            done, self._sending = self._sending, set()
        return (done or None), None
