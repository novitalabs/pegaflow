"""Typed public client API for PegaFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import pegaflow as _native


@dataclass(frozen=True)
class KvCacheLayer:
    """One registered KV cache layer."""

    name: str
    wrapper_bytes: bytes
    num_blocks: int
    bytes_per_block: int
    kv_stride_bytes: int
    segments: int


@dataclass(frozen=True)
class KvCacheRegistration:
    """KV cache memory and topology registered by one worker process."""

    instance_id: str
    namespace: str
    tp_rank: int
    tp_size: int
    world_size: int
    device_id: int
    num_layers: int
    layers: tuple[KvCacheLayer, ...]


@dataclass(frozen=True)
class PrepareLoadRequest:
    """Scheduler facts needed to prepare KV blocks for a later load."""

    instance_id: str
    request_id: str
    block_hashes: list[bytes] = field(default_factory=list)
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    virtual_block_size: int = 0
    decode_request_id: str | None = None
    decode_expected_writes: int = 0


@dataclass(frozen=True)
class LoadPlan:
    """Terminal load plan returned by ``prepare_load``."""

    request_id: str
    plan_id: int
    num_tokens: int


@dataclass(frozen=True)
class PrepareLoadResult:
    """Scheduler-facing prepare state.

    ``preparing`` means retry later. ``plan is None`` means no load plan.
    """

    preparing: bool
    plan: LoadPlan | None = None

    @classmethod
    def in_progress(cls) -> PrepareLoadResult:
        return cls(preparing=True)

    @classmethod
    def done(cls, plan: LoadPlan | None = None) -> PrepareLoadResult:
        return cls(preparing=False, plan=plan)


@dataclass(frozen=True)
class LoadItem:
    """One prepared request span to load into destination block IDs."""

    plan: LoadPlan
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class LoadRequest:
    """Load prepared KV into local KV cache blocks."""

    instance_id: str
    tp_rank: int
    device_id: int
    layer_names: tuple[str, ...]
    items: tuple[LoadItem, ...]


@dataclass(frozen=True)
class LayerSave:
    """Blocks to save for one layer."""

    layer_name: str
    block_ids: tuple[int, ...]
    block_hashes: tuple[bytes, ...]


@dataclass(frozen=True)
class SaveRequest:
    """Save local KV cache blocks to PegaFlow."""

    instance_id: str
    tp_rank: int
    device_id: int
    layers: tuple[LayerSave, ...]


class LoadHandle:
    """Completion handle for an asynchronous load submitted by PegaFlow."""

    def __init__(self) -> None:
        self._native: _native._NativeLoadState = _native._NativeLoadState()

    @property
    def shm_name(self) -> str:
        return self._native.shm_name()

    @property
    def state(self) -> int:
        return int(self._native.get_state())

    @property
    def done(self) -> bool:
        return bool(self._native.is_ready())

    @property
    def ok(self) -> bool:
        return self.done and self.state >= 0


class PrepareLoadHandle:
    """Shared-memory handle for an asynchronous prepare-load transaction."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._native: _native._NativePrepareLoadState = _native._NativePrepareLoadState()

    @property
    def shm_name(self) -> str:
        return self._native.shm_name()

    def result(self) -> PrepareLoadResult:
        return _parse_prepare_load_snapshot(self._native.snapshot(), self.request_id)


class PegaClient:
    """Long-lived synchronous client for PegaFlow."""

    def __init__(self, endpoint: str | None = None) -> None:
        self._native = _native._NativeEngineClient(endpoint)
        self._closed = False

    @property
    def endpoint(self) -> str:
        return self._native.endpoint()

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> PegaClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        self.close()
        return False

    def register_kv_cache(self, registration: KvCacheRegistration) -> None:
        self._ensure_open()
        if not registration.layers:
            raise ValueError("registration.layers must not be empty")
        ok, message = self._native.register_context_batch(
            registration.instance_id,
            registration.namespace,
            int(registration.tp_rank),
            int(registration.tp_size),
            int(registration.world_size),
            int(registration.device_id),
            int(registration.num_layers),
            [layer.name for layer in registration.layers],
            [bytes(layer.wrapper_bytes) for layer in registration.layers],
            [int(layer.num_blocks) for layer in registration.layers],
            [int(layer.bytes_per_block) for layer in registration.layers],
            [int(layer.kv_stride_bytes) for layer in registration.layers],
            [int(layer.segments) for layer in registration.layers],
        )
        _expect_ok("register_kv_cache", ok, message)

    def begin_prepare_load(self, request: PrepareLoadRequest) -> PrepareLoadHandle:
        self._ensure_open()
        handle = PrepareLoadHandle(request.request_id)
        ok, message = self._native.prepare_load(
            request.instance_id,
            request.request_id,
            request.block_hashes,
            request.num_prompt_tokens,
            request.num_computed_tokens,
            request.virtual_block_size,
            handle.shm_name,
            request.decode_request_id,
            request.decode_expected_writes,
        )
        _expect_ok("prepare_load", ok, message)
        return handle

    def prepare_load(self, request: PrepareLoadRequest) -> PrepareLoadResult:
        """Begin a prepare-load transaction and return its current shm snapshot."""
        return self.begin_prepare_load(request).result()

    def load(self, request: LoadRequest) -> LoadHandle:
        self._ensure_open()
        if not request.items:
            raise ValueError("load request must contain at least one item")
        if not request.layer_names:
            raise ValueError("load request must include at least one layer")

        handle = LoadHandle()
        self._load_prepared(request, handle)
        return handle

    def save(self, request: SaveRequest) -> None:
        self._ensure_open()
        saves = [
            (
                layer.layer_name,
                [int(block_id) for block_id in layer.block_ids],
                [bytes(block_hash) for block_hash in layer.block_hashes],
            )
            for layer in request.layers
            if layer.block_ids
        ]
        if not saves:
            return
        ok, message = self._native.save(
            request.instance_id,
            int(request.tp_rank),
            int(request.device_id),
            saves,
        )
        _expect_ok("save", ok, message)

    def _load_prepared(self, request: LoadRequest, handle: LoadHandle) -> None:
        items: list[tuple[int, list[int]]] = []
        for item in request.items:
            _check_ready(item.plan)
            block_ids = [int(block_id) for block_id in item.block_ids]
            if not block_ids:
                raise ValueError("load item has no block_ids")
            items.append((int(item.plan.plan_id), block_ids))
        ok, message = self._native.load(
            request.instance_id,
            int(request.tp_rank),
            int(request.device_id),
            handle.shm_name,
            list(request.layer_names),
            items,
        )
        _expect_ok("load", ok, message)

    def _health(self) -> bool:
        self._ensure_open()
        ok, _ = self._native.health()
        return bool(ok)

    def _start_session_watcher(
        self,
        instance_id: str,
        namespace: str,
        tp_size: int,
        world_size: int,
    ) -> None:
        self._ensure_open()
        self._native.start_session_watcher(
            instance_id,
            namespace,
            int(tp_size),
            int(world_size),
        )

    def _unregister_context(self, instance_id: str) -> None:
        self._ensure_open()
        ok, message = self._native.unregister_context(instance_id)
        _expect_ok("unregister_context", ok, message)

    def _get_staged_load_descriptor(
        self,
        remote_instance_id: str,
        request_id: str,
        receive_rank: int = -1,
        handle: str | None = None,
    ) -> dict:
        self._ensure_open()
        return self._native.get_pd_receive_descriptor(
            remote_instance_id,
            request_id,
            int(receive_rank),
            handle,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PegaClient is closed")


def _check_ready(plan: LoadPlan) -> None:
    if plan.plan_id <= 0:
        raise ValueError("load plan has no plan_id")
    if plan.num_tokens <= 0:
        raise ValueError("load plan has no tokens")


def _parse_prepare_load_snapshot(snapshot: dict, request_id: str) -> PrepareLoadResult:
    if snapshot.get("preparing", False):
        return PrepareLoadResult.in_progress()
    if snapshot.get("ready_no_plan", False):
        return PrepareLoadResult.done(None)
    if not snapshot.get("ready_plan", False):
        return PrepareLoadResult.done(None)

    plan_id = int(snapshot.get("plan_id") or 0)
    if plan_id <= 0:
        return PrepareLoadResult.done(None)

    load_plan = LoadPlan(
        request_id=request_id,
        plan_id=plan_id,
        num_tokens=int(snapshot.get("num_tokens") or 0),
    )
    if load_plan.num_tokens <= 0:
        return PrepareLoadResult.done(None)
    return PrepareLoadResult.done(load_plan)


def _expect_ok(operation: str, ok: bool, message: str) -> None:
    if not ok:
        _raise_business(operation, message)


def _raise_business(operation: str, message: object) -> None:
    raise _native.PegaFlowBusinessError(f"{operation} failed: {message}")


__all__ = [
    "KvCacheLayer",
    "KvCacheRegistration",
    "LayerSave",
    "LoadHandle",
    "LoadItem",
    "LoadPlan",
    "LoadRequest",
    "PegaClient",
    "PrepareLoadHandle",
    "PrepareLoadResult",
    "PrepareLoadRequest",
    "SaveRequest",
]
