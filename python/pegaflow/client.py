"""Typed public client API for PegaFlow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from . import pegaflow as _native


class LoadSourceKind(Enum):
    CACHE = "cache"
    STAGED = "staged"


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
    block_hashes: tuple[bytes, ...] = ()
    num_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    virtual_block_size: int = 0
    decode_request_id: str | None = None
    decode_expected_writes: int = 0


@dataclass(frozen=True)
class LoadPlan:
    """Terminal load plan returned by ``prepare_load``."""

    request_id: str
    source: LoadSourceKind
    num_tokens: int
    num_blocks: int = 0
    block_hashes: tuple[bytes, ...] = ()
    token: str | None = None


@dataclass(frozen=True)
class PrepareLoadResult:
    """Scheduler-facing prepare state.

    ``preparing`` means retry later. ``plan is None`` in a terminal result
    means the scheduler should fall back to local computation.
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
    receive_rank: int | None = None


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

    def __init__(self, native: _native._NativeLoadState | None = None) -> None:
        self._native: _native._NativeLoadState = (
            native if native is not None else _native._NativeLoadState()
        )

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

    def prepare_load(self, request: PrepareLoadRequest) -> PrepareLoadResult:
        self._ensure_open()
        native_result = self._native.prepare_load(
            request.instance_id,
            request.request_id,
            [bytes(block_hash) for block_hash in request.block_hashes],
            int(request.num_prompt_tokens),
            int(request.num_computed_tokens),
            int(request.virtual_block_size),
            request.decode_request_id,
            int(request.decode_expected_writes),
        )
        return _parse_prepare_load_result(native_result)

    def load(self, request: LoadRequest) -> LoadHandle:
        self._ensure_open()
        if not request.items:
            raise ValueError("load request must contain at least one item")
        if not request.layer_names:
            raise ValueError("load request must include at least one layer")

        sources = {item.plan.source for item in request.items}
        if len(sources) != 1:
            raise ValueError("one load request cannot mix source kinds")
        source = next(iter(sources))

        handle = LoadHandle()
        {
            LoadSourceKind.CACHE: self._load_cache,
            LoadSourceKind.STAGED: self._load_staged,
        }[source](request, handle)
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

    def _load_cache(self, request: LoadRequest, handle: LoadHandle) -> None:
        block_ids: list[int] = []
        block_hashes: list[bytes] = []
        for item in request.items:
            _check_ready(item.plan)
            block_ids.extend(int(block_id) for block_id in item.block_ids)
            block_hashes.extend(bytes(block_hash) for block_hash in item.plan.block_hashes)
        if len(block_ids) != len(block_hashes):
            raise ValueError(
                "cache load block id/hash length mismatch "
                f"({len(block_ids)} != {len(block_hashes)})"
            )
        ok, message = self._native.load(
            request.instance_id,
            int(request.tp_rank),
            int(request.device_id),
            handle.shm_name,
            list(request.layer_names),
            block_ids,
            block_hashes,
        )
        _expect_ok("load", ok, message)

    def _load_staged(self, request: LoadRequest, handle: LoadHandle) -> None:
        items: list[tuple[str, str, list[int]]] = []
        for item in request.items:
            _check_ready(item.plan)
            items.append(
                (
                    item.plan.request_id,
                    item.plan.token or "",
                    [int(block_id) for block_id in item.block_ids],
                )
            )
        ok, message = self._native.load_pd_receive(
            request.instance_id,
            int(request.tp_rank),
            int(request.device_id),
            handle.shm_name,
            list(request.layer_names),
            items,
            int(request.receive_rank if request.receive_rank is not None else -1),
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
    if plan.num_blocks <= 0:
        raise ValueError("load plan has no blocks")


def _parse_prepare_load_result(result: dict) -> PrepareLoadResult:
    if result.get("preparing", False):
        return PrepareLoadResult.in_progress()

    return PrepareLoadResult.done(_parse_prepare_load_plan(result.get("plan")))


def _parse_prepare_load_plan(plan: object) -> LoadPlan | None:
    if plan is None:
        return None
    if not isinstance(plan, dict):
        raise TypeError(f"prepare_load returned invalid plan: {type(plan)!r}")

    try:
        source = LoadSourceKind(str(plan["source"]))
    except KeyError as e:
        raise TypeError("prepare_load plan missing source") from e
    except ValueError as e:
        raise TypeError(f"prepare_load plan has invalid source: {plan['source']!r}") from e

    load_plan = LoadPlan(
        request_id=str(plan.get("request_id") or ""),
        source=source,
        num_tokens=int(plan.get("num_tokens") or 0),
        num_blocks=int(plan.get("num_blocks") or 0),
        block_hashes=tuple(bytes(block_hash) for block_hash in plan.get("block_hashes") or ()),
        token=str(plan["token"]) if plan.get("token") is not None else None,
    )
    if load_plan.num_tokens <= 0:
        return None
    return load_plan


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
    "LoadSourceKind",
    "PegaClient",
    "PrepareLoadResult",
    "PrepareLoadRequest",
    "SaveRequest",
]
