"""Request-level KV transfer schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from pegaflow.client import PrepareLoadRequest

if TYPE_CHECKING:
    from vllm.v1.request import Request

_ROOT_KEY = "pegaflow"
_TYPE_KEY = "type"
_DECODE_LOAD = "decode_load"
_PREFILL_PUSH = "prefill_push"

TransferParam: TypeAlias = str | int | None
TransferParams: TypeAlias = dict[str, TransferParam]


@dataclass(frozen=True)
class DecodeLoad:
    request_id: str
    expected_writes: int = 0


@dataclass(frozen=True)
class PrefillPush:
    request_id: str
    decode_endpoint: str
    decode_instance_id: str
    handle: str | None = None


def decode_load_from_request(request: Request) -> DecodeLoad | None:
    params = _transfer_params(request)
    if params is None or params.get(_TYPE_KEY) != _DECODE_LOAD:
        return None
    expected_writes = _optional_int_param(params, "expected_writes")
    return DecodeLoad(
        request_id=_request_id(params, request.request_id),
        expected_writes=0 if expected_writes is None else expected_writes,
    )


def prefill_push_from_request(request: Request) -> PrefillPush | None:
    params = _transfer_params(request)
    if params is None or params.get(_TYPE_KEY) != _PREFILL_PUSH:
        return None

    decode_endpoint = _optional_str_param(params, "decode_endpoint")
    decode_instance_id = _optional_str_param(params, "decode_instance_id")
    if decode_endpoint is None or decode_instance_id is None:
        return None

    return PrefillPush(
        request_id=_request_id(params, request.request_id),
        decode_endpoint=decode_endpoint,
        decode_instance_id=decode_instance_id,
        handle=_optional_str_param(params, "handle"),
    )


def prepare_load_request_from_request(
    request: Request,
    instance_id: str,
    num_computed_tokens: int,
    virtual_block_size: int,
) -> PrepareLoadRequest:
    transfer = decode_load_from_request(request)
    return PrepareLoadRequest(
        instance_id=instance_id,
        request_id=request.request_id,
        block_hashes=tuple(bytes(block_hash) for block_hash in request.block_hashes),
        num_prompt_tokens=request.num_prompt_tokens,
        num_computed_tokens=int(num_computed_tokens),
        virtual_block_size=int(virtual_block_size),
        decode_request_id=transfer.request_id if transfer is not None else None,
        decode_expected_writes=transfer.expected_writes if transfer is not None else 0,
    )


def _transfer_params(request: Request) -> TransferParams | None:
    params = request.kv_transfer_params
    if not isinstance(params, Mapping):
        return None
    raw_params = params.get(_ROOT_KEY)
    if not isinstance(raw_params, Mapping):
        return None

    typed_params: TransferParams = {}
    for key, value in raw_params.items():
        if not isinstance(key, str):
            continue
        if value is None or isinstance(value, str):
            typed_params[key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            typed_params[key] = value
    return typed_params


def _request_id(params: TransferParams, fallback: str) -> str:
    value = params.get("request_id")
    if not isinstance(value, str) or not value:
        return fallback
    return value


def _optional_str_param(params: TransferParams, key: str) -> str | None:
    value = params.get(key)
    if not isinstance(value, str):
        return None
    return value or None


def _optional_int_param(params: TransferParams, key: str) -> int | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return int(value)


__all__ = [
    "DecodeLoad",
    "PrefillPush",
    "decode_load_from_request",
    "prepare_load_request_from_request",
    "prefill_push_from_request",
]
