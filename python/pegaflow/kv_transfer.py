"""Request-level KV transfer schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pegaflow.client import PrepareLoadRequest

if TYPE_CHECKING:
    from vllm.v1.request import Request

_ROOT_KEY = "pegaflow"
_TYPE_KEY = "type"
_DECODE_LOAD = "decode_load"
_PREFILL_PUSH = "prefill_push"

TransferParams = Mapping[str, object]


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
    return DecodeLoad(
        request_id=_request_id(params, request.request_id),
        expected_writes=_int_param(params, "expected_writes"),
    )


def prefill_push_from_request(request: Request) -> PrefillPush | None:
    params = _transfer_params(request)
    if params is None or params.get(_TYPE_KEY) != _PREFILL_PUSH:
        return None

    decode_endpoint = _str_param(params, "decode_endpoint")
    decode_instance_id = _str_param(params, "decode_instance_id")
    if not decode_endpoint or not decode_instance_id:
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
    pegaflow_params = params.get(_ROOT_KEY)
    if not isinstance(pegaflow_params, Mapping):
        return None
    return pegaflow_params


def _request_id(params: TransferParams, fallback: str) -> str:
    value = params.get("request_id")
    if value in (None, ""):
        return fallback
    return str(value)


def _optional_str_param(params: TransferParams, key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    string_value = str(value)
    return string_value or None


def _str_param(params: TransferParams, key: str) -> str:
    value = _optional_str_param(params, key)
    if value is None:
        return ""
    return value


def _int_param(params: TransferParams, key: str) -> int:
    value = params.get(key)
    if value is None:
        return 0
    return int(value)


__all__ = [
    "DecodeLoad",
    "PrefillPush",
    "decode_load_from_request",
    "prepare_load_request_from_request",
    "prefill_push_from_request",
]
