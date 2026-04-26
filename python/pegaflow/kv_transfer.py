"""Request-level KV transfer schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pegaflow.client import PrepareLoadRequest

if TYPE_CHECKING:
    from vllm.v1.request import Request

_ROOT_KEY = "pegaflow"
_TYPE_KEY = "type"
_DECODE_LOAD = "decode_load"
_PREFILL_PUSH = "prefill_push"
_EMPTY_PARAMS: dict[str, Any] = {}


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


def decode_load_from_request(request: "Request") -> DecodeLoad | None:
    params = _transfer_params(request)
    if _value(params, _TYPE_KEY) != _DECODE_LOAD:
        return None
    req_id = _str(params, "request_id") or request.request_id
    return DecodeLoad(
        request_id=req_id,
        expected_writes=_int(params, "expected_writes"),
    )


def prefill_push_from_request(request: "Request") -> PrefillPush | None:
    params = _transfer_params(request)
    if _value(params, _TYPE_KEY) != _PREFILL_PUSH:
        return None

    decode_endpoint = _str(params, "decode_endpoint")
    decode_instance_id = _str(params, "decode_instance_id")
    if not decode_endpoint or not decode_instance_id:
        return None

    req_id = _str(params, "request_id") or request.request_id
    handle = _str(params, "handle") or None
    return PrefillPush(
        request_id=req_id,
        decode_endpoint=decode_endpoint,
        decode_instance_id=decode_instance_id,
        handle=handle,
    )


def prepare_load_request_from_request(
    request: "Request",
    instance_id: str,
    num_computed_tokens: int,
    virtual_block_size: int,
) -> PrepareLoadRequest:
    transfer = decode_load_from_request(request)
    return PrepareLoadRequest(
        instance_id=instance_id,
        request_id=request.request_id,
        block_hashes=tuple(bytes(h) for h in getattr(request, "block_hashes", ())),
        num_prompt_tokens=_prompt_token_count(request),
        num_computed_tokens=int(num_computed_tokens),
        virtual_block_size=int(virtual_block_size),
        decode_request_id=transfer.request_id if transfer is not None else None,
        decode_expected_writes=transfer.expected_writes if transfer is not None else 0,
    )


def _prompt_token_count(request: "Request") -> int:
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    try:
        return len(prompt_token_ids)
    except TypeError:
        pass

    num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
    if num_prompt_tokens is not None:
        return int(num_prompt_tokens)

    return int(getattr(request, "num_tokens", 0) or 0)


def _transfer_params(request: "Request") -> Any:
    try:
        return getattr(request, "kv_transfer_params")[_ROOT_KEY]
    except (AttributeError, KeyError, TypeError):
        return _EMPTY_PARAMS


def _str(params: Any, key: str) -> str:
    value = _value(params, key)
    if value is None:
        return ""
    return str(value)


def _int(params: Any, key: str) -> int:
    value = _value(params, key)
    if value is None:
        return 0
    return int(value)


def _value(params: Any, key: str) -> Any:
    try:
        return params.get(key)
    except AttributeError:
        return None


__all__ = [
    "DecodeLoad",
    "PrefillPush",
    "decode_load_from_request",
    "prepare_load_request_from_request",
    "prefill_push_from_request",
]
