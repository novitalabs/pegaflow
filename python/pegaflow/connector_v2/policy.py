"""Policy: choices inside the legal range computed by ``pegaflow.semantics``.

This is the main RSI surface. A policy can only lower what semantics allows
(accept a shorter hit, save fewer blocks). It receives immutable copies, and
the core validates every answer against what it offered: ``accept`` is capped
at the legal ``p`` and re-checked against the server answers, and
``choose_saves`` may only return offered items (anything else raises). So a
bad policy costs hit rate, never correctness.

Knobs live in ``models/<profile>.toml``: ``default.toml`` is loaded first, then
the selected profile (``kv_connector_extra_config["model_profile"]``, else
vLLM's ``model_config.architecture``) overrides it when present.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).with_name("models")
_PROFILE_NAME = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class SaveItem:
    """One block the core may save this step."""

    req_id: str
    group: int  # vLLM kv_cache_group index
    kind: str  # semantics Group kind: full / window / checkpoint
    block: int  # logical block index within the group
    block_id: int  # GPU block id
    block_hash: bytes


@dataclass
class Policy:
    # Defaults equal models/default.toml (a test pins this).
    # How long a request may wait while the server prefetches, and, on its own
    # clock, while another request's async load of its blocks lands; 0 = never.
    loading_wait_ms: int = 1000
    # Hits of at most this many tokens load before the step's forward (where the
    # layout allows it), longer ones asynchronously between steps.
    sync_load_tokens: int = 256
    # Merge consecutive steps' saves into one save RPC: send when the oldest
    # held step has seen save_batch_steps steps (<= 1 sends every step), when
    # the GPU KV cache is under pressure, or after the worker's staleness cap
    # (Worker.SAVE_HOLD_MAX_S). Preemption flushes, fences on a held save
    # and shutdown send at once.
    save_batch_steps: int = 32
    # Send held saves at once while vLLM's GPU KV cache usage (0..1, after
    # the step's allocations) is at or above this: freed blocks are reused
    # soon, and a fence on a held save would wait for the batch. > 1 disables it.
    save_batch_max_kv_usage: float = 0.9

    def accept(self, p: int, present: Sequence[frozenset[int]]) -> int:
        """Hit length to use, <= ``p``; ``present[sid]`` = blocks the server has."""
        return p  # default: the longest legal hit

    def choose_saves(self, items: Sequence[SaveItem]) -> Sequence[SaveItem]:
        """Subset of ``items`` to save this step."""
        return items  # default: save everything semantics allows


def load_policy(profile: str | None) -> Policy:
    knobs = _read_toml(MODELS_DIR / "default.toml")
    if profile and not _PROFILE_NAME.fullmatch(profile):
        raise ValueError(f"model_profile {profile!r} must match {_PROFILE_NAME.pattern}")
    if profile:
        path = MODELS_DIR / f"{profile}.toml"
        if path.is_file():
            knobs.update(_read_toml(path))
        else:
            logger.info("[PegaKVConnectorV2] no model profile %s, using defaults", path.name)
    known = {f.name for f in dataclasses.fields(Policy)}
    unknown = sorted(set(knobs) - known)
    if unknown:
        raise ValueError(f"unknown policy knobs {unknown}; known: {sorted(known)}")
    return Policy(**knobs)


def _read_toml(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)
    try:
        import tomli
    except ImportError:
        return _parse_flat_toml(text, path)
    return tomli.loads(text)


def _parse_flat_toml(text: str, path: Path) -> dict[str, Any]:
    """`key = value` lines with int/float/bool/"string" values; enough for knobs."""
    out: dict[str, Any] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = (s.strip() for s in line.partition("="))
        if not sep or not key:
            raise ValueError(f"{path}:{lineno}: expected `key = value`")
        if value in ("true", "false"):
            out[key] = value == "true"
        elif len(value) >= 2 and value[0] == value[-1] == '"':
            out[key] = value[1:-1]
        else:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = float(value)
    return out
