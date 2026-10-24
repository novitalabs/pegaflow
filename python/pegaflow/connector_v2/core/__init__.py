"""Model-agnostic connector glue (no vLLM / torch imports)."""

from __future__ import annotations

import os
from typing import Any

from .layout import GroupInfo, Layout, build_layout, layer_specs, may_zero_or_copy_blocks
from .scheduler import GroupSave, LoadPlan, Scheduler, StepPlan
from .worker import Worker

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 50055


def endpoint_of(extra: dict[str, Any]) -> str:
    """Server endpoint: kv_connector_extra_config, then PEGAFLOW_HOST/PEGAFLOW_PORT."""
    host = extra.get("pegaflow_host") or os.environ.get("PEGAFLOW_HOST") or DEFAULT_HOST
    port = extra.get("pegaflow_port") or os.environ.get("PEGAFLOW_PORT") or DEFAULT_PORT
    if "://" not in str(host):
        host = f"http://{host}"
    return f"{host}:{int(port)}"


__all__ = [
    "GroupInfo",
    "GroupSave",
    "Layout",
    "LoadPlan",
    "Scheduler",
    "StepPlan",
    "Worker",
    "build_layout",
    "endpoint_of",
    "layer_specs",
    "may_zero_or_copy_blocks",
]
