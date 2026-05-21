#!/usr/bin/env python3
"""Probe the native P/D RDMA binding without starting vLLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_pd_rdma_engine():
    try:
        from pegaflow.pegaflow import PdRdmaEngine

        return PdRdmaEngine
    except ImportError:
        repo = Path(__file__).resolve().parents[1]
        native = repo / "target" / "debug" / "libpegaflow.so"
        if not native.exists():
            raise
        spec = importlib.util.spec_from_file_location("pegaflow.pegaflow", native)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["pegaflow.pegaflow"] = module
        spec.loader.exec_module(module)
        return module.PdRdmaEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--numa-node", type=int)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--device", choices=("cuda", "host"), default="cuda")
    args = parser.parse_args()

    PdRdmaEngine = load_pd_rdma_engine()
    engine = PdRdmaEngine(
        cuda_device=args.cuda_device,
        numa_node=args.numa_node,
        domains=args.domain or None,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "main_address": engine.main_address(),
                "num_domains": engine.num_domains(),
                "num_groups": engine.num_groups(),
                "aggregated_link_speed": engine.aggregated_link_speed(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
