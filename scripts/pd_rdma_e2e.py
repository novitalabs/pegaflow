#!/usr/bin/env python3
"""Run a minimal GPU-buffer P/D RDMA WRITE+IMM correctness check."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def load_native():
    repo = Path(__file__).resolve().parents[1]
    native = repo / "target" / "release" / "libpegaflow.so"
    if not native.exists():
        native = repo / "target" / "debug" / "libpegaflow.so"
    if native.exists():
        spec = importlib.util.spec_from_file_location("pegaflow.pegaflow", native)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["pegaflow.pegaflow"] = module
        spec.loader.exec_module(module)
        return module
    from pegaflow import pegaflow

    return pegaflow


@dataclass(frozen=True)
class BlockSlice:
    block_id: int
    src_offset_bytes: int
    bytes: int


@dataclass(frozen=True)
class LayerBlockSlices:
    k: BlockSlice
    v: BlockSlice


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    base_addr: int
    block_bytes: int
    block_ids: tuple[int, ...]
    k_block_addrs: tuple[int, ...]
    v_block_addrs: tuple[int, ...]
    mr_desc: object | None = None


@dataclass(frozen=True)
class PdHandshake:
    request_id: str
    engine_id: str
    tp_rank: int
    tp_size: int
    block_size: int
    kv_layout: str
    layers: tuple[LayerRemoteLayout, ...]


class RealRdmaPort:
    def __init__(self, engine):
        self.engine = engine

    def register_local_layers(self, layers):
        native_layers = [_layer_to_native(layer) for layer in layers]
        return tuple(
            _layer_from_native(layer)
            for layer in self.engine.register_local_layers(native_layers)
        )

    def register_remote(self, req_id, handshake):
        self.engine.register_remote(
            req_id,
            {
                "request_id": handshake.request_id,
                "engine_id": handshake.engine_id,
                "tp_rank": handshake.tp_rank,
                "tp_size": handshake.tp_size,
                "block_size": handshake.block_size,
                "kv_layout": handshake.kv_layout,
                "layers": [_layer_to_native(layer) for layer in handshake.layers],
            },
        )

    def push_layer(self, req_id, layer_idx, blocks):
        self.engine.push_layer(
            req_id,
            layer_idx,
            [
                {
                    "k": {
                        "block_id": block.k.block_id,
                        "src_offset_bytes": block.k.src_offset_bytes,
                        "bytes": block.k.bytes,
                    },
                    "v": {
                        "block_id": block.v.block_id,
                        "src_offset_bytes": block.v.src_offset_bytes,
                        "bytes": block.v.bytes,
                    },
                }
                for block in blocks
            ],
        )

    def push_done(self, req_id):
        self.engine.push_done(req_id)


def _layer_to_native(layer):
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "base_addr": layer.base_addr,
        "block_bytes": layer.block_bytes,
        "block_ids": list(layer.block_ids),
        "k_block_addrs": list(layer.k_block_addrs),
        "v_block_addrs": list(layer.v_block_addrs),
        "mr_desc": layer.mr_desc,
    }


def _layer_from_native(layer):
    return LayerRemoteLayout(
        layer_name=layer["layer_name"],
        layer_idx=layer["layer_idx"],
        base_addr=layer["base_addr"],
        block_bytes=layer["block_bytes"],
        block_ids=tuple(layer["block_ids"]),
        k_block_addrs=tuple(layer["k_block_addrs"]),
        v_block_addrs=tuple(layer["v_block_addrs"]),
        mr_desc=layer["mr_desc"],
    )


def build_layer(
    buf_ptr: int, block_bytes: int, block_ids: tuple[int, ...]
) -> LayerRemoteLayout:
    num_blocks = len(block_ids)
    return LayerRemoteLayout(
        layer_name="layer.0",
        layer_idx=0,
        base_addr=buf_ptr,
        block_bytes=block_bytes,
        block_ids=block_ids,
        k_block_addrs=tuple(buf_ptr + block_id * block_bytes for block_id in block_ids),
        v_block_addrs=tuple(
            buf_ptr + (num_blocks + block_id) * block_bytes for block_id in block_ids
        ),
    )


def build_slices(
    block_bytes: int, block_ids: tuple[int, ...]
) -> list[LayerBlockSlices]:
    num_blocks = len(block_ids)
    return [
        LayerBlockSlices(
            k=BlockSlice(
                block_id=block_id,
                src_offset_bytes=block_id * block_bytes,
                bytes=block_bytes,
            ),
            v=BlockSlice(
                block_id=block_id,
                src_offset_bytes=(num_blocks + block_id) * block_bytes,
                bytes=block_bytes,
            ),
        )
        for block_id in block_ids
    ]


def stage(name: str) -> None:
    print(f"[pd-rdma-e2e] {name}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-device", type=int, default=2)
    parser.add_argument("--block-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--decode-worker-cpu", type=int)
    parser.add_argument("--prefill-worker-cpu", type=int)
    parser.add_argument("--src-byte", type=lambda x: int(x, 0), default=0x5A)
    parser.add_argument("--dst-byte", type=lambda x: int(x, 0), default=0x00)
    args = parser.parse_args()

    assert args.blocks > 0
    assert args.block_bytes > 0
    native = load_native()
    total_bytes = args.block_bytes * args.blocks * 2
    block_ids = tuple(range(args.blocks))

    stage("create decode engine")
    decode = native.PdRdmaEngine(
        cuda_device=args.cuda_device,
        device="cuda",
        pin_worker_cpu=args.decode_worker_cpu,
    )
    stage("create prefill engine")
    prefill = native.PdRdmaEngine(
        cuda_device=args.cuda_device,
        device="cuda",
        pin_worker_cpu=args.prefill_worker_cpu,
    )
    stage("allocate and initialize test buffers")
    src = native.PdRdmaTestBuffer(size=total_bytes, cuda_device=args.cuda_device)
    dst = native.PdRdmaTestBuffer(size=total_bytes, cuda_device=args.cuda_device)
    src.fill(args.src_byte)
    dst.fill(args.dst_byte)

    stage("register local memory")
    decode_port = RealRdmaPort(decode)
    prefill_port = RealRdmaPort(prefill)
    decode_layers = decode_port.register_local_layers(
        (build_layer(dst.ptr(), args.block_bytes, block_ids),)
    )
    prefill_port.register_local_layers(
        (build_layer(src.ptr(), args.block_bytes, block_ids),)
    )
    handshake = PdHandshake(
        request_id="decode-req",
        engine_id="decode",
        tp_rank=0,
        tp_size=1,
        block_size=args.blocks,
        kv_layout="test-contiguous-kv",
        layers=decode_layers,
    )

    stage("register remote memory")
    prefill_port.register_remote("prefill-req", handshake)
    decode.register_remote(
        "decode-req",
        {
            "request_id": "decode-req",
            "engine_id": "decode",
            "tp_rank": 0,
            "tp_size": 1,
            "block_size": args.blocks,
            "kv_layout": "test-contiguous-kv",
            "layers": [_layer_to_native(layer) for layer in decode_layers],
        },
    )

    started = time.perf_counter()
    stage("submit RDMA WRITE blocks")
    prefill_port.push_layer("prefill-req", 0, build_slices(args.block_bytes, block_ids))
    stage("submit RDMA IMM done")
    prefill_port.push_done("prefill-req")
    stage("wait decode done")
    decode.wait_done("decode-req")
    elapsed_s = time.perf_counter() - started

    stage("copy destination buffer to host and verify")
    data = dst.to_bytes()
    expected = bytes([args.src_byte]) * total_bytes
    assert data == expected, "RDMA copied bytes do not match source fill pattern"
    bandwidth_gbps = total_bytes * 8 / elapsed_s / 1e9
    print(
        json.dumps(
            {
                "ok": True,
                "cuda_device": args.cuda_device,
                "bytes": total_bytes,
                "elapsed_ms": elapsed_s * 1000,
                "bandwidth_gbps": bandwidth_gbps,
                "domains": prefill.num_domains(),
                "groups": prefill.num_groups(),
                "aggregated_link_speed": prefill.aggregated_link_speed(),
                "decode_pin_worker_cpu": decode.pin_worker_cpu(),
                "prefill_pin_worker_cpu": prefill.pin_worker_cpu(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
