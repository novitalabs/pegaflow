#!/usr/bin/env python3
"""Two-node RDMA-only integration benchmark for the P/D transfer path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RANK_MAP = {
    "0": {"cuda_device": 0, "nic": "mlx5_1", "worker_cpu": 16},
    "1": {"cuda_device": 1, "nic": "mlx5_1", "worker_cpu": 30},
    "2": {"cuda_device": 2, "nic": "mlx5_2", "worker_cpu": 60},
    "3": {"cuda_device": 3, "nic": "mlx5_2", "worker_cpu": 90},
    "4": {"cuda_device": 4, "nic": "mlx5_3", "worker_cpu": 120},
    "5": {"cuda_device": 5, "nic": "mlx5_3", "worker_cpu": 150},
    "6": {"cuda_device": 6, "nic": "mlx5_4", "worker_cpu": 180},
    "7": {"cuda_device": 7, "nic": "mlx5_4", "worker_cpu": 210},
}


@dataclass(frozen=True)
class TransferRegionLayout:
    region_idx: int
    base_addr: int
    block_len: int


@dataclass(frozen=True)
class LayerRemoteLayout:
    layer_name: str
    layer_idx: int
    block_ids: tuple[int, ...]
    regions: tuple[TransferRegionLayout, ...]
    mr_desc: object | None = None


@dataclass(frozen=True)
class RankContext:
    rank: int
    cuda_device: int
    nic: str
    worker_cpu: int
    engine: Any
    buffer: Any
    layers: tuple[LayerRemoteLayout, ...]


def load_native() -> Any:
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


def parse_rank_map(value: str | None) -> dict[str, dict[str, int | str]]:
    if value is None:
        return DEFAULT_RANK_MAP
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise ValueError("rank map must be a JSON object")
    out: dict[str, dict[str, int | str]] = {}
    for rank, config in loaded.items():
        if not isinstance(config, dict):
            raise ValueError(f"rank_map[{rank}] must be an object")
        rank_key = str(rank)
        cuda_device = int(config.get("cuda_device", rank_key))
        out[rank_key] = {
            "cuda_device": cuda_device,
            "nic": str(config["nic"]),
            "worker_cpu": int(config["worker_cpu"]),
        }
    return out


def selected_ranks(rank_map: dict[str, dict[str, int | str]], ranks: int) -> list[int]:
    out = sorted(int(rank) for rank in rank_map)
    if ranks > len(out):
        raise ValueError(
            f"requested ranks={ranks}, but rank_map only has {len(out)} entries"
        )
    return out[:ranks]


def build_layers(
    base_addr: int,
    *,
    layer_count: int,
    block_count: int,
    block_bytes: int,
) -> tuple[LayerRemoteLayout, ...]:
    layer_bytes = block_count * block_bytes
    block_ids = tuple(range(block_count))
    return tuple(
        LayerRemoteLayout(
            layer_name=f"layer.{layer_idx}",
            layer_idx=layer_idx,
            block_ids=block_ids,
            regions=(
                TransferRegionLayout(
                    region_idx=0,
                    base_addr=base_addr + layer_idx * layer_bytes,
                    block_len=block_bytes,
                ),
            ),
        )
        for layer_idx in range(layer_count)
    )


def layer_to_native(layer: LayerRemoteLayout) -> dict[str, Any]:
    return {
        "layer_name": layer.layer_name,
        "layer_idx": layer.layer_idx,
        "block_ids": list(layer.block_ids),
        "regions": [
            {
                "region_idx": region.region_idx,
                "base_addr": region.base_addr,
                "block_len": region.block_len,
            }
            for region in layer.regions
        ],
        "mr_desc": layer.mr_desc,
    }


def layer_from_native(layer: dict[str, Any]) -> LayerRemoteLayout:
    return LayerRemoteLayout(
        layer_name=str(layer["layer_name"]),
        layer_idx=int(layer["layer_idx"]),
        block_ids=tuple(int(block_id) for block_id in layer["block_ids"]),
        regions=tuple(
            TransferRegionLayout(
                region_idx=int(region["region_idx"]),
                base_addr=int(region["base_addr"]),
                block_len=int(region["block_len"]),
            )
            for region in layer["regions"]
        ),
        mr_desc=layer.get("mr_desc"),
    )


def register_local_layers(
    engine: Any, layers: tuple[LayerRemoteLayout, ...]
) -> tuple[LayerRemoteLayout, ...]:
    registered = engine.register_local_layers(
        [layer_to_native(layer) for layer in layers]
    )
    return tuple(layer_from_native(layer) for layer in registered)


def build_coalesced_layer_blocks(
    block_count: int, block_bytes: int
) -> list[dict[str, Any]]:
    return [
        {
            "regions": [
                {
                    "region_idx": 0,
                    "block_id": 0,
                    "src_offset_bytes": 0,
                    "bytes": block_count * block_bytes,
                }
            ]
        }
    ]


def create_rank_contexts(
    native: Any,
    rank_map: dict[str, dict[str, int | str]],
    ranks: list[int],
    *,
    layer_count: int,
    block_count: int,
    block_bytes: int,
    device: str,
    fill_byte: int,
) -> list[RankContext]:
    contexts: list[RankContext] = []
    size = layer_count * block_count * block_bytes
    for rank in ranks:
        config = rank_map[str(rank)]
        cuda_device = int(config["cuda_device"])
        nic = str(config["nic"])
        worker_cpu = int(config["worker_cpu"])
        stage(
            "create rank engine "
            f"rank={rank} cuda={cuda_device} nic={nic} worker_cpu={worker_cpu}"
        )
        engine = native.PdRdmaEngine(
            cuda_device=cuda_device,
            domains=[nic],
            device=device,
            pin_worker_cpu=worker_cpu,
        )
        if device != "cuda":
            raise ValueError("pd_rdma_two_node_it currently requires CUDA test buffers")
        buffer = native.PdRdmaTestBuffer(size=size, cuda_device=cuda_device)
        buffer.fill(fill_byte)
        layers = register_local_layers(
            engine,
            build_layers(
                buffer.ptr(),
                layer_count=layer_count,
                block_count=block_count,
                block_bytes=block_bytes,
            ),
        )
        contexts.append(
            RankContext(
                rank=rank,
                cuda_device=cuda_device,
                nic=nic,
                worker_cpu=worker_cpu,
                engine=engine,
                buffer=buffer,
                layers=layers,
            )
        )
    return contexts


def handshake(
    *,
    request_id: str,
    engine_id: str,
    rank: int,
    ranks: int,
    block_size: int,
    layers: tuple[LayerRemoteLayout, ...],
    imm_id: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "engine_id": engine_id,
        "tp_rank": rank,
        "tp_size": ranks,
        "block_size": block_size,
        "layers": [layer_to_native(layer) for layer in layers],
        "imm_id": imm_id,
    }


def register_remote(engine: Any, req_id: str, remote_handshake: dict[str, Any]) -> None:
    native_handshake = {
        **remote_handshake,
        "layers": [layer_dict_to_native(layer) for layer in remote_handshake["layers"]],
    }
    engine.register_remote(req_id, native_handshake)


def layer_dict_to_native(layer: dict[str, Any]) -> dict[str, Any]:
    return {
        **layer,
        "mr_desc": mr_desc_to_native(layer.get("mr_desc")),
    }


def mr_desc_to_native(mr_desc: Any | None) -> Any | None:
    if not isinstance(mr_desc, dict):
        return mr_desc
    addr_rkey_list = mr_desc.get("addr_rkey_list")
    if addr_rkey_list is None:
        return mr_desc
    return {
        **mr_desc,
        "addr_rkey_list": [
            (str(addr_rkey[0]), int(addr_rkey[1])) for addr_rkey in addr_rkey_list
        ],
    }


def send_json(sock_file: Any, payload: dict[str, Any]) -> None:
    sock_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sock_file.flush()


def recv_json(sock_file: Any) -> dict[str, Any]:
    line = sock_file.readline()
    if not line:
        raise EOFError("control socket closed")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError("control payload must be a JSON object")
    return payload


def nic_counters(nics: set[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for nic in sorted(nics):
        counters = Path("/sys/class/infiniband") / nic / "ports" / "1" / "counters"
        out[nic] = {
            "xmit": int((counters / "port_xmit_data").read_text().strip()),
            "rcv": int((counters / "port_rcv_data").read_text().strip()),
        }
    return out


def nic_delta(
    before: dict[str, dict[str, int]],
    after: dict[str, dict[str, int]],
    elapsed_s: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for nic in sorted(before):
        xmit_bytes = (after[nic]["xmit"] - before[nic]["xmit"]) * 4
        rcv_bytes = (after[nic]["rcv"] - before[nic]["rcv"]) * 4
        out[nic] = {
            "xmit_GB": xmit_bytes / 1e9,
            "rcv_GB": rcv_bytes / 1e9,
            "xmit_gbps": xmit_bytes * 8 / elapsed_s / 1e9 if elapsed_s > 0 else 0.0,
            "rcv_gbps": rcv_bytes * 8 / elapsed_s / 1e9 if elapsed_s > 0 else 0.0,
        }
    return out


def run_rank_push(
    ctx: RankContext,
    *,
    requests: list[dict[str, Any]],
    block_count: int,
    block_bytes: int,
) -> None:
    blocks = build_coalesced_layer_blocks(block_count, block_bytes)
    for request in requests:
        prefill_req = request["prefill_req"]
        for layer in ctx.layers:
            ctx.engine.push_layer(prefill_req, layer.layer_idx, blocks)
        ctx.engine.wait_for_pushes(prefill_req)
        ctx.engine.push_done(prefill_req)


def run_prefill(args: argparse.Namespace) -> None:
    native = load_native()
    rank_map = parse_rank_map(args.rank_map)
    ranks = selected_ranks(rank_map, args.ranks)
    contexts = create_rank_contexts(
        native,
        rank_map,
        ranks,
        layer_count=args.layers,
        block_count=args.blocks,
        block_bytes=args.block_bytes,
        device=args.device,
        fill_byte=args.src_byte,
    )
    rank_by_id = {ctx.rank: ctx for ctx in contexts}
    nics = {ctx.nic for ctx in contexts}

    with socket.create_connection(
        (args.decode_host, args.port), timeout=args.connect_timeout_s
    ) as sock:
        sock_file = sock.makefile("rw", encoding="utf-8", newline="\n")
        send_json(
            sock_file,
            {
                "event": "prefill_ready",
                "ranks": ranks,
                "layers": args.layers,
                "blocks": args.blocks,
                "block_bytes": args.block_bytes,
                "iterations": args.iterations,
            },
        )
        setup = recv_json(sock_file)
        assert setup["event"] == "setup"
        requests_by_rank: dict[int, list[dict[str, Any]]] = {rank: [] for rank in ranks}
        for request in setup["requests"]:
            rank = int(request["rank"])
            ctx = rank_by_id[rank]
            register_remote(ctx.engine, request["prefill_req"], request["handshake"])
            requests_by_rank[rank].append(request)
        send_json(sock_file, {"event": "registered"})
        start = recv_json(sock_file)
        assert start["event"] == "start"

        before = nic_counters(nics)
        started = time.perf_counter()
        threads = [
            threading.Thread(
                target=run_rank_push,
                kwargs={
                    "ctx": ctx,
                    "requests": requests_by_rank[ctx.rank],
                    "block_count": args.blocks,
                    "block_bytes": args.block_bytes,
                },
                name=f"rdma-it-prefill-rank-{ctx.rank}",
            )
            for ctx in contexts
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed_s = time.perf_counter() - started
        after = nic_counters(nics)

        bytes_total = (
            args.ranks * args.iterations * args.layers * args.blocks * args.block_bytes
        )
        result = {
            "role": "prefill",
            "ok": True,
            "ranks": args.ranks,
            "layers": args.layers,
            "blocks": args.blocks,
            "block_bytes": args.block_bytes,
            "iterations": args.iterations,
            "bytes": bytes_total,
            "elapsed_ms": elapsed_s * 1000,
            "bandwidth_gbps": bytes_total * 8 / elapsed_s / 1e9,
            "rank_domains": {
                str(ctx.rank): {
                    "cuda_device": ctx.cuda_device,
                    "nic": ctx.nic,
                    "worker_cpu": ctx.worker_cpu,
                    "domains": ctx.engine.num_domains(),
                    "groups": ctx.engine.num_groups(),
                    "link_speed": ctx.engine.aggregated_link_speed(),
                }
                for ctx in contexts
            },
            "nic_delta": nic_delta(before, after, elapsed_s),
        }
        send_json(sock_file, {"event": "prefill_done", "result": result})
        print_result(result, args.json_out)


def run_decode(args: argparse.Namespace) -> None:
    native = load_native()
    rank_map = parse_rank_map(args.rank_map)
    ranks = selected_ranks(rank_map, args.ranks)
    contexts = create_rank_contexts(
        native,
        rank_map,
        ranks,
        layer_count=args.layers,
        block_count=args.blocks,
        block_bytes=args.block_bytes,
        device=args.device,
        fill_byte=args.dst_byte,
    )
    rank_by_id = {ctx.rank: ctx for ctx in contexts}
    nics = {ctx.nic for ctx in contexts}

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.listen_host, args.port))
        server.listen(1)
        stage(f"decode listening on {args.listen_host}:{args.port}")
        sock, addr = server.accept()
        stage(f"decode accepted prefill control connection from {addr}")
        with sock, sock.makefile("rw", encoding="utf-8", newline="\n") as sock_file:
            ready = recv_json(sock_file)
            assert ready["event"] == "prefill_ready"
            assert ready["ranks"] == ranks
            assert ready["layers"] == args.layers
            assert ready["blocks"] == args.blocks
            assert ready["block_bytes"] == args.block_bytes
            assert ready["iterations"] == args.iterations

            requests: list[dict[str, Any]] = []
            for iteration in range(args.iterations):
                for rank in ranks:
                    ctx = rank_by_id[rank]
                    decode_req = f"rdma-it-d-r{rank}-i{iteration}"
                    prefill_req = f"rdma-it-p-r{rank}-i{iteration}"
                    imm_id = ((iteration + 1) << 8) + rank + 1
                    hs = handshake(
                        request_id=decode_req,
                        engine_id="decode",
                        rank=rank,
                        ranks=args.ranks,
                        block_size=args.block_size,
                        layers=ctx.layers,
                        imm_id=imm_id,
                    )
                    register_remote(ctx.engine, decode_req, hs)
                    requests.append(
                        {
                            "rank": rank,
                            "iteration": iteration,
                            "decode_req": decode_req,
                            "prefill_req": prefill_req,
                            "handshake": hs,
                        }
                    )
            send_json(sock_file, {"event": "setup", "requests": requests})
            registered = recv_json(sock_file)
            assert registered["event"] == "registered"

            before = nic_counters(nics)
            started = time.perf_counter()
            send_json(sock_file, {"event": "start"})
            wait_threads = [
                threading.Thread(
                    target=ctx.engine.wait_done,
                    args=(request["decode_req"],),
                    name=f"rdma-it-decode-rank-{ctx.rank}-iter-{request['iteration']}",
                )
                for request in requests
                for ctx in (rank_by_id[int(request["rank"])],)
            ]
            for thread in wait_threads:
                thread.start()
            for thread in wait_threads:
                thread.join()
            elapsed_s = time.perf_counter() - started
            after = nic_counters(nics)
            done = recv_json(sock_file)
            assert done["event"] == "prefill_done"

    bytes_total = (
        args.ranks * args.iterations * args.layers * args.blocks * args.block_bytes
    )
    result = {
        "role": "decode",
        "ok": True,
        "ranks": args.ranks,
        "layers": args.layers,
        "blocks": args.blocks,
        "block_bytes": args.block_bytes,
        "iterations": args.iterations,
        "bytes": bytes_total,
        "elapsed_ms": elapsed_s * 1000,
        "bandwidth_gbps": bytes_total * 8 / elapsed_s / 1e9,
        "prefill_result": done["result"],
        "rank_domains": {
            str(ctx.rank): {
                "cuda_device": ctx.cuda_device,
                "nic": ctx.nic,
                "worker_cpu": ctx.worker_cpu,
                "domains": ctx.engine.num_domains(),
                "groups": ctx.engine.num_groups(),
                "link_speed": ctx.engine.aggregated_link_speed(),
            }
            for ctx in contexts
        },
        "nic_delta": nic_delta(before, after, elapsed_s),
    }
    print_result(result, args.json_out)


def print_result(result: dict[str, Any], json_out: str | None) -> None:
    text = json.dumps(result, sort_keys=True)
    print(text)
    if json_out is not None:
        Path(json_out).write_text(text + "\n")


def stage(message: str) -> None:
    print(f"[pd-rdma-two-node-it] {message}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="role", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--rank-map")
        subparser.add_argument("--ranks", type=int, default=8)
        subparser.add_argument("--layers", type=int, default=61)
        subparser.add_argument("--blocks", type=int, default=1024)
        subparser.add_argument("--block-size", type=int, default=16)
        subparser.add_argument("--block-bytes", type=int, default=18432)
        subparser.add_argument("--iterations", type=int, default=1)
        subparser.add_argument("--device", choices=("cuda",), default="cuda")
        subparser.add_argument("--json-out")

    decode = subparsers.add_parser("decode")
    add_common(decode)
    decode.add_argument("--listen-host", default="0.0.0.0")
    decode.add_argument("--port", type=int, default=19190)
    decode.add_argument("--dst-byte", type=lambda x: int(x, 0), default=0x00)

    prefill = subparsers.add_parser("prefill")
    add_common(prefill)
    prefill.add_argument("--decode-host", required=True)
    prefill.add_argument("--port", type=int, default=19190)
    prefill.add_argument("--connect-timeout-s", type=float, default=120.0)
    prefill.add_argument("--src-byte", type=lambda x: int(x, 0), default=0x5A)

    args = parser.parse_args()
    if (
        args.ranks <= 0
        or args.layers <= 0
        or args.blocks <= 0
        or args.block_size <= 0
        or args.block_bytes <= 0
    ):
        raise ValueError(
            "ranks, layers, blocks, block-size, and block-bytes must be positive"
        )
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.role == "decode":
        run_decode(args)
    else:
        run_prefill(args)


if __name__ == "__main__":
    main()
