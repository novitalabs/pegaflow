#!/usr/bin/env python3
"""
Start a vLLM server with PegaFlow KV cache connector.

Usage:
    python run_vllm_with_pega.py --model <model_path>
    python run_vllm_with_pega.py --model meta-llama/Llama-3.1-8B --port 8000
    python run_vllm_with_pega.py --model gpt2 --kv-events  # Enable KV events publishing
"""
import argparse
import json
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start vLLM server with PegaFlow KV cache connector"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--kv-events",
        action="store_true",
        help="Enable KV cache events publishing via ZMQ (default port 5557)",
    )
    parser.add_argument(
        "--kv-events-port",
        type=int,
        default=5557,
        help="ZMQ port for KV events publishing (default: 5557)",
    )
    parser.add_argument(
        "--kv-events-topic",
        type=str,
        default="kv-events",
        help="Topic for KV events (default: kv-events)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure PegaFlow KV connector
    kv_transfer_config = {
        "kv_connector": "PegaKVConnector",
        "kv_role": "kv_both",  # Both scheduler and worker roles
        "kv_connector_module_path": "pegaflow.connector",
    }

    # Build vllm serve command
    cmd = [
        "vllm", "serve",
        args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--kv-transfer-config", json.dumps(kv_transfer_config),
    ]

    # KV events requires prefix caching to be enabled
    if args.kv_events:
        cmd.append("--enable-prefix-caching")
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://*:{args.kv_events_port}",
            "replay_endpoint": f"tcp://*:{args.kv_events_port + 1}",
            "topic": args.kv_events_topic,
        }
        cmd.extend(["--kv-events-config", json.dumps(kv_events_config)])
    else:
        cmd.append("--no-enable-prefix-caching")

    print(f"Starting vLLM server with PegaFlow...")
    print(f"Model: {args.model}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    if args.kv_events:
        print(f"Prefix Caching: enabled (required for KV events)")
        print(f"KV Events: enabled (zmq://localhost:{args.kv_events_port}, topic: {args.kv_events_topic})")
        print(f"           Monitor with: python monitor_kv_events.py --endpoint tcp://localhost:{args.kv_events_port} --topic {args.kv_events_topic}")
    print(f"\nCommand: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
