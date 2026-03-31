#!/usr/bin/env python3
"""
Layer-wise KV Cache Benchmark

Tests the benefit of layer-wise KV cache transfer by:
1. Sending a warmup request (base prompt) to populate the cache
2. Sending a test request (base prompt + N appended tokens) to trigger partial prefill
3. Measuring TTFT difference to quantify layer-wise transfer benefit

The test request reuses cached KV for the base prompt portion and only computes
the new appended tokens, demonstrating layer-wise prefill overlap.

Usage:
    python examples/bench_layer_wise.py --model /path/to/model

Arguments:
    --model         Model path (required)
    --base-len      Base prompt length in tokens (default: model-dependent, 10000 for Qwen3, 4000 for Qwen3.5)
    --append-len    Additional tokens appended for test request (default: 128)
    --output-len    Output tokens to generate (default: 1)
    --port          vLLM server port (default: 8001)
    --output-dir    Results directory (default: examples/bench_results)
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence

import requests
from openai import OpenAI


class VLLMServer:
    """Context manager for vLLM server lifecycle with PegaFlow connector."""

    def __init__(
        self,
        model: str,
        port: int,
        log_file: Optional[Path] = None,
        torch_profile_dir: Optional[Path] = None,
        health_endpoints: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.port = port
        self.log_file = log_file
        self.torch_profile_dir = torch_profile_dir
        self.health_endpoints = (
            list(health_endpoints)
            if health_endpoints
            else [
                "/health",
                "/metrics",
            ]
        )
        self.process: Optional[subprocess.Popen] = None
        self.log_handle = None

    def __enter__(self):
        """Start the vLLM server with PegaFlow connector."""
        env = os.environ.copy()
        env["LMCACHE_CHUNK_SIZE"] = "256"
        env["LMCACHE_LOCAL_CPU"] = "True"
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "32.0"

        # Enable torch profiler if directory specified
        if self.torch_profile_dir:
            self.torch_profile_dir.mkdir(parents=True, exist_ok=True)
            env["VLLM_TORCH_PROFILER_DIR"] = str(self.torch_profile_dir)

        # Build server command
        kv_config = {
            "kv_connector": "PegaKVConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "pegaflow.connector",
        }

        # lmcache_config = {
        #     "kv_connector": "LMCacheConnectorV1",
        #     "kv_role": "kv_both",
        # }

        cmd = [
            "vllm",
            "serve",
            self.model,
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--max-num-batched-tokens",
            "65536",
            "--no-enable-prefix-caching",
            "--kv-transfer-config",
            json.dumps(kv_config),
        ]

        print(f"\n[PegaFlow] Starting vLLM server on port {self.port}")
        print(f"[PegaFlow] Torch profiler dir: {self.torch_profile_dir}")

        # Redirect output to log file
        if self.log_file:
            print(f"[PegaFlow] Server log: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        else:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the vLLM server."""
        if self.process:
            print("\n[PegaFlow] Stopping vLLM server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[PegaFlow] Server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            print("[PegaFlow] Server stopped.")

        if self.log_handle:
            self.log_handle.close()

    def _wait_for_ready(self, timeout: int = 180):
        """Wait for the server to be ready."""
        start_time = time.time()
        print("[PegaFlow] Waiting for server to be ready...")

        while time.time() - start_time < timeout:
            for endpoint in self.health_endpoints:
                url = f"http://localhost:{self.port}{endpoint}"
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        print(f"[PegaFlow] Server is ready! (checked {endpoint})\n")
                        time.sleep(2)
                        return
                except requests.exceptions.RequestException:
                    continue
            time.sleep(2)

        raise TimeoutError(f"Server did not become ready within {timeout} seconds")

    def start_profile(self):
        """Start torch profiling via HTTP endpoint."""
        url = f"http://localhost:{self.port}/start_profile"
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print("[PegaFlow] Torch profiling started")
            else:
                print(f"[PegaFlow] Failed to start profiling: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[PegaFlow] Failed to start profiling: {e}")

    def stop_profile(self):
        """Stop torch profiling via HTTP endpoint."""
        url = f"http://localhost:{self.port}/stop_profile"
        try:
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                print("[PegaFlow] Torch profiling stopped")
            else:
                print(f"[PegaFlow] Failed to stop profiling: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[PegaFlow] Failed to stop profiling: {e}")


def generate_prompt(num_tokens: int, prefix: str = "") -> str:
    """Generate a prompt with approximately num_tokens tokens.

    Uses repeated 'hi ' pattern which is roughly 1 token per word.
    """
    # Add prefix to make prompts unique
    base = prefix + " " if prefix else ""
    # Each "hi " is approximately 1 token
    return base + " ".join(["hi"] * num_tokens)


def send_request(client: OpenAI, model: str, prompt: str, output_len: int) -> dict:
    """Send a completion request and measure timing.

    Returns:
        dict with ttft_ms, total_ms, and success status
    """
    start_time = time.time()
    first_token_time = None

    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=output_len,
            temperature=0.0,
            stream=True,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].text:
                if first_token_time is None:
                    first_token_time = time.time()

        end_time = time.time()

        if first_token_time is None:
            return {"ttft_ms": -1, "total_ms": -1, "success": False}

        return {
            "ttft_ms": (first_token_time - start_time) * 1000,
            "total_ms": (end_time - start_time) * 1000,
            "success": True,
        }
    except Exception as e:
        print(f"Request failed: {e}")
        return {"ttft_ms": -1, "total_ms": -1, "success": False}


def run_benchmark(args) -> dict:
    """Run the layer-wise benchmark."""
    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / f"layer_wise_bench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch_trace_dir = run_dir / "torch_trace"
    server_log = run_dir / "server.log"

    print("\n" + "=" * 70)
    print("LAYER-WISE KV CACHE BENCHMARK")
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Base Length:     {args.base_len} tokens")
    print(f"Append Length:   {args.append_len} tokens")
    print(f"Output Length:   {args.output_len} tokens")
    print(f"Port:            {args.port}")
    print(f"Results Dir:     {run_dir}")
    print("=" * 70)

    # Generate prompts
    base_prompt = generate_prompt(args.base_len, prefix="0")
    # Test prompt = base prompt + appended tokens
    test_prompt = base_prompt + " " + generate_prompt(args.append_len, prefix="extra")

    print(f"\nBase prompt: ~{args.base_len} tokens")
    print(f"Test prompt: ~{args.base_len + args.append_len} tokens")

    results = {
        "config": {
            "model": args.model,
            "base_len": args.base_len,
            "append_len": args.append_len,
            "output_len": args.output_len,
            "timestamp": timestamp,
        },
        "warmup": None,
        "test": None,
        "speedup": None,
    }

    with VLLMServer(
        model=args.model,
        port=args.port,
        log_file=server_log,
        torch_profile_dir=torch_trace_dir,
    ) as server:
        # Create OpenAI client
        client = OpenAI(
            base_url=f"http://localhost:{args.port}/v1",
            api_key="dummy",
            timeout=None,
        )

        # Get model name from server
        models = client.models.list()
        model_name = models.data[0].id
        print(f"Using model: {model_name}")

        # Phase 1: Warmup request (populates cache)
        print("\n" + "-" * 50)
        print("PHASE 1: Warmup Request (populating cache)")
        print("-" * 50)

        warmup_result = send_request(client, model_name, base_prompt, args.output_len)
        results["warmup"] = warmup_result

        if warmup_result["success"]:
            print(f"Warmup TTFT: {warmup_result['ttft_ms']:.2f} ms")
            print(f"Warmup Total: {warmup_result['total_ms']:.2f} ms")
        else:
            print("Warmup request failed!")
            return results

        # Small delay between requests
        time.sleep(1)

        # Phase 2: Test request with torch profiling
        print("\n" + "-" * 50)
        print("PHASE 2: Test Request (with appended tokens + profiling)")
        print("-" * 50)

        server.start_profile()
        time.sleep(0.5)  # Give profiler time to start

        test_result = send_request(client, model_name, test_prompt, args.output_len)
        results["test"] = test_result

        time.sleep(0.5)  # Let profiler capture everything
        server.stop_profile()

        if test_result["success"]:
            print(f"Test TTFT: {test_result['ttft_ms']:.2f} ms")
            print(f"Test Total: {test_result['total_ms']:.2f} ms")
        else:
            print("Test request failed!")

    # Calculate metrics
    if results["warmup"]["success"] and results["test"]["success"]:
        # Speedup: how much faster is the test TTFT compared to expected full prefill
        # Expected full prefill time = warmup_ttft * (total_tokens / base_tokens)
        total_tokens = args.base_len + args.append_len
        expected_ttft = results["warmup"]["ttft_ms"] * (total_tokens / args.base_len)
        actual_ttft = results["test"]["ttft_ms"]

        if actual_ttft > 0:
            results["speedup"] = expected_ttft / actual_ttft
            results["expected_ttft_ms"] = expected_ttft

    # Save results
    results_file = run_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    if results["warmup"]["success"]:
        print(
            f"Warmup TTFT:      {results['warmup']['ttft_ms']:>10.2f} ms  (base: {args.base_len} tokens)"
        )

    if results["test"]["success"]:
        print(
            f"Test TTFT:        {results['test']['ttft_ms']:>10.2f} ms  (base + {args.append_len} tokens)"
        )

    if results.get("expected_ttft_ms"):
        print(
            f"Expected TTFT:    {results['expected_ttft_ms']:>10.2f} ms  (if no cache reuse)"
        )

    if results.get("speedup"):
        print("-" * 70)
        print(f"Layer-wise Speedup: {results['speedup']:.2f}x")

        if results["speedup"] > 1.5:
            print("Result: Layer-wise KV cache transfer is effective!")
        elif results["speedup"] > 1.0:
            print("Result: Marginal benefit from layer-wise transfer")
        else:
            print("Result: No benefit observed (may need investigation)")

    print("=" * 70)
    print(f"\nResults saved to: {results_file}")
    print(f"Torch traces saved to: {torch_trace_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark layer-wise KV cache transfer benefit"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model path or identifier"
    )
    parser.add_argument(
        "--base-len",
        type=int,
        default=None,
        help="Base prompt length in tokens (default: model-dependent, 10000 for Qwen3, 4000 for Qwen3.5)",
    )
    parser.add_argument(
        "--append-len",
        type=int,
        default=1024,
        help="Additional tokens appended for test request (default: 128)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="Output tokens to generate (default: 1)",
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="vLLM server port (default: 8001)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/bench_results",
        help="Results directory (default: examples/bench_results)",
    )

    args = parser.parse_args()

    if args.base_len is None:
        model_lower = args.model.lower()
        if "qwen3.5" in model_lower or "qwen35" in model_lower:
            args.base_len = 4000
        else:
            args.base_len = 10000

    run_benchmark(args)


if __name__ == "__main__":
    main()
