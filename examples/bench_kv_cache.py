#!/usr/bin/env python3
"""
Benchmark script to compare vLLM baseline performance against PegaFlow's KV cache connector.

This script automates the following workflow:
1. Start baseline vLLM server (prefix caching disabled)
2. Run benchmark CLI - first execution (cold cache)
3. Run benchmark CLI - second execution (warm cache)
4. Stop baseline vLLM server
5. Start vLLM server with PegaFlow connector enabled
6. Run benchmark CLI - first execution (cold cache)
7. Run benchmark CLI - second execution (warm cache)
8. Stop vLLM server with PegaFlow
9. Collect and compare all four sets of results
10. Save results to a file

Usage:
    python examples/bench_kv_cache.py [--model MODEL] [--num-prompts N] [--input-len L] [--output-len O]
"""

import argparse
import json
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional


class VLLMServer:
    """Context manager for vLLM server lifecycle."""
    
    def __init__(self, model: str, port: int, use_pegaflow: bool = False, 
                 enable_prefix_caching: bool = False):
        self.model = model
        self.port = port
        self.use_pegaflow = use_pegaflow
        self.enable_prefix_caching = enable_prefix_caching
        self.process: Optional[subprocess.Popen] = None
        
    def __enter__(self):
        """Start the vLLM server."""
        cmd = [
            "vllm", "serve", self.model,
            "--port", str(self.port),
            "--enforce-eager",
            "--trust-remote-code",
        ]

        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")
        
        if self.use_pegaflow:
            # Add PegaFlow KV connector configuration using JSON format
            kv_config = {
                "kv_connector": "PegaKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
            cmd.extend([
                "--kv-transfer-config", json.dumps(kv_config),
            ])
        
        server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
        cache_state = "enabled" if self.enable_prefix_caching else "disabled"
        print(
            f"\n[{server_label}] Starting vLLM server on port {self.port} "
            f"(prefix caching {cache_state})"
        )

        # Redirect output to /dev/null to avoid blocking
        with open('/dev/null', 'w') as devnull:
            self.process = subprocess.Popen(
                cmd,
                stdout=devnull,
                stderr=devnull,
            )
        
        # Wait for server to be ready
        self._wait_for_ready()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: U100
        """Stop the vLLM server."""
        if self.process:
            server_label = "PegaFlow" if self.use_pegaflow else "Baseline"
            print(f"\n[{server_label}] Stopping vLLM server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            print("Server stopped.\n")
    
    def _wait_for_ready(self, timeout: int = 120):
        """Wait for the server to be ready to accept requests."""
        import requests
        
        start_time = time.time()
        url = f"http://localhost:{self.port}/health"
        
        print("Waiting for server to be ready...")
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=1)
                if response.status_code == 200:
                    print("✓ Server is ready!\n")
                    time.sleep(2)  # Extra buffer
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        
        raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def run_benchmark(model: str, port: int, num_prompts: int, input_len: int, output_len: int,
                  result_file: Path, label: str, request_rate: float = 1.0,
                  seed: int = 42) -> dict:
    """Run vllm bench serve and return the results."""
    cmd = [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--host", "localhost",
        "--port", str(port),
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--request-rate", str(request_rate),  # Control request arrival rate
        "--seed", str(seed),  # Fixed seed for reproducible requests
        "--save-result",
        "--result-filename", result_file.name,
        "--result-dir", str(result_file.parent),
        "--label", label,
    ]

    print(f"- {label}: running vLLM bench serve on port {port}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Benchmark failed: {label}")

    print(f"  {label}: complete -> {result_file.name}")

    # Load and return the results
    with open(result_file, 'r') as f:
        return json.load(f)


def print_comparison(results: dict, args):
    """Print a concise summary of the four benchmark runs."""
    print("\n" + "=" * 80)
    print("KV CACHE BENCHMARK SUMMARY (TTFT FOCUSED)")
    print("=" * 80)

    print(
        "Model: {model} | Prompts: {num_prompts} | Input: {input_len} tok | "
        "Output: {output_len} tok | Rate: {request_rate} req/s | Seed: {seed}".format(
            model=args.model,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
            output_len=args.output_len,
            request_rate=args.request_rate,
            seed=args.seed,
        )
    )

    configs = [
        ("baseline_cold", "Baseline (Cold)"),
        ("baseline_warm", "Baseline (Warm)"),
        ("pegaflow_cold", "PegaFlow (Cold)"),
        ("pegaflow_warm", "PegaFlow (Warm)"),
    ]

    columns = [
        ("mean_ttft_ms", "TTFT mean (ms)", 16),
        ("p99_ttft_ms", "TTFT p99 (ms)", 15),
        ("request_throughput", "Req/s", 10),
        ("output_throughput", "Tok/s", 11),
        ("duration", "Duration (s)", 14),
    ]

    header = "{:<22}".format("Configuration") + "".join(
        f"{title:>{width}}" for _, title, width in columns
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    def _format_value(value):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
        return str(value)

    for key, label in configs:
        data = results.get(key)
        if not data:
            continue
        row = f"{label:<22}"
        for metric_key, _, width in columns:
            value = _format_value(data.get(metric_key))
            row += f"{value:>{width}}"
        print(row)

    print("-" * len(header))
    print("TTFT columns are the primary signals; throughput/duration are included for context.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM baseline vs PegaFlow KV cache connector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model to benchmark (default: gpt2)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to benchmark (default: 20)"
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=524,
        help="Input prompt length in tokens (default: 1024)"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=1,
        help="Output length in tokens (default: 1)"
    )
    parser.add_argument(
        "--baseline-port",
        type=int,
        default=8000,
        help="Port for baseline vLLM server (default: 8000)"
    )
    parser.add_argument(
        "--pegaflow-port",
        type=int,
        default=8001,
        help="Port for PegaFlow vLLM server (default: 8001)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/bench_results",
        help="Directory to save benchmark results (default: examples/bench_results)"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Request rate in requests per second. Use 1.0 to send requests one by one. "
             "Use 'inf' for sending all at once (default: 1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible request generation. "
             "Same seed ensures cold and warm runs use identical requests (default: 42)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this benchmark run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"kv_cache_bench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PEGAFLOW KV CACHE BENCHMARK")
    print("="*70)
    print(f"Model:           {args.model}")
    print(f"Num Prompts:     {args.num_prompts}")
    print(f"Input Length:    {args.input_len} tokens")
    print(f"Output Length:   {args.output_len} tokens")
    print(f"Request Rate:    {args.request_rate} req/s")
    print(f"Random Seed:     {args.seed}")
    print(f"Baseline Port:   {args.baseline_port}")
    print(f"PegaFlow Port:   {args.pegaflow_port}")
    print(f"Results Dir:     {run_dir}")
    print("="*70)

    all_results = {}

    # Phase 1: Baseline vLLM (no prefix caching)
    print("\n" + "="*70)
    print("PHASE 1: BASELINE vLLM (Prefix Caching Disabled)")
    print("="*70)

    with VLLMServer(args.model, args.baseline_port, use_pegaflow=False,
                    enable_prefix_caching=False):
        # Cold cache run
        result_file = run_dir / "baseline_cold.json"
        all_results["baseline_cold"] = run_benchmark(
            args.model, args.baseline_port, args.num_prompts, args.input_len,
            args.output_len, result_file, "baseline_cold", args.request_rate, args.seed
        )

        # Warm cache run (same requests again - using same seed for identical requests)
        result_file = run_dir / "baseline_warm.json"
        all_results["baseline_warm"] = run_benchmark(
            args.model, args.baseline_port, args.num_prompts, args.input_len,
            args.output_len, result_file, "baseline_warm", args.request_rate, args.seed
        )

    # Phase 2: PegaFlow vLLM (with KV connector)
    print("\n" + "="*70)
    print("PHASE 2: PEGAFLOW vLLM (KV Cache Connector Enabled)")
    print("="*70)

    with VLLMServer(args.model, args.pegaflow_port, use_pegaflow=True,
                    enable_prefix_caching=False):
        # Cold cache run
        result_file = run_dir / "pegaflow_cold.json"
        all_results["pegaflow_cold"] = run_benchmark(
            args.model, args.pegaflow_port, args.num_prompts, args.input_len,
            args.output_len, result_file, "pegaflow_cold", args.request_rate, args.seed
        )

        # Warm cache run (same requests again - using same seed for identical requests)
        result_file = run_dir / "pegaflow_warm.json"
        all_results["pegaflow_warm"] = run_benchmark(
            args.model, args.pegaflow_port, args.num_prompts, args.input_len,
            args.output_len, result_file, "pegaflow_warm", args.request_rate, args.seed
        )

    # Save combined results
    combined_file = run_dir / "combined_results.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to: {run_dir}")

    # Print comparison
    print_comparison(all_results, args)

    print("\n" + "="*70)
    print("BENCHMARK COMPLETED!")
    print("="*70)
    print(f"\nResults directory: {run_dir}")
    print(f"Combined results:  {combined_file}")
    print()


if __name__ == "__main__":
    main()
