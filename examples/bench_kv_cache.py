#!/usr/bin/env python3
"""
Benchmark script to compare LMCache performance against PegaFlow's KV cache connector.

This script automates the following workflow:
1. Start LMCache vLLM server (local CPU backend) [optional, enable with --with-lmcache]
2. Run benchmark CLI - first execution (cold cache)
3. Run benchmark CLI - second execution (warm cache)
4. Stop LMCache vLLM server
5. Start vLLM server with PegaFlow connector enabled
6. Run benchmark CLI - first execution (cold cache)
7. Run benchmark CLI - second execution (warm cache)
8. Stop vLLM server with PegaFlow
9. Collect and compare all sets of results
10. Save results to a file

Usage:
    # Default: PegaFlow only (faster for development):
    python examples/bench_kv_cache.py [--model MODEL] [--num-prompts N] [--input-len L] [--output-len O]

    # Full benchmark with LMCache comparison:
    python examples/bench_kv_cache.py --with-lmcache [--model MODEL] [--num-prompts N] [--input-len L] [--output-len O]
"""

import argparse
import json
import shutil
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, Sequence


class VLLMServer:
    """Context manager for vLLM server lifecycle."""

    def __init__(
        self,
        model: str,
        port: int,
        use_pegaflow: bool = False,
        use_lmcache: bool = False,
        enable_prefix_caching: bool = False,
        log_file: Optional[Path] = None,
        health_endpoints: Optional[Sequence[str]] = None,
        profile_output: Optional[Path] = None,
        torch_profile_output: Optional[Path] = None,
        max_model_len: Optional[int] = None,
    ):
        self.model = model
        self.port = port
        self.use_pegaflow = use_pegaflow
        self.use_lmcache = use_lmcache
        self.enable_prefix_caching = enable_prefix_caching
        self.max_model_len = max_model_len
        self.log_file = log_file
        self.health_endpoints = (
            list(health_endpoints)
            if health_endpoints
            else [
                "/health",
                "/metrics",
                "/invocations",
            ]
        )
        self.profile_output = profile_output
        self.torch_profile_output = torch_profile_output
        self.process: Optional[subprocess.Popen] = None
        self.log_handle = None

    def __enter__(self):
        """Start the vLLM server."""
        import os

        env = os.environ.copy()

        # Set up LMCache environment variables if using LMCache
        if self.use_lmcache:
            env["LMCACHE_CHUNK_SIZE"] = "256"
            env["LMCACHE_LOCAL_CPU"] = "True"
            env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "32.0"

        if self.torch_profile_output:
            env["VLLM_TORCH_PROFILER_DIR"] = str(self.torch_profile_output)

        cmd = []
        if self.profile_output:
            cmd.extend(
                [
                    "nsys",
                    "profile",
                    "--output",
                    str(self.profile_output),
                    "--force-overwrite",
                    "true",
                ]
            )

        cmd.extend(
            [
                "vllm",
                "serve",
                self.model,
                "--port",
                str(self.port),
                "--trust-remote-code",
                "--block-size",
                "64",
            ]
        )

        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        if not self.enable_prefix_caching:
            cmd.append("--no-enable-prefix-caching")

        if self.use_pegaflow:
            # Add PegaFlow KV connector configuration using JSON format
            kv_config = {
                "kv_connector": "PegaKVConnector",
                "kv_role": "kv_both",
                "kv_connector_module_path": "pegaflow.connector",
            }
            cmd.extend(
                [
                    "--kv-transfer-config",
                    json.dumps(kv_config),
                ]
            )
        elif self.use_lmcache:
            # Add LMCache connector configuration using JSON format
            kv_config = {
                "kv_connector": "LMCacheConnectorV1",
                "kv_role": "kv_both",
            }
            cmd.extend(
                [
                    "--kv-transfer-config",
                    json.dumps(kv_config),
                ]
            )

        server_label = (
            "PegaFlow"
            if self.use_pegaflow
            else ("LMCache" if self.use_lmcache else "Baseline")
        )
        cache_state = "enabled" if self.enable_prefix_caching else "disabled"
        print(
            f"\n[{server_label}] Starting vLLM server on port {self.port} "
            f"(prefix caching {cache_state})"
        )

        # Redirect output to log file if provided, otherwise to /dev/null
        if self.log_file:
            print(f"[{server_label}] Logging to: {self.log_file}")
            self.log_handle = open(self.log_file, "w")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                env=env,
            )
        else:
            with open("/dev/null", "w") as devnull:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=devnull,
                    stderr=devnull,
                    env=env,
                )

        # Wait for server to be ready
        self._wait_for_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: U100
        """Stop the vLLM server."""
        if self.process:
            server_label = (
                "PegaFlow"
                if self.use_pegaflow
                else ("LMCache" if self.use_lmcache else "Baseline")
            )
            print(f"\n[{server_label}] Stopping vLLM server...")
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
            print("Server stopped.\n")

        # Close log file if it was opened
        if self.log_handle:
            self.log_handle.close()

    def _wait_for_ready(self, timeout: int = 180):
        """Wait for the server to be ready to accept requests."""
        import requests

        start_time = time.time()
        print("Waiting for server to be ready...")
        while time.time() - start_time < timeout:
            for endpoint in self.health_endpoints:
                url = f"http://localhost:{self.port}{endpoint}"
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        print(f"✓ Server is ready! (checked {endpoint})\n")
                        time.sleep(2)  # Extra buffer
                        return
                except requests.exceptions.RequestException:
                    continue
            time.sleep(2)

        endpoints_str = ", ".join(self.health_endpoints)
        raise TimeoutError(
            f"Server did not become ready within {timeout} seconds "
            f"(endpoints tried: {endpoints_str})"
        )


def run_benchmark(
    model: str,
    port: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    result_file: Path,
    label: str,
    request_rate: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run vllm bench serve and return the results."""
    cmd = [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "localhost",
        "--port",
        str(port),
        "--model",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(request_rate),  # Control request arrival rate
        "--seed",
        str(seed),  # Fixed seed for reproducible requests
        "--ready-check-timeout",
        "0",  # Skip endpoint ready check to avoid extra requests
        "--save-result",
        "--result-filename",
        result_file.name,
        "--result-dir",
        str(result_file.parent),
        "--label",
        label,
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
    with open(result_file, "r") as f:
        return json.load(f)


def print_comparison(results: dict, args):
    """Print a concise summary of the benchmark runs."""
    print("\n" + "=" * 80)
    if args.with_lmcache:
        print("LMCACHE vs PEGAFLOW KV CACHE BENCHMARK SUMMARY (TTFT FOCUSED)")
    else:
        print("PEGAFLOW KV CACHE BENCHMARK SUMMARY (TTFT FOCUSED)")
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

    # Build configs list based on what benchmarks were included
    configs = []
    if args.with_baseline:
        configs.append(("baseline", "Baseline (No KV)"))
    if args.with_lmcache:
        configs.extend(
            [
                ("lmcache_cold", "LMCache (Cold)"),
                ("lmcache_warm", "LMCache (Warm)"),
            ]
        )
    configs.extend(
        [
            ("pegaflow_cold", "PegaFlow (Cold)"),
            ("pegaflow_warm", "PegaFlow (Warm)"),
        ]
    )

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
    print(
        "TTFT columns are the primary signals; throughput/duration are included for context."
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LMCache local CPU vs PegaFlow KV cache connector"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Path or identifier for the model to benchmark (e.g. /work/models/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=50,
        help="Number of prompts to benchmark (default: 20)",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=2048,
        help="Input prompt length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--output-len", type=int, default=1, help="Output length in tokens (default: 1)"
    )
    parser.add_argument(
        "--lmcache-port",
        type=int,
        default=8000,
        help="Port for LMCache vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--pegaflow-port",
        type=int,
        default=8001,
        help="Port for PegaFlow vLLM server (default: 8001)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/bench_results",
        help="Directory to save benchmark results (default: examples/bench_results)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=4.0,
        help="Request rate in requests per second. Use 1.0 to send requests one by one. "
        "Use 'inf' for sending all at once (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible request generation. "
        "Same seed ensures cold and warm runs use identical requests (default: 42)",
    )
    parser.add_argument(
        "--with-lmcache",
        action="store_true",
        help="Include LMCache benchmark for comparison. "
        "By default, only PegaFlow is tested (faster for development).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable nsys profiling for the vLLM server. Profiling results will be saved to the output directory.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Enable torch profiling for both cold and warm runs (saved to separate directories)",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="Include pure vLLM baseline (no KV connector, single run) for comparison.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length. If not specified, uses the model's default.",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this benchmark run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"kv_cache_bench_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    benchmark_title = (
        "PEGAFLOW vs LMCACHE KV CACHE BENCHMARK"
        if args.with_lmcache
        else "PEGAFLOW KV CACHE BENCHMARK"
    )
    print(benchmark_title)
    print("=" * 70)
    print(f"Model:           {args.model}")
    print(f"Num Prompts:     {args.num_prompts}")
    print(f"Input Length:    {args.input_len} tokens")
    print(f"Output Length:   {args.output_len} tokens")
    print(f"Request Rate:    {args.request_rate} req/s")
    print(f"Random Seed:     {args.seed}")
    if args.with_lmcache:
        print(f"LMCache Port:    {args.lmcache_port}")
    print(f"PegaFlow Port:   {args.pegaflow_port}")
    if args.max_model_len:
        print(f"Max Model Len:   {args.max_model_len}")
    print(f"Results Dir:     {run_dir}")
    print(f"Profiling:       {'Enabled' if args.profile else 'Disabled'}")
    if not args.with_baseline:
        print("Baseline:        Skipped (use --with-baseline to enable)")
    if not args.with_lmcache:
        print("LMCache:         Skipped (use --with-lmcache to enable)")
    print("=" * 70)

    all_results = {}

    # Track phase numbering dynamically
    phase_counter = 1

    # Phase: Pure vLLM Baseline (no KV connector, single run) - only run if requested
    if args.with_baseline:
        print("\n" + "=" * 70)
        print(f"PHASE {phase_counter}: PURE vLLM BASELINE (No KV Connector)")
        print("=" * 70)
        phase_counter += 1

        baseline_log = run_dir / "baseline_server.log"
        profile_path = run_dir / "baseline" if args.profile else None
        with VLLMServer(
            args.model,
            args.pegaflow_port,
            use_pegaflow=False,
            use_lmcache=False,
            enable_prefix_caching=False,
            log_file=baseline_log,
            profile_output=profile_path,
            max_model_len=args.max_model_len,
        ):
            # Single run (no warm run needed since there's no external cache)
            result_file = run_dir / "baseline.json"
            all_results["baseline"] = run_benchmark(
                args.model,
                args.pegaflow_port,
                args.num_prompts,
                args.input_len,
                args.output_len,
                result_file,
                "baseline",
                args.request_rate,
                args.seed,
            )
    else:
        print("\n" + "=" * 70)
        print(
            f"PHASE {phase_counter}: BASELINE - SKIPPED (use --with-baseline to enable)"
        )
        print("=" * 70)
        phase_counter += 1

    # Phase: LMCache vLLM (local CPU backend) - only run if requested
    if args.with_lmcache:
        print("\n" + "=" * 70)
        print(f"PHASE {phase_counter}: LMCACHE vLLM (Local CPU Backend)")
        print("=" * 70)
        phase_counter += 1

        lmcache_log = run_dir / "lmcache_server.log"
        profile_path = run_dir / "lmcache" if args.profile else None
        with VLLMServer(
            args.model,
            args.lmcache_port,
            use_lmcache=True,
            enable_prefix_caching=False,
            log_file=lmcache_log,
            profile_output=profile_path,
            max_model_len=args.max_model_len,
        ):
            # Cold cache run
            result_file = run_dir / "lmcache_cold.json"
            all_results["lmcache_cold"] = run_benchmark(
                args.model,
                args.lmcache_port,
                args.num_prompts,
                args.input_len,
                args.output_len,
                result_file,
                "lmcache_cold",
                args.request_rate,
                args.seed,
            )

            # Warm cache run (same requests again - using same seed for identical requests)
            result_file = run_dir / "lmcache_warm.json"
            all_results["lmcache_warm"] = run_benchmark(
                args.model,
                args.lmcache_port,
                args.num_prompts,
                args.input_len,
                args.output_len,
                result_file,
                "lmcache_warm",
                args.request_rate,
                args.seed,
            )
    else:
        print("\n" + "=" * 70)
        print(
            f"PHASE {phase_counter}: LMCACHE - SKIPPED (use --with-lmcache to enable)"
        )
        print("=" * 70)
        phase_counter += 1

    # Phase: PegaFlow vLLM (with KV connector)
    print("\n" + "=" * 70)
    print(f"PHASE {phase_counter}: PEGAFLOW vLLM (KV Cache Connector Enabled)")
    print("=" * 70)

    pegaflow_log = run_dir / "pegaflow_server.log"
    profile_path = run_dir / "pegaflow" if args.profile else None
    # Use a temp directory for torch profiler, we'll move files after each run
    torch_profile_base = run_dir / "torch_trace_tmp" if args.torch_profile else None
    torch_profile_cold = run_dir / "torch_trace_cold" if args.torch_profile else None
    torch_profile_warm = run_dir / "torch_trace_warm" if args.torch_profile else None

    if args.torch_profile:
        torch_profile_base.mkdir(parents=True, exist_ok=True)
        torch_profile_cold.mkdir(parents=True, exist_ok=True)
        torch_profile_warm.mkdir(parents=True, exist_ok=True)

    with VLLMServer(
        args.model,
        args.pegaflow_port,
        use_pegaflow=True,
        enable_prefix_caching=False,
        log_file=pegaflow_log,
        profile_output=profile_path,
        torch_profile_output=torch_profile_base,
        max_model_len=args.max_model_len,
    ):
        # Cold cache run
        if args.torch_profile:
            import requests

            print("Starting torch profile for cold run...")
            requests.post(f"http://localhost:{args.pegaflow_port}/start_profile")

        result_file = run_dir / "pegaflow_cold.json"
        all_results["pegaflow_cold"] = run_benchmark(
            args.model,
            args.pegaflow_port,
            args.num_prompts,
            args.input_len,
            args.output_len,
            result_file,
            "pegaflow_cold",
            args.request_rate,
            args.seed,
        )

        if args.torch_profile:
            import requests

            print("Stopping torch profile for cold run...")
            requests.post(f"http://localhost:{args.pegaflow_port}/stop_profile")
            time.sleep(1)  # Wait for trace files to be written
            # Move trace files to cold directory
            for f in torch_profile_base.iterdir():
                shutil.move(str(f), str(torch_profile_cold / f.name))
            print(f"  Cold trace saved to: {torch_profile_cold}")

        # Warm cache run (same requests again - using same seed for identical requests)
        if args.torch_profile:
            import requests

            print("Starting torch profile for warm run...")
            requests.post(f"http://localhost:{args.pegaflow_port}/start_profile")

        result_file = run_dir / "pegaflow_warm.json"
        all_results["pegaflow_warm"] = run_benchmark(
            args.model,
            args.pegaflow_port,
            args.num_prompts,
            args.input_len,
            args.output_len,
            result_file,
            "pegaflow_warm",
            args.request_rate,
            args.seed,
        )

        if args.torch_profile:
            import requests

            print("Stopping torch profile for warm run...")
            requests.post(f"http://localhost:{args.pegaflow_port}/stop_profile")
            time.sleep(1)  # Wait for trace files to be written
            # Move trace files to warm directory
            for f in torch_profile_base.iterdir():
                shutil.move(str(f), str(torch_profile_warm / f.name))
            print(f"  Warm trace saved to: {torch_profile_warm}")
            # Remove temp directory
            torch_profile_base.rmdir()

    # Save combined results
    combined_file = run_dir / "combined_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to: {run_dir}")

    # Print comparison
    print_comparison(all_results, args)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED!")
    print("=" * 70)
    print(f"\nResults directory: {run_dir}")
    print(f"Combined results:  {combined_file}")
    print()


if __name__ == "__main__":
    main()
