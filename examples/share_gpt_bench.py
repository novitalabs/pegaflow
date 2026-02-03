#!/usr/bin/env python3
"""
Benchmark vLLM server using ShareGPT dataset with multi-turn conversations.

Usage:
    python share_gpt_bench.py --model meta-llama/Llama-3.1-8B
    python share_gpt_bench.py --model Qwen/Qwen2.5-7B --num-conversations 50
"""

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Default ShareGPT dataset URL
DEFAULT_DATASET_URL = "https://huggingface.co/datasets/philschmid/sharegpt-raw/resolve/main/sharegpt_20230401_clean_lang_split.json"


def download_dataset(url: str, output_path: Path):
    """Download ShareGPT dataset if not exists."""
    if output_path.exists():
        print(f"Dataset already exists at: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShareGPT dataset from {url}...")

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
        print(f"✓ Dataset downloaded to: {output_path}\n")
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        print(
            "\nYou can manually download the dataset and specify it with --dataset-path"
        )
        sys.exit(1)


def convert_sharegpt_dataset(
    input_path: Path,
    output_path: Path,
    max_items: int,
    seed: int,
) -> Path:
    """Convert ShareGPT dataset to OpenAI format for multi-turn benchmark."""
    if output_path.exists():
        print(f"Converted dataset already exists at: {output_path}")
        return output_path

    print("Converting ShareGPT dataset to OpenAI format...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Max items: {max_items}, Seed: {seed}\n")

    # Find convert_sharegpt_to_openai.py script
    # Try to find it in vLLM benchmarks directory
    vllm_bench_dir = (
        Path(__file__).parent.parent.parent / "vllm" / "benchmarks" / "multi_turn"
    )
    convert_script = vllm_bench_dir / "convert_sharegpt_to_openai.py"

    if not convert_script.exists():
        # Try alternative locations
        import shutil

        convert_script_path = shutil.which("convert_sharegpt_to_openai.py")
        if convert_script_path:
            convert_script = Path(convert_script_path)
        else:
            raise FileNotFoundError(
                f"Cannot find convert_sharegpt_to_openai.py. "
                f"Expected at: {convert_script}\n"
                f"Please ensure vLLM is installed and the script is available."
            )

    cmd = [
        sys.executable,
        str(convert_script),
        str(input_path),
        str(output_path),
        "--seed",
        str(seed),
        "--max-items",
        str(max_items),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Conversion failed with return code {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError("Failed to convert ShareGPT dataset")

    print(f"✓ Dataset converted to: {output_path}\n")
    return output_path


def run_benchmark(args, converted_dataset_path: Path, output_file: Path = None):
    """Run benchmark_serving_multi_turn.py with ShareGPT dataset."""
    # Find benchmark_serving_multi_turn.py script
    vllm_bench_dir = (
        Path(__file__).parent.parent.parent / "vllm" / "benchmarks" / "multi_turn"
    )
    benchmark_script = vllm_bench_dir / "benchmark_serving_multi_turn.py"

    if not benchmark_script.exists():
        # Try alternative locations
        import shutil

        benchmark_script_path = shutil.which("benchmark_serving_multi_turn.py")
        if benchmark_script_path:
            benchmark_script = Path(benchmark_script_path)
        else:
            raise FileNotFoundError(
                f"Cannot find benchmark_serving_multi_turn.py. "
                f"Expected at: {benchmark_script}\n"
                f"Please ensure vLLM is installed and the script is available."
            )

    url = f"http://{args.host}:{args.port}"
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--input-file",
        str(converted_dataset_path),
        "--model",
        args.model,
        "--url",
        url,
        "--seed",
        str(args.seed),
        "--request-rate",
        str(args.request_rate),
        "--num-clients",
        str(args.num_clients),
    ]

    # Add optional arguments
    if args.served_model_name:
        cmd.extend(["--served-model-name", args.served_model_name])
    else:
        cmd.extend(["--served-model-name", args.model])

    if args.max_active_conversations:
        cmd.extend(["--max-active-conversations", str(args.max_active_conversations)])

    if args.max_num_requests:
        cmd.extend(["--max-num-requests", str(args.max_num_requests)])

    if args.max_turns:
        cmd.extend(["--max-turns", str(args.max_turns)])

    if args.warmup_step:
        cmd.append("--warmup-step")

    if args.verbose:
        cmd.append("--verbose")

    if args.excel_output:
        cmd.append("--excel-output")

    if output_file:
        cmd.extend(["--output-file", str(output_file)])

    print(f"Running ShareGPT multi-turn benchmark: {args.label}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"Benchmark failed with return code {result.returncode}")
        raise RuntimeError(f"Benchmark failed: {args.label}")

    print("\n✓ Benchmark complete")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM server with ShareGPT dataset (multi-turn conversations)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g., meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Model name used in API (default: same as --model)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ShareGPT dataset JSON file (if not provided, will auto-download)",
    )
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=100,
        help="Number of conversations to test (default: 50)",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=2,
        help="Number of parallel clients (default: 2)",
    )
    parser.add_argument(
        "--max-active-conversations",
        type=int,
        default=None,
        help="Max number of active conversations at a time (default: None)",
    )
    parser.add_argument(
        "--max-num-requests",
        type=int,
        default=None,
        help="Max number of requests to send (default: None)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns per conversation (default: None, use all turns)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=0.5,
        help="Request rate per client in requests/sec (default: 0.0, no delay)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--warmup-step",
        action="store_true",
        help="Run warmup step before benchmark",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--excel-output",
        action="store_true",
        help="Export results to Excel file",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="sharegpt_multiturn",
        help="Label for this benchmark run (default: sharegpt_multiturn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/bench_results",
        help="Directory to save benchmark results (default: examples/bench_results)",
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / f"sharegpt_multiturn_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Handle dataset download
    if not args.dataset_path:
        dataset_path = output_dir / "sharegpt_raw.json"
        download_dataset(DEFAULT_DATASET_URL, dataset_path)
        args.dataset_path = str(dataset_path)

    # Convert ShareGPT dataset to OpenAI format
    converted_dataset_path = run_dir / "sharegpt_converted.json"
    convert_sharegpt_dataset(
        Path(args.dataset_path),
        converted_dataset_path,
        max_items=args.num_conversations,
        seed=args.seed,
    )

    print("\n" + "=" * 70)
    print("SHAREGPT MULTI-TURN BENCHMARK")
    print("=" * 70)
    print(f"Model:                    {args.model}")
    print(f"Served Model Name:        {args.served_model_name or args.model}")
    print(f"Server:                   http://{args.host}:{args.port}")
    print(f"Num Conversations:        {args.num_conversations}")
    print(f"Num Clients:              {args.num_clients}")
    print(f"Max Active Conversations: {args.max_active_conversations or 'None'}")
    print(f"Max Turns:                {args.max_turns or 'All'}")
    print(f"Request Rate:             {args.request_rate} req/s per client")
    print(f"Random Seed:              {args.seed}")
    print(f"Raw Dataset:             {args.dataset_path}")
    print(f"Converted Dataset:        {converted_dataset_path}")
    print(f"Results Dir:              {run_dir}")
    print("=" * 70 + "\n")

    try:
        # Run benchmark
        output_file = (
            run_dir / "conversations_output.json" if args.excel_output else None
        )
        run_benchmark(args, converted_dataset_path, output_file)

        print(f"\n✓ Results saved to: {run_dir}")
        if output_file:
            print(f"  Output conversations: {output_file}")
        print()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
