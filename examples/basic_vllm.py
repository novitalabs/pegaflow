"""
Basic vLLM example - Test KV cache save and load with PegaFlow connector

This example demonstrates:
1. First run: Generate text and save KV cache to CPU
2. Second run: Generate same prompt again and load KV cache from CPU
"""

import random
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def generate_long_prompt(length: int) -> str:
    """Randomly generate a prompt string with (at least) the given length."""
    words = [
        "apple",
        "banana",
        "orange",
        "kiwi",
        "mango",
        "grape",
        "quantum",
        "tensor",
        "network",
        "compute",
        "memory",
        "rdma",
        "cache",
        "pega",
        "flow",
        "mooncake",
        "nvlink",
        "distributed",
        "system",
        "prefill",
        "decode",
        "chunk",
    ]

    parts = []
    total_len = 0

    while total_len < length:
        w = random.choice(words)
        parts.append(w)
        total_len += len(w) + 1

    full = " ".join(parts)
    return full[:length]


def real_long_prompt() -> str:
    return (
        "Hello, this is a longer piece of text written for testing purposes. "
        "Imagine a quiet morning in a small coastal town, the kind of place "
        "where the streets are still empty when the sun rises and the smell of "
        "the sea drifts in through every open window. The houses stand close "
        "together, their walls painted in soft colors that have faded slightly "
        "over the years, giving the town a gentle and familiar feeling. "
        "A bakery opens its doors early, letting warm air spill out along with "
        "the scent of fresh bread. A few fishermen prepare their boats at the "
        "harbor, moving slowly and without hurry, as if the rhythm of the town "
        "is set by the tide itself. "
        "This text continues in a calm and steady way, describing nothing more "
        "than simple scenes and small moments. It does not introduce dramatic "
        "events or complicated ideas, because its purpose is simply to provide "
        "a long, predictable prompt. You can use it to observe whether the "
        "system handles extended context normally, without sudden changes in "
        "behavior. The tone stays even, the sentences unfold at a relaxed pace, "
        "and the details remain grounded in everyday life. "
        "If you need an even longer prompt, you can extend this one by adding "
        "more scenes from the same quiet town, or simply repeat the structure "
        "to create additional length for stress testing."
    )


def main():
    print("=" * 70)
    print("PegaFlow KV Cache Test - Save and Load")
    print("=" * 70)

    # Configure vLLM to use our PegaKVConnector
    kv_transfer_config = KVTransferConfig(
        kv_connector="PegaKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="pegaflow.connector",
    )

    # Initialize vLLM with GPT-2
    print("\n[1/4] Loading model...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        enable_prefix_caching=False,  # Disable vLLM's internal prefix cache
        kv_transfer_config=kv_transfer_config,
    )
    print("✓ Model loaded successfully!")

    # Test prompt - use a long prompt (~2048 characters)
    print("\n[Generating long prompt (~2048 chars)...]")
    prompt = real_long_prompt()
    print(f"✓ Generated prompt with {len(prompt)} characters")
    print(f"Preview: {prompt[:100]}...")

    # Sampling parameters - use temperature=0 for deterministic output
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=50,
    )

    # First run: Generate and save KV cache
    print("\n" + "=" * 70)
    print("[2/4] First run - Generating text (will save KV cache to CPU)...")
    print("=" * 70)
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    first_run_time = time.time() - start_time

    first_output = outputs[0].outputs[0].text
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Prompt preview: {prompt[:100]}...")
    print(f"Generated: {first_output}")
    print(f"Time: {first_run_time:.3f}s")

    # Second run: Same prompt, should load from CPU cache
    print("\n" + "=" * 70)
    print("[3/4] Second run - Same prompt (should load KV cache from CPU)...")
    print("=" * 70)
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    second_run_time = time.time() - start_time

    second_output = outputs[0].outputs[0].text
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Prompt preview: {prompt[:100]}...")
    print(f"Generated: {second_output}")
    print(f"Time: {second_run_time:.3f}s")

    # Compare results
    print("\n" + "=" * 70)
    print("[4/4] Results Summary")
    print("=" * 70)
    print(f"First run time:  {first_run_time:.3f}s")
    print(f"Second run time: {second_run_time:.3f}s")

    if first_output == second_output:
        print("✓ Outputs match (deterministic generation working)")
    else:
        print("✗ Outputs differ (unexpected)")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
