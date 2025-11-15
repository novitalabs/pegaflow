"""
Basic vLLM example - Start a small language model for development
"""
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def main():
    # Initialize vLLM with a small 125M parameter model
    # Using GPT-2 125M as it's lightweight and good for development
    print("Loading model...")

    # Configure vLLM to use our local PegaKVConnector (v1 KV connector).
    kv_transfer_config = KVTransferConfig(
        kv_connector="PegaKVConnector",
        kv_role="kv_both",
        kv_connector_module_path="pegaflow.connector",
    )

    llm = LLM(
        model="gpt2",  # GPT-2 125M model
        trust_remote_code=True,
        max_model_len=512,  # Smaller context for faster loading
        enforce_eager=True,  # Disable CUDA graphs to avoid PTX issues
        kv_transfer_config=kv_transfer_config,
    )

    print("Model loaded successfully!")

    # Simple test prompt
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,
    )
    
    # Generate
    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    print("\nvLLM is running successfully!")

if __name__ == "__main__":
    main()

