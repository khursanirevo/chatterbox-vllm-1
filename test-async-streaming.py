#!/usr/bin/env python3
"""Test vLLM AsyncLLMEngine for true token streaming"""

import os
import asyncio
import time
import torch
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# IMPORTANT: Import this first to register the custom tokenizer
from chatterbox_vllm.models.t3 import T3VllmModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Enable debug mode to track token validation
os.environ["CHATTERBOX_DEBUG_TOKENS"] = "1"

async def test_vllm_streaming():
    """Test if vLLM can stream tokens incrementally"""

    print("Initializing AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model="./t3-model",
        tokenizer="EnTokenizer",
        tokenizer_mode="custom",
        gpu_memory_utilization=0.90,
        max_model_len=2000,
        enforce_eager=True,
        # Recommended settings for async debugging:
        disable_log_stats=False,  # Enable logging to debug issues
        tensor_parallel_size=1,  # Explicitly set to 1 (single GPU)
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Engine ready!\n")

    # Note: Device verification is handled by the _validate_token_ids debug function
    # Set CHATTERBOX_DEBUG_TOKENS=1 to see token validation output

    # Test prompt
    prompt = "[START]Hello world, this is a test.[STOP]"

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        stop_token_ids=[6561 + 10000],  # Adjust based on config
    )

    print(f"Prompt: {prompt}")
    print("="*70)
    print("STREAMING TOKENS:")
    print("="*70)

    start_time = time.time()
    token_count = 0
    all_token_ids = []

    request_id = "test-request-1"

    async for request_output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        current_time = time.time()
        elapsed = current_time - start_time

        # Get the first output (index 0)
        if request_output.outputs:
            output = request_output.outputs[0]
            new_tokens = output.token_ids
            all_token_ids = list(new_tokens)  # Cumulative tokens

            # Check if we got new tokens
            if len(new_tokens) > token_count:
                new_token_count = len(new_tokens) - token_count
                token_count = len(new_tokens)
                print(f"[{elapsed:.3f}s] Got {new_token_count} new tokens (total: {token_count} tokens)")
                print(f"  Latest tokens: {new_tokens[-5:] if len(new_tokens) > 5 else new_tokens}")

        if request_output.finished:
            print(f"\n[STREAMING COMPLETE] Total time: {elapsed:.3f}s, Tokens: {token_count}")
            break

    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"Total tokens generated: {token_count}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Time per token: {elapsed/token_count*1000:.1f}ms")
    print(f"Tokens/second: {token_count/elapsed:.1f}")

    # Cleanup
    del engine

if __name__ == "__main__":
    asyncio.run(test_vllm_streaming())
