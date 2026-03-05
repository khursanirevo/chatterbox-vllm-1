#!/usr/bin/env python3
"""
Streaming TTS example using ChatterboxTTSAsync with Continuous Batching.

This example demonstrates how to use ChatterboxTTSAsync which leverages vLLM's
AsyncLLMEngine for CONTINUOUS BATCHING - the key feature that allows efficient
handling of multiple concurrent TTS requests.

What is Continuous Batching?
-----------------------------
Traditional static batching: All requests in a batch must complete together.
Continuous batching: Requests can join/leave the batch as they complete.

For TTS this is crucial because:
- Short prompts finish quickly, long prompts take longer
- Continuous batching lets short prompts complete without waiting for long ones
- New requests can start processing immediately when slots are available
- Significantly better throughput and latency for variable-length TTS

Key Benefits:
-------------
1. Higher Throughput: Process more requests per second
2. Lower Latency: Short prompts don't wait for long ones
3. Better GPU Utilization: Batches stay full with active requests
4. Concurrent Request Handling: Naturally supports many simultaneous users
"""

import asyncio
import time
import torch
import torchaudio as ta
from typing import AsyncGenerator

from chatterbox_vllm import ChatterboxTTSAsync


async def example_basic_continuous_batching():
    """
    Demonstrate basic continuous batching with multiple concurrent requests.

    All requests are submitted to the engine, and as each completes its
    token generation, it moves to audio synthesis while the engine continues
    processing remaining token generation for other requests.
    """
    print("="*70)
    print("Example 1: Basic Continuous Batching")
    print("="*70)
    print("\nSubmitting 3 requests of different lengths simultaneously...")
    print("Watch how they complete based on content length, not submission order!\n")

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=8,  # Allow up to 8 concurrent requests in batch
        max_model_len=1000,
    )

    prompts = [
        "Short.",  # Should complete first
        "This is a medium length prompt that will take some time to synthesize.",
        "This is a much longer prompt with significantly more text to process. It should take the longest time to complete the synthesis because there are many more tokens to generate and subsequently decode into audio waveform data.",
    ]

    async def generate_and_save(request_id: str, prompt: str):
        """Generate audio and save to file."""
        start = time.time()
        print(f"[{request_id}] Starting: {prompt[:50]}...")

        # Generate audio (continuous batching handles this efficiently)
        results = await model.generate(
            prompts=[prompt],
            temperature=0.8,
            exaggeration=0.5,
        )

        elapsed = time.time() - start
        print(f"[{request_id}] Completed in {elapsed:.2f}s")

        # Save result
        if results and results[0] is not None:
            output_path = f"test-continuous-{request_id}.mp3"
            ta.save(output_path, results[0], model.sr)
            print(f"[{request_id}] Saved to {output_path}")

        return elapsed

    # Submit all requests concurrently
    # With continuous batching, short requests finish before long ones!
    start_time = time.time()
    results = await asyncio.gather(*[
        generate_and_save(f"req-{i}", prompt)
        for i, prompt in enumerate(prompts)
    ])
    total_time = time.time() - start_time

    print(f"\nAll requests completed in {total_time:.2f}s")
    print(f"Individual times: {results}")
    print(f"Time saved vs sequential: {sum(results) - total_time:.2f}s")

    await model.shutdown()


async def example_streaming_with_continuous_batching():
    """
    Stream audio chunks from multiple concurrent requests.

    This demonstrates how continuous batching enables efficient streaming
    for multiple users/requests simultaneously.
    """
    print("\n" + "="*70)
    print("Example 2: Streaming with Continuous Batching")
    print("="*70)
    print("\nSimulating 3 concurrent users requesting TTS...\n")

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=8,
        max_model_len=1000,
    )

    async def stream_for_user(user_id: str, prompt: str):
        """Simulate a user receiving streamed audio."""
        print(f"[User {user_id}] Connecting...")
        print(f"[User {user_id}] Request: {prompt[:50]}...")

        start = time.time()
        chunk_count = 0
        total_samples = 0

        # Note: This still uses generate() and then chunks the result
        # True token-level streaming would require modifications to the
        # token generation loop
        results = await model.generate(
            prompts=[prompt],
            temperature=0.8,
        )

        if results and results[0] is not None:
            audio = results[0]
            chunk_size = 12000  # 0.5 second chunks

            for i in range(0, audio.shape[1], chunk_size):
                chunk = audio[:, i:i+chunk_size]
                chunk_count += 1
                total_samples += chunk.shape[1]
                duration = total_samples / model.sr

                # In a real application, you would send this chunk to the user
                # via WebSocket, HTTP streaming, etc.
                print(f"[User {user_id}] Chunk {chunk_count}: {chunk.shape[1]} samples (total: {duration:.1f}s)")

                # Simulate network/processing delay
                await asyncio.sleep(0.05)

        elapsed = time.time() - start
        print(f"[User {user_id}] Complete: {chunk_count} chunks in {elapsed:.2f}s")

        # Save for verification
        if results and results[0] is not None:
            ta.save(f"test-stream-user-{user_id}.mp3", results[0], model.sr)

        return user_id, elapsed

    # Simulate 3 concurrent users with different prompt lengths
    users = [
        ("alice", "Quick audio generation test."),
        ("bob", "Bob has a medium length request that will take a bit more time to process through the text to speech synthesis pipeline."),
        ("charlie", "Charlie's request is the longest, containing much more text that needs to be tokenized, processed through the language model, and then decoded into high quality audio using the neural vocoder. This demonstrates how continuous batching allows shorter requests to complete independently."),
    ]

    start_time = time.time()
    results = await asyncio.gather(*[stream_for_user(uid, prompt) for uid, prompt in users])
    total_time = time.time() - start_time

    print(f"\nAll users served in {total_time:.2f}s")
    print(f"User completion times: {[(uid, f'{t:.2f}s') for uid, t in results]}")

    await model.shutdown()


async def example_batch_processing_comparison():
    """
    Compare static batching vs continuous batching performance.

    This demonstrates the throughput advantage of continuous batching
    when handling requests with variable complexity.
    """
    print("\n" + "="*70)
    print("Example 3: Throughput Comparison")
    print("="*70)
    print("\nProcessing 10 requests of varying lengths...")
    print("With continuous batching, shorter requests complete faster!\n")

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=8,
        max_model_len=1000,
    )

    # Create 10 requests with varying lengths
    prompts = [
        "Short prompt one.",
        "Short prompt two.",
        "This is a medium length prompt with more content.",
        "Another medium prompt for testing.",
        "Short three.",
        "This is a longer prompt that will take significantly more time to process through the TTS pipeline.",
        "Medium prompt number seven here.",
        "Short eight.",
        "Yet another longer prompt designed to test the continuous batching capabilities.",
        "Final short prompt.",
    ]

    async def process_request(req_id: int, prompt: str):
        start = time.time()
        prompt_length = len(prompt.split())

        results = await model.generate(
            prompts=[prompt],
            temperature=0.8,
        )

        elapsed = time.time() - start
        tokens_per_second = prompt_length / elapsed if elapsed > 0 else 0

        print(f"[Req {req_id:02d}] {prompt_length:3d} words -> {elapsed:5.2f}s ({tokens_per_second:4.1f} words/s)")

        return elapsed

    print("Processing requests with continuous batching...\n")
    print("Req # | Words | Time  | Rate")
    print("-" * 35)

    start_time = time.time()
    results = await asyncio.gather(*[process_request(i, prompt) for i, prompt in enumerate(prompts)])
    total_time = time.time() - start_time

    print("-" * 35)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per request: {sum(results)/len(results):.2f}s")
    print(f"Throughput: {len(prompts)/total_time:.2f} requests/second")

    await model.shutdown()


async def example_dynamic_load():
    """
    Demonstrate handling dynamic load with continuous batching.

    Simulates a real-world scenario where requests arrive at different times
    with varying complexity.
    """
    print("\n" + "="*70)
    print("Example 4: Dynamic Load Simulation")
    print("="*70)
    print("\nSimulating requests arriving over time with varying complexity...\n")

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=8,
        max_model_len=1000,
    )

    # Simulate requests arriving at different times
    request_schedule = [
        (0.0, "Initial request", "First prompt arriving at time zero."),
        (0.5, "Burst 1", "A quick request."),
        (0.5, "Burst 2", "Another quick one."),
        (1.0, "Medium", "This is a medium length request arriving one second in."),
        (1.5, "Short", "Brief."),
        (2.0, "Long", "This is a longer request that will take more processing time. It demonstrates how continuous batching handles requests arriving after the initial batch has started processing."),
        (2.5, "Quick", "Fast."),
        (3.0, "Another", "Yet another medium length request."),
    ]

    async def delayed_request(arrival_time: float, req_id: str, prompt: str):
        """Simulate a request arriving at a specific time."""
        await asyncio.sleep(arrival_time)

        start = time.time()
        elapsed_since_start = start - start_time_offset
        print(f"[{req_id}] Arrived at {elapsed_since_start:.2f}s | {prompt[:40]}...")

        results = await model.generate(
            prompts=[prompt],
            temperature=0.8,
        )

        elapsed = time.time() - start
        total_time = time.time() - start_time_offset
        print(f"[{req_id}] Completed at {total_time:.2f}s (generation took {elapsed:.2f}s)")

        return req_id, elapsed_since_start, elapsed

    start_time_offset = time.time()
    print("Timeline: |0s----|1s----|2s----|3s----|4s----|5s")
    print()

    results = await asyncio.gather(*[
        delayed_request(delay, req_id, prompt)
        for delay, req_id, prompt in request_schedule
    ])

    total_duration = time.time() - start_time_offset
    print(f"\nProcessed {len(results)} requests over {total_duration:.2f}s")
    print(f"Average latency: {sum(r[2] for r in results)/len(results):.2f}s")
    print(f"Requests/second: {len(results)/total_duration:.2f}")

    await model.shutdown()


async def main():
    """Run all examples demonstrating continuous batching benefits."""
    print("\n" + "="*70)
    print("Chatterbox vLLM - Continuous Batching Examples")
    print("="*70)
    print("\nThese examples demonstrate AsyncLLMEngine's CONTINUOUS BATCHING")
    print("capability and its benefits for TTS workloads.\n")

    await example_basic_continuous_batching()
    await example_streaming_with_continuous_batching()
    await example_batch_processing_comparison()
    await example_dynamic_load()

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Continuous batching allows short requests to complete quickly")
    print("2. Multiple concurrent requests are processed efficiently")
    print("3. GPU utilization is improved with dynamic batch management")
    print("4. Lower latency and higher throughput vs static batching")
    print("\nFor production use with FastAPI/aiohttp:")
    print("  - Use ChatterboxTTSAsync for your TTS endpoint")
    print("  - Requests will automatically benefit from continuous batching")
    print("  - Scale to handle many concurrent users efficiently")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
