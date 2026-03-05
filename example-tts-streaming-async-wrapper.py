#!/usr/bin/env python3
"""
Async Streaming TTS example using executor-based approach.

This wraps the synchronous ChatterboxTTS in an async interface using run_in_executor,
allowing async/await usage while leveraging the existing synchronous implementation.
This is the most practical approach without modifying the core TTS class.
"""

from typing import AsyncGenerator, Optional
import asyncio
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from chatterbox_vllm.tts import ChatterboxTTS


class AsyncChatterboxTTS:
    """
    Async wrapper for ChatterboxTTS using thread pool executor.

    This provides an async interface without modifying the core TTS implementation.
    """

    def __init__(self, model: ChatterboxTTS, executor: Optional[ThreadPoolExecutor] = None):
        self.model = model
        self._executor = executor
        self._loop = None

    @classmethod
    async def from_pretrained(cls, *args, **kwargs) -> 'AsyncChatterboxTTS':
        """Async factory method."""
        loop = asyncio.get_event_loop()

        # Run the synchronous from_pretrained in executor
        model = await loop.run_in_executor(
            None,
            lambda: ChatterboxTTS.from_pretrained(*args, **kwargs)
        )

        return cls(model)

    async def stream_audio_chunks(
        self,
        prompt: str,
        audio_prompt_path: Optional[str] = None,
        language_id: str = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        chunk_size_samples: int = 24000,
        **generation_kwargs
    ) -> AsyncGenerator[torch.Tensor, None]:
        """
        Async generator that yields audio chunks.

        The generation runs in a thread pool, allowing the event loop to remain responsive.
        """
        loop = asyncio.get_event_loop()

        def generate_audio():
            """Synchronous generation in thread pool."""
            return self.model.generate(
                prompts=[prompt],
                audio_prompt_path=audio_prompt_path,
                language_id=language_id,
                exaggeration=exaggeration,
                temperature=temperature,
                **generation_kwargs
            )

        # Run generation in thread pool
        results = await loop.run_in_executor(self._executor, generate_audio)

        if results and results[0] is not None:
            audio = results[0]
            total_samples = audio.shape[1]

            # Yield chunks asynchronously
            for start_idx in range(0, total_samples, chunk_size_samples):
                end_idx = min(start_idx + chunk_size_samples, total_samples)
                chunk = audio[:, start_idx:end_idx]

                # Small delay to simulate streaming behavior
                await asyncio.sleep(0.01)
                yield chunk

    async def generate(
        self,
        prompts,
        audio_prompt_path: Optional[str] = None,
        **kwargs
    ):
        """Async wrapper for generate method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.model.generate(prompts, audio_prompt_path, **kwargs)
        )

    async def shutdown(self):
        """Async shutdown."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self.model.shutdown)


async def example_basic_async_streaming():
    """Basic async streaming example."""
    print("="*60)
    print("Example 1: Basic async streaming")
    print("="*60)

    model = await AsyncChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    prompt = "This is an example of async streaming text to speech generation."
    print(f"\nGenerating: {prompt}")

    chunks = []
    chunk_count = 0

    async for chunk in model.stream_audio_chunks(
        prompt=prompt,
        chunk_size_samples=12000,  # 0.5 second chunks
    ):
        chunk_count += 1
        chunks.append(chunk.cpu())
        print(f"  Received chunk {chunk_count}: {chunk.shape[1]} samples ({chunk.shape[1]/model.model.sr:.2f}s)")

    # Save result
    if chunks:
        full_audio = torch.cat(chunks, dim=1)
        ta.save("test-async-basic.mp3", full_audio, model.model.sr)
        print(f"\nSaved to test-async-basic.mp3")

    await model.shutdown()


async def example_concurrent_requests():
    """Example of handling multiple concurrent requests."""
    print("\n" + "="*60)
    print("Example 2: Concurrent streaming requests")
    print("="*60)

    model = await AsyncChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    async def process_request(request_id: str, prompt: str, delay: float = 0):
        """Process a single streaming request."""
        if delay > 0:
            await asyncio.sleep(delay)

        print(f"[{request_id}] Starting: {prompt[:50]}...")
        chunk_count = 0
        start_time = asyncio.get_event_loop().time()

        async for chunk in model.stream_audio_chunks(
            prompt=prompt,
            chunk_size_samples=12000,
        ):
            chunk_count += 1
            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"[{request_id}] Chunk {chunk_count} at {elapsed:.2f}s: {chunk.shape[1]} samples")

        elapsed = asyncio.get_event_loop().time() - start_time
        print(f"[{request_id}] Complete in {elapsed:.2f}s! Total chunks: {chunk_count}")
        return request_id, chunk_count

    # Run multiple requests concurrently with staggered starts
    tasks = [
        process_request("req-1", "First request with some text to synthesize.", delay=0),
        process_request("req-2", "Second request running concurrently.", delay=0.5),
        process_request("req-3", "Third request for parallel processing.", delay=1.0),
    ]

    results = await asyncio.gather(*tasks)

    print(f"\nAll requests complete:")
    for req_id, count in results:
        print(f"  {req_id}: {count} chunks")

    await model.shutdown()


async def example_with_web_framework():
    """
    Example showing how to use with web frameworks (FastAPI, aiohttp, etc.).

    This demonstrates the pattern you would use in a web server.
    """
    print("\n" + "="*60)
    print("Example 3: Web framework pattern (simulation)")
    print("="*60)

    model = await AsyncChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    async def stream_tts_endpoint(prompt: str):
        """
        Simulated FastAPI/aiohttp endpoint that streams audio.

        In a real implementation, you would return this generator
        directly from your endpoint handler.
        """
        print(f"Endpoint received request: {prompt[:50]}...")

        # In FastAPI, you would use:
        # return StreamingResponse(stream_audio_chunks(), media_type="audio/mpeg")
        #
        # In aiohttp, you would use:
        # return web.StreamResponse(stream_audio_chunks())

        chunks = []
        async for chunk in model.stream_audio_chunks(prompt=prompt):
            chunks.append(chunk)
            # In real implementation, yield chunk data here
            yield chunk

        return chunks

    # Simulate multiple concurrent web requests
    async def simulate_web_request(request_id: int, prompt: str):
        print(f"[Request {request_id}] Incoming connection...")
        chunk_count = 0

        async for chunk in stream_tts_endpoint(prompt):
            chunk_count += 1
            # In real implementation, chunk would be sent to client

        print(f"[Request {request_id}] Sent {chunk_count} chunks to client")
        return chunk_count

    # Simulate 5 concurrent requests
    prompts = [
        "Request one with text to synthesize.",
        "Request two for concurrent processing.",
        "Request three showing async handling.",
        "Request four with more text here.",
        "Request five completing the batch.",
    ]

    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*[
        simulate_web_request(i, prompt) for i, prompt in enumerate(prompts, 1)
    ])
    elapsed = asyncio.get_event_loop().time() - start_time

    print(f"\nProcessed {len(prompts)} requests in {elapsed:.2f}s")
    print(f"Total chunks sent: {sum(results)}")

    await model.shutdown()


async def example_queue_based_streaming():
    """
    Example using asyncio queues for producer-consumer pattern.

    This is useful for decoupling generation from playback/transmission.
    """
    print("\n" + "="*60)
    print("Example 4: Queue-based streaming")
    print("="*60)

    model = await AsyncChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=1000,
    )

    async def audio_producer(queue: asyncio.Queue, prompt: str):
        """Producer: generates audio chunks and puts them in queue."""
        print(f"[Producer] Starting generation for: {prompt[:50]}...")
        async for chunk in model.stream_audio_chunks(prompt=prompt):
            await queue.put(chunk)
        await queue.put(None)  # Signal end of stream
        print(f"[Producer] Complete")

    async def audio_consumer(queue: asyncio.Queue, consumer_id: str):
        """Consumer: processes audio chunks from queue."""
        print(f"[{consumer_id}] Waiting for chunks...")
        chunk_count = 0
        total_samples = 0

        while True:
            chunk = await queue.get()
            if chunk is None:
                print(f"[{consumer_id}] Received end-of-stream signal")
                break

            chunk_count += 1
            total_samples += chunk.shape[1]
            duration = total_samples / model.model.sr
            print(f"[{consumer_id}] Processing chunk {chunk_count}: {chunk.shape[1]} samples (total: {duration:.2f}s)")

            # Simulate processing time (e.g., encoding, network transmission)
            await asyncio.sleep(0.05)

        print(f"[{consumer_id}] Complete: {chunk_count} chunks, {total_samples} samples")

    # Create queue and run producer/consumer concurrently
    queue = asyncio.Queue(maxsize=10)  # Backpressure handling

    prompt = "This example demonstrates a producer-consumer pattern for audio streaming."
    await asyncio.gather(
        audio_producer(queue, prompt),
        audio_consumer(queue, "Consumer-1"),
    )

    await model.shutdown()


async def main():
    """Run all examples."""
    await example_basic_async_streaming()
    await example_concurrent_requests()
    await example_with_web_framework()
    await example_queue_based_streaming()
    print("\n" + "="*60)
    print("All async streaming examples complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
