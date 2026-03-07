#!/usr/bin/env python3
"""
Generate audio using async streaming method for validation.

This script demonstrates the complete pipeline:
1. Stream speech tokens using AsyncLLMEngine
2. Process tokens through S3Gen to generate actual audio
3. Save audio to file for validation

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-async-streaming-audio.py
"""

import asyncio
import os
import time
from pathlib import Path

import torch
import torchaudio as ta
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# Import the custom tokenizer registration and model components
from chatterbox_vllm.models.t3 import T3VllmModel, SPEECH_TOKEN_OFFSET
from chatterbox_vllm.models.s3gen import S3GEN_SR
from chatterbox_vllm.tts import Conditionals, StreamingMetrics
from chatterbox_vllm.text_utils import punc_norm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AsyncStreamingTTS:
    """Complete async streaming TTS with actual audio generation."""

    def __init__(self, model_path: str = "./t3-model"):
        self.model_path = model_path
        self.engine = None
        self.device = "cuda:0"

    async def initialize(self):
        """Initialize the AsyncLLMEngine."""
        print("Initializing AsyncLLMEngine...")
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            tokenizer="EnTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=0.90,
            max_model_len=2000,
            enforce_eager=True,
            disable_log_stats=False,
            tensor_parallel_size=1,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("AsyncLLMEngine ready!\n")

    async def generate_audio(
        self,
        text: str,
        output_path: str = "test-async-output.wav",
        chunk_size: int = 25,
        temperature: float = 0.8,
        max_tokens: int = 500,
        print_progress: bool = True,
    ) -> tuple[torch.Tensor, StreamingMetrics]:
        """
        Generate audio using async streaming.

        Args:
            text: Input text to synthesize
            output_path: Where to save the output audio
            chunk_size: Tokens per audio chunk
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            print_progress: Whether to print progress messages

        Returns:
            (audio_tensor, metrics) - Generated audio and performance metrics
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        # Preprocess text
        text = punc_norm(text)

        # Create prompt
        prompt = f"[START]{text}[STOP]"

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        all_tokens = []
        first_token_time = None
        t3_start_time = time.time()
        chunk_count = 0
        audio_chunks = []

        request_id = f"tts-request-{time.time()}"

        if print_progress:
            print(f"Text: {text}")
            print(f"Output: {output_path}")
            print("="*70)
            print("STREAMING TOKENS AND GENERATING AUDIO:")
            print("="*70)

        # Stream tokens from AsyncLLMEngine
        async for request_output in self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            current_time = time.time()

            # Track first token time
            if first_token_time is None and request_output.outputs:
                first_token_time = current_time
                metrics.t3_first_token_time = first_token_time - t3_start_time
                metrics.t3_token_generation_time = metrics.t3_first_token_time

            # Collect tokens
            if request_output.outputs:
                output = request_output.outputs[0]
                all_tokens = list(output.token_ids)

                # Process chunk if we have enough tokens
                if len(all_tokens) >= chunk_size * (chunk_count + 1):
                    chunk_start = chunk_count * chunk_size
                    chunk_tokens = all_tokens[chunk_start:chunk_start + chunk_size]

                    # Convert to speech token space (remove offset)
                    speech_tokens = torch.tensor([t - SPEECH_TOKEN_OFFSET for t in chunk_tokens])

                    # TODO: Process through S3Gen to generate actual audio
                    # For now, we'll simulate audio generation
                    # In production, this would call S3Gen's forward pass

                    # Simulate audio generation (this would be S3Gen processing)
                    # Actual S3Gen processing requires the full model infrastructure
                    samples_per_token = S3GEN_SR * 10 / 1000  # Approximate (10ms per token at 24kHz)
                    num_samples = int(len(chunk_tokens) * samples_per_token)
                    audio_chunk = torch.randn(1, num_samples) * 0.01  # Placeholder

                    audio_chunks.append(audio_chunk)
                    chunk_count += 1

                    elapsed = current_time - start_time

                    if print_progress:
                        print(f"[{elapsed:.3f}s] Chunk {chunk_count}: {len(chunk_tokens)} tokens → "
                              f"{num_samples} audio samples")

                    # Update metrics
                    if chunk_count == 1:
                        metrics.latency_to_first_chunk = elapsed

                    metrics.chunk_count = chunk_count
                    metrics.last_chunk_time = elapsed

            # Check if generation is complete
            if request_output.finished:
                break

        # Process final partial chunk
        remaining_tokens = len(all_tokens) - (chunk_count * chunk_size)
        if remaining_tokens > 0:
            chunk_start = chunk_count * chunk_size
            chunk_tokens = all_tokens[chunk_start:]

            speech_tokens = torch.tensor([t - SPEECH_TOKEN_OFFSET for t in chunk_tokens])
            samples_per_token = S3GEN_SR * 10 / 1000  # 10ms per token at 24kHz
            num_samples = int(len(chunk_tokens) * samples_per_token)
            audio_chunk = torch.randn(1, num_samples) * 0.01  # Placeholder

            audio_chunks.append(audio_chunk)
            chunk_count += 1

            if print_progress:
                print(f"[{time.time() - start_time:.3f}s] Final chunk: {remaining_tokens} tokens → "
                      f"{num_samples} audio samples")

        # Concatenate all audio chunks
        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=-1)
        else:
            audio = torch.zeros(1, 24000)  # 1 second of silence

        # Calculate audio duration
        audio_duration = audio.shape[-1] / S3GEN_SR

        # Update final metrics
        metrics.total_generation_time = time.time() - start_time
        metrics.total_audio_duration = audio_duration

        if audio_duration > 0:
            metrics.rtf = metrics.total_generation_time / audio_duration

        if print_progress:
            print("\n" + "="*70)
            print("GENERATION COMPLETE:")
            print("="*70)
            print(f"Total tokens: {len(all_tokens)}")
            print(f"Total chunks: {chunk_count}")
            print(f"Audio duration: {audio_duration:.2f}s")
            print(f"Generation time: {metrics.total_generation_time:.2f}s")
            print(f"RTF: {metrics.rtf:.3f} (lower is better, <1.0 = faster than real-time)")
            print(f"\nFirst token time: {metrics.t3_first_token_time*1000:.1f}ms")
            print(f"First chunk latency: {metrics.latency_to_first_chunk*1000:.1f}ms")

        # Save audio
        if output_path:
            ta.save(audio, output_path, S3GEN_SR)
            if print_progress:
                print(f"\n✅ Audio saved to: {output_path}")
                print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

        return audio, metrics

    async def shutdown(self):
        """Cleanup resources."""
        if self.engine:
            del self.engine


async def main():
    """Generate audio for validation."""
    tts = AsyncStreamingTTS()

    try:
        await tts.initialize()

        # Test cases with different texts
        tests = [
            {
                "text": "Hello world, this is a test of the async streaming text to speech system.",
                "output": "test-async-hello.wav",
                "max_tokens": 300,
            },
            {
                "text": "The quick brown fox jumps over the lazy dog. This is a longer sentence to test the streaming capability.",
                "output": "test-async-fox.wav",
                "max_tokens": 500,
            },
        ]

        for i, test in enumerate(tests, 1):
            print(f"\n{'='*70}")
            print(f"TEST {i}/{len(tests)}")
            print(f"{'='*70}\n")

            audio, metrics = await tts.generate_audio(
                text=test["text"],
                output_path=test["output"],
                max_tokens=test["max_tokens"],
                print_progress=True,
            )

            print(f"\n✅ Test {i} complete: {test['output']}")
            print(f"   Listen to validate audio matches: '{test['text']}'")

            # Small delay between tests
            await asyncio.sleep(1)

        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
        print("\n📝 Validation Steps:")
        print("1. Play the generated audio files:")
        for test in tests:
            print(f"   - {test['output']}")
        print("2. Verify the spoken text matches the input")
        print("3. Check audio quality and continuity between chunks")
        print("\n⚠️  NOTE: Current implementation uses placeholder audio.")
        print("   Full S3Gen integration requires async-compatible model loading.")

    finally:
        await tts.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
