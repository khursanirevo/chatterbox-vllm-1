#!/usr/bin/env python3
"""
Generate audio files using the stream pool verification script.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python generate_stream_pool_audio.py
"""

import asyncio
import sys
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox_vllm.tts_async import AsyncChatterboxTTS

async def main():
    print("🎵 Generating Audio with Stream Pool\n")

    # Setup output directory
    output_dir = Path("output/stream_pool_audio")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model with stream pool
    print("📦 Loading model with stream pool...")
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=4,
        gpu_memory_utilization=0.3,
    )

    # Check stream pool exists
    assert model.s3gen_stream_pool is not None, "Stream pool not initialized"
    print(f"✅ Stream pool created: {model.s3gen_stream_pool.num_streams} streams")

    # Warmup model
    print("\nWarming up model...")
    async for _ in model.generate_stream("Warmup.", print_metrics=False):
        pass
    print("✅ Model warmed up\n")

    # Test texts
    texts = [
        "Hello world, this is a test of the stream pool audio generation.",
        "The quick brown fox jumps over the lazy dog.",
        "This is the third test for concurrent audio generation with stream pool.",
    ]

    # Generate audio for each text
    for i, text in enumerate(texts):
        print(f"📝 Generating audio {i+1}/{len(texts)}: '{text[:50]}...'")

        audio_chunks = []
        async for audio_chunk, metrics in model.generate_stream(text, print_metrics=True):
            audio_chunks.append(audio_chunk)

        # Combine chunks
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            from chatterbox_vllm.models.s3gen import S3GEN_SR
            duration = full_audio.shape[-1] / S3GEN_SR

            # Save audio
            output_file = output_dir / f"test_{i+1}.wav"
            ta.save(str(output_file), full_audio.cpu(), S3GEN_SR)

            print(f"   ✅ Saved: {output_file.name} ({duration:.2f}s)\n")

    # Test concurrent generation
    print("📝 Testing concurrent generation...")

    async def generate_and_save(text, index):
        audio_chunks = []
        async for chunk, _ in model.generate_stream(text, print_metrics=False):
            audio_chunks.append(chunk)

        if audio_chunks:
            from chatterbox_vllm.models.s3gen import S3GEN_SR
            full_audio = torch.cat(audio_chunks, dim=-1)
            output_file = output_dir / f"concurrent_{index+1}.wav"
            ta.save(str(output_file), full_audio.cpu(), S3GEN_SR)
            return output_file.name, full_audio.shape[-1] / S3GEN_SR
        return None, 0

    concurrent_texts = [
        "Concurrent test number one.",
        "Concurrent test number two.",
        "Concurrent test number three.",
    ]

    start = asyncio.get_event_loop().time()
    tasks = [generate_and_save(text, i) for i, text in enumerate(concurrent_texts)]
    results = await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start

    for filename, duration in results:
        if filename:
            print(f"   ✅ {filename}: {duration:.2f}s")

    print(f"\n   Completed in {elapsed:.2f}s")

    # Print metrics
    print(f"\n📊 Stream Pool Metrics:")
    metrics = model.s3gen_stream_pool.metrics
    print(f"   Total requests: {metrics.total_requests}")
    print(f"   Active streams: {metrics.active_streams}")
    print(f"   Avg queue wait: {metrics.avg_queue_wait_ms:.2f}ms")
    print(f"   Queue depth: {metrics.queue_depth}")

    print(f"\n📁 All audio files saved to: {output_dir}/")
    print(f"\nFiles generated:")
    for audio_file in sorted(output_dir.glob("*.wav")):
        print(f"   - {audio_file.name}")

    print("\n✅ Audio generation complete!")

    await model.shutdown()
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
