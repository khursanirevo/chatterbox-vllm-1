#!/usr/bin/env python3
"""
Comprehensive concurrent load test with streaming fix.

Tests incremental T3 streaming + S3Gen stream pool to verify
that we can achieve <1s first chunk for 8-16 concurrent requests.

Saves:
- Individual audio chunks
- Full audio
- Text input
- Profiling metrics

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test_concurrent_profiling.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import torch
import torchaudio as ta

from chatterbox_vllm.tts_async import AsyncChatterboxTTS
from chatterbox_vllm.models.s3gen import S3GEN_SR


@dataclass
class ConcurrentTestResult:
    """Results for a single concurrent test."""
    concurrent_level: int
    test_text: str

    # Per-request results
    first_chunk_times: List[float] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    audio_durations: List[float] = field(default_factory=list)
    chunk_counts: List[int] = field(default_factory=list)

    # Metrics
    avg_first_chunk_ms: float = 0.0
    min_first_chunk_ms: float = 0.0
    max_first_chunk_ms: float = 0.0
    median_first_chunk_ms: float = 0.0

    under_1s_count: int = 0
    under_1s_pct: float = 0.0

    # Stream pool metrics
    avg_queue_wait_ms: float = 0.0
    total_requests_processed: int = 0


async def test_concurrent_level(
    model: AsyncChatterboxTTS,
    num_concurrent: int,
    output_dir: Path,
    print_debug: bool = False,
) -> ConcurrentTestResult:
    """Test a specific concurrent level."""

    # Create unique texts for each request
    texts = [
        f"This is concurrent test number {i+1} of {num_concurrent}. "
        f"We are testing streaming TTS with incremental token generation "
        f"and parallel S3Gen processing using the CUDA stream pool."
        for i in range(num_concurrent)
    ]

    # Create output subdirectory
    test_dir = output_dir / f"concurrent_{num_concurrent}"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save text input
    (test_dir / "test_text.txt").write_text("\n".join(texts))

    # Track results
    result = ConcurrentTestResult(
        concurrent_level=num_concurrent,
        test_text=texts[0] if num_concurrent == 1 else texts[0][:100] + "..."
    )

    async def generate_and_track(request_id: int, text: str) -> Dict:
        """Generate audio and track detailed metrics."""
        start = time.time()
        first_chunk_time = None
        audio_chunks = []
        chunk_count = 0

        request_dir = test_dir / f"request_{request_id:03d}"
        request_dir.mkdir(exist_ok=True)

        # Save text for this request
        (request_dir / "input.txt").write_text(text)

        # Generate with metrics
        async for audio_chunk, metrics in model.generate_stream(
            text,
            chunk_size=15,  # Optimized for latency
            diffusion_steps=5,
            print_metrics=print_debug,
        ):
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start) * 1000

            audio_chunks.append(audio_chunk)
            chunk_count += 1

            # Save individual chunk
            chunk_path = request_dir / f"chunk_{chunk_count:03d}.wav"
            ta.save(str(chunk_path), audio_chunk.cpu(), S3GEN_SR)

        total_time = time.time() - start

        # Save full audio
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            full_path = request_dir / "full_audio.wav"
            ta.save(str(full_path), full_audio.cpu(), S3GEN_SR)

            duration = full_audio.shape[-1] / S3GEN_SR
        else:
            duration = 0.0

        # Save metrics
        metrics_path = request_dir / "metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"Request ID: {request_id}\n")
            f.write(f"Text: {text}\n")
            f.write(f"\nTiming:\n")
            if first_chunk_time:
                f.write(f"  First chunk: {first_chunk_time:.1f}ms\n")
            else:
                f.write("  First chunk: N/A\n")
                
            f.write(f"  Audio duration: {duration:.2f}s\n")
            f.write(f"  Chunks: {chunk_count}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  T3 first token: {metrics.t3_first_token_time*1000:.1f}ms\n")
            f.write(f"  S3Gen first chunk: {metrics.s3gen_first_chunk_time*1000:.1f}ms\n")
            f.write(f"  Latency to first chunk: {metrics.latency_to_first_chunk*1000:.1f}ms\n")
            f.write(f"  RTF: {metrics.rtf:.3f}\n")

        return {
            'request_id': request_id,
            'first_chunk_time': first_chunk_time,
            'total_time': total_time,
            'duration': duration,
            'chunk_count': chunk_count,
        }

    # Launch all requests concurrently
    print(f"\n▶ Testing {num_concurrent} concurrent requests...")
    start_time = time.time()

    tasks = [generate_and_track(i, text) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)

    total_elapsed = time.time() - start_time

    # Collect results
    for r in results:
        if r['first_chunk_time']:
            result.first_chunk_times.append(r['first_chunk_time'])
            result.total_times.append(r['total_time'])
            result.audio_durations.append(r['duration'])
            result.chunk_counts.append(r['chunk_count'])

    # Calculate statistics
    if result.first_chunk_times:
        result.avg_first_chunk_ms = sum(result.first_chunk_times) / len(result.first_chunk_times)
        result.min_first_chunk_ms = min(result.first_chunk_times)
        result.max_first_chunk_ms = max(result.first_chunk_times)

        sorted_times = sorted(result.first_chunk_times)
        result.median_first_chunk_ms = sorted_times[len(sorted_times) // 2]

        result.under_1s_count = sum(1 for t in result.first_chunk_times if t < 1000)
        result.under_1s_pct = (result.under_1s_count / len(result.first_chunk_times)) * 100

    # Get stream pool metrics
    if model.s3gen_stream_pool:
        pool_metrics = model.s3gen_stream_pool.metrics
        result.avg_queue_wait_ms = pool_metrics.avg_queue_wait_ms
        result.total_requests_processed = pool_metrics.total_requests

    # Print summary
    print(f"  First chunk times:")
    print(f"    Average: {result.avg_first_chunk_ms:.1f}ms")
    print(f"    Median:  {result.median_first_chunk_ms:.1f}ms")
    print(f"    Min:     {result.min_first_chunk_ms:.1f}ms")
    print(f"    Max:     {result.max_first_chunk_ms:.1f}ms")
    print(f"  Under 1s: {result.under_1s_count}/{len(result.first_chunk_times)} ({result.under_1s_pct:.1f}%)")
    print(f"  Stream pool queue wait: {result.avg_queue_wait_ms:.2f}ms")
    print(f"  Total time: {total_elapsed:.2f}s")

    # Save summary
    summary_path = test_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Concurrent Level: {num_concurrent}\n")
        f.write(f"\nPerformance Summary:\n")
        f.write(f"  Average first chunk: {result.avg_first_chunk_ms:.1f}ms\n")
        f.write(f"  Median first chunk:  {result.median_first_chunk_ms:.1f}ms\n")
        f.write(f"  Min first chunk:     {result.min_first_chunk_ms:.1f}ms\n")
        f.write(f"  Max first chunk:     {result.max_first_chunk_ms:.1f}ms\n")
        f.write(f"  Under 1s: {result.under_1s_count}/{len(result.first_chunk_times)} ({result.under_1s_pct:.1f}%)\n")
        f.write(f"\nStream Pool:\n")
        f.write(f"  Queue wait: {result.avg_queue_wait_ms:.2f}ms\n")
        f.write(f"  Requests processed: {result.total_requests_processed}\n")
        f.write(f"\nPer-Request Results:\n")
        for i, r in enumerate(results):
            f.write(f"\n  Request {i}:\n")
            f.write(f"    First chunk: {r['first_chunk_time']:.1f}ms\n")
            f.write(f"    Total time: {r['total_time']*1000:.1f}ms\n")
            f.write(f"    Duration: {r['duration']:.2f}s\n")
            f.write(f"    Chunks: {r['chunk_count']}\n")

    return result


async def main():
    print("="*70)
    print("Concurrent Load Test with Streaming Fix")
    print("="*70)
    print()
    print("Testing incremental T3 streaming + S3Gen stream pool")
    print("Goal: Achieve <1s first chunk for 8-16 concurrent requests")
    print()

    # Setup output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/concurrent_profiling_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Load model
    print("📦 Loading model with stream pool...")
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=12,
        gpu_memory_utilization=0.5,
        default_chunk_size=15,
    )

    print(f"✅ Stream pool: {model.s3gen_stream_pool.num_streams} streams")
    print()

    # Warmup
    print("Warming up model...")
    async for _ in model.generate_stream("Warmup.", print_metrics=False):
        pass
    print("✅ Warmup complete\n")

    # Test different concurrent levels
    concurrent_levels = [1, 2, 4, 8, 16]
    results = []

    print("="*70)
    print("Running Concurrent Load Tests")
    print("="*70)

    for concurrent in concurrent_levels:
        result = await test_concurrent_level(
            model,
            concurrent,
            output_dir,
            print_debug=False,
        )
        results.append(result)

        # Stop if we're getting close to 1s average
        if result.avg_first_chunk_ms > 1500:
            print(f"\n  ⚠️  Approaching 1.5s average, stopping tests")
            break

    # Print final summary
    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    print()
    print(f"{'Concurrent':<12} {'Avg First':<12} {'Median':<10} {'<1s':<10} {'Queue':<10}")
    print("-" * 70)

    for r in results:
        status = "✅" if r.avg_first_chunk_ms < 1000 else "❌"
        print(
            f"{r.concurrent_level:<8} {status}  "
            f"{r.avg_first_chunk_ms:>8.1f}ms  "
            f"{r.median_first_chunk_ms:>6.1f}ms  "
            f"{r.under_1s_pct:>5.0f}%  "
            f"{r.avg_queue_wait_ms:>6.2f}ms"
        )

    print()

    # Find maximum concurrent with <1s first chunk
    under_1s_results = [r for r in results if r.avg_first_chunk_ms < 1000]
    if under_1s_results:
        max_concurrent = max(r.concurrent_level for r in under_1s_results)
        result = under_1s_results[-1]
        print(f"🎯 Maximum concurrent with <1s first chunk: {max_concurrent}")
        print(f"   Average first chunk: {result.avg_first_chunk_ms:.1f}ms")
        print(f"   ({result.under_1s_count}/{result.concurrent_level} requests under 1s)")
    else:
        print("❌ No concurrent level achieved <1s average")

    print()
    print(f"📁 All outputs saved to: {output_dir}/")

    # Save overall summary
    summary_path = output_dir / "overall_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Concurrent Load Test Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: t3-model\n")
        f.write(f"Stream pool: {model.s3gen_stream_pool.num_streams} streams\n")
        f.write(f"Chunk size: 15\n")
        f.write(f"Diffusion steps: 5\n\n")

        f.write("Results:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Concurrent':<12} {'Avg First':<12} {'Median':<10} {'<1s':<10}\n")
        f.write("-"*70 + "\n")

        for r in results:
            status = "✅" if r.avg_first_chunk_ms < 1000 else "❌"
            f.write(
                f"{r.concurrent_level:<8} {status}  "
                f"{r.avg_first_chunk_ms:>8.1f}ms  "
                f"{r.median_first_chunk_ms:>6.1f}ms  "
                f"{r.under_1s_pct:>5.0f}%\n"
            )

        if under_1s_results:
            f.write(f"\n🎯 Max concurrent with <1s: {max_concurrent}\n")

    await model.shutdown()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
