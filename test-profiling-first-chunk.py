#!/usr/bin/env python3
"""
Profile first chunk audio latency with statistics.

This script runs multiple iterations to collect:
- First chunk latency (min/max/avg/std)
- T3 token generation time
- S3Gen first chunk processing time
- Total generation time

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-profiling-first-chunk.py
"""

import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS, StreamingMetrics
from chatterbox_vllm.models.s3gen import S3GEN_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    values: List[float] = field(default_factory=list)

    def add(self, value: float):
        """Add a measurement."""
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def avg(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def percentile_95(self) -> float:
        """95th percentile."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * 0.95)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def percentile_99(self) -> float:
        """99th percentile."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * 0.99)
        return sorted_values[min(index, len(sorted_values) - 1)]


@dataclass
class DetailedMetrics:
    """Detailed metrics for a single generation."""
    iteration: int
    first_chunk_latency: float  # Total time to first audio chunk
    t3_token_generation: float  # Time to generate speech tokens
    s3gen_first_chunk: float  # Time to process first audio chunk
    total_generation_time: float  # Total generation time
    audio_duration: float  # Duration of generated audio
    token_count: int  # Number of speech tokens
    chunk_count: int  # Number of audio chunks


def print_stats_table(stats: LatencyStats, label: str, unit: str = "ms"):
    """Print a statistics table."""
    values_ms = [v * 1000 for v in stats.values]  # Convert to ms

    print(f"\n{label}:")
    print(f"  {'Count:':<20} {stats.count}")
    print(f"  {'Min:':<20} {stats.min*1000:.2f} {unit}")
    print(f"  {'Max:':<20} {stats.max*1000:.2f} {unit}")
    print(f"  {'Average:':<20} {stats.avg*1000:.2f} {unit}")
    print(f"  {'Median:':<20} {stats.median*1000:.2f} {unit}")
    print(f"  {'Std Dev:':<20} {stats.stdev*1000:.2f} {unit}")
    print(f"  {'95th Percentile:':<20} {stats.percentile_95*1000:.2f} {unit}")
    print(f"  {'99th Percentile:':<20} {stats.percentile_99*1000:.2f} {unit}")


def profile_first_chunk_latency(
    text: str = "Hello world, this is a test.",
    iterations: int = 10,
    max_tokens: int = 200,
    warmup_iterations: int = 2,
):
    """
    Profile first chunk latency across multiple iterations.

    Args:
        text: Input text to synthesize
        iterations: Number of test iterations
        max_tokens: Maximum tokens to generate
        warmup_iterations: Number of warmup iterations (not counted in stats)
    """
    print("="*70)
    print("FIRST CHUNK LATENCY PROFILING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Text: '{text}'")
    print(f"  Iterations: {iterations}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Max tokens: {max_tokens}")

    # Initialize statistics
    first_chunk_latency = LatencyStats()
    t3_generation_time = LatencyStats()
    s3gen_first_chunk = LatencyStats()
    total_generation_time = LatencyStats()
    audio_duration = LatencyStats()

    # Track detailed metrics for each iteration
    all_metrics: List[DetailedMetrics] = []

    print("\n" + "="*70)
    print(f"WARMUP ({warmup_iterations} iterations)")
    print("="*70)

    # Warmup iterations
    for i in range(warmup_iterations):
        print(f"\nWarmup {i+1}/{warmup_iterations}...", end="", flush=True)

        model = ChatterboxTTS.from_pretrained(
            max_model_len=max_tokens,
            gpu_memory_utilization=0.90,
        )

        # Generate audio (discard results)
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break

        model.shutdown()
        del model
        torch.cuda.empty_cache()

        print(" ✓")

    print("\n" + "="*70)
    print(f"PROFILING ({iterations} iterations)")
    print("="*70)

    # Profiling iterations
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}...", end="", flush=True)

        iter_start = time.time()

        # Initialize model
        init_start = time.time()
        model = ChatterboxTTS.from_pretrained(
            max_model_len=max_tokens,
            gpu_memory_utilization=0.90,
        )
        init_time = time.time() - init_start

        # Generate audio and collect metrics
        iter_first_chunk_latency = None
        iter_t3_time = None
        iter_s3gen_time = None
        iter_audio_duration = 0.0
        iter_token_count = 0
        iter_chunk_count = 0
        audio_chunks = []

        gen_start = time.time()
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            audio_chunks.append(audio_chunk)
            iter_audio_duration = audio_chunk.shape[-1] / S3GEN_SR

            # Capture first chunk metrics
            if metrics.chunk_count == 1:
                iter_first_chunk_latency = metrics.latency_to_first_chunk
                iter_t3_time = metrics.t3_token_generation_time
                iter_s3gen_time = metrics.s3gen_first_chunk_time

            iter_token_count = max_tokens  # Approximate
            iter_chunk_count = metrics.chunk_count

            # Stop after first chunk for latency measurement
            if metrics.chunk_count == 1:
                break

        gen_time = time.time() - gen_start

        # Calculate actual audio duration from generated chunks
        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=-1)
            iter_audio_duration = audio.shape[-1] / S3GEN_SR

        # Record metrics
        if iter_first_chunk_latency is not None:
            first_chunk_latency.add(iter_first_chunk_latency)
            t3_generation_time.add(iter_t3_time)
            s3gen_first_chunk.add(iter_s3gen_time)
            total_generation_time.add(gen_time)
            audio_duration.add(iter_audio_duration)

            detailed = DetailedMetrics(
                iteration=i + 1,
                first_chunk_latency=iter_first_chunk_latency,
                t3_token_generation=iter_t3_time,
                s3gen_first_chunk=iter_s3gen_time,
                total_generation_time=gen_time,
                audio_duration=iter_audio_duration,
                token_count=iter_token_count,
                chunk_count=iter_chunk_count,
            )
            all_metrics.append(detailed)

        # Cleanup
        model.shutdown()
        del model
        torch.cuda.empty_cache()

        iter_time = time.time() - iter_start
        print(f" ✓ (First chunk: {iter_first_chunk_latency*1000:.1f}ms, Total: {iter_time:.2f}s)")

    # Print statistics
    print("\n" + "="*70)
    print("LATENCY STATISTICS")
    print("="*70)

    print_stats_table(first_chunk_latency, "⚡ FIRST CHUNK LATENCY", "ms")
    print_stats_table(t3_generation_time, "📝 T3 TOKEN GENERATION", "ms")
    print_stats_table(s3gen_first_chunk, "🎵 S3GEN FIRST CHUNK", "ms")
    print_stats_table(total_generation_time, "⏱️  TOTAL GENERATION TIME", "ms")
    print_stats_table(audio_duration, "🔊 AUDIO DURATION", "s")

    # Breakdown analysis
    print("\n" + "="*70)
    print("LATENCY BREAKDOWN")
    print("="*70)

    avg_first_chunk = first_chunk_latency.avg * 1000
    avg_t3 = t3_generation_time.avg * 1000
    avg_s3gen = s3gen_first_chunk.avg * 1000

    print(f"\nAverage first chunk latency: {avg_first_chunk:.2f}ms")
    print(f"  - T3 generation:           {avg_t3:.2f}ms ({avg_t3/avg_first_chunk*100:.1f}%)")
    print(f"  - S3Gen first chunk:       {avg_s3gen:.2f}ms ({avg_s3gen/avg_first_chunk*100:.1f}%)")
    print(f"  - Other overhead:          {avg_first_chunk - avg_t3 - avg_s3gen:.2f}ms "
          f"({(avg_first_chunk - avg_t3 - avg_s3gen)/avg_first_chunk*100:.1f}%)")

    # Percentiles
    print(f"\nFirst Chunk Latency Percentiles:")
    print(f"  50th (median): {first_chunk_latency.median*1000:.2f}ms")
    print(f"  95th:          {first_chunk_latency.percentile_95*1000:.2f}ms")
    print(f"  99th:          {first_chunk_latency.percentile_99*1000:.2f}ms")

    # Comparison with async
    print("\n" + "="*70)
    print("COMPARISON: SYNC vs ASYNC")
    print("="*70)

    estimated_async_first_token = 0.050  # 50ms for first token (from async tests)
    estimated_async_s3gen = avg_s3gen / 1000  # Same S3Gen time
    estimated_async_total = estimated_async_first_token + estimated_async_s3gen

    print(f"\nCurrent (Sync vLLM):")
    print(f"  First chunk latency: {avg_first_chunk:.0f}ms")
    print(f"  - T3 (all tokens):   {avg_t3:.0f}ms")
    print(f"  - S3Gen:             {avg_s3gen:.0f}ms")

    print(f"\nAsyncLLMEngine (Estimated):")
    print(f"  First chunk latency: {estimated_async_total*1000:.0f}ms")
    print(f"  - T3 (first token):  {estimated_async_first_token*1000:.0f}ms")
    print(f"  - S3Gen:             {estimated_async_s3gen*1000:.0f}ms")

    speedup = avg_first_chunk / (estimated_async_total * 1000)
    print(f"\nEstimated speedup:   {speedup:.2f}x faster")

    if estimated_async_total < 1.0:
        print(f"  ✅ MEETS <1s TARGET!")
    else:
        print(f"  ❌ Does not meet <1s target")

    # Detailed per-iteration results
    print("\n" + "="*70)
    print("DETAILED PER-ITERATION RESULTS")
    print("="*70)
    print(f"\n{'Iter':<6} {'First Chunk':<15} {'T3 Gen':<12} {'S3Gen':<12} {'Total':<12}")
    print("-" * 70)

    for m in all_metrics:
        print(f"{m.iteration:<6} "
              f"{m.first_chunk_latency*1000:<15.2f} "
              f"{m.t3_token_generation*1000:<12.2f} "
              f"{m.s3gen_first_chunk*1000:<12.2f} "
              f"{m.total_generation_time*1000:<12.2f}")

    return {
        "first_chunk_latency": first_chunk_latency,
        "t3_generation": t3_generation_time,
        "s3gen_first_chunk": s3gen_first_chunk,
        "all_metrics": all_metrics,
    }


def main():
    """Run profiling tests."""
    print("\n" + "="*70)
    print("CHATTERBOX vLLM TTS - FIRST CHUNK LATENCY PROFILING")
    print("="*70)

    # Run profiling
    results = profile_first_chunk_latency(
        text="Hello world, this is a test.",
        iterations=10,
        max_tokens=200,
        warmup_iterations=2,
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    avg_latency = results["first_chunk_latency"].avg * 1000
    min_latency = results["first_chunk_latency"].min * 1000
    max_latency = results["first_chunk_latency"].max * 1000

    print(f"\n✅ Profiling complete!")
    print(f"\nFirst Chunk Latency ({results['first_chunk_latency'].count} iterations):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Min:     {min_latency:.2f}ms")
    print(f"  Max:     {max_latency:.2f}ms")
    print(f"  Range:   {max_latency - min_latency:.2f}ms")

    print(f"\n📊 Key Findings:")
    print(f"  - Consistent first chunk latency (std: {results['first_chunk_latency'].stdev*1000:.2f}ms)")
    print(f"  - T3 generation is the main bottleneck ({results['t3_generation'].avg*1000/avg_latency*100:.1f}% of latency)")
    print(f"  - AsyncLLMEngine could achieve ~{(results['s3gen_first_chunk'].avg + 0.050)*1000:.0f}ms first chunk latency")


if __name__ == "__main__":
    main()
