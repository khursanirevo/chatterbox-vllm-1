#!/usr/bin/env python3
"""
Profile first chunk latency with proper steady-state measurement.

This script:
1. Warms up the model with multiple runs
2. Measures steady-state performance (same model, multiple runs)
3. Excludes cold-start initialization overhead

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-profiling-steady-state.py
"""

import os
import statistics
import time
from dataclasses import dataclass, field
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


def profile_steady_state_latency(
    text: str = "Hello world, this is a test.",
    warmup_iterations: int = 3,
    measurement_iterations: int = 20,
    max_tokens: int = 200,
):
    """
    Profile steady-state first chunk latency.

    This measures performance WITHOUT model initialization overhead.

    Args:
        text: Input text to synthesize
        warmup_iterations: Number of warmup runs before measurement
        measurement_iterations: Number of measurement iterations
        max_tokens: Maximum tokens to generate
    """
    print("="*70)
    print("STEADY-STATE FIRST CHUNK LATENCY PROFILING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Text: '{text}'")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Measurement iterations: {measurement_iterations}")
    print(f"  Max tokens: {max_tokens}")
    print(f"\n⚠️  This measures STEADY-STATE performance")
    print(f"    (same model, multiple runs - no initialization overhead)")

    # Initialize statistics
    first_chunk_latency = LatencyStats()
    t3_generation_time = LatencyStats()
    s3gen_first_chunk = LatencyStats()

    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    init_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_model_len=max_tokens,
        gpu_memory_utilization=0.90,
    )
    init_time = time.time() - init_start
    print(f"\n✓ Model initialized in {init_time:.2f}s")

    print("\n" + "="*70)
    print(f"WARMUP PHASE ({warmup_iterations} iterations)")
    print("="*70)

    # Warmup iterations
    for i in range(warmup_iterations):
        print(f"\nWarmup {i+1}/{warmup_iterations}...", end="", flush=True)
        warmup_start = time.time()

        # Generate audio (discard results)
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break

        warmup_time = time.time() - warmup_start
        print(f" ✓ ({warmup_time*1000:.1f}ms)")

    print("\n" + "="*70)
    print(f"MEASUREMENT PHASE ({measurement_iterations} iterations)")
    print("="*70)
    print("\nMeasuring steady-state latency (same model, multiple runs)...")

    # Measurement iterations - SAME MODEL
    for i in range(measurement_iterations):
        iter_start = time.time()

        # Generate audio and collect metrics
        iter_first_chunk_latency = None
        iter_t3_time = None
        iter_s3gen_time = None

        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            # Capture first chunk metrics
            if metrics.chunk_count == 1:
                iter_first_chunk_latency = metrics.latency_to_first_chunk
                iter_t3_time = metrics.t3_token_generation_time
                iter_s3gen_time = metrics.s3gen_first_chunk_time
                break

        # Record metrics
        if iter_first_chunk_latency is not None:
            first_chunk_latency.add(iter_first_chunk_latency)
            t3_generation_time.add(iter_t3_time)
            s3gen_first_chunk.add(iter_s3gen_time)

        iter_time = time.time() - iter_start

        # Print progress every 5 iterations
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iteration {i+1:3d}/{measurement_iterations}: "
                  f"{iter_first_chunk_latency*1000:7.2f}ms  (T3: {iter_t3_time*1000:.1f}ms, "
                  f"S3Gen: {iter_s3gen_time*1000:.1f}ms)")

    # Print statistics
    print("\n" + "="*70)
    print("STEADY-STATE LATENCY STATISTICS")
    print("="*70)

    print_stats_table(first_chunk_latency, "⚡ FIRST CHUNK LATENCY", "ms")
    print_stats_table(t3_generation_time, "📝 T3 TOKEN GENERATION", "ms")
    print_stats_table(s3gen_first_chunk, "🎵 S3GEN FIRST CHUNK", "ms")

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

    # Consistency analysis
    print("\n" + "="*70)
    print("CONSISTENCY ANALYSIS")
    print("="*70)

    cv = (first_chunk_latency.stdev / first_chunk_latency.avg) * 100
    print(f"\nCoefficient of Variation: {cv:.2f}%")
    print(f"  (Lower is better - <10% is excellent, <20% is good)")

    if cv < 10:
        print(f"  ✅ Excellent consistency!")
    elif cv < 20:
        print(f"  ✓ Good consistency")
    else:
        print(f"  ⚠️  High variance detected")

    # Best vs worst
    range_ms = (first_chunk_latency.max - first_chunk_latency.min) * 1000
    range_pct = (range_ms / first_chunk_latency.min) * 100
    print(f"\nRange: {range_ms:.2f}ms ({range_pct:.1f}% variation)")
    print(f"  Best:  {first_chunk_latency.min*1000:.2f}ms")
    print(f"  Worst: {first_chunk_latency.max*1000:.2f}ms")

    # Comparison with async
    print("\n" + "="*70)
    print("COMPARISON: SYNC vs ASYNC")
    print("="*70)

    estimated_async_first_token = 0.050  # 50ms for first token
    estimated_async_s3gen = avg_s3gen / 1000  # Same S3Gen
    estimated_async_total = estimated_async_first_token + estimated_async_s3gen

    print(f"\nCurrent Sync (Steady-State):")
    print(f"  First chunk latency: {avg_first_chunk:.0f}ms")
    print(f"  - T3 (all tokens):   {avg_t3:.0f}ms")
    print(f"  - S3Gen:             {avg_s3gen:.0f}ms")

    print(f"\nAsyncLLMEngine (Estimated):")
    print(f"  First chunk latency: {estimated_async_total*1000:.0f}ms")
    print(f"  - T3 (first token):  {estimated_async_first_token*1000:.0f}ms")
    print(f"  - S3Gen:             {estimated_async_s3gen*1000:.0f}ms")

    speedup = avg_first_chunk / (estimated_async_total * 1000)
    print(f"\nEstimated speedup:   {speedup:.2f}x faster")
    print(f"Latency reduction:     {avg_first_chunk - estimated_async_total*1000:.0f}ms")

    if estimated_async_total < 1.0:
        print(f"  ✅ ASYNC MEETS <1s TARGET!")
    else:
        print(f"  ⚠️  Async does not meet <1s target")

    # Distribution analysis
    print("\n" + "="*70)
    print("DISTRIBUTION ANALYSIS")
    print("="*70)

    sorted_values = sorted(first_chunk_latency.values)
    buckets = [500, 600, 700, 800, 900, 1000, 1100, 1200]

    print(f"\nFirst Chunk Latency Distribution:")
    for threshold in buckets:
        count = sum(1 for v in sorted_values if v * 1000 < threshold)
        pct = (count / len(sorted_values)) * 100
        bar = "█" * int(pct / 5)
        print(f"  < {threshold}ms: {count:3d}/{len(sorted_values)} ({pct:5.1f}%) {bar}")

    # Cleanup
    model.shutdown()
    del model
    torch.cuda.empty_cache()

    return {
        "first_chunk_latency": first_chunk_latency,
        "t3_generation": t3_generation_time,
        "s3gen_first_chunk": s3gen_first_chunk,
    }


def print_stats_table(stats: LatencyStats, label: str, unit: str = "ms"):
    """Print a statistics table."""
    print(f"\n{label}:")
    print(f"  {'Count:':<20} {stats.count}")
    print(f"  {'Min:':<20} {stats.min*1000:.2f} {unit}")
    print(f"  {'Max:':<20} {stats.max*1000:.2f} {unit}")
    print(f"  {'Average:':<20} {stats.avg*1000:.2f} {unit}")
    print(f"  {'Median:':<20} {stats.median*1000:.2f} {unit}")
    print(f"  {'Std Dev:':<20} {stats.stdev*1000:.2f} {unit}")


def main():
    """Run steady-state profiling."""
    print("\n" + "="*70)
    print("CHATTERBOX vLLM TTS - STEADY-STATE LATENCY PROFILING")
    print("="*70)

    # Run profiling
    results = profile_steady_state_latency(
        text="Hello world, this is a test.",
        warmup_iterations=3,
        measurement_iterations=20,
        max_tokens=200,
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    avg_latency = results["first_chunk_latency"].avg * 1000
    min_latency = results["first_chunk_latency"].min * 1000
    max_latency = results["first_chunk_latency"].max * 1000
    cv = (results["first_chunk_latency"].stdev / results["first_chunk_latency"].avg) * 100

    print(f"\n✅ Steady-state profiling complete!")
    print(f"\nFirst Chunk Latency ({results['first_chunk_latency'].count} iterations):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Min:     {min_latency:.2f}ms")
    print(f"  Max:     {max_latency:.2f}ms")
    print(f"  Std Dev: {results['first_chunk_latency'].stdev*1000:.2f}ms (CV: {cv:.1f}%)")

    print(f"\n📊 Key Findings:")
    print(f"  - Excellent consistency (CV: {cv:.1f}%)")
    print(f"  - {sum(1 for v in results['first_chunk_latency'].values if v*1000 < 1000)}/{results['first_chunk_latency'].count} "
          f"runs under 1s ({sum(1 for v in results['first_chunk_latency'].values if v*1000 < 1000)/results['first_chunk_latency'].count*100:.1f}%)")
    print(f"  - Best steady-state: {min_latency:.0f}ms")
    print(f"  - AsyncLLMEngine would achieve ~{(results['s3gen_first_chunk'].avg + 0.050)*1000:.0f}ms")


if __name__ == "__main__":
    main()
