#!/usr/bin/env python3
"""
Profile first chunk latency to identify bottlenecks.

Tests:
1. Cold start (no warmup) - measures initialization overhead
2. Steady state (with warmup) - measures true performance
3. S3Gen warmup analysis - checks if S3Gen needs warmup

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python profile_first_chunk.py
"""

import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS, StreamingMetrics
from chatterbox_vllm.models.s3gen import S3GEN_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


TEST_TEXT = "Hello world, this is a test of the streaming TTS system."


def profile_cold_start(model: ChatterboxTTS, text: str, max_tokens: int = 200) -> dict:
    """Profile first chunk with COLD model (no warmup)."""
    print("\n" + "="*70)
    print("COLD START PROFILING (No Warmup)")
    print("="*70)

    start_time = time.time()
    metrics = None

    for audio_chunk, m in model.generate_stream(
        text=text,
        max_tokens=max_tokens,
        chunk_size=25,
        print_metrics=True,
    ):
        metrics = m
        break  # Only need first chunk

    total_time = time.time() - start_time

    return {
        "first_chunk_latency_s": total_time,
        "t3_time_s": metrics.t3_token_generation_time,
        "s3gen_time_s": metrics.s3gen_first_chunk_time,
        "metrics": metrics,
    }


def profile_steady_state(model: ChatterboxTTS, text: str, max_tokens: int = 200, warmup_runs: int = 3) -> dict:
    """Profile first chunk with WARMED model."""
    print("\n" + "="*70)
    print(f"STEADY-STATE PROFILING (After {warmup_runs} warmup runs)")
    print("="*70)

    print(f"\nWarming up model ({warmup_runs} runs)...")
    warmup_times = []

    for i in range(warmup_runs):
        w_start = time.time()
        for audio_chunk, _ in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            break
        w_time = time.time() - w_start
        warmup_times.append(w_time)
        print(f"  Warmup {i+1}/{warmup_runs}: {w_time*1000:.1f}ms")

    print(f"\n✓ Warmup complete")
    print(f"  Average warmup time: {sum(warmup_times)/len(warmup_times)*1000:.1f}ms")
    print(f"  Last warmup: {warmup_times[-1]*1000:.1f}ms")

    print("\nMeasuring steady-state first chunk...")
    start_time = time.time()
    metrics = None

    for audio_chunk, m in model.generate_stream(
        text=text,
        max_tokens=max_tokens,
        chunk_size=25,
        print_metrics=True,
    ):
        metrics = m
        break

    total_time = time.time() - start_time

    return {
        "first_chunk_latency_s": total_time,
        "t3_time_s": metrics.t3_token_generation_time,
        "s3gen_time_s": metrics.s3gen_first_chunk_time,
        "metrics": metrics,
        "warmup_times": warmup_times,
    }


def profile_s3gen_compilation():
    """Test if S3Gen has one-time compilation overhead."""
    print("\n" + "="*70)
    print("S3GEN COMPILATION OVERHEAD ANALYSIS")
    print("="*70)

    from chatterbox_vllm.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )

    # Get some tokens to process
    tokens = []
    for audio_chunk, _ in model.generate_stream(
        text=TEST_TEXT,
        max_tokens=200,
        chunk_size=25,
        print_metrics=False,
    ):
        break

    # Now measure S3Gen processing time repeatedly
    print("\nMeasuring S3Gen first chunk processing time (10 iterations)...")
    s3gen_times = []

    for i in range(10):
        # This will re-process the same tokens
        start = time.time()
        for audio_chunk, _ in model.generate_stream(
            text=TEST_TEXT,
            max_tokens=200,
            chunk_size=25,
            print_metrics=False,
        ):
            s3gen_time = time.time() - start
            s3gen_times.append(s3gen_time)
            break

    print("\nS3Gen First Chunk Times:")
    for i, t in enumerate(s3gen_times, 1):
        status = "🔥 FIRST (cold)" if i == 1 else ""
        print(f"  {i:2d}. {t*1000:7.2f}ms  {status}")

    avg_after_first = sum(s3gen_times[1:]) / len(s3gen_times[1:]) if len(s3gen_times) > 1 else 0
    first_overhead = s3gen_times[0] - avg_after_first if len(s3gen_times) > 1 else 0

    print(f"\nS3Gen Analysis:")
    print(f"  First call:     {s3gen_times[0]*1000:.2f}ms")
    print(f"  Average (2-10): {avg_after_first*1000:.2f}ms")
    print(f"  Overhead:       {first_overhead*1000:.2f}ms ({first_overhead/s3gen_times[0]*100:.1f}%)")

    model.shutdown()

    return {
        "s3gen_times": s3gen_times,
        "first_overhead_ms": first_overhead * 1000,
        "avg_after_first_ms": avg_after_first * 1000,
    }


def profile_detailed_breakdown():
    """Detailed breakdown of T3 vs S3Gen contributions."""
    print("\n" + "="*70)
    print("DETAILED COMPONENT BREAKDOWN")
    print("="*70)

    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )

    print("\nMeasuring with detailed metrics...")
    results = []

    for run in range(5):
        print(f"\n--- Run {run+1}/5 ---")

        start = time.time()
        metrics = None

        for audio_chunk, m in model.generate_stream(
            text=TEST_TEXT,
            max_tokens=200,
            chunk_size=25,
            print_metrics=True,
        ):
            metrics = m
            break

        total = time.time() - start

        results.append({
            "total": total,
            "t3": metrics.t3_token_generation_time,
            "s3gen": metrics.s3gen_first_chunk_time,
        })

    model.shutdown()

    # Calculate statistics
    totals = [r["total"] for r in results]
    t3s = [r["t3"] for r in results]
    s3gens = [r["s3gen"] for r in results]

    print("\n" + "="*70)
    print("COMPONENT STATISTICS (5 runs)")
    print("="*70)

    print(f"\n{'Component':<15} {'Avg':<12} {'Min':<12} {'Max':<12} {'% of Total'}")
    print("-" * 60)

    avg_total = sum(totals) / len(totals)
    avg_t3 = sum(t3s) / len(t3s)
    avg_s3gen = sum(s3gens) / len(s3gens)

    print(f"{'Total':<15} {avg_total*1000:<12.1f} {min(totals)*1000:<12.1f} {max(totals)*1000:<12.1f} 100.0%")
    print(f"{'T3 Gen':<15} {avg_t3*1000:<12.1f} {min(t3s)*1000:<12.1f} {max(t3s)*1000:<12.1f} {avg_t3/avg_total*100:>5.1f}%")
    print(f"{'S3Gen':<15} {avg_s3gen*1000:<12.1f} {min(s3gens)*1000:<12.1f} {max(s3gens)*1000:<12.1f} {avg_s3gen/avg_total*100:>5.1f}%")
    print(f"{'Other':<15} {(avg_total-avg_t3-avg_s3gen)*1000:<12.1f} {'-':<12} {'-':<12} {(avg_total-avg_t3-avg_s3gen)/avg_total*100:>5.1f}%")

    return results


def main():
    print("="*70)
    print("FIRST CHUNK LATENCY PROFILING")
    print("="*70)
    print(f"\nTest text: '{TEST_TEXT}'")

    # Test 1: Cold start
    print("\n" + "#"*70)
    print("# TEST 1: COLD START (Fresh model)")
    print("#"*70)

    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )

    cold_result = profile_cold_start(model, TEST_TEXT)
    model.shutdown()

    # Cleanup
    torch.cuda.empty_cache()
    time.sleep(2)

    # Test 2: Steady state
    print("\n" + "#"*70)
    print("# TEST 2: STEADY STATE (With warmup)")
    print("#"*70)

    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )

    steady_result = profile_steady_state(model, TEST_TEXT, warmup_runs=3)
    model.shutdown()

    # Cleanup
    torch.cuda.empty_cache()
    time.sleep(2)

    # Test 3: S3Gen compilation
    print("\n" + "#"*70)
    print("# TEST 3: S3GEN COMPILATION OVERHEAD")
    print("#"*70)

    s3gen_result = profile_s3gen_compilation()

    # Cleanup
    torch.cuda.empty_cache()
    time.sleep(2)

    # Test 4: Detailed breakdown
    print("\n" + "#"*70)
    print("# TEST 4: DETAILED COMPONENT BREAKDOWN")
    print("#"*70)

    detailed_results = profile_detailed_breakdown()

    # SUMMARY
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Test':<20} {'First Chunk':<15} {'T3 Time':<12} {'S3Gen Time':<12}")
    print("-" * 65)

    cold_latency = cold_result["first_chunk_latency_s"] * 1000
    cold_t3 = cold_result["t3_time_s"] * 1000
    cold_s3gen = cold_result["s3gen_time_s"] * 1000

    steady_latency = steady_result["first_chunk_latency_s"] * 1000
    steady_t3 = steady_result["t3_time_s"] * 1000
    steady_s3gen = steady_result["s3gen_time_s"] * 1000

    print(f"{'Cold Start':<20} {cold_latency:>8.1f}ms      {cold_t3:>8.1f}ms     {cold_s3gen:>8.1f}ms")
    print(f"{'Steady State':<20} {steady_latency:>8.1f}ms      {steady_t3:>8.1f}ms     {steady_s3gen:>8.1f}ms")

    improvement = cold_latency - steady_latency
    improvement_pct = (improvement / cold_latency) * 100

    print(f"\n📊 Warmup Improvement:")
    print(f"  Latency reduction: {improvement:.1f}ms ({improvement_pct:.1f}%)")
    print(f"  Cold -> Steady: {cold_latency:.1f}ms -> {steady_latency:.1f}ms")

    print(f"\n🔍 S3Gen Overhead:")
    print(f"  First call overhead: {s3gen_result['first_overhead_ms']:.1f}ms")
    print(f"  After warmup: {s3gen_result['avg_after_first_ms']:.1f}ms")

    print(f"\n🎯 Analysis:")
    if steady_latency < 1000:
        print(f"  ✅ Steady state meets <1s target!")
    else:
        print(f"  ⚠️  Steady state still above <1s target: {steady_latency:.1f}ms")
        print(f"      Bottleneck: T3 ({steady_t3:.1f}ms) + S3Gen ({steady_s3gen:.1f}ms)")

    print(f"\n💡 Key Findings:")
    print(f"  1. Cold start overhead: ~{improvement:.0f}ms")
    print(f"  2. S3Gen first-call overhead: ~{s3gen_result['first_overhead_ms']:.0f}ms")
    print(f"  3. Main bottleneck in steady state: T3 token generation ({steady_t3:.0f}ms)")
    print(f"  4. For <1s target, need to use AsyncLLMEngine for T3")


if __name__ == "__main__":
    main()
