#!/usr/bin/env python3
"""
Benchmark FP16 vs FP32 performance for S3Gen.

This script compares the performance (latency and quality) of:
1. FP32 mode (baseline)
2. FP16 mode (optimized)
"""

import asyncio
import time
import torch
from pathlib import Path
from chatterbox_vllm import ChatterboxTTSAsync


async def benchmark_mode(use_fp16: bool, num_runs: int = 5):
    """Benchmark TTS generation with specific precision mode."""

    mode_name = "FP16" if use_fp16 else "FP32"
    print(f"\n{'='*60}")
    print(f"Benchmarking {mode_name} mode")
    print(f"{'='*60}")

    # Initialize model
    print(f"\nInitializing model in {mode_name} mode...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=use_fp16,
        s3gen_compile_model=False,
    )

    # Verify precision
    flow = model.s3gen.flow
    affine_dtype = flow.spk_embed_affine_layer.weight.dtype
    print(f"✓ Affine layer dtype: {affine_dtype}")

    if use_fp16 and affine_dtype != torch.float16:
        print(f"✗ ERROR: Expected FP16 but got {affine_dtype}")
        return None

    # Test prompts of varying lengths
    test_cases = [
        ("Short prompt", "Hello world."),
        ("Medium prompt", "This is a medium length prompt that will take some time to synthesize into speech output."),
        ("Long prompt", "This is a much longer prompt with significantly more text to process. It should take the longest time to complete the synthesis because there are many more tokens to generate and subsequently decode into audio waveform data."),
    ]

    results = {}

    for category, text in test_cases:
        print(f"\nTesting: {category}")

        timings = []

        for i in range(num_runs):
            start = time.time()

            try:
                audio = await model.generate(
                    prompts=[text],
                    audio_prompt_path=None,
                    temperature=0.8,
                    exaggeration=0.5,
                )

                if audio and len(audio) > 0:
                    elapsed = time.time() - start
                    timings.append(elapsed)
                    print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s (audio: {audio[0].shape[1]} samples)")
                else:
                    print(f"  Run {i+1}/{num_runs}: FAILED (no output)")

            except Exception as e:
                print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
                return None

        if timings:
            mean_time = sum(timings) / len(timings)
            min_time = min(timings)
            max_time = max(timings)
            results[category] = {
                "mean": mean_time,
                "min": min_time,
                "max": max_time,
                "all": timings,
            }
            print(f"  Mean: {mean_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s")

    return results


async def main():
    print("="*60)
    print("S3Gen FP16 Benchmark")
    print("="*60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires GPU.")
        return 1

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    num_runs = 3

    # Benchmark FP32 (baseline)
    print("\n" + "="*60)
    print("BASELINE: FP32 mode")
    print("="*60)
    fp32_results = await benchmark_mode(use_fp16=False, num_runs=num_runs)

    if not fp32_results:
        print("\n✗ FP32 benchmark failed!")
        return 1

    # Benchmark FP16 (optimized)
    print("\n" + "="*60)
    print("OPTIMIZED: FP16 mode")
    print("="*60)
    fp16_results = await benchmark_mode(use_fp16=True, num_runs=num_runs)

    if not fp16_results:
        print("\n✗ FP16 benchmark failed!")
        return 1

    # Compare results
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    print(f"\n{'Category':<20} {'FP32':<12} {'FP16':<12} {'Speedup':<10}")
    print("-"*60)

    overall_speedup = []

    for category in fp32_results.keys():
        fp32_time = fp32_results[category]["mean"]
        fp16_time = fp16_results[category]["mean"]
        speedup = fp32_time / fp16_time
        improvement = ((fp32_time - fp16_time) / fp32_time) * 100

        print(f"{category:<20} {fp32_time:<12.3f} {fp16_time:<12.3f} {speedup:<10.2f}x ({improvement:+.1f}%)")

        overall_speedup.append(speedup)

    avg_speedup = sum(overall_speedup) / len(overall_speedup)
    avg_improvement = ((1 - 1/avg_speedup) * -100) if avg_speedup > 1 else ((1 - avg_speedup) * 100)

    print("-"*60)
    print(f"{'Average':<20} {'':<12} {'':<12} {avg_speedup:<10.2f}x ({avg_improvement:+.1f}%)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if avg_speedup > 1.0:
        print(f"\n✅ FP16 mode is {avg_speedup:.2f}x FASTER on average")
        print(f"   Average latency improvement: {avg_improvement:.1f}%")
        print("\nThe FP16 dtype mismatch has been successfully fixed!")
        print("You can now use s3gen_use_fp16=True for production deployments.")
    else:
        print(f"\n⚠ FP16 mode is {1/avg_speedup:.2f}x SLOWER on average")
        print(f"This might be due to:")
        print("  - GPU architecture not optimized for FP16")
        print("  - Overhead of dtype conversion")
        print("  - Mixed precision operations not fully supported")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
