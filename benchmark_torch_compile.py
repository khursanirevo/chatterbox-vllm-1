#!/usr/bin/env python3
"""
Benchmark torch.compile() optimization for S3Gen.

Tests:
1. Baseline: No compilation
2. compile(mode="default")
3. compile(mode="reduce-overhead")
4. compile(mode="max-autotune")

Expected improvement: 30-50% speedup
"""

import asyncio
import time
import statistics
from pathlib import Path
import torch

from chatterbox_vllm import ChatterboxTTSAsync


# Test prompts
TEST_PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": "The weather today is quite nice, with clear skies and mild temperatures.",
    "long": "This is a significantly longer text passage designed to test the text to speech synthesis pipeline with more content to process through multiple stages including tokenization and generation.",
}


async def benchmark_configuration(
    use_compile: bool = False,
    compile_mode: str = "default",
    num_iterations: int = 5,
    num_warmup: int = 2,
) -> dict:
    """
    Benchmark S3Gen with torch.compile().

    Args:
        use_compile: Whether to use torch.compile()
        compile_mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
        num_iterations: Number of test iterations
        num_warmup: Number of warmup iterations

    Returns:
        Dictionary with benchmark results
    """
    config_name = f"compile={use_compile}_{compile_mode}" if use_compile else "no_compile"

    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")

    # Initialize model
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=False,
    )

    # Apply torch.compile() if requested
    if use_compile:
        print(f"Applying torch.compile(mode='{compile_mode}')...")
        # Compile the flow and mel2wav modules
        model.s3gen.flow = torch.compile(
            model.s3gen.flow,
            mode=compile_mode,
            fullgraph=False,
        )
        model.s3gen.mel2wav = torch.compile(
            model.s3gen.mel2wav,
            mode=compile_mode,
            fullgraph=False,
        )
        print("Compilation complete")

    results_by_category = {cat: [] for cat in TEST_PROMPTS.keys()}

    try:
        # Warmup iterations (important for compile to optimize)
        if num_warmup > 0:
            print(f"\nWarming up ({num_warmup} iterations)...")
            for i in range(num_warmup):
                for category, text in TEST_PROMPTS.items():
                    await model.generate(
                        prompts=[text],
                        temperature=0.8,
                        exaggeration=0.5,
                        max_tokens=1000,
                        diffusion_steps=5,
                    )
                print(f"  Warmup {i+1}/{num_warmup} complete")

        # Benchmark iterations
        print(f"\nBenchmarking ({num_iterations} iterations)...")
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            for category, text in TEST_PROMPTS.items():
                start_time = time.time()

                result = await model.generate(
                    prompts=[text],
                    temperature=0.8,
                    exaggeration=0.5,
                    max_tokens=1000,
                    diffusion_steps=5,
                )

                end_time = time.time()
                latency = end_time - start_time

                results_by_category[category].append(latency)

                if iteration == 0:
                    print(f"  {category.capitalize():<8}: {latency:.3f}s")

    finally:
        await model.shutdown()

    # Calculate statistics
    stats = {}
    for category in TEST_PROMPTS.keys():
        latencies = results_by_category[category]
        if latencies:
            stats[category] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            }

    return {
        "config": config_name,
        "use_compile": use_compile,
        "compile_mode": compile_mode,
        "stats": stats,
    }


def print_results_table(all_results: list):
    """Print comparison table."""
    print("\n" + "="*100)
    print("TORCH.COMPILE() OPTIMIZATION BENCHMARK RESULTS")
    print("="*100)

    # Table header
    print(f"\n{'Configuration':<30} {'Short (s)':<15} {'Medium (s)':<15} {'Long (s)':<15} {'Speedup'}")
    print("-"*100)

    baseline = all_results[0] if all_results else None
    baseline_short_mean = baseline["stats"]["short"]["mean"] if baseline else 1.0

    for result in all_results:
        config = result["config"]
        short_mean = result["stats"]["short"]["mean"]
        medium_mean = result["stats"]["medium"]["mean"]
        long_mean = result["stats"]["long"]["mean"]

        speedup = baseline_short_mean / short_mean if config != baseline.get("config") else 1.0

        print(f"{config:<30} {short_mean:>6.3f}s     {medium_mean:>6.3f}s     {long_mean:>6.3f}s     {speedup:>5.2f}x")

    print("="*100)

    # P95 table
    print(f"\n{'Configuration':<30} {'Short P95 (s)':<15} {'Medium P95 (s)':<15} {'Long P95 (s)':<15}")
    print("-"*100)

    for result in all_results:
        config = result["config"]
        short_p95 = result["stats"]["short"]["p95"]
        medium_p95 = result["stats"]["medium"]["p95"]
        long_p95 = result["stats"]["long"]["p95"]

        print(f"{config:<30} {short_p95:>6.3f}s       {medium_p95:>6.3f}s       {long_p95:>6.3f}s")

    print("="*100)


def calculate_improvements(all_results: list):
    """Calculate and print improvement metrics."""
    print("\n" + "="*100)
    print("IMPROVEMENT ANALYSIS")
    print("="*100)

    baseline = all_results[0] if all_results else None
    if not baseline:
        return

    baseline_short = baseline["stats"]["short"]["mean"]
    baseline_medium = baseline["stats"]["medium"]["mean"]
    baseline_long = baseline["stats"]["long"]["mean"]

    print(f"\nBaseline: {baseline['config']}")
    print(f"  Short: {baseline_short:.3f}s, Medium: {baseline_medium:.3f}s, Long: {baseline_long:.3f}s")
    print()

    for result in all_results[1:]:
        config = result["config"]
        short = result["stats"]["short"]["mean"]
        medium = result["stats"]["medium"]["mean"]
        long = result["stats"]["long"]["mean"]

        short_improvement = ((baseline_short - short) / baseline_short) * 100
        medium_improvement = ((baseline_medium - medium) / baseline_medium) * 100
        long_improvement = ((baseline_long - long) / baseline_long) * 100

        print(f"{config}:")
        print(f"  Short:  {short:.3f}s ({short_improvement:+.1f}%)")
        print(f"  Medium: {medium:.3f}s ({medium_improvement:+.1f}%)")
        print(f"  Long:   {long:.3f}s ({long_improvement:+.1f}%)")
        print()

    print("="*100)


async def main():
    """Run torch.compile() benchmark."""
    print("\n" + "="*100)
    print("TORCH.COMPILE() OPTIMIZATION BENCHMARK")
    print("="*100)

    # Define configurations to test
    configurations = [
        # Baseline
        (False, "default"),  # No compilation

        # Test different compilation modes
        (True, "default"),          # Standard compilation
        (True, "reduce-overhead"),  # Optimized for inference
        # Skip max-autotune for now - takes too long
        # (True, "max-autotune"),    # Maximum optimization (slow compilation)
    ]

    all_results = []
    num_iterations = 3
    num_warmup = 2  # Important for compile!

    print(f"\nConfigurations to test: {len(configurations)}")
    print(f"Warmup iterations: {num_warmup}")
    print(f"Benchmark iterations: {num_iterations}")
    print(f"Total tests: {len(configurations) * (num_iterations + num_warmup) * len(TEST_PROMPTS)}")

    for use_compile, compile_mode in configurations:
        result = await benchmark_configuration(
            use_compile=use_compile,
            compile_mode=compile_mode,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        all_results.append(result)

    # Print results
    print_results_table(all_results)
    calculate_improvements(all_results)

    # Find best configuration
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    # Find best configuration for short requests (TTFA focus)
    best_short = min(all_results, key=lambda r: r["stats"]["short"]["mean"])
    best_config = best_short["config"]
    best_latency = best_short["stats"]["short"]["mean"]
    best_improvement = ((all_results[0]["stats"]["short"]["mean"] - best_latency) /
                       all_results[0]["stats"]["short"]["mean"]) * 100

    print(f"\nBest for TTFA (short requests): {best_config}")
    print(f"  Latency: {best_latency:.3f}s ({best_improvement:+.1f}% improvement)")
    print(f"  P95: {best_short['stats']['short']['p95']:.3f}s")

    # Check if P95 < 1s target
    if best_short["stats"]["short"]["p95"] < 1.0:
        print(f"  ✅ Meets P95 < 1s target!")
    else:
        print(f"  ⚠️  Does not meet P95 < 1s target")

    # Check if P95 < 0.5s (aggressive target)
    if best_short["stats"]["short"]["p95"] < 0.5:
        print(f"  🎯 Also meets aggressive P95 < 0.5s target!")

    print("\n" + "="*100)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())
