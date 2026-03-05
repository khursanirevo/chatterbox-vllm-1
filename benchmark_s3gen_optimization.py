#!/usr/bin/env python3
"""
S3Gen Optimization Benchmark

Tests different diffusion step counts and FP16 settings to measure TTFA improvement.

Configurations to test:
1. Baseline: n_timesteps=10, fp16=False
2. Reduced steps: n_timesteps=5, fp16=False
3. Minimal steps: n_timesteps=3, fp16=False
4. FP16 only: n_timesteps=10, fp16=True
5. Combined: n_timesteps=5, fp16=True
6. Aggressive: n_timesteps=3, fp16=True
"""

import asyncio
import time
import statistics
from pathlib import Path
from typing import Dict, List
import json

from chatterbox_vllm import ChatterboxTTSAsync


# Test prompts by category
TEST_PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": "The weather today is quite nice, with clear skies and mild temperatures.",
    "long": "This is a significantly longer text passage designed to test the text to speech synthesis pipeline with more content to process through multiple stages including tokenization and generation.",
}


async def benchmark_configuration(
    n_timesteps: int,
    use_fp16: bool,
    num_iterations: int = 5,
) -> Dict:
    """
    Benchmark a specific S3Gen configuration.

    Args:
        n_timesteps: Number of diffusion steps
        use_fp16: Whether to use FP16
        num_iterations: Number of test iterations

    Returns:
        Dictionary with benchmark results
    """
    config_name = f"timesteps={n_timesteps}_fp16={use_fp16}"

    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")

    # Initialize model with specific configuration
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=use_fp16,
    )

    results_by_category = {cat: [] for cat in TEST_PROMPTS.keys()}

    try:
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            for category, text in TEST_PROMPTS.items():
                start_time = time.time()

                result = await model.generate(
                    prompts=[text],
                    temperature=0.8,
                    exaggeration=0.5,
                    max_tokens=1000,
                    diffusion_steps=n_timesteps,  # Key parameter
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
                "p99": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) >= 2 else max(latencies),
            }

    return {
        "config": config_name,
        "n_timesteps": n_timesteps,
        "use_fp16": use_fp16,
        "stats": stats,
        "all_latencies": results_by_category,
    }


def print_results_table(all_results: List[Dict]):
    """Print comparison table of all configurations."""
    print("\n" + "="*100)
    print("S3GEN OPTIMIZATION BENCHMARK RESULTS")
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

    # Detailed P95 table
    print(f"\n{'Configuration':<30} {'Short P95 (s)':<15} {'Medium P95 (s)':<15} {'Long P95 (s)':<15}")
    print("-"*100)

    for result in all_results:
        config = result["config"]
        short_p95 = result["stats"]["short"]["p95"]
        medium_p95 = result["stats"]["medium"]["p95"]
        long_p95 = result["stats"]["long"]["p95"]

        print(f"{config:<30} {short_p95:>6.3f}s       {medium_p95:>6.3f}s       {long_p95:>6.3f}s")

    print("="*100)


def calculate_improvements(all_results: List[Dict]):
    """Calculate and print improvement metrics."""
    print("\n" + "="*100)
    print("IMPROVEMENT ANALYSIS vs BASELINE")
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


def save_results(all_results: List[Dict], output_path: str = "s3gen_benchmark_results.json"):
    """Save benchmark results to JSON."""
    output_path = Path(output_path)

    # Convert to serializable format
    serializable_results = []
    for result in all_results:
        serializable_results.append({
            "config": result["config"],
            "n_timesteps": result["n_timesteps"],
            "use_fp16": result["use_fp16"],
            "stats": result["stats"],
        })

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


async def main():
    """Run all S3Gen optimization benchmarks."""
    print("\n" + "="*100)
    print("S3GEN OPTIMIZATION BENCHMARK")
    print("="*100)

    # Define configurations to test
    configurations = [
        # Baseline
        (10, False),  # Baseline: 10 steps, no FP16

        # Single optimizations
        (5, False),   # Reduced steps: 5 steps, no FP16
        (3, False),   # Minimal steps: 3 steps, no FP16

        # FP16 currently has dtype mismatch - skip for now
        # (10, True),   # FP16 only: 10 steps, FP16
        # (5, True),    # Balanced: 5 steps, FP16
        # (3, True),    # Aggressive: 3 steps, FP16
    ]

    all_results = []
    num_iterations = 3  # Reduced for faster initial testing

    print(f"\nConfigurations to test: {len(configurations)}")
    print(f"Iterations per configuration: {num_iterations}")
    print(f"Total tests: {len(configurations) * num_iterations * len(TEST_PROMPTS)}")

    for n_timesteps, use_fp16 in configurations:
        result = await benchmark_configuration(
            n_timesteps=n_timesteps,
            use_fp16=use_fp16,
            num_iterations=num_iterations,
        )
        all_results.append(result)

    # Print results
    print_results_table(all_results)
    calculate_improvements(all_results)

    # Save results
    save_results(all_results)

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

    print("\n" + "="*100)

    return all_results


if __name__ == "__main__":
    results = asyncio.run(main())
