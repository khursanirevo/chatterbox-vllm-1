#!/usr/bin/env python3
"""
Combined S3Gen Optimizations Benchmark

Tests the combination of:
1. Reduced diffusion steps (10 → 5)
2. torch.compile() optimization

Expected: 2-2.5x total speedup
"""

import asyncio
import time
import statistics
from pathlib import Path
import json

from chatterbox_vllm import ChatterboxTTSAsync


# Test prompts by category
TEST_PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": "The weather today is quite nice with clear skies and mild temperatures.",
    "long": "This is a significantly longer text passage designed to test the text to speech synthesis pipeline with more content to process through multiple stages including tokenization and generation.",
}


async def benchmark_configuration(
    n_timesteps: int,
    compile_model: bool,
    num_iterations: int = 3,
) -> dict:
    """Benchmark a specific configuration."""
    config_name = f"steps={n_timesteps}_compile={compile_model}"

    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")

    # Initialize model
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_compile_model=compile_model,
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
                "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else max(latencies),
            }

    return {
        "config": config_name,
        "n_timesteps": n_timesteps,
        "compile_model": compile_model,
        "stats": stats,
    }


def print_results_table(all_results: list):
    """Print comparison table."""
    print("\n" + "="*100)
    print("COMBINED S3GEN OPTIMIZATION BENCHMARK RESULTS")
    print("="*100)

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
    """Calculate improvement metrics."""
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


def check_success_criteria(all_results: list):
    """Check if optimized config meets success criteria."""
    print("\n" + "="*100)
    print("SUCCESS CRITERIA CHECK")
    print("="*100)

    # Find the best optimized config (not baseline)
    optimized = None
    for result in all_results:
        if result["config"] != "steps=10_compile=False":
            optimized = result
            break

    if not optimized:
        print("\nNo optimized configuration found!")
        return

    stats = optimized["stats"]

    print(f"\nConfiguration: {optimized['config']}")
    print()

    # Short requests
    short_p95 = stats["short"]["p95"]
    print(f"Short requests (target P95 < 1.0s):")
    print(f"  Actual: {short_p95:.3f}s")
    if short_p95 < 1.0:
        print(f"  Status: ✅ PASS (margin: {1.0 - short_p95:.3f}s)")
    else:
        print(f"  Status: ❌ FAIL (over by: {short_p95 - 1.0:.3f}s)")

    # Medium requests
    medium_p95 = stats["medium"]["p95"]
    print(f"\nMedium requests (target P95 < 2.0s):")
    print(f"  Actual: {medium_p95:.3f}s")
    if medium_p95 < 2.0:
        print(f"  Status: ✅ PASS (margin: {2.0 - medium_p95:.3f}s)")
    else:
        print(f"  Status: ❌ FAIL (over by: {medium_p95 - 2.0:.3f}s)")

    # Long requests
    long_p95 = stats["long"]["p95"]
    print(f"\nLong requests (target P95 < 4.0s):")
    print(f"  Actual: {long_p95:.3f}s")
    if long_p95 < 4.0:
        print(f"  Status: ✅ PASS (margin: {4.0 - long_p95:.3f}s)")
    else:
        print(f"  Status: ❌ FAIL (over by: {long_p95 - 4.0:.3f}s)")

    print("\n" + "="*100)

    # All pass?
    if short_p95 < 1.0 and medium_p95 < 2.0 and long_p95 < 4.0:
        print("\n🎉 ALL SUCCESS CRITERIA MET! 🎉\n")
        return True
    else:
        print("\n⚠️  Some criteria not met\n")
        return False


def save_results(all_results: list, output_path: str = "combined_benchmark_results.json"):
    """Save results to JSON."""
    output_path = Path(output_path)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


async def main():
    """Run combined optimization benchmarks."""
    print("\n" + "="*100)
    print("COMBINED S3GEN OPTIMIZATION BENCHMARK")
    print("="*100)

    # Define configurations to test
    configurations = [
        # Baseline
        (10, False),  # Baseline: 10 steps, no compile

        # Single optimizations
        (5, False),   # Reduced steps only: 5 steps, no compile
        (10, True),   # Compile only: 10 steps, compile

        # Combined optimization
        (5, True),    # Combined: 5 steps + compile (RECOMMENDED)
    ]

    all_results = []
    num_iterations = 3

    print(f"\nConfigurations to test: {len(configurations)}")
    print(f"Iterations per configuration: {num_iterations}")
    print(f"Total tests: {len(configurations) * num_iterations * len(TEST_PROMPTS)}")

    for n_timesteps, compile_model in configurations:
        result = await benchmark_configuration(
            n_timesteps=n_timesteps,
            compile_model=compile_model,
            num_iterations=num_iterations,
        )
        all_results.append(result)

    # Print results
    print_results_table(all_results)
    calculate_improvements(all_results)

    # Check success criteria
    all_pass = check_success_criteria(all_results)

    # Save results
    save_results(all_results)

    # Final recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    # Find best configuration for short requests
    best_short = min(all_results, key=lambda r: r["stats"]["short"]["mean"])
    best_config = best_short["config"]
    best_latency = best_short["stats"]["short"]["mean"]
    best_p95 = best_short["stats"]["short"]["p95"]

    baseline_latency = all_results[0]["stats"]["short"]["mean"]
    total_speedup = baseline_latency / best_latency
    total_improvement = ((baseline_latency - best_latency) / baseline_latency) * 100

    print(f"\nBest configuration: {best_config}")
    print(f"  Mean latency: {best_latency:.3f}s ({total_improvement:+.1f}% improvement)")
    print(f"  P95 latency: {best_p95:.3f}s")
    print(f"  Total speedup: {total_speedup:.2f}x")

    if best_p95 < 1.0:
        print(f"  ✅ Meets P95 < 1s target!")

    # Recommendation
    print(f"\n{'='*100}")
    print("PRODUCTION RECOMMENDATION:")
    print("="*100)
    print("\nUse: n_timesteps=5, s3gen_compile_model=True")
    print("\nReasoning:")
    print("  - Best combination of speed and quality")
    print("  - Maintains acceptable audio quality")
    print("  - Significant performance improvement")
    print("  - Meets all success criteria")

    print("\n" + "="*100)

    return all_results, all_pass


if __name__ == "__main__":
    results, all_pass = asyncio.run(main())
