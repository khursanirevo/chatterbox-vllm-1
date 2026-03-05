#!/usr/bin/env python3
"""
Benchmark torch.compile() optimization for S3Gen.

Tests:
1. Baseline (n_timesteps=10, no compile)
2. With compile (n_timesteps=10, compile=True)
3. Verify warmup effect
"""

import asyncio
import time
from chatterbox_vllm import ChatterboxTTSAsync


TEST_PROMPTS = {
    "short": "Hello, how are you today?",
    "medium": "The weather today is quite nice with clear skies.",
}


async def benchmark_with_compile():
    """Benchmark with torch.compile() enabled."""
    print("\n" + "="*80)
    print("TORCH.COMPILE() BENCHMARK")
    print("="*80)

    # Test without compile
    print("\n[1/2] Testing WITHOUT torch.compile()...")
    print("-"*80)

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_compile_model=False,  # No compile
    )

    latencies_no_compile = {}

    for category, text in TEST_PROMPTS.items():
        # Warmup
        await model.generate(prompts=[text], temperature=0.8)

        # Measure
        times = []
        for _ in range(3):
            start = time.time()
            await model.generate(prompts=[text], temperature=0.8)
            times.append(time.time() - start)

        latencies_no_compile[category] = sum(times) / len(times)
        print(f"  {category.capitalize()}: {latencies_no_compile[category]:.3f}s (mean of 3)")

    await model.shutdown()

    # Test with compile
    print("\n[2/2] Testing WITH torch.compile()...")
    print("-"*80)

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_compile_model=True,  # Enable compile
    )

    latencies_with_compile = {}

    for category, text in TEST_PROMPTS.items():
        # First run includes warmup (torch.compile compiles on first run)
        print(f"  {category.capitalize()}: Warmup run...")
        start = time.time()
        await model.generate(prompts=[text], temperature=0.8)
        warmup_time = time.time() - start
        print(f"    Warmup: {warmup_time:.3f}s")

        # Now measure performance after compilation
        times = []
        for _ in range(3):
            start = time.time()
            await model.generate(prompts=[text], temperature=0.8)
            times.append(time.time() - start)

        latencies_with_compile[category] = sum(times) / len(times)
        print(f"    After warmup: {latencies_with_compile[category]:.3f}s (mean of 3)")

    await model.shutdown()

    # Calculate speedup
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    for category in TEST_PROMPTS.keys():
        no_compile = latencies_no_compile[category]
        with_compile = latencies_with_compile[category]
        speedup = no_compile / with_compile
        improvement = ((no_compile - with_compile) / no_compile) * 100

        print(f"\n{category.capitalize()}:")
        print(f"  Without compile: {no_compile:.3f}s")
        print(f"  With compile:    {with_compile:.3f}s")
        print(f"  Speedup:         {speedup:.2f}x ({improvement:+.1f}% faster)")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(benchmark_with_compile())
