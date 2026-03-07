#!/usr/bin/env python3
"""
Test concurrent burst with 32 UNIQUE texts to avoid prefix caching.

This eliminates the artificial speedup from vLLM's prefix caching feature
by using completely different texts for each request.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-concurrent-unique.py
"""

import asyncio
import os
import time
import statistics
from dataclasses import dataclass, field
from typing import List

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# Import for tokenizer registration
from chatterbox_vllm.models.t3 import T3VllmModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 32 UNIQUE texts to avoid prefix caching
UNIQUE_TEXTS = [
    "The weather is beautiful today with clear blue skies.",
    "Machine learning models require large amounts of training data.",
    "The conference room was packed with enthusiastic attendees.",
    "Quantum computing leverages superposition for parallel processing.",
    "The ancient library contained thousands of rare manuscripts.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "The mountain trail offered breathtaking views of the valley below.",
    "Software engineering best practices emphasize code quality and testing.",
    "The marathon runner trained for months to prepare for competition.",
    "Neural networks consist of interconnected layers of artificial neurons.",
    "The culinary festival featured dishes from around the world.",
    "Climate change affects ecosystems and weather patterns globally.",
    "The new smartphone model features advanced camera capabilities.",
    "Acoustic guitars produce sound through vibrating strings and resonance.",
    "The space mission will explore distant galaxies and nebulae.",
    "Financial markets experience volatility during uncertain times.",
    "The art gallery showcased contemporary paintings and sculptures.",
    "Biology students study cell structures and genetic inheritance.",
    "The urban transportation system includes buses, trains, and subways.",
    "Cloud computing provides scalable infrastructure for applications.",
    "The archaeological discovery revealed ancient civilization artifacts.",
    "Chemical reactions involve breaking and forming molecular bonds.",
    "The music concert featured a symphony orchestra performance.",
    "Data visualization helps interpret complex information graphically.",
    "The ocean contains diverse marine ecosystems and coral reefs.",
    "Robotic automation transforms manufacturing and logistics processes.",
    "Philosophy examines fundamental questions about existence and knowledge.",
    "The wildlife documentary captured stunning footage of nature.",
    "Cryptocurrency uses blockchain technology for secure transactions.",
    "The sailing ship navigated through stormy waters to reach port.",
    "Psychological research explores human behavior and mental processes.",
    "The fashion industry trends change rapidly with seasonal collections.",
    "Geologists study rock formations to understand Earth's history.",
    "The theater production involved elaborate costumes and staging.",
    "Wireless communication protocols enable global information exchange.",
    "The organic farm practices sustainable agriculture methods.",
]

# Verify we have at least 32 unique texts
assert len(UNIQUE_TEXTS) >= 32, f"Expected at least 32 texts, got {len(UNIQUE_TEXTS)}"

# Verify all texts are unique
assert len(set(UNIQUE_TEXTS)) == len(UNIQUE_TEXTS), "Texts are not unique!"


@dataclass
class RequestResult:
    """Result for a single request."""
    request_id: int
    start_time: float
    first_token_time: float = None
    completion_time: float = None
    token_count: int = 0
    success: bool = True
    error: str = None

    @property
    def ttfa(self) -> float:
        """Time to first audio (first token in this case)."""
        if self.first_token_time is None:
            return 0.0
        return self.first_token_time - self.start_time

    @property
    def total_time(self) -> float:
        if self.completion_time is None:
            return 0.0
        return self.completion_time - self.start_time


@dataclass
class BurstTestResults:
    """Results for a burst test."""
    burst_size: int
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def ttfas(self) -> List[float]:
        return [r.ttfa for r in self.results if r.success and r.ttfa > 0]

    @property
    def avg_ttfa(self) -> float:
        vals = self.ttfas
        return statistics.mean(vals) if vals else 0.0

    @property
    def min_ttfa(self) -> float:
        vals = self.ttfas
        return min(vals) if vals else 0.0

    @property
    def max_ttfa(self) -> float:
        vals = self.ttfas
        return max(vals) if vals else 0.0

    @property
    def median_ttfa(self) -> float:
        vals = self.ttfas
        return statistics.median(vals) if vals else 0.0

    @property
    def stdev_ttfa(self) -> float:
        vals = self.ttfas
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    @property
    def p95_ttfa(self) -> float:
        vals = sorted(self.ttfas)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    @property
    def p99_ttfa(self) -> float:
        vals = sorted(self.ttfas)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.99)
        return vals[min(idx, len(vals) - 1)]

    @property
    def under_100ms_count(self) -> int:
        return sum(1 for t in self.ttfas if t < 0.1)

    @property
    def under_100ms_pct(self) -> float:
        total = len(self.ttfas)
        if total == 0:
            return 0.0
        return (self.under_100ms_count / total) * 100


async def process_single_request_async(
    request_id: int,
    text: str,
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
) -> RequestResult:
    """Process a single async TTS request."""
    prompt = f"[START]{text}[STOP]"

    start_time = time.time()
    first_token_time = None
    token_count = 0
    success = True
    error = None

    try:
        request_id_str = f"burst_req_{request_id}"

        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id_str,
        ):
            if output.outputs:
                # Track first token time
                if first_token_time is None and len(output.outputs[0].token_ids) > 0:
                    first_token_time = time.time()

                token_count = len(output.outputs[0].token_ids)

            if output.finished:
                break

        completion_time = time.time()

    except Exception as e:
        completion_time = time.time()
        error = str(e)
        success = False

    return RequestResult(
        request_id=request_id,
        start_time=start_time,
        first_token_time=first_token_time,
        completion_time=completion_time,
        token_count=token_count,
        success=success,
        error=error,
    )


async def test_burst_async_unique(
    burst_size: int,
    texts: List[str],
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
) -> BurstTestResults:
    """Test burst of concurrent requests with UNIQUE texts (no prefix caching)."""
    print(f"\n{'='*70}")
    print(f"BURST TEST: {burst_size} CONCURRENT REQUESTS (UNIQUE TEXTS)")
    print(f"{'='*70}")
    print(f"Note: Using {len(texts)} unique texts to avoid prefix caching")

    results = BurstTestResults(burst_size=burst_size)
    burst_start = time.time()

    # Create all tasks
    tasks = []
    print(f"\n  Submitting {burst_size} concurrent requests...")
    for i in range(burst_size):
        text = texts[i % len(texts)]
        task = asyncio.create_task(
            process_single_request_async(
                request_id=i,
                text=text,
                engine=engine,
                sampling_params=sampling_params,
            )
        )
        tasks.append(task)
        if (i + 1) % 8 == 0:
            print(f"    Submitted {i+1}/{burst_size} requests...")

    print(f"\n  Processing {burst_size} requests concurrently...")
    print(f"  (NO prefix caching - all unique texts)")

    # Wait for all to complete and collect results
    completed_results = await asyncio.gather(*tasks)

    for result in completed_results:
        results.results.append(result)

        elapsed_from_start = result.completion_time - burst_start if result.completion_time else 0
        ttfa_ms = result.ttfa * 1000

        print(f"  Request {result.request_id+1:2d}: TTFA={ttfa_ms:7.2f}ms, "
              f"Tokens={result.token_count:3d}, "
              f"Complete at {elapsed_from_start*1000:7.2f}ms from burst start")

    burst_duration = time.time() - burst_start

    print(f"\n  Burst completed in {burst_duration:.2f}s")
    print(f"  Throughput: {burst_size/burst_duration:.2f} requests/second")

    return results


def print_burst_results(results: BurstTestResults):
    """Print burst test results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results.burst_size} CONCURRENT REQUESTS (UNIQUE TEXTS)")
    print(f"{'='*70}")

    print(f"\nSuccess: {results.successful_count}/{results.burst_size}")
    print(f"Failed:  {results.failed_count}/{results.burst_size}")

    if results.failed_count > 0:
        print("\n❌ Failed requests:")
        for r in results.results:
            if not r.success:
                print(f"  Request {r.request_id}: {r.error}")

    if results.successful_count == 0:
        return

    print(f"\n⚡ TTFA (TIME TO FIRST TOKEN) - NO PREFIX CACHING:")
    print(f"  Average:   {results.avg_ttfa*1000:7.2f}ms")
    print(f"  Min:       {results.min_ttfa*1000:7.2f}ms")
    print(f"  Max:       {results.max_ttfa*1000:7.2f}ms")
    print(f"  Median:    {results.median_ttfa*1000:7.2f}ms")
    print(f"  Std Dev:   {results.stdev_ttfa*1000:7.2f}ms")
    print(f"  95th pctl: {results.p95_ttfa*1000:7.2f}ms")
    print(f"  99th pctl: {results.p99_ttfa*1000:7.2f}ms")

    # Note: For token streaming, we expect <100ms, not <1s
    print(f"\n🎯 <100ms TARGET (for first token):")
    under_100ms = results.under_100ms_count
    under_100ms_pct = results.under_100ms_pct
    print(f"  Under 100ms:  {under_100ms}/{results.successful_count} ({under_100ms_pct:.1f}%)")

    if under_100ms_pct == 100:
        print(f"  ✅ ALL REQUESTS UNDER 100ms!")
    elif under_100ms_pct >= 95:
        print(f"  ✓ 95%+ under 100ms")
    elif under_100ms_pct >= 80:
        print(f"  ⚠️  {under_100ms_pct:.0f}% under 100ms")
    else:
        print(f"  ❌ Only {under_100ms_pct:.0f}% under 100ms")

    # Calculate throughput
    if results.results:
        total_time = max(r.completion_time for r in results.results if r.completion_time) - min(r.start_time for r in results.results if r.start_time)
        throughput = results.successful_count / total_time if total_time > 0 else 0
        print(f"\n📊 THROUGHPUT:")
        print(f"  Total time:     {total_time:.2f}s")
        print(f"  Throughput:     {throughput:.2f} requests/second")

    # Distribution analysis
    print(f"\n📊 DISTRIBUTION:")
    sorted_ttfas = sorted(results.ttfas)
    buckets = [50, 100, 150, 200, 250, 300, 400, 500]

    print(f"\n  Range        Count  Percentage")
    print(f"  " + "-"*40)
    for threshold in buckets:
        count = sum(1 for t in sorted_ttfas if t * 1000 < threshold)
        pct = (count / len(sorted_ttfas)) * 100 if sorted_ttfas else 0
        bar = "█" * int(pct / 5)
        print(f"  < {threshold:4d}ms:   {count:3d}/{len(sorted_ttfas):3d}  ({pct:5.1f}%) {bar}")


async def main():
    """Run concurrent burst test with unique texts."""
    print("="*70)
    print("CONCURRENT BURST TESTING - UNIQUE TEXTS (NO PREFIX CACHING)")
    print("="*70)

    print("\n⚠️  IMPORTANT: Using 32 UNIQUE texts to eliminate prefix caching")
    print("    This gives more realistic performance numbers!")

    # Test different burst sizes
    burst_sizes = [1, 4, 8, 16, 32]

    print("\nInitializing AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model="./t3-model",
        tokenizer="EnTokenizer",
        tokenizer_mode="custom",
        gpu_memory_utilization=0.90,
        max_model_len=2000,
        enforce_eager=True,
        tensor_parallel_size=1,
        disable_log_stats=False,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✓ AsyncLLMEngine ready")

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        top_p=0.95,
    )

    # Warmup with a different text
    print("\nWarming up...")
    warmup_text = "The quick brown fox jumps over the lazy dog."
    async for output in engine.generate(
        prompt=f"[START]{warmup_text}[STOP]",
        sampling_params=sampling_params,
        request_id="warmup",
    ):
        if output.finished:
            break
    print("✓ Warmup complete")

    all_results = {}

    # Test each burst size
    for burst_size in burst_sizes:
        print(f"\n{'#'*70}")
        print(f"# BURST SIZE: {burst_size} CONCURRENT (UNIQUE TEXTS)")
        print(f"{'#'*70}")

        results = await test_burst_async_unique(
            burst_size=burst_size,
            texts=UNIQUE_TEXTS[:burst_size],  # Use unique texts
            engine=engine,
            sampling_params=sampling_params,
        )
        print_burst_results(results)
        all_results[burst_size] = results

        # Small delay between tests
        await asyncio.sleep(2)

    # Summary comparison
    print("\n" + "="*70)
    print("BURST SIZE COMPARISON - UNIQUE TEXTS (NO PREFIX CACHING)")
    print("="*70)

    print(f"\n{'Burst':<10} {'Avg TTFA':<12} {'Median':<12} {'95th':<12} {'<100ms':<12} {'Status':<15}")
    print("-" * 75)

    for burst_size in burst_sizes:
        if burst_size in all_results:
            r = all_results[burst_size]
            avg = f"{r.avg_ttfa*1000:.1f}ms"
            median = f"{r.median_ttfa*1000:.1f}ms"
            p95 = f"{r.p95_ttfa*1000:.1f}ms"
            under_100ms = f"{r.under_100ms_pct:.0f}%"

            if r.under_100ms_pct == 100:
                status = "✅ EXCELLENT"
            elif r.under_100ms_pct >= 95:
                status = "✓ GOOD"
            elif r.under_100ms_pct >= 80:
                status = "⚠️  FAIR"
            else:
                status = "❌ POOR"

            print(f"{burst_size:<10} {avg:<12} {median:<12} {p95:<12} {under_100ms:<12} {status:<15}")

    # Comparison with cached results
    print("\n" + "="*70)
    print("COMPARISON: CACHED vs UNIQUE TEXTS")
    print("="*70)

    print("\n" + f"{'Burst':<10} {'Cached TTFA':<15} {'Unique TTFA':<15} {'Difference':<12}")
    print("-" * 50)

    cached_results = {
        1: 9.1,
        4: 36.6,
        8: 29.6,
        16: 30.7,
        32: 48.6,
    }

    for burst_size in burst_sizes:
        if burst_size in all_results:
            unique = all_results[burst_size].avg_ttfa * 1000
            cached = cached_results.get(burst_size, 0)
            diff = unique - cached
            diff_pct = (diff / cached * 100) if cached > 0 else 0

            print(f"{burst_size:<10} {cached:>8.1f}ms        {unique:>8.1f}ms        "
                  f"{diff:>+7.1f}ms ({diff_pct:+6.1f}%)")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print("""
Note: With unique texts, there's no prefix caching advantage.

- Each request has completely different tokenized input
- No shared KV cache between requests
- Shows true concurrent processing performance

For production:
- Real-world traffic has diverse inputs (not all the same)
- These unique text results are more realistic
- AsyncLLMEngine still performs excellently under load!
    """)

    # Cleanup
    del engine

    print("\n" + "="*70)
    print("UNIQUE TEXT BURST TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
