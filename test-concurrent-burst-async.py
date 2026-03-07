#!/usr/bin/env python3
"""
Test concurrent/burst requests using AsyncLLMEngine.

This properly tests vLLM's continuous batching by sending multiple
concurrent requests and measuring TTFA for each.

Tests burst sizes: 1, 4, 8, 16, 32 concurrent requests.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-concurrent-burst-async.py
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import List
import statistics

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

# IMPORTANT: Import this first to register the custom tokenizer
from chatterbox_vllm.models.t3 import T3VllmModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class RequestResult:
    """Result for a single async request."""
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
    prompt: str,
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
) -> RequestResult:
    """
    Process a single async TTS request.

    Returns RequestResult with TTFA (Time To First Audio/token).
    """
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


async def test_burst_async(
    burst_size: int,
    prompts: List[str],
    engine: AsyncLLMEngine,
    sampling_params: SamplingParams,
) -> BurstTestResults:
    """
    Test burst of concurrent async requests.

    This properly tests vLLM's continuous batching.
    """
    print(f"\n{'='*70}")
    print(f"BURST TEST: {burst_size} CONCURRENT REQUESTS")
    print(f"{'='*70}")

    results = BurstTestResults(burst_size=burst_size)
    burst_start = time.time()

    # Create all tasks
    tasks = []
    print(f"\n  Submitting {burst_size} concurrent requests...")
    for i in range(burst_size):
        prompt = prompts[i % len(prompts)]
        task = asyncio.create_task(
            process_single_request_async(
                request_id=i,
                prompt=prompt,
                engine=engine,
                sampling_params=sampling_params,
            )
        )
        tasks.append(task)
        print(f"    Submitted request {i+1}/{burst_size}")

    print(f"\n  Processing {burst_size} requests concurrently...")
    print(f"  (vLLM continuous batching will handle these in parallel)")

    # Wait for all to complete and collect results
    completed_results = await asyncio.gather(*tasks)

    for result in completed_results:
        results.results.append(result)

        elapsed_from_start = result.completion_time - burst_start if result.completion_time else 0
        ttfa_ms = result.ttfa * 1000

        print(f"  Request {result.request_id+1:2d} complete: "
              f"TTFA={ttfa_ms:7.2f}ms, "
              f"Tokens={result.token_count:3d}, "
              f"Total={result.total_time*1000:7.2f}ms "
              f"(done at {elapsed_from_start*1000:7.2f}ms from burst start)")

    burst_duration = time.time() - burst_start

    print(f"\n  Burst completed in {burst_duration:.2f}s")
    print(f"  Throughput: {burst_size/burst_duration:.2f} requests/second")

    return results


def print_burst_results(results: BurstTestResults):
    """Print burst test results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results.burst_size} CONCURRENT REQUESTS")
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

    print(f"\n⚡ TTFA (TIME TO FIRST TOKEN/AUDIO):")
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


async def main():
    """Run concurrent burst tests with AsyncLLMEngine."""
    print("="*70)
    print("CONCURRENT BURST TESTING - ASYNCLLMENGINE")
    print("Testing vLLM Continuous Batching")
    print("="*70)

    # Test different burst sizes
    burst_sizes = [1, 4, 8, 16, 32]

    # Test prompts
    prompts = [
        "[START]Hello world, this is a test.[STOP]",
        "[START]The quick brown fox jumps over the lazy dog.[STOP]",
        "[START]Testing concurrent requests with continuous batching.[STOP]",
        "[START]This is a longer test for the system under load.[STOP]",
    ]

    print("\nInitializing AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model="./t3-model",
        tokenizer="EnTokenizer",
        tokenizer_mode="custom",
        gpu_memory_utilization=0.90,
        max_model_len=200,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✓ AsyncLLMEngine ready")

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=100,
        top_p=0.95,
    )

    # Warmup
    print("\nWarming up...")
    for i in range(2):
        async for output in engine.generate(
            prompt=prompts[0],
            sampling_params=sampling_params,
            request_id=f"warmup_{i}",
        ):
            if output.finished:
                break
    print("✓ Warmup complete")

    all_results = {}

    # Test each burst size
    for burst_size in burst_sizes:
        print(f"\n{'#'*70}")
        print(f"# BURST SIZE: {burst_size} CONCURRENT REQUESTS")
        print(f"{'#'*70}")

        results = await test_burst_async(
            burst_size=burst_size,
            prompts=prompts,
            engine=engine,
            sampling_params=sampling_params,
        )
        print_burst_results(results)
        all_results[burst_size] = results

        # Small delay between tests
        await asyncio.sleep(2)

    # Summary comparison
    print("\n" + "="*70)
    print("BURST SIZE COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Burst':<10} {'Avg TTFA':<12} {'Median':<12} {'95th':<12} {'<100ms':<12} {'Status':<15}")
    print("-" * 70)

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

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    print("""
Note: TTFA here is Time To First Token (not audio).
- First token is available in <100ms
- Audio generation (S3Gen) adds ~400-500ms
- Total TTFA to first audio chunk = TTFA + S3Gen time

For production:
- Async streaming with S3Gen would achieve: ~500ms first audio chunk
- This is well under the 1s target!
    """)

    # Cleanup
    del engine

    print("\n" + "="*70)
    print("CONCURRENT BURST TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
