#!/usr/bin/env python3
"""
Test concurrent/burst requests to validate continuous batching.

This script tests if the TTS system can maintain <1s Time To First Audio (TTFA)
under load with multiple concurrent requests.

Tests burst sizes: 1, 4, 8, 16, 32 concurrent requests.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-concurrent-burst.py
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List
import statistics

import torch

from chatterbox_vllm.tts import ChatterboxTTS, StreamingMetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class RequestResult:
    """Result for a single request."""
    request_id: int
    start_time: float
    end_time: float
    first_chunk_time: float
    total_time: float
    t3_time: float
    s3gen_time: float
    audio_duration: float
    success: bool
    error: str = None


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
    def first_chunk_latencies(self) -> List[float]:
        return [r.first_chunk_time for r in self.results if r.success]

    @property
    def avg_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.mean(vals) if vals else 0.0

    @property
    def min_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return min(vals) if vals else 0.0

    @property
    def max_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return max(vals) if vals else 0.0

    @property
    def median_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.median(vals) if vals else 0.0

    @property
    def stdev_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    @property
    def p95_first_chunk(self) -> float:
        """95th percentile."""
        vals = sorted(self.first_chunk_latencies)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    @property
    def p99_first_chunk(self) -> float:
        """99th percentile."""
        vals = sorted(self.first_chunk_latencies)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.99)
        return vals[min(idx, len(vals) - 1)]

    @property
    def under_1s_count(self) -> int:
        return sum(1 for t in self.first_chunk_latencies if t < 1.0)

    @property
    def under_1s_pct(self) -> float:
        total = len(self.first_chunk_latencies)
        if total == 0:
            return 0.0
        return (self.under_1s_count / total) * 100


def process_single_request(
    request_id: int,
    text: str,
    model: ChatterboxTTS,
    max_tokens: int = 200,
) -> RequestResult:
    """
    Process a single TTS request.

    Returns RequestResult with timing metrics.
    """
    start_time = time.time()
    first_chunk_time = None
    t3_time = None
    s3gen_time = None
    audio_duration = 0.0
    success = False
    error = None

    try:
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            if first_chunk_time is None:
                first_chunk_time = metrics.latency_to_first_chunk
                t3_time = metrics.t3_token_generation_time
                s3gen_time = metrics.s3gen_first_chunk_time

            # Stop after first chunk for TTFA measurement
            if metrics.chunk_count == 1:
                audio_duration = audio_chunk.shape[-1] / 24000  # S3GEN_SR
                break

        end_time = time.time()
        total_time = end_time - start_time
        success = True

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        error = str(e)
        success = False

    return RequestResult(
        request_id=request_id,
        start_time=start_time,
        end_time=end_time,
        first_chunk_time=first_chunk_time or 0.0,
        total_time=total_time,
        t3_time=t3_time or 0.0,
        s3gen_time=s3gen_time or 0.0,
        audio_duration=audio_duration,
        success=success,
        error=error,
    )


def test_burst_concurrent(
    burst_size: int,
    texts: List[str],
    model: ChatterboxTTS,
    max_tokens: int = 200,
) -> BurstTestResults:
    """
    Test burst of concurrent requests.

    Uses ThreadPoolExecutor to simulate concurrent requests.
    """
    print(f"\n{'='*70}")
    print(f"BURST TEST: {burst_size} CONCURRENT REQUESTS")
    print(f"{'='*70}")

    results = BurstTestResults(burst_size=burst_size)
    burst_start = time.time()

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=burst_size) as executor:
        # Submit all requests
        futures = []
        for i in range(burst_size):
            text = texts[i % len(texts)]
            future = executor.submit(
                process_single_request,
                request_id=i,
                text=text,
                model=model,
                max_tokens=max_tokens,
            )
            futures.append(future)
            print(f"  Submitted request {i+1}/{burst_size}")

        # Collect results as they complete
        print("\n  Processing requests...")
        for i, future in enumerate(futures):
            result = future.result()
            results.results.append(result)

            elapsed = result.end_time - burst_start
            print(f"  Request {result.request_id+1:2d} complete: "
                  f"TTFA={result.first_chunk_time*1000:7.2f}ms, "
                  f"Total={result.total_time*1000:7.2f}ms "
                  f"(at {elapsed*1000:7.2f}ms from burst start)")

    burst_duration = time.time() - burst_start

    return results


def test_burst_sequential(
    burst_size: int,
    texts: List[str],
    model: ChatterboxTTS,
    max_tokens: int = 200,
) -> BurstTestResults:
    """
    Test sequential processing (for comparison).

    Processes requests one at a time.
    """
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL TEST: {burst_size} REQUESTS (one at a time)")
    print(f"{'='*70}")

    results = BurstTestResults(burst_size=burst_size)
    burst_start = time.time()

    for i in range(burst_size):
        text = texts[i % len(texts)]

        result = process_single_request(
            request_id=i,
            text=text,
            model=model,
            max_tokens=max_tokens,
        )
        results.results.append(result)

        elapsed = result.end_time - burst_start
        print(f"  Request {i+1:2d} complete: "
              f"TTFA={result.first_chunk_time*1000:7.2f}ms, "
              f"Total={result.total_time*1000:7.2f}ms "
              f"(at {elapsed*1000:7.2f}ms from burst start)")

    burst_duration = time.time() - burst_start

    return results


def print_burst_results(results: BurstTestResults):
    """Print burst test results."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {results.burst_size} REQUESTS")
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

    print(f"\n⚡ FIRST CHUNK LATENCY (TTFA):")
    print(f"  Average:   {results.avg_first_chunk*1000:7.2f}ms")
    print(f"  Min:       {results.min_first_chunk*1000:7.2f}ms")
    print(f"  Max:       {results.max_first_chunk*1000:7.2f}ms")
    print(f"  Median:    {results.median_first_chunk*1000:7.2f}ms")
    print(f"  Std Dev:   {results.stdev_first_chunk*1000:7.2f}ms")
    print(f"  95th pctl: {results.p95_first_chunk*1000:7.2f}ms")
    print(f"  99th pctl: {results.p99_first_chunk*1000:7.2f}ms")

    print(f"\n🎯 <1s TARGET:")
    under_1s = results.under_1s_count
    under_1s_pct = results.under_1s_pct
    print(f"  Under 1s:  {under_1s}/{results.successful_count} ({under_1s_pct:.1f}%)")

    if under_1s_pct == 100:
        print(f"  ✅ ALL REQUESTS UNDER 1s!")
    elif under_1s_pct >= 95:
        print(f"  ✓ 95%+ under 1s")
    elif under_1s_pct >= 80:
        print(f"  ⚠️  {under_1s_pct:.0f}% under 1s")
    else:
        print(f"  ❌ Only {under_1s_pct:.0f}% under 1s")

    # Calculate throughput
    if results.results:
        total_time = max(r.end_time for r in results.results) - min(r.start_time for r in.results.results)
        throughput = results.successful_count / total_time if total_time > 0 else 0
        print(f"\n📊 THROUGHPUT:")
        print(f"  Total time:     {total_time:.2f}s")
        print(f"  Throughput:     {throughput:.2f} requests/second")


def main():
    """Run concurrent burst tests."""
    print("="*70)
    print("CONCURRENT BURST TESTING - CONTINUOUS BATCHING")
    print("="*70)

    # Test different burst sizes
    burst_sizes = [1, 4, 8, 16, 32]

    # Test texts
    texts = [
        "Hello world, this is a test of concurrent text to speech.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence to test the system under load.",
        "Testing continuous batching with multiple concurrent requests.",
    ]

    all_results = {}

    print("\nInitializing model...")
    init_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )
    init_time = time.time() - init_start
    print(f"✓ Model initialized in {init_time:.2f}s")

    # Warmup
    print("\nWarming up model...")
    for i in range(2):
        for audio_chunk, metrics in model.generate_stream(
            text=texts[0],
            max_tokens=200,
            chunk_size=25,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break
    print("✓ Warmup complete")

    # Test each burst size
    for burst_size in burst_sizes:
        print(f"\n{'#'*70}")
        print(f"# BURST SIZE: {burst_size} CONCURRENT REQUESTS")
        print(f"{'#'*70}")

        # Test concurrent
        results = test_burst_concurrent(
            burst_size=burst_size,
            texts=texts,
            model=model,
            max_tokens=200,
        )
        print_burst_results(results)
        all_results[f"concurrent_{burst_size}"] = results

        # Small delay between tests
        time.sleep(2)

    # Summary comparison
    print("\n" + "="*70)
    print("BURST SIZE COMPARISON SUMMARY")
    print("="*70)

    print(f"\n{'Burst':<10} {'Avg TTFA':<12} {'Median':<12} {'95th':<12} {'<1s':<10} {'Status':<15}")
    print("-" * 70)

    for burst_size in burst_sizes:
        key = f"concurrent_{burst_size}"
        if key in all_results:
            r = all_results[key]
            avg = f"{r.avg_first_chunk*1000:.1f}ms"
            median = f"{r.median_first_chunk*1000:.1f}ms"
            p95 = f"{r.p95_first_chunk*1000:.1f}ms"
            under_1s = f"{r.under_1s_pct:.0f}%"

            if r.under_1s_pct == 100:
                status = "✅ EXCELLENT"
            elif r.under_1s_pct >= 95:
                status = "✓ GOOD"
            elif r.under_1s_pct >= 80:
                status = "⚠️  FAIR"
            else:
                status = "❌ POOR"

            print(f"{burst_size:<10} {avg:<12} {median:<12} {p95:<12} {under_1s:<10} {status:<15}")

    # Cleanup
    model.shutdown()

    print("\n" + "="*70)
    print("CONCURRENT BURST TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
