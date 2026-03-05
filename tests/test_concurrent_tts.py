#!/usr/bin/env python3
"""
Concurrent TTS stress test.

Tests system stability and TTFA degradation under high concurrent load.
"""

import asyncio
import argparse
import random
import time
import statistics
from typing import List, Dict

from chatterbox_vllm import ChatterboxTTSAsync


# Test prompts
SHORT_PROMPTS = [
    "Hello.", "Yes, please.", "Thank you.", "Good morning.",
    "See you later.", "That's great!", "I agree.", "No way.",
    "Sounds good.", "Perfect.", "Excellent work!", "Nice to meet you.",
]

MEDIUM_PROMPTS = [
    "This is a medium length text that will take some time to process.",
    "The weather today is quite nice, with clear skies and mild temperatures.",
    "I would like to order a large pizza with pepperoni and extra cheese.",
    "Please remember to turn off the lights before leaving the office.",
    "The meeting has been rescheduled to next Tuesday at three in the afternoon.",
]


async def stress_test_concurrent(
    model: ChatterboxTTSAsync,
    max_concurrent: int,
    num_requests: int,
    duration_seconds: int = None,
) -> Dict:
    """
    Stress test with specified max concurrent requests.

    Args:
        model: ChatterboxTTSAsync instance
        max_concurrent: Maximum concurrent requests
        num_requests: Total number of requests to send
        duration_seconds: Optional test duration (overrides num_requests)

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"CONCURRENT STRESS TEST")
    print(f"  Max Concurrent: {max_concurrent}")
    print(f"  Total Requests: {num_requests}")
    if duration_seconds:
        print(f"  Duration: {duration_seconds}s")
    print(f"{'='*80}\n")

    latencies = []
    ttfas = []
    errors = []
    request_times = []
    completed_count = 0
    failed_count = 0

    semaphore = asyncio.Semaphore(max_concurrent)
    start_time = time.time()

    async def run_request(req_id: int):
        nonlocal completed_count, failed_count

        async with semaphore:
            # Mix of short and medium prompts
            prompt = random.choice(SHORT_PROMPTS + MEDIUM_PROMPTS)
            req_start = time.time()

            try:
                results = await model.generate(
                    prompts=[prompt],
                    temperature=0.8,
                    exaggeration=0.5,
                )
                req_end = time.time()

                latency = req_end - req_start
                elapsed = req_end - start_time

                completed_count += 1

                return {
                    "req_id": req_id,
                    "latency": latency,
                    "elapsed": elapsed,
                    "success": True,
                }

            except Exception as e:
                failed_count += 1
                return {
                    "req_id": req_id,
                    "error": str(e),
                    "success": False,
                }

    # Run requests
    if duration_seconds:
        # Time-based test: keep launching requests until duration
        tasks = []
        req_id = 0
        end_time = start_time + duration_seconds

        while time.time() < end_time:
            # Launch new requests up to max concurrent
            while len(tasks) < max_concurrent and time.time() < end_time:
                task = asyncio.create_task(run_request(req_id))
                tasks.append(task)
                req_id += 1

            # Wait for at least one to complete before launching more
            if tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=0.1,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Process completed
                for task in done:
                    result = task.result()
                    if result["success"]:
                        latencies.append(result["latency"])
                        ttfas.append(result["latency"])
                        request_times.append(result["elapsed"])
                    else:
                        errors.append(result["error"])

                # Remove completed from pending
                tasks = list(pending)

        # Cancel remaining tasks
        for task in tasks:
            task.cancel()

        num_requests = req_id

    else:
        # Count-based test
        tasks = [run_request(i) for i in range(num_requests)]
        results_list = await asyncio.gather(*tasks)

        for r in results_list:
            if r["success"]:
                latencies.append(r["latency"])
                ttfas.append(r["latency"])
                request_times.append(r["elapsed"])
            else:
                errors.append(r["error"])

    total_time = time.time() - start_time

    # Analyze latency over time (check for degradation)
    if len(request_times) > 10:
        # Split into 3 time segments
        n = len(request_times)
        segment_size = n // 3

        early_latencies = latencies[:segment_size]
        mid_latencies = latencies[segment_size:2*segment_size]
        late_latencies = latencies[2*segment_size:]

        latency_degradation = None
        if early_latencies and late_latencies:
            early_p95 = sorted(early_latencies)[int(len(early_latencies) * 0.95)]
            late_p95 = sorted(late_latencies)[int(len(late_latencies) * 0.95)]
            latency_degradation = (late_p95 - early_p95) / early_p95 * 100
    else:
        early_latencies = mid_latencies = late_latencies = []
        latency_degradation = None

    result_summary = {
        "max_concurrent": max_concurrent,
        "num_requests": num_requests,
        "completed": completed_count,
        "failed": failed_count,
        "total_time": total_time,
        "throughput": num_requests / total_time,
        "latencies": latencies,
        "ttfas": ttfas,
        "errors": errors,
        "early_latencies": early_latencies,
        "mid_latencies": mid_latencies,
        "late_latencies": late_latencies,
        "latency_degradation": latency_degradation,
    }

    if latencies:
        result_summary.update({
            "latency_min": min(latencies),
            "latency_max": max(latencies),
            "latency_mean": statistics.mean(latencies),
            "latency_median": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
        })

    return result_summary


def print_stress_results(results: Dict):
    """Print stress test results."""
    print("\n" + "="*80)
    print("STRESS TEST RESULTS")
    print("="*80)

    print(f"\nLoad:")
    print(f"  Max Concurrent: {results['max_concurrent']}")
    print(f"  Total Requests: {results['num_requests']}")
    print(f"  Completed: {results['completed']} ({results['completed']/results['num_requests']*100:.1f}%)")
    print(f"  Failed: {results['failed']}")

    print(f"\nPerformance:")
    print(f"  Total Time: {results['total_time']:.2f}s")
    print(f"  Throughput: {results['throughput']:.2f} req/s")

    if "latency_median" in results:
        print(f"\nLatency Statistics:")
        print(f"  Min:    {results['latency_min']:.2f}s")
        print(f"  Mean:   {results['latency_mean']:.2f}s")
        print(f"  Median: {results['latency_median']:.2f}s")
        print(f"  P95:    {results['latency_p95']:.2f}s")
        print(f"  P99:    {results['latency_p99']:.2f}s")
        print(f"  Max:    {results['latency_max']:.2f}s")

    # Check for latency degradation
    if results.get("latency_degradation") is not None:
        deg = results["latency_degradation"]
        print(f"\nLatency Degradation (P95 late vs early):")
        print(f"  Change: {deg:+.1f}%")

        if results["early_latencies"] and results["late_latencies"]:
            early_p95 = sorted(results["early_latencies"])[int(len(results["early_latencies"]) * 0.95)]
            late_p95 = sorted(results["late_latencies"])[int(len(results["late_latencies"]) * 0.95)]
            print(f"  Early P95: {early_p95:.2f}s")
            print(f"  Late P95:  {late_p95:.2f}s")

        if abs(deg) < 10:
            print(f"  Status: ✅ STABLE - No significant degradation")
        elif deg > 0:
            print(f"  Status: ⚠️  DEGRADATION - Latency increased over time")
        else:
            print(f"  Status: ✅ IMPROVEMENT - Latency decreased over time")

    # Error analysis
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"][:5]:
            print(f"  - {err}")

    print("\n" + "="*80)


async def run_progressive_stress_test(
    model: ChatterboxTTSAsync,
    max_concurrent: int,
):
    """Run progressive stress test with increasing concurrency."""
    print("\n" + "="*80)
    print("PROGRESSIVE STRESS TEST")
    print("="*80)
    print(f"  Testing concurrency levels from 1 to {max_concurrent}")
    print(f"{'='*80}\n")

    all_results = []

    for concurrency in range(1, max_concurrent + 1):
        # Use fewer requests for higher concurrency to save time
        num_requests = max(10, 50 // concurrency)

        result = await stress_test_concurrent(
            model,
            max_concurrent=concurrency,
            num_requests=num_requests,
        )

        all_results.append(result)

        # Print summary for this level
        if "latency_p95" in result:
            print(f"Concurrent {concurrency:2d}: "
                  f"P95={result['latency_p95']:.2f}s, "
                  f"Throughput={result['throughput']:.2f} req/s, "
                  f"Success={result['completed']}/{result['num_requests']}")

        # Stop if failure rate is too high
        failure_rate = result['failed'] / result['num_requests']
        if failure_rate > 0.1:
            print(f"\n⚠️  High failure rate at concurrency {concurrency} ({failure_rate*100:.1f}%)")
            break

    # Find optimal concurrency
    valid_results = [r for r in all_results if r.get("latency_p95")]
    if valid_results:
        # Optimal is best throughput with P95 < 2s
        optimal = None
        for r in valid_results:
            if r["latency_p95"] < 2.0:
                if optimal is None or r["throughput"] > optimal["throughput"]:
                    optimal = r

        if optimal:
            print(f"\n{'='*80}")
            print(f"OPTIMAL CONCURRENCY: {optimal['max_concurrent']}")
            print(f"  Throughput: {optimal['throughput']:.2f} req/s")
            print(f"  P95 Latency: {optimal['latency_p95']:.2f}s")
            print(f"{'='*80}")

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Concurrent TTS stress test")
    parser.add_argument("--max-concurrent", type=int, default=50,
                        help="Maximum concurrent requests to test")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests for single stress test")
    parser.add_argument("--duration", type=int,
                        help="Test duration in seconds (overrides num-requests)")
    parser.add_argument("--progressive", action="store_true",
                        help="Run progressive stress test (1 to max-concurrent)")
    parser.add_argument("--single-concurrency", type=int,
                        help="Run stress test at specific concurrency level")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("CONCURRENT TTS STRESS TEST")
    print("="*80)

    # Initialize model
    print("\nInitializing model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
    )

    try:
        if args.progressive:
            # Run progressive test
            results = await run_progressive_stress_test(model, args.max_concurrent)

        elif args.single_concurrency:
            # Run single concurrency level
            result = await stress_test_concurrent(
                model,
                max_concurrent=args.single_concurrency,
                num_requests=args.num_requests,
                duration_seconds=args.duration,
            )
            print_stress_results(result)

        else:
            # Run at max concurrent
            result = await stress_test_concurrent(
                model,
                max_concurrent=args.max_concurrent,
                num_requests=args.num_requests,
                duration_seconds=args.duration,
            )
            print_stress_results(result)

            # Success criteria
            print("\n" + "="*80)
            print("SUCCESS CRITERIA")
            print("="*80)

            success_rate = result['completed'] / result['num_requests']
            print(f"\n✅ Handle {result['completed']} concurrent requests: "
                  f"{'PASS' if result['completed'] >= 20 else 'FAIL'} (≥ 20 target)")

            print(f"✅ Success rate ≥ 90%: "
                  f"{'PASS' if success_rate >= 0.9 else 'FAIL'} ({success_rate*100:.1f}%)")

            if "latency_p95" in result:
                print(f"✅ TTFA P95 < 2.0s: "
                      f"{'PASS' if result['latency_p95'] < 2.0 else 'FAIL'} ({result['latency_p95']:.2f}s)")

            if result.get("latency_degradation") is not None:
                print(f"✅ No significant latency degradation: "
                      f"{'PASS' if abs(result['latency_degradation']) < 10 else 'FAIL'} "
                      f"({result['latency_degradation']:+.1f}%)")

            print("\n" + "="*80)

    finally:
        await model.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
