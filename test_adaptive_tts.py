#!/usr/bin/env python3
"""
Testing suite for adaptive TTS configuration.

Tests single-category, mixed workload, and stress testing scenarios.
"""

import asyncio
import argparse
import random
import time
import statistics
from pathlib import Path
from typing import List, Dict

from chatterbox_vllm import ChatterboxTTSAsync
from chatterbox_vllm.profiling import TTFAProfiler


# Test prompts by category
TEST_PROMPTS = {
    "short": [
        "Hello.",
        "Yes, please.",
        "Thank you.",
        "Good morning.",
        "See you later.",
        "That's great!",
        "I agree.",
        "No way.",
        "Sounds good.",
        "Perfect.",
        "Excellent work!",
        "Nice to meet you.",
        "How are you?",
        "Fine, thanks.",
        "Goodbye!",
    ],
    "medium": [
        "This is a medium length text that will take some time to process.",
        "The weather today is quite nice, with clear skies and mild temperatures.",
        "I would like to order a large pizza with pepperoni and extra cheese.",
        "Please remember to turn off the lights before leaving the office.",
        "The meeting has been rescheduled to next Tuesday at three in the afternoon.",
        "Could you please send me the report by the end of the day tomorrow?",
        "I'm looking forward to seeing you at the conference next week.",
        "The project deadline has been extended to give us more time to complete it.",
    ],
    "long": [
        "This is a significantly longer text passage that will require more processing time through the text to speech synthesis pipeline, including tokenization, language model inference, and audio decoding.",
        "The history of artificial intelligence dates back to ancient times, but the modern field of AI research was founded in 1956 at a conference held at Dartmouth College.",
        "When preparing for a long journey, it's important to pack all the essentials including clothing appropriate for the climate, toiletries, important documents.",
    ],
}


async def test_single_category(
    model: ChatterboxTTSAsync,
    category: str,
    num_requests: int,
) -> Dict:
    """Test a single category of requests."""
    print(f"\n{'='*80}")
    print(f"Testing {category.upper()} requests ({num_requests} total)")
    print(f"{'='*80}")

    prompts = TEST_PROMPTS[category]
    ttfas = []
    latencies = []
    errors = []

    start_time = time.time()

    for i in range(num_requests):
        prompt = random.choice(prompts)
        req_id = f"{category}_req_{i}"

        try:
            req_start = time.time()
            results = await model.generate(
                prompts=[prompt],
                temperature=0.8,
                exaggeration=0.5,
            )
            req_end = time.time()

            latency = req_end - req_start
            latencies.append(latency)
            ttfas.append(latency)  # For non-streaming, TTFA ≈ total latency

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")

        except Exception as e:
            errors.append(str(e))
            print(f"  Error on request {i}: {e}")

    total_time = time.time() - start_time

    results = {
        "category": category,
        "num_requests": num_requests,
        "successful": num_requests - len(errors),
        "errors": errors,
        "total_time": total_time,
        "throughput": num_requests / total_time,
        "latencies": latencies,
        "ttfas": ttfas,
    }

    if latencies:
        results.update({
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
            "latency_mean": statistics.mean(latencies),
        })

    return results


async def test_mixed_workload(
    model: ChatterboxTTSAsync,
    num_requests: int,
    distribution: Dict[str, float] = None,
) -> Dict:
    """Test mixed workload with specified distribution."""
    if distribution is None:
        distribution = {"short": 0.7, "medium": 0.2, "long": 0.1}

    print(f"\n{'='*80}")
    print(f"Testing MIXED workload ({num_requests} total)")
    print(f"  Distribution: {distribution}")
    print(f"{'='*80}")

    categories = list(distribution.keys())
    weights = list(distribution.values())

    ttfas_by_category = {cat: [] for cat in categories}
    latencies = []
    errors = []

    start_time = time.time()

    for i in range(num_requests):
        category = random.choices(categories, weights=weights, k=1)[0]
        prompt = random.choice(TEST_PROMPTS[category])
        req_id = f"mixed_{category}_req_{i}"

        try:
            req_start = time.time()
            results = await model.generate(
                prompts=[prompt],
                temperature=0.8,
                exaggeration=0.5,
            )
            req_end = time.time()

            latency = req_end - req_start
            latencies.append(latency)
            ttfas_by_category[category].append(latency)

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_requests} requests")

        except Exception as e:
            errors.append(str(e))

    total_time = time.time() - start_time

    results = {
        "num_requests": num_requests,
        "successful": num_requests - len(errors),
        "errors": errors,
        "total_time": total_time,
        "throughput": num_requests / total_time,
        "latencies": latencies,
        "ttfas_by_category": ttfas_by_category,
    }

    if latencies:
        results.update({
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
            "latency_mean": statistics.mean(latencies),
        })

    # Per-category statistics
    for cat in categories:
        cat_ttfas = ttfas_by_category[cat]
        if cat_ttfas:
            results[f"{cat}_p50"] = statistics.median(cat_ttfas)
            results[f"{cat}_p95"] = sorted(cat_ttfas)[int(len(cat_ttfas) * 0.95)] if len(cat_ttfas) >= 20 else max(cat_ttfas)
            results[f"{cat}_mean"] = statistics.mean(cat_ttfas)

    return results


async def test_concurrent_load(
    model: ChatterboxTTSAsync,
    max_concurrent: int,
    num_requests: int,
) -> Dict:
    """Test system behavior under concurrent load."""
    print(f"\n{'='*80}")
    print(f"Testing CONCURRENT load (max {max_concurrent} concurrent, {num_requests} total)")
    print(f"{'='*80}")

    categories = ["short", "medium", "long"]
    latencies = []
    ttfas = []
    errors = []
    completion_times = []

    async def run_request(req_id: int):
        category = random.choice(categories)
        prompt = random.choice(TEST_PROMPTS[category])

        req_start = time.time()
        try:
            results = await model.generate(
                prompts=[prompt],
                temperature=0.8,
                exaggeration=0.5,
            )
            req_end = time.time()

            latency = req_end - req_start
            return {
                "req_id": req_id,
                "category": category,
                "latency": latency,
                "success": True,
            }
        except Exception as e:
            return {
                "req_id": req_id,
                "category": category,
                "error": str(e),
                "success": False,
            }

    start_time = time.time()

    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(req_id: int):
        async with semaphore:
            return await run_request(req_id)

    tasks = [bounded_request(i) for i in range(num_requests)]
    results_list = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    for r in results_list:
        if r["success"]:
            latencies.append(r["latency"])
            ttfas.append(r["latency"])
        else:
            errors.append(r["error"])

    successful = len(latencies)
    failed = len(errors)

    result_summary = {
        "max_concurrent": max_concurrent,
        "num_requests": num_requests,
        "successful": successful,
        "failed": failed,
        "total_time": total_time,
        "throughput": num_requests / total_time,
        "latencies": latencies,
        "ttfas": ttfas,
        "errors": errors,
    }

    if latencies:
        result_summary.update({
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
            "latency_mean": statistics.mean(latencies),
        })

    return result_summary


def print_results(results: Dict, test_type: str):
    """Print test results summary."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {test_type.upper()}")
    print(f"{'='*80}")

    if test_type == "single":
        cat = results["category"]
        print(f"\nCategory: {cat.upper()}")
        print(f"  Requests: {results['successful']}/{results['num_requests']} successful")
        print(f"  Total Time: {results['total_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.2f} req/s")

        if "latency_p50" in results:
            print(f"\nLatency Statistics:")
            print(f"  P50: {results['latency_p50']:.2f}s")
            print(f"  P95: {results['latency_p95']:.2f}s")
            print(f"  P99: {results['latency_p99']:.2f}s")
            print(f"  Mean: {results['latency_mean']:.2f}s")

    elif test_type == "mixed":
        print(f"\nMixed Workload:")
        print(f"  Requests: {results['successful']}/{results['num_requests']} successful")
        print(f"  Total Time: {results['total_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.2f} req/s")

        if "latency_p50" in results:
            print(f"\nOverall Latency:")
            print(f"  P50: {results['latency_p50']:.2f}s")
            print(f"  P95: {results['latency_p95']:.2f}s")
            print(f"  P99: {results['latency_p99']:.2f}s")

        print(f"\nPer-Category Latency:")
        for cat in ["short", "medium", "long"]:
            if f"{cat}_p50" in results:
                print(f"  {cat.capitalize():<8}: P50={results[f'{cat}_p50']:.2f}s, "
                      f"P95={results[f'{cat}_p95']:.2f}s")

    elif test_type == "concurrent":
        print(f"\nConcurrent Load:")
        print(f"  Max Concurrent: {results['max_concurrent']}")
        print(f"  Requests: {results['successful']}/{results['num_requests']} successful")
        print(f"  Total Time: {results['total_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.2f} req/s")

        if "latency_p50" in results:
            print(f"\nLatency Statistics:")
            print(f"  P50: {results['latency_p50']:.2f}s")
            print(f"  P95: {results['latency_p95']:.2f}s")
            print(f"  P99: {results['latency_p99']:.2f}s")
            print(f"  Mean: {results['latency_mean']:.2f}s")

    if results.get("errors"):
        print(f"\nErrors: {len(results['errors'])}")
        for err in results["errors"][:3]:
            print(f"  - {err}")

    print(f"{'='*80}\n")


async def main():
    parser = argparse.ArgumentParser(description="Test adaptive TTS configuration")
    parser.add_argument("--profile", type=str, choices=["short", "medium", "long"],
                        help="Test single category profile")
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests for single-category test")
    parser.add_argument("--mixed", action="store_true",
                        help="Run mixed workload test")
    parser.add_argument("--concurrent", type=int,
                        help="Run concurrent load test with max concurrent requests")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests")
    parser.add_argument("--enable-ttfa", action="store_true",
                        help="Enable TTFA tracking")

    args = parser.parse_args()

    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING CHATTERBOX TTS WITH ADAPTIVE CONFIGURATION")
    print("="*80)

    from chatterbox_vllm.adaptive_config import print_profile_summary
    print_profile_summary()

    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        enable_ttfa_tracking=args.enable_ttfa,
    )

    all_results = {}

    try:
        # Single category test
        if args.profile or args.all:
            categories = [args.profile] if args.profile else ["short", "medium", "long"]
            for cat in categories:
                num_req = 20 if cat == "long" else (50 if cat == "medium" else 100)
                result = await test_single_category(model, cat, num_req)
                print_results(result, "single")
                all_results[f"single_{cat}"] = result

        # Mixed workload test
        if args.mixed or args.all:
            result = await test_mixed_workload(model, 200)
            print_results(result, "mixed")
            all_results["mixed"] = result

        # Concurrent load test
        if args.concurrent or args.all:
            max_conc = args.concurrent or 20
            result = await test_concurrent_load(model, max_conc, 100)
            print_results(result, "concurrent")
            all_results["concurrent"] = result

        # Print TTFA summary if enabled
        if args.enable_ttfa:
            model.print_ttfa_summary()
            model.save_ttfa_metrics("adaptive_test_metrics.csv")

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        print("\nSuccess Criteria:")
        if all_results:
            # Check short request TTFA
            if "single_short" in all_results:
                short_p95 = all_results["single_short"].get("latency_p95", 0)
                print(f"  Short request TTFA P95: {short_p95:.2f}s - {'✅ PASS' if short_p95 < 1.0 else '❌ FAIL'} (< 1.0s target)")

            # Check medium request TTFA
            if "single_medium" in all_results:
                med_p95 = all_results["single_medium"].get("latency_p95", 0)
                print(f"  Medium request TTFA P95: {med_p95:.2f}s - {'✅ PASS' if med_p95 < 2.0 else '❌ FAIL'} (< 2.0s target)")

            # Check long request TTFA
            if "single_long" in all_results:
                long_p95 = all_results["single_long"].get("latency_p95", 0)
                print(f"  Long request TTFA P95: {long_p95:.2f}s - {'✅ PASS' if long_p95 < 4.0 else '❌ FAIL'} (< 4.0s target)")

            # Check concurrent handling
            if "concurrent" in all_results:
                conc_success = all_results["concurrent"]["successful"]
                print(f"  Concurrent requests: {conc_success} - {'✅ PASS' if conc_success >= 20 else '❌ FAIL'} (≥ 20 target)")

        print("\n" + "="*80)

    finally:
        await model.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
