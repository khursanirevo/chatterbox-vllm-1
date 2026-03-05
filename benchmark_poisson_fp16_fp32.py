#!/usr/bin/env python3
"""
FP16 vs FP32 Poisson Traffic Simulation Benchmark

Compares real-world performance of FP16 and FP32 modes under
realistic concurrent traffic with Poisson arrival patterns.
"""

import asyncio
import time
import random
import torch
import statistics
from typing import List, Tuple, Dict
from pathlib import Path
import json
import datetime

from chatterbox_vllm import ChatterboxTTSAsync


# Diverse text prompts with varying lengths (same as example-poisson-traffic.py)
TEXT_CORPUS = {
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
    ],
    "medium": [
        "This is a medium length text that will take some time to process.",
        "The weather today is quite nice, with clear skies and mild temperatures.",
        "I would like to order a large pizza with pepperoni and extra cheese.",
        "Please remember to turn off the lights before leaving the office.",
        "The meeting has been rescheduled to next Tuesday at three in the afternoon.",
        "Could you please send me the report by the end of the day tomorrow?",
        "The new software update includes many improvements and bug fixes.",
        "I'm really looking forward to the weekend trip we have planned.",
        "The train will be arriving at platform four in approximately ten minutes.",
        "Please make sure to review all the documents before the meeting starts.",
    ],
    "long": [
        "This is a significantly longer text passage that will require more processing time through the text to speech synthesis pipeline, including tokenization, language model inference, and audio decoding.",
        "The history of artificial intelligence dates back to ancient times, but the modern field of AI research was founded in 1956 at a conference held at Dartmouth College, where researchers gathered to discuss the possibility of creating machines that could think.",
        "When preparing for a long journey, it's important to pack all the essentials including clothing appropriate for the climate, toiletries, important documents, electronic devices with their chargers, and any medications you might need during your trip.",
        "The process of photosynthesis in plants involves converting light energy into chemical energy, which is stored in glucose molecules, and this process occurs in the chloroplasts of plant cells using chlorophyll to capture the light energy.",
        "In today's competitive job market, having a diverse set of skills is crucial for career success, including technical skills relevant to your field, soft skills like communication and teamwork, and the ability to adapt to new technologies and methodologies.",
    ],
}


def generate_poisson_requests(
    num_requests: int,
    avg_requests_per_second: float,
    seed: int = 42
) -> List[Tuple[float, str, str, str]]:
    """
    Generate requests following a Poisson process.

    Args:
        num_requests: Total number of requests to generate
        avg_requests_per_second: Average arrival rate (lambda)
        seed: Random seed for reproducibility

    Returns:
        List of (arrival_time, request_id, text, category) tuples
    """
    random.seed(seed)
    requests = []
    current_time = 0.0

    for i in range(num_requests):
        inter_arrival = random.expovariate(avg_requests_per_second)
        current_time += inter_arrival

        # Select text length (weighted towards medium length)
        text_type = random.choices(
            ["short", "medium", "long"],
            weights=[0.30, 0.50, 0.20],
            k=1
        )[0]

        text = random.choice(TEXT_CORPUS[text_type])
        requests.append((current_time, f"req-{i:04d}", text, text_type))

    return requests


async def simulate_request(
    model: ChatterboxTTSAsync,
    request_id: str,
    text: str,
    category: str,
    arrival_time: float,
    start_time: float,
    results_queue: asyncio.Queue
):
    """Process a single TTS request and record metrics."""
    # Wait until arrival time
    wait_time = arrival_time - (time.time() - start_time)
    if wait_time > 0:
        await asyncio.sleep(wait_time)

    actual_start_time = time.time()
    queue_time = actual_start_time - start_time - arrival_time

    word_count = len(text.split())

    try:
        # Generate audio
        await model.generate(
            prompts=[text],
            temperature=0.8,
            exaggeration=0.5,
        )

        actual_end_time = time.time()
        generation_time = actual_end_time - actual_start_time
        total_time = actual_end_time - start_time

        result = {
            "request_id": request_id,
            "category": category,
            "word_count": word_count,
            "arrival_time": arrival_time,
            "queue_time": queue_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "success": True,
        }

        await results_queue.put(result)

    except Exception as e:
        actual_end_time = time.time()
        result = {
            "request_id": request_id,
            "category": category,
            "word_count": word_count,
            "arrival_time": arrival_time,
            "queue_time": queue_time,
            "generation_time": actual_end_time - actual_start_time,
            "total_time": actual_end_time - start_time,
            "success": False,
            "error": str(e),
        }
        await results_queue.put(result)


async def run_simulation(
    model: ChatterboxTTSAsync,
    requests: List[Tuple[float, str, str, str]]
) -> Tuple[List[dict], float]:
    """Run the Poisson traffic simulation."""
    results_queue = asyncio.Queue()
    start_time = time.time()

    # Create all request tasks
    tasks = [
        simulate_request(model, req_id, text, category, arrival, start_time, results_queue)
        for arrival, req_id, text, category in requests
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(await results_queue.get())

    total_time = time.time() - start_time
    return results, total_time


def analyze_results(results: List[dict], mode_name: str) -> Dict:
    """Analyze simulation results and return metrics."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "mode": mode_name,
            "success_rate": 0,
            "failed_count": len(failed),
        }

    # Overall metrics
    queue_times = [r["queue_time"] for r in successful]
    gen_times = [r["generation_time"] for r in successful]
    total_times = [r["total_time"] for r in successful]

    # Per-category metrics
    by_category = {}
    for category in ["short", "medium", "long"]:
        cat_results = [r for r in successful if r["category"] == category]
        if cat_results:
            by_category[category] = {
                "count": len(cat_results),
                "avg_gen_time": statistics.mean([r["generation_time"] for r in cat_results]),
                "p95_gen_time": sorted([r["generation_time"] for r in cat_results])[int(len(cat_results) * 0.95)] if len(cat_results) >= 20 else max([r["generation_time"] for r in cat_results]),
                "p99_gen_time": sorted([r["generation_time"] for r in cat_results])[int(len(cat_results) * 0.99)] if len(cat_results) >= 100 else max([r["generation_time"] for r in cat_results]),
                "avg_total_time": statistics.mean([r["total_time"] for r in cat_results]),
            }

    return {
        "mode": mode_name,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) * 100,
        "avg_queue_time": statistics.mean(queue_times),
        "p95_queue_time": sorted(queue_times)[int(len(queue_times) * 0.95)],
        "p99_queue_time": sorted(queue_times)[int(len(queue_times) * 0.99)],
        "avg_generation_time": statistics.mean(gen_times),
        "p95_generation_time": sorted(gen_times)[int(len(gen_times) * 0.95)],
        "p99_generation_time": sorted(gen_times)[int(len(gen_times) * 0.99)],
        "avg_total_time": statistics.mean(total_times),
        "p95_total_time": sorted(total_times)[int(len(total_times) * 0.95)],
        "p99_total_time": sorted(total_times)[int(len(total_times) * 0.99)],
        "by_category": by_category,
    }


def print_comparison(fp16_metrics: Dict, fp32_metrics: Dict, num_requests: int, rate: float):
    """Print detailed comparison report."""
    print("\n" + "="*100)
    print(f"{'FP16 vs FP32 POISSON TRAFFIC SIMULATION':^100}")
    print("="*100)

    print(f"\nSimulation Parameters:")
    print(f"  Total Requests: {num_requests}")
    print(f"  Average Arrival Rate: {rate:.2f} requests/second")
    print(f"  Traffic Pattern: Poisson (random inter-arrival times)")
    print(f"  Text Distribution: 30% short, 50% medium, 20% long")

    # Overall comparison
    print(f"\n{'='*100}")
    print(f"{'OVERALL PERFORMANCE COMPARISON':^100}")
    print(f"{'='*100}\n")

    print(f"{'Metric':<30} {'FP16':<20} {'FP32':<20} {'Speedup':<15} {'Improvement':<15}")
    print("-"*100)

    metrics_to_compare = [
        ("Success Rate", "%", fp16_metrics["success_rate"], fp32_metrics["success_rate"], True),
        ("Avg Queue Time", "s", fp16_metrics["avg_queue_time"], fp32_metrics["avg_queue_time"], False),
        ("P95 Queue Time", "s", fp16_metrics["p95_queue_time"], fp32_metrics["p95_queue_time"], False),
        ("P99 Queue Time", "s", fp16_metrics["p99_queue_time"], fp32_metrics["p99_queue_time"], False),
        ("Avg Generation Time", "s", fp16_metrics["avg_generation_time"], fp32_metrics["avg_generation_time"], False),
        ("P95 Generation Time", "s", fp16_metrics["p95_generation_time"], fp32_metrics["p95_generation_time"], False),
        ("P99 Generation Time", "s", fp16_metrics["p99_generation_time"], fp32_metrics["p99_generation_time"], False),
        ("Avg Total Time", "s", fp16_metrics["avg_total_time"], fp32_metrics["avg_total_time"], False),
        ("P95 Total Time", "s", fp16_metrics["p95_total_time"], fp32_metrics["p95_total_time"], False),
        ("P99 Total Time", "s", fp16_metrics["p99_total_time"], fp32_metrics["p99_total_time"], False),
    ]

    for name, unit, fp16_val, fp32_val, higher_is_better in metrics_to_compare:
        if higher_is_better:
            speedup = fp16_val / fp32_val
            improvement = ((fp16_val - fp32_val) / fp32_val) * 100
        else:
            speedup = fp32_val / fp16_val
            improvement = ((fp32_val - fp16_val) / fp32_val) * 100

        fp16_str = f"{fp16_val:.2f} {unit}"
        fp32_str = f"{fp32_val:.2f} {unit}"

        if speedup > 1.0:
            speedup_str = f"{speedup:.2f}x ✅"
            improve_str = f"{improvement:+.1f}%"
        else:
            speedup_str = f"{speedup:.2f}x ⚠️"
            improve_str = f"{improvement:+.1f}%"

        print(f"{name:<30} {fp16_str:<20} {fp32_str:<20} {speedup_str:<15} {improve_str:<15}")

    # Per-category comparison
    print(f"\n{'='*100}")
    print(f"{'PERFORMANCE BY TEXT LENGTH':^100}")
    print(f"{'='*100}\n")

    for category in ["short", "medium", "long"]:
        fp16_cat = fp16_metrics["by_category"].get(category, {})
        fp32_cat = fp32_metrics["by_category"].get(category, {})

        if not fp16_cat or not fp32_cat:
            continue

        print(f"{category.upper()} (n={fp16_cat['count']}):")
        print(f"  {'Metric':<20} {'FP16':<15} {'FP32':<15} {'Speedup':<15}")
        print(f"  {'-'*60}")

        for metric in ["avg_gen_time", "p95_gen_time", "p99_gen_time", "avg_total_time"]:
            metric_name = metric.replace("_", " ").title()
            fp16_val = fp16_cat[metric]
            fp32_val = fp32_cat[metric]
            speedup = fp32_val / fp16_val
            improvement = ((fp32_val - fp16_val) / fp32_val) * 100

            if speedup > 1.0:
                speedup_str = f"{speedup:.2f}x ✅"
            else:
                speedup_str = f"{speedup:.2f}x"

            print(f"  {metric_name:<20} {fp16_val:<15.2f} {fp32_val:<15.2f} {speedup_str:<15}")

        print()

    # Key findings
    print("="*100)
    print("KEY FINDINGS")
    print("="*100)

    overall_speedup = fp32_metrics["avg_generation_time"] / fp16_metrics["avg_generation_time"]

    if overall_speedup > 1.0:
        print(f"\n✅ FP16 is {overall_speedup:.2f}x FASTER on average")
    else:
        print(f"\n⚠️ FP16 is {1/overall_speedup:.2f}x SLOWER on average")

    # Calculate category-specific insights
    for category in ["short", "medium", "long"]:
        fp16_cat = fp16_metrics["by_category"].get(category, {})
        fp32_cat = fp32_metrics["by_category"].get(category, {})

        if fp16_cat and fp32_cat:
            cat_speedup = fp32_cat["avg_gen_time"] / fp16_cat["avg_gen_time"]
            if cat_speedup > 1.1:
                print(f"  ✅ {category.capitalize()}: {cat_speedup:.2f}x faster")
            elif cat_speedup < 0.95:
                print(f"  ⚠️ {category.capitalize()}: {1/cat_speedup:.2f}x slower")

    print("\n" + "="*100)


async def main():
    """Run FP16 vs FP32 Poisson traffic benchmark."""
    print("\n" + "="*100)
    print(f"{'FP16 vs FP32 Real-World Traffic Simulation':^100}")
    print("="*100)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA is required for this benchmark")
        return 1

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # Simulation parameters
    NUM_REQUESTS = 50
    AVG_REQUESTS_PER_SECOND = 2.0

    # Generate same traffic pattern for both modes (seeded)
    requests = generate_poisson_requests(
        num_requests=NUM_REQUESTS,
        avg_requests_per_second=AVG_REQUESTS_PER_SECOND,
        seed=42
    )

    print(f"\nTraffic Configuration:")
    print(f"  Total Requests: {NUM_REQUESTS}")
    print(f"  Average Arrival Rate: {AVG_REQUESTS_PER_SECOND} req/s")
    print(f"  Traffic Pattern: Poisson (random inter-arrival times)")
    print(f"  Text Distribution: 30% short, 50% medium, 20% long")

    # ========== FP32 BASELINE ==========
    print("\n" + "="*100)
    print("TESTING FP32 (BASELINE)")
    print("="*100 + "\n")

    model_fp32 = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=False,
        s3gen_compile_model=False,
    )

    print("Running FP32 simulation...")
    results_fp32, time_fp32 = await run_simulation(model_fp32, requests)
    metrics_fp32 = analyze_results(results_fp32, "FP32")

    print(f"\nFP32 Simulation Complete:")
    print(f"  Duration: {time_fp32:.2f}s")
    print(f"  Throughput: {NUM_REQUESTS/time_fp32:.2f} req/s")
    print(f"  Success Rate: {metrics_fp32['success_rate']:.1f}%")

    # Cleanup FP32 model
    await model_fp32.shutdown()

    # Small delay to let GPU memory clear
    await asyncio.sleep(2)

    # ========== FP16 OPTIMIZED ==========
    print("\n" + "="*100)
    print("TESTING FP16 (OPTIMIZED)")
    print("="*100 + "\n")

    model_fp16 = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        s3gen_compile_model=False,
    )

    print("Running FP16 simulation...")
    results_fp16, time_fp16 = await run_simulation(model_fp16, requests)
    metrics_fp16 = analyze_results(results_fp16, "FP16")

    print(f"\nFP16 Simulation Complete:")
    print(f"  Duration: {time_fp16:.2f}s")
    print(f"  Throughput: {NUM_REQUESTS/time_fp16:.2f} req/s")
    print(f"  Success Rate: {metrics_fp16['success_rate']:.1f}%")

    # Cleanup FP16 model
    await model_fp16.shutdown()

    # ========== COMPARISON ==========
    print_comparison(metrics_fp16, metrics_fp32, NUM_REQUESTS, AVG_REQUESTS_PER_SECOND)

    # Save results to JSON
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": {
            "num_requests": NUM_REQUESTS,
            "avg_requests_per_second": AVG_REQUESTS_PER_SECOND,
        },
        "fp32": metrics_fp32,
        "fp16": metrics_fp16,
    }

    output_file = Path("poisson_fp16_fp32_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file.absolute()}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
