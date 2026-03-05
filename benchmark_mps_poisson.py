#!/usr/bin/env python3
"""
MPS vs Sequential Poisson Traffic Benchmark

Tests real-world performance with Poisson arrival patterns.
Compares sequential processing vs MPS parallel processing.
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
import os
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from chatterbox_vllm.tts_async import ChatterboxTTSAsync


# Diverse text prompts with varying lengths
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
        "This is a significantly longer text passage that will require more processing time through the text to speech synthesis pipeline.",
        "The history of artificial intelligence dates back to ancient times, but the modern field of AI research was founded in 1956.",
        "When preparing for a long journey, it's important to pack all the essentials including clothing and toiletries.",
        "The process of photosynthesis in plants involves converting light energy into chemical energy stored in glucose.",
        "In today's competitive job market, having a diverse set of skills is crucial for career success and advancement.",
    ],
}


def generate_poisson_requests(
    num_requests: int,
    avg_requests_per_second: float,
    seed: int = 42
) -> List[Tuple[float, str, str, str]]:
    """Generate requests following a Poisson process."""
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


def print_comparison(seq_metrics: Dict, mps_metrics: Dict, num_requests: int, rate: float):
    """Print detailed comparison report."""
    print("\n" + "="*100)
    print(f"{'MPS vs SEQUENTIAL POISSON TRAFFIC SIMULATION':^100}")
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

    print(f"{'Metric':<30} {'Sequential':<20} {'MPS Parallel':<20} {'Speedup':<15} {'Improvement':<15}")
    print("-"*100)

    metrics_to_compare = [
        ("Success Rate", "%", seq_metrics["success_rate"], mps_metrics["success_rate"], True),
        ("Avg Queue Time", "s", seq_metrics["avg_queue_time"], mps_metrics["avg_queue_time"], False),
        ("P95 Queue Time", "s", seq_metrics["p95_queue_time"], mps_metrics["p95_queue_time"], False),
        ("P99 Queue Time", "s", seq_metrics["p99_queue_time"], mps_metrics["p99_queue_time"], False),
        ("Avg Generation Time", "s", seq_metrics["avg_generation_time"], mps_metrics["avg_generation_time"], False),
        ("P95 Generation Time", "s", seq_metrics["p95_generation_time"], mps_metrics["p95_generation_time"], False),
        ("P99 Generation Time", "s", seq_metrics["p99_generation_time"], mps_metrics["p99_generation_time"], False),
        ("Avg Total Time", "s", seq_metrics["avg_total_time"], mps_metrics["avg_total_time"], False),
        ("P95 Total Time", "s", seq_metrics["p95_total_time"], mps_metrics["p95_total_time"], False),
        ("P99 Total Time", "s", seq_metrics["p99_total_time"], mps_metrics["p99_total_time"], False),
    ]

    for name, unit, seq_val, mps_val, higher_is_better in metrics_to_compare:
        if higher_is_better:
            speedup = mps_val / seq_val
            improvement = ((mps_val - seq_val) / seq_val) * 100
        else:
            speedup = seq_val / mps_val
            improvement = ((seq_val - mps_val) / mps_val) * 100

        seq_str = f"{seq_val:.2f} {unit}"
        mps_str = f"{mps_val:.2f} {unit}"

        if speedup > 1.0:
            speedup_str = f"{speedup:.2f}x ✅"
            improve_str = f"{improvement:+.1f}%"
        else:
            speedup_str = f"{speedup:.2f}x ⚠️"
            improve_str = f"{improvement:+.1f}%"

        print(f"{name:<30} {seq_str:<20} {mps_str:<20} {speedup_str:<15} {improve_str:<15}")

    # Per-category comparison
    print(f"\n{'='*100}")
    print(f"{'PERFORMANCE BY TEXT LENGTH':^100}")
    print(f"{'='*100}\n")

    for category in ["short", "medium", "long"]:
        seq_cat = seq_metrics["by_category"].get(category, {})
        mps_cat = mps_metrics["by_category"].get(category, {})

        if not seq_cat or not mps_cat:
            continue

        print(f"{category.upper()} (n={seq_cat['count']}):")
        print(f"  {'Metric':<20} {'Sequential':<15} {'MPS':<15} {'Speedup':<15}")
        print(f"  {'-'*60}")

        for metric in ["avg_gen_time", "p95_gen_time", "avg_total_time"]:
            metric_name = metric.replace("_", " ").title()
            seq_val = seq_cat[metric]
            mps_val = mps_cat[metric]
            speedup = seq_val / mps_val
            improvement = ((seq_val - mps_val) / mps_val) * 100

            speedup_str = f"{speedup:.2f}x"
            if speedup > 1.1:
                speedup_str += " ✅"

            print(f"  {metric_name:<20} {seq_val:<15.2f} {mps_val:<15.2f} {speedup_str:<15}")
        print()

    # Key findings
    print("="*100)
    print("KEY FINDINGS")
    print("="*100)

    overall_speedup = seq_metrics["avg_generation_time"] / mps_metrics["avg_generation_time"]

    if overall_speedup > 1.0:
        print(f"\n✅ MPS is {overall_speedup:.2f}x FASTER on average")
    else:
        print(f"\n⚠️ MPS is {1/overall_speedup:.2f}x SLOWER on average")

    print(f"\nThroughput Comparison:")
    print(f"  Sequential: {seq_metrics['successful']} requests processed")
    print(f"  MPS Parallel: {mps_metrics['successful']} requests processed")

    print("\n" + "="*100)


async def main():
    """Run MPS vs Sequential Poisson traffic benchmark."""
    print("\n" + "="*100)
    print(f"{'MPS vs Sequential Real-World Traffic Simulation':^100}")
    print("="*100)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA is required for this benchmark")
        return 1

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Get checkpoint directory
    ckpt_dir = os.environ.get('CHATTERBOX_CKPT')
    if not ckpt_dir:
        # Try to find it
        for path in [
            "/mnt/data/shared/hf/hub/models--ResembleAI--chatterbox/snapshots/1b475dffa71fb191cb6d5901215eb6f55635a9b6/",
            "./models/chatterbox",
        ]:
            if Path(path).exists():
                ckpt_dir = path
                break
    
    if not ckpt_dir:
        print("ERROR: Set CHATTERBOX_CKPT environment variable")
        return 1

    print(f"Checkpoint: {ckpt_dir}")

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

    # ========== SEQUENTIAL BASELINE ==========
    print("\n" + "="*100)
    print("TESTING SEQUENTIAL (BASELINE - MPS DISABLED)")
    print("="*100 + "\n")

    # Make sure MPS is disabled
    old_mps = os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    
    model_seq = await ChatterboxTTSAsync.from_local(
        ckpt_dir,
        target_device="cuda:0",
        variant="english",
        s3gen_use_fp16=False,
        s3gen_compile_model=False,
    )

    print("Running Sequential simulation...")
    results_seq, time_seq = await run_simulation(model_seq, requests)
    metrics_seq = analyze_results(results_seq, "Sequential")

    print(f"\nSequential Complete:")
    print(f"  Duration: {time_seq:.2f}s")
    print(f"  Success Rate: {metrics_seq['success_rate']:.1f}%")

    # Cleanup Sequential model
    await model_seq.shutdown()

    # Restore MPS setting for next test
    if old_mps:
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = old_mps

    # ========== MPS PARALLEL ==========
    print("\n" + "="*100)
    print("TESTING MPS PARALLEL (OPTIMIZED)")
    print("="*100 + "\n")

    # Enable MPS
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'

    model_mps = await ChatterboxTTSAsync.from_local(
        ckpt_dir,
        target_device="cuda:0",
        variant="english",
        s3gen_use_fp16=False,
        s3gen_compile_model=False,
    )

    print("Running MPS simulation...")
    results_mps, time_mps = await run_simulation(model_mps, requests)
    metrics_mps = analyze_results(results_mps, "MPS")

    print(f"\nMPS Complete:")
    print(f"  Duration: {time_mps:.2f}s")
    print(f"  Success Rate: {metrics_mps['success_rate']:.1f}%")

    # Cleanup MPS model
    await model_mps.shutdown()

    # ========== COMPARISON ==========
    print_comparison(metrics_seq, metrics_mps, NUM_REQUESTS, AVG_REQUESTS_PER_SECOND)

    # Save results to JSON
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": {
            "num_requests": NUM_REQUESTS,
            "avg_requests_per_second": AVG_REQUESTS_PER_SECOND,
        },
        "sequential": metrics_seq,
        "mps_parallel": metrics_mps,
    }

    output_file = Path("poisson_mps_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file.absolute()}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
