#!/usr/bin/env python3
"""
MPS vs Sequential with High-Rate Poisson Traffic

Tests with higher arrival rate where requests naturally accumulate into batches.
"""

import asyncio
import time
import random
import torch
import statistics
from typing import List, Tuple, Dict
import os
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

from chatterbox_vllm.tts_async import ChatterboxTTSAsync


TEXT_CORPUS = [
    "Hello, this is a test.",
    "The weather is nice today.",
    "Thank you for your help.",
    "I'll see you tomorrow.",
    "This is a medium length text.",
    "The meeting has been rescheduled.",
    "Please review the documents.",
    "The software update is ready.",
]


async def simulate_poisson_arrivals(
    model,
    requests: List[Tuple[float, int, str]],
    mode_name: str
):
    """Simulate Poisson arrivals with concurrent processing."""
    results_queue = asyncio.Queue()
    start_time = time.time()
    
    async def process_request(arrival_time, req_id, text):
        # Wait until arrival time
        wait_time = arrival_time - (time.time() - start_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        req_start = time.time()
        queue_time = req_start - start_time - arrival_time
        
        try:
            result = await model.generate([text])
            req_end = time.time()
            
            await results_queue.put({
                "req_id": req_id,
                "arrival_time": arrival_time,
                "queue_time": queue_time,
                "generation_time": req_end - req_start,
                "total_time": req_end - start_time,
                "success": True,
            })
        except Exception as e:
            await results_queue.put({
                "req_id": req_id,
                "arrival_time": arrival_time,
                "queue_time": queue_time,
                "generation_time": 0,
                "total_time": 0,
                "success": False,
                "error": str(e),
            })
    
    # Launch all request tasks immediately (they will sleep until their arrival time)
    tasks = [
        process_request(arrival, req_id, text)
        for arrival, req_id, text in requests
    ]
    
    await asyncio.gather(*tasks)
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(await results_queue.get())
    
    total_time = time.time() - start_time
    return results, total_time


def analyze_results(results: List[dict]) -> Dict:
    """Analyze simulation results."""
    successful = [r for r in results if r["success"]]
    
    if not successful:
        return {"success_rate": 0}
    
    gen_times = [r["generation_time"] for r in successful]
    queue_times = [r["queue_time"] for r in successful]
    total_times = [r["total_time"] for r in successful]
    
    return {
        "success_rate": len(successful) / len(results) * 100,
        "avg_gen_time": statistics.mean(gen_times),
        "p95_gen_time": sorted(gen_times)[int(len(gen_times) * 0.95)],
        "p99_gen_time": sorted(gen_times)[int(len(gen_times) * 0.99)],
        "avg_queue_time": statistics.mean(queue_times),
        "p95_queue_time": sorted(queue_times)[int(len(queue_times) * 0.95)],
        "p99_queue_time": sorted(queue_times)[int(len(queue_times) * 0.99)],
        "avg_total_time": statistics.mean(total_times),
        "p95_total_time": sorted(total_times)[int(len(total_times) * 0.95)],
        "p99_total_time": sorted(total_times)[int(len(total_times) * 0.99)],
        "throughput": len(successful) / (max([r["total_time"] for r in results]) if results else 1),
    }


async def main():
    print("\n" + "="*80)
    print(f"{'MPS High-Rate Poisson Traffic Benchmark':^80}")
    print("="*80)
    
    ckpt_dir = os.environ.get('CHATTERBOX_CKPT',
        "/mnt/data/shared/hf/hub/models--ResembleAI--chatterbox/snapshots/1b475dffa71fb191cb6d5901215eb6f55635a9b6/")
    
    # Test different arrival rates
    NUM_REQUESTS = 50
    RATES = [2.0, 5.0, 10.0]  # requests per second
    
    for rate in RATES:
        print(f"\n{'='*80}")
        print(f"ARRIVAL RATE: {rate:.1f} requests/second ({NUM_REQUESTS} total requests)")
        print(f"{'='*80}")
        
        # Generate Poisson requests
        random.seed(42)
        requests = []
        current_time = 0.0
        for i in range(NUM_REQUESTS):
            inter_arrival = random.expovariate(rate)
            current_time += inter_arrival
            text = random.choice(TEXT_CORPUS)
            requests.append((current_time, i, text))
        
        # Calculate expected span
        total_span = requests[-1][0] - requests[0][0]
        print(f"Expected time span: {total_span:.1f}s")
        print(f"Avg inter-arrival: {1/rate:.2f}s")
        
        # Sequential (MPS disabled)
        print(f"\n  Sequential (MPS disabled)...")
        old_mps = os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
        
        model_seq = await ChatterboxTTSAsync.from_local(
            ckpt_dir,
            target_device="cuda:0",
            variant="english",
            s3gen_use_fp16=False,
            s3gen_compile_model=False,
        )
        
        results_seq, time_seq = await simulate_poisson_arrivals(model_seq, requests, "Sequential")
        metrics_seq = analyze_results(results_seq)
        
        print(f"    Total: {time_seq:.2f}s | Avg gen: {metrics_seq['avg_gen_time']:.2f}s")
        print(f"    P95 gen: {metrics_seq['p95_gen_time']:.2f}s | Throughput: {NUM_REQUESTS/time_seq:.2f} req/s")
        
        await model_seq.shutdown()
        
        # MPS Parallel
        print(f"  MPS Parallel...")
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
        
        model_mps = await ChatterboxTTSAsync.from_local(
            ckpt_dir,
            target_device="cuda:0",
            variant="english",
            s3gen_use_fp16=False,
            s3gen_compile_model=False,
        )
        
        results_mps, time_mps = await simulate_poisson_arrivals(model_mps, requests, "MPS")
        metrics_mps = analyze_results(results_mps)
        
        print(f"    Total: {time_mps:.2f}s | Avg gen: {metrics_mps['avg_gen_time']:.2f}s")
        print(f"    P95 gen: {metrics_mps['p95_gen_time']:.2f}s | Throughput: {NUM_REQUESTS/time_mps:.2f} req/s")
        
        await model_mps.shutdown()
        
        # Comparison
        speedup = time_seq / time_mps
        improve = ((time_seq - time_mps) / time_seq) * 100
        
        print(f"\n  → Speedup: {speedup:.2f}x ({improve:+.1f}%)")
        
        if speedup > 1.0:
            print(f"  ✅ MPS is {speedup:.2f}x FASTER")
        else:
            print(f"  ⚠️ MPS is {1/speedup:.2f}x SLOWER")
    
    print("\n" + "="*80)
    print("Use higher rates for better batching!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
