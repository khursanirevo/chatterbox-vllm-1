#!/usr/bin/env python3
"""
MPS vs Sequential with Batched Poisson Traffic

Accumulates Poisson arrivals into batches before processing.
This demonstrates the real benefit of MPS parallelism.
"""

import asyncio
import time
import random
import torch
import statistics
from typing import List, Dict
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


async def benchmark_batched(
    model, 
    num_requests: int = 50, 
    batch_size: int = 8,
    rate: float = 2.0
):
    """Benchmark with batched processing."""
    random.seed(42)
    
    # Generate requests with Poisson timing
    requests = []
    current_time = 0.0
    for i in range(num_requests):
        inter_arrival = random.expovariate(rate)
        current_time += inter_arrival
        text = random.choice(TEXT_CORPUS)
        requests.append((current_time, i, text))
    
    # Process in batches
    start_time = time.time()
    all_results = []
    
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        batch_texts = [r[2] for r in batch]
        
        # Generate for this batch
        results = await model.generate(batch_texts)
        all_results.extend(results)
        
        if i % 10 == 0:
            print(f"  Processed {min(i+batch_size, len(requests))}/{len(requests)} requests")
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    gen_times = []
    for result in all_results:
        # Approximate generation time based on audio length
        gen_time = result.shape[1] / 24000  # 24kHz sample rate
        gen_times.append(gen_time)
    
    return {
        "total_time": total_time,
        "avg_gen_time": statistics.mean(gen_times) if gen_times else 0,
        "throughput": len(requests) / total_time,
    }


async def main():
    print("\n" + "="*80)
    print(f"{'MPS Batched Processing Benchmark':^80}")
    print("="*80)
    
    ckpt_dir = os.environ.get('CHATTERBOX_CKPT',
        "/mnt/data/shared/hf/hub/models--ResembleAI--chatterbox/snapshots/1b475dffa71fb191cb6d5901215eb6f55635a9b6/")
    
    print(f"\nCheckpoint: {ckpt_dir}")
    print(f"Requests: 50")
    print(f"Batch size: 8")
    print(f"Arrival rate: 2 req/s (Poisson)")
    
    # Sequential (MPS disabled)
    print("\n" + "-"*80)
    print("SEQUENTIAL (MPS disabled)")
    print("-"*80)
    
    old_mps = os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    
    model_seq = await ChatterboxTTSAsync.from_local(
        ckpt_dir,
        target_device="cuda:0",
        variant="english",
        s3gen_use_fp16=False,
        s3gen_compile_model=False,
    )
    
    metrics_seq = await benchmark_batched(model_seq)
    print(f"Total time: {metrics_seq['total_time']:.2f}s")
    print(f"Throughput: {metrics_seq['throughput']:.2f} req/s")
    
    await model_seq.shutdown()
    
    # MPS Parallel
    print("\n" + "-"*80)
    print("MPS PARALLEL (with batching)")
    print("-"*80)
    
    os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
    
    model_mps = await ChatterboxTTSAsync.from_local(
        ckpt_dir,
        target_device="cuda:0",
        variant="english",
        s3gen_use_fp16=False,
        s3gen_compile_model=False,
    )
    
    metrics_mps = await benchmark_batched(model_mps)
    print(f"Total time: {metrics_mps['total_time']:.2f}s")
    print(f"Throughput: {metrics_mps['throughput']:.2f} req/s")
    
    await model_mps.shutdown()
    
    # Comparison
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Sequential: {metrics_seq['total_time']:.2f}s ({metrics_seq['throughput']:.2f} req/s)")
    print(f"MPS:        {metrics_mps['total_time']:.2f}s ({metrics_mps['throughput']:.2f} req/s)")
    
    speedup = metrics_seq['total_time'] / metrics_mps['total_time']
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print("✅ MPS is FASTER with batched processing")
    else:
        print("⚠️ MPS is slower (unexpected!)")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
