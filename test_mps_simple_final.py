#!/usr/bin/env python3
"""
Simple CUDA MPS test - no device queries, just computation.
"""
import os
import sys

# Set MPS environment before importing torch
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

import time
import torch
import torch.multiprocessing as mp


def gpu_worker(worker_id, iterations=3):
    """Simulate S3Gen-like GPU work."""
    try:
        print(f"[Worker {worker_id}] Starting")
        start = time.time()

        # Simulate S3Gen computation
        size = 1200
        for i in range(iterations):
            # Flow matching-like operations
            a = torch.randn(size, size, device='cuda:0')
            b = torch.randn(size, size, device='cuda:0')
            c = torch.mm(a, b)
            d = torch.randn(size, size, device='cuda:0')
            e = c + d
            torch.cuda.synchronize()

        elapsed = time.time() - start
        print(f"[Worker {worker_id}] ✓ Completed in {elapsed:.2f}s")
        return worker_id, elapsed
    except Exception as e:
        print(f"[Worker {worker_id}] ✗ Error: {e}")
        return worker_id, -1


def test_sequential():
    """Test sequential execution."""
    print("=" * 70)
    print("Test 1: SEQUENTIAL (Baseline)")
    print("=" * 70)

    num_workers = 4
    start = time.time()

    results = []
    for i in range(num_workers):
        result = gpu_worker(i)
        results.append(result)

    total = time.time() - start
    successful = [r for r in results if r[1] > 0]

    if successful:
        avg = sum(r[1] for r in successful) / len(successful)
        print(f"\nSEQUENTIAL: {total:.2f}s total, {avg:.2f}s per worker")
        return total
    return 0


def test_parallel_mps():
    """Test parallel execution with MPS."""
    print("\n" + "=" * 70)
    print("Test 2: PARALLEL with CUDA MPS")
    print("=" * 70)

    # Start MPS
    print("\n[1/3] Starting MPS daemon...")
    import subprocess
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control 2>/dev/null'],
                  capture_output=True)
    time.sleep(0.5)
    subprocess.run(['nvidia-cuda-mps-control', '-d'], capture_output=True)
    time.sleep(1)
    print("  ✓ MPS started")

    # Run parallel
    print("\n[2/3] Running 4 workers in parallel...")
    num_workers = 4
    start = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(gpu_worker, range(num_workers))

    total = time.time() - start
    successful = [r for r in results if r and r[1] > 0]

    if successful:
        avg = sum(r[1] for r in successful) / len(successful)
        efficiency = (num_workers * avg / total) * 100
        print(f"\n[3/3] PARALLEL: {total:.2f}s total, {avg:.2f}s per worker")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Speedup: {num_workers * avg / total:.2f}x")
        return total
    return 0


if __name__ == "__main__":
    print("=" * 70)
    print("CUDA MPS Test")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n✗ CUDA not available")
        sys.exit(1)

    print(f"\n✓ CUDA available ({torch.cuda.device_count()} GPUs)")

    # Run tests
    time_seq = test_sequential()
    time.sleep(1)
    torch.cuda.empty_cache()

    time_par = test_parallel_mps()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if time_seq > 0 and time_par > 0:
        speedup = time_seq / time_par
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Parallel:   {time_par:.2f}s")
        print(f"Speedup:    {speedup:.2f}x ({(1 - time_par/time_seq) * 100:.1f}% faster)")

        if speedup >= 2.0:
            print("\n✓✓✓ MPS working great!")
        elif speedup >= 1.3:
            print("\n✓✓ MPS working well")
        elif speedup >= 1.1:
            print("\n✓ MPS showing some benefit")
        else:
            print("\n⚠ MPS may not be fully utilized")
    print("=" * 70)

    # Cleanup
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control 2>/dev/null'],
                  capture_output=True)
    print("\n✓ Test completed!")
