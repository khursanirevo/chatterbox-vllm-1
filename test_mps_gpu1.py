#!/usr/bin/env python3
"""
Simple CUDA MPS test using GPU 1.

This script must be run with CUDA_VISIBLE_DEVICES=1
"""
import os
import sys

# MUST set CUDA_VISIBLE_DEVICES before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

import time
import torch
import torch.multiprocessing as mp


def gpu_worker(worker_id, iterations=5):
    """Simulate GPU work."""
    try:
        print(f"[Worker {worker_id}] Starting")

        # Create tensor on GPU
        size = 1000
        start = time.time()

        for i in range(iterations):
            # Matrix multiplication (GPU intensive)
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            c = torch.mm(a, b)
            torch.cuda.synchronize()

        elapsed = time.time() - start
        print(f"[Worker {worker_id}] Completed in {elapsed:.2f}s")
        return worker_id, elapsed
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        return worker_id, -1


def test_sequential():
    """Test sequential execution."""
    print("=" * 60)
    print("Test 1: Sequential Execution (without MPS)")
    print("=" * 60)

    num_workers = 4
    start = time.time()

    results = []
    for i in range(num_workers):
        result = gpu_worker(i)
        results.append(result)

    total = time.time() - start

    successful = [r for r in results if r[1] > 0]
    if successful:
        avg_time = sum(r[1] for r in successful) / len(successful)
        print(f"\nTotal time: {total:.2f}s")
        print(f"Average per worker: {avg_time:.2f}s")
        return total

    print("ERROR: No workers completed successfully")
    return 0


def test_parallel_mps():
    """Test parallel execution with MPS."""
    print("\n" + "=" * 60)
    print("Test 2: Parallel Execution (with MPS)")
    print("=" * 60)

    # Start MPS daemon
    print("\n[1/3] Starting MPS daemon...")
    import subprocess
    result = subprocess.run(['nvidia-cuda-mps-control', '-d'],
                          capture_output=True, text=True)
    if result.returncode != 0 and 'already running' not in result.stderr.lower():
        print(f"Warning: {result.stderr}")

    time.sleep(1)

    # Verify MPS
    print("[2/3] Verifying MPS...")
    try:
        result = subprocess.run(['echo', 'get_state', '|', 'nvidia-cuda-mps-control'],
                              shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout:
            print(f"MPS status: {result.stdout.strip()}")
    except:
        print("MPS verification failed (may still work)")

    # Run workers in parallel
    print("\n[3/3] Running 4 workers in parallel...")
    num_workers = 4
    start = time.time()

    ctx = mp.get_context('spawn')
    try:
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(gpu_worker, range(num_workers))

        total = time.time() - start

        successful = [r for r in results if r and r[1] > 0]
        if successful:
            avg_time = sum(r[1] for r in successful) / len(successful)
            print(f"\nTotal time: {total:.2f}s")
            print(f"Average per worker: {avg_time:.2f}s")
            print(f"Speedup: {4 * avg_time / total:.2f}x")
            return total
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("=" * 60)
    print("CUDA MPS Test")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available!")
        print("Make sure you have CUDA GPUs and PyTorch with CUDA support")
        sys.exit(1)

    print(f"\nCUDA available: Yes")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

    # Test 1: Sequential
    time_seq = test_sequential()

    # Test 2: Parallel with MPS
    time_par = test_parallel_mps()

    # Summary
    if time_seq > 0 and time_par > 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Sequential: {time_seq:.2f}s")
        print(f"Parallel (MPS): {time_par:.2f}s")
        print(f"Speedup: {time_seq / time_par:.2f}x")
        print("=" * 60)

    # Cleanup
    print("\nStopping MPS...")
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control'],
                  capture_output=True)

    print("\n✅ Test completed!")
