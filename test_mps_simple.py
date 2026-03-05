#!/usr/bin/env python3
"""
Simple test for CUDA MPS with S3Gen.

This demonstrates that CUDA MPS allows multiple processes to share GPU resources.
"""
import os
import time
import torch
import torch.multiprocessing as mp

# Set MPS environment variables - GPU 0 only as specified
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

def s3gen_worker(gpu_id, worker_id, duration=2):
    """
    Simulate S3Gen work on a specific GPU.

    With MPS, multiple workers can share the same GPU.
    """
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}")
    start = time.time()

    # Simulate GPU work (matrix multiplication)
    # Use cuda:0 which maps to the actual GPU selected by CUDA_VISIBLE_DEVICES
    with torch.cuda.device(0):
        # Create large tensors (simulating S3Gen computation)
        size = 2000
        for _ in range(10):
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            c = torch.mm(a, b)
            torch.cuda.synchronize()

    elapsed = time.time() - start
    print(f"[Worker {worker_id}] Completed in {elapsed:.2f}s")
    return worker_id, elapsed


def test_without_mps():
    """Test without MPS (sequential)."""
    print("=" * 80)
    print("Test WITHOUT MPS (Sequential)")
    print("=" * 80)

    # CUDA_VISIBLE_DEVICES is already set by the caller
    num_workers = 4
    start = time.time()

    for i in range(num_workers):
        s3gen_worker(0, i)

    total = time.time() - start
    print(f"\nTotal time: {total:.2f}s for {num_workers} workers")
    print(f"Average per worker: {total / num_workers:.2f}s")

    return total


def test_with_mps():
    """Test with MPS (parallel)."""
    print("\n" + "=" * 80)
    print("Test WITH MPS (Parallel)")
    print("=" * 80)

    # Start MPS daemon
    print("\n[1/3] Starting MPS daemon...")
    import subprocess
    subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)
    time.sleep(1)

    # Run workers in parallel
    print(f"\n[2/3] Running 4 workers in parallel with MPS...")
    num_workers = 4
    start = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.starmap(s3gen_worker, [(0, i) for i in range(num_workers)])

    total = time.time() - start
    print(f"\n[3/3] Results:")
    print(f"Total time: {total:.2f}s for {num_workers} workers")
    print(f"Average per worker: {total / num_workers:.2f}s")

    return total


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("\n" + "=" * 80)
    print("CUDA MPS Test for S3Gen")
    print("=" * 80)
    print("\nThis test demonstrates GPU resource sharing with CUDA MPS.")
    print("Multiple processes will share GPU 1 (the one with minimal usage).")

    # Test without MPS (baseline)
    time_without = test_without_mps()

    # Test with MPS
    time_with = test_with_mps()

    # Calculate speedup
    speedup = time_without / time_with

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Without MPS: {time_without:.2f}s")
    print(f"With MPS:    {time_with:.2f}s")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"Improvement: {(1 - time_with/time_without) * 100:.1f}%")
    print("=" * 80)

    # Cleanup
    print("\nStopping MPS daemon...")
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control'], check=False)

    print("\n✅ Test completed!")
