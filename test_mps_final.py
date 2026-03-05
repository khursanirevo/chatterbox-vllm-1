#!/usr/bin/env python3
"""
Simple CUDA MPS test using GPU 0.

This demonstrates GPU resource sharing with CUDA MPS.
"""
import os
import sys

# Set environment before importing torch
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

import time
import torch
import torch.multiprocessing as mp


def gpu_worker(worker_id, iterations=5):
    """Simulate S3Gen-like GPU work."""
    try:
        print(f"[Worker {worker_id}] Starting")
        start = time.time()

        # Simulate S3Gen computation (matrix operations)
        size = 1500
        for i in range(iterations):
            # Flow matching-like operations
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            c = torch.mm(a, b)  # Matrix multiply
            d = torch.randn(size, size, device='cuda')
            e = c + d  # Element-wise add
            torch.cuda.synchronize()

        elapsed = time.time() - start
        print(f"[Worker {worker_id}] Completed in {elapsed:.2f}s")
        return worker_id, elapsed
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return worker_id, -1


def test_sequential():
    """Test sequential execution (baseline)."""
    print("=" * 70)
    print("Test 1: SEQUENTIAL Execution (Baseline)")
    print("=" * 70)

    num_workers = 4
    start = time.time()

    results = []
    for i in range(num_workers):
        print(f"\n[Main] Starting worker {i+1}/{num_workers}...")
        result = gpu_worker(i)
        results.append(result)

    total = time.time() - start

    successful = [r for r in results if r[1] > 0]
    if successful:
        avg_time = sum(r[1] for r in successful) / len(successful)
        print(f"\n{'='*70}")
        print(f"SEQUENTIAL Results:")
        print(f"  Total time:     {total:.2f}s")
        print(f"  Per worker:     {avg_time:.2f}s")
        print(f"  Workers:        {len(successful)}/{num_workers}")
        print(f"{'='*70}")
        return total

    print("ERROR: No workers completed successfully")
    return 0


def test_parallel_mps():
    """Test parallel execution with CUDA MPS."""
    print("\n" + "=" * 70)
    print("Test 2: PARALLEL Execution (with CUDA MPS)")
    print("=" * 70)

    # Start MPS daemon
    print("\n[1/4] Starting CUDA MPS daemon...")
    import subprocess

    # Stop any existing MPS
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control 2>/dev/null'],
                  capture_output=True)
    time.sleep(0.5)

    # Start fresh MPS
    result = subprocess.run(['nvidia-cuda-mps-control', '-d'],
                          capture_output=True, text=True)
    time.sleep(1)

    if result.returncode == 0 or 'already running' in result.stderr.lower():
        print("  ✓ MPS daemon started")
    else:
        print(f"  ⚠ Warning: {result.stderr}")

    # Verify GPU status
    print("\n[2/4] Checking GPU status...")
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
                            '--format=csv,noheader'],
                          capture_output=True, text=True)
    print(f"  GPU Status:\n{result.stdout}")

    # Run workers in parallel
    print("\n[3/4] Running 4 workers in PARALLEL with MPS...")
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
            print(f"\n{'='*70}")
            print(f"PARALLEL (MPS) Results:")
            print(f"  Total time:     {total:.2f}s")
            print(f"  Per worker:     {avg_time:.2f}s")
            print(f"  Workers:        {len(successful)}/{num_workers}")
            print(f"  Efficiency:     {4 * avg_time / total:.1f}%")
            print(f"{'='*70}")
            return total
    except Exception as e:
        print(f"ERROR in parallel execution: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Main test function."""
    print("=" * 70)
    print("CUDA MPS Multi-Processing Test")
    print("=" * 70)
    print("\nThis test demonstrates GPU resource sharing with CUDA MPS.")
    print("Multiple processes will share GPU 0 for parallel computation.")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA not available!")
        print("   Check your PyTorch and CUDA installation")
        sys.exit(1)

    print(f"\n✓ CUDA available")
    print(f"  Device count:  {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name:    {torch.cuda.get_device_name(0)}")
    print(f"  Memory total:   {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test 1: Sequential
    time_seq = test_sequential()

    # Small pause between tests
    time.sleep(1)
    torch.cuda.empty_cache()

    # Test 2: Parallel with MPS
    time_par = test_parallel_mps()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if time_seq > 0 and time_par > 0:
        speedup = time_seq / time_par
        print(f"  Sequential time:  {time_seq:.2f}s")
        print(f"  Parallel time:    {time_par:.2f}s")
        print(f"  Speedup:          {speedup:.2f}x")
        print(f"  Improvement:      {(1 - time_par/time_seq) * 100:.1f}%")

        if speedup > 1.5:
            print(f"\n  ✓✓✓ CUDA MPS is working! ({speedup:.1f}x speedup)")
        elif speedup > 1.1:
            print(f"\n  ✓ CUDA MPS shows some benefit ({speedup:.1f}x speedup)")
        else:
            print(f"\n  ⚠ CUDA MPS may not be fully utilized")
    else:
        print("  Test incomplete - check errors above")

    print("=" * 70)

    # Cleanup
    print("\n[4/4] Stopping MPS daemon...")
    subprocess.run(['bash', '-c', 'echo quit | nvidia-cuda-mps-control 2>/dev/null'],
                  capture_output=True)
    print("  ✓ MPS daemon stopped")

    print("\n✅ Test completed!")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
