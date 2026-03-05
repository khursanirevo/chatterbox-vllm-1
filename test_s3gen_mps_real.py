#!/usr/bin/env python3
"""
Test S3Gen with CUDA MPS (GPU 0 only).

This uses the actual Chatterbox model to test MPS performance.
"""
import os
import sys
import asyncio
import time

# IMPORTANT: Set GPU 0 ONLY before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

# Now import torch and model
import torch

from chatterbox_vllm import ChatterboxTTSAsync

# Test prompts
PROMPTS = [
    "Hello world, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "CUDA MPS allows multiple processes to share GPU resources.",
    "This is a demonstration of parallel S3Gen inference.",
]


async def test_sequential():
    """Test sequential S3Gen (baseline)."""
    print("=" * 70)
    print("Test 1: SEQUENTIAL S3Gen (Baseline)")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=False,
    )

    # Generate
    print("\n[2/3] Generating audio sequentially...")
    request_ids = [f"seq_{i}" for i in range(len(PROMPTS))]

    start = time.time()
    results = await model.generate(
        prompts=PROMPTS,
        request_ids=request_ids,
        ref_audio=None,
        ref_sr=24000,
        temperature=0.8,
    )
    total = time.time() - start

    print(f"\n[3/3] Sequential Results:")
    print(f"  Requests:   {len(results)}")
    print(f"  Total time: {total:.2f}s")
    print(f"  Per request: {total / len(results):.2f}s")

    await model.shutdown()
    return total


async def test_parallel_mps():
    """Test parallel S3Gen with CUDA MPS."""
    print("\n" + "=" * 70)
    print("Test 2: PARALLEL S3Gen with CUDA MPS")
    print("=" * 70)

    # Check MPS is running
    print("\n[1/4] Checking MPS status...")
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,compute_mode', '--format=csv,noheader'],
                          capture_output=True, text=True)
    print(f"  GPU modes:\n{result.stdout}")

    # Load model
    print("\n[2/4] Loading model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=False,
    )

    # Generate (should auto-detect MPS and use multiprocessing)
    print("\n[3/4] Generating audio with MPS...")
    request_ids = [f"mps_{i}" for i in range(len(PROMPTS))]

    start = time.time()
    results = await model.generate(
        prompts=PROMPTS,
        request_ids=request_ids,
        ref_audio=None,
        ref_sr=24000,
        temperature=0.8,
    )
    total = time.time() - start

    print(f"\n[4/4] Parallel (MPS) Results:")
    print(f"  Requests:   {len(results)}")
    print(f"  Total time: {total:.2f}s")
    print(f"  Per request: {total / len(results):.2f}s")

    await model.shutdown()
    return total


async def main():
    """Main test."""
    print("=" * 70)
    print("S3Gen CUDA MPS Test (GPU 0 only)")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"  CUDA_MPS_PIPE_DIRECTORY: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}")
    print(f"  Number of prompts: {len(PROMPTS)}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n✗ CUDA not available!")
        sys.exit(1)

    print(f"\n✓ CUDA available")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Using GPU: 0 ({torch.cuda.get_device_name(0)})")

    # Run tests
    time_seq = await test_sequential()
    time.sleep(2)
    torch.cuda.empty_cache()

    time_par = await test_parallel_mps()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if time_seq > 0 and time_par > 0:
        speedup = time_seq / time_par
        print(f"  Sequential: {time_seq:.2f}s")
        print(f"  Parallel (MPS): {time_par:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {(1 - time_par/time_seq) * 100:.1f}%")

        if speedup >= 1.3:
            print("\n  ✓✓ CUDA MPS is working!")
        elif speedup >= 1.1:
            print("\n  ✓ CUDA MPS showing benefit")
        else:
            print("\n  ⚠ MPS may not be fully utilized (only 4 requests)")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n✓ Test completed!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
