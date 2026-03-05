#!/usr/bin/env python3
"""
Benchmark script for CUDA MPS parallel S3Gen.

This script benchmarks:
1. Sequential processing (baseline)
2. MPS parallel processing (if available)

Expected performance: 3-4x speedup with MPS for batches of 4+ requests.
"""
import os
import sys
import time
import asyncio
from pathlib import Path


async def benchmark_sequential(model, prompts, audio_prompt_path=None):
    """Benchmark sequential S3Gen processing."""
    print("\n" + "=" * 70)
    print("Benchmark 1: Sequential Processing (Baseline)")
    print("=" * 70)

    # Force sequential processing by disabling MPS
    original_mps = os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    if original_mps:
        del os.environ['CUDA_MPS_PIPE_DIRECTORY']
        print("[Benchmark] Disabled MPS for sequential test")

    start_time = time.time()

    try:
        results = await model.generate(
            prompts=prompts,
            audio_prompt_path=audio_prompt_path,
            temperature=0.8,
            max_tokens=1000,
            diffusion_steps=5,
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Generated {len(results)} audio samples")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average time per sample: {elapsed / len(results):.2f}s")
        print(f"  Throughput: {len(results) / elapsed:.2f} samples/sec")

        return elapsed, len(results)

    except Exception as e:
        print(f"\n✗ Sequential processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0
    finally:
        # Restore MPS setting
        if original_mps:
            os.environ['CUDA_MPS_PIPE_DIRECTORY'] = original_mps
            print("[Benchmark] Restored MPS setting")


async def benchmark_mps_parallel(model, prompts, audio_prompt_path=None):
    """Benchmark MPS parallel S3Gen processing."""
    print("\n" + "=" * 70)
    print("Benchmark 2: MPS Parallel Processing")
    print("=" * 70)

    # Check if MPS is available
    if not os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):
        print("\n⚠ MPS not enabled (CUDA_MPS_PIPE_DIRECTORY not set)")
        print("To enable MPS:")
        print("  1. Start MPS daemon: nvidia-cuda-mps-control -d")
        print("  2. Set environment: export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps")
        return None, 0

    print(f"[Benchmark] MPS enabled: {os.environ.get('CUDA_MPS_PIPE_DIRECTORY')}")

    start_time = time.time()

    try:
        results = await model.generate(
            prompts=prompts,
            audio_prompt_path=audio_prompt_path,
            temperature=0.8,
            max_tokens=1000,
            diffusion_steps=5,
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Generated {len(results)} audio samples")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average time per sample: {elapsed / len(results):.2f}s")
        print(f"  Throughput: {len(results) / elapsed:.2f} samples/sec")

        return elapsed, len(results)

    except Exception as e:
        print(f"\n✗ MPS parallel processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


async def main():
    """Run benchmarks."""
    print("=" * 70)
    print("CUDA MPS Parallel S3Gen Benchmark")
    print("=" * 70)

    # Check checkpoint directory
    ckpt_dir = os.environ.get('CHATTERBOX_CKPT', './models/chatterbox')

    if not Path(ckpt_dir).exists():
        print(f"\n✗ Checkpoint directory not found: {ckpt_dir}")
        print("Set CHATTERBOX_CKPT environment variable to checkpoint path")
        return 1

    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Load model
    print("\n" + "=" * 70)
    print("Loading ChatterboxTTSAsync model...")
    print("=" * 70)

    try:
        from chatterbox_vllm.tts_async import ChatterboxTTSAsync

        model = await ChatterboxTTSAsync.from_local(
            ckpt_dir,
            target_device="cuda:0",
            max_model_len=1000,
            variant="english",
            s3gen_use_fp16=False,
            s3gen_compile_model=False,
        )

        print("✓ Model loaded successfully")

    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test prompts
    test_prompts = [
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "CUDA MPS enables parallel processing on a single GPU.",
        "This benchmark tests the performance improvements.",
    ]

    # Check if we should use more prompts
    if len(sys.argv) > 1:
        num_prompts = int(sys.argv[1])
        if num_prompts > len(test_prompts):
            # Repeat prompts to reach desired count
            test_prompts = (test_prompts * ((num_prompts // len(test_prompts)) + 1))[:num_prompts]

    print(f"\nTest prompts: {len(test_prompts)}")

    # Run benchmarks
    sequential_time, count1 = await benchmark_sequential(model, test_prompts)
    parallel_time, count2 = await benchmark_mps_parallel(model, test_prompts)

    # Calculate speedup
    print("\n" + "=" * 70)
    print("Benchmark Results Summary")
    print("=" * 70)

    if sequential_time and parallel_time:
        speedup = sequential_time / parallel_time
        efficiency = speedup / 4 * 100  # 4 workers

        print(f"\nSequential time: {sequential_time:.2f}s")
        print(f"Parallel time:   {parallel_time:.2f}s")
        print(f"Speedup:         {speedup:.2f}x")
        print(f"Efficiency:      {efficiency:.1f}% (theoretical max: 4x)")

        if speedup >= 3.0:
            print("\n🎉 Excellent performance! Achieved 3x+ speedup")
        elif speedup >= 2.0:
            print("\n✓ Good performance! Achieved 2x+ speedup")
        elif speedup >= 1.5:
            print("\n✓ Moderate improvement")
        else:
            print("\n⚠ Low speedup - check MPS configuration")

    elif sequential_time:
        print("\n⚠ Parallel processing failed or not available")
        print(f"Sequential time: {sequential_time:.2f}s")
    else:
        print("\n✗ Both benchmarks failed")

    # Cleanup
    await model.shutdown()

    print("\n" + "=" * 70)
    print("Benchmark complete")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
