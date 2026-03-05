#!/usr/bin/env python3
"""
Test CUDA MPS (Multi-Process Service) for S3Gen parallelism.

This script demonstrates how CUDA MPS allows multiple processes to share
a single GPU for parallel S3Gen inference.
"""
import os
import subprocess
import time
import asyncio
from pathlib import Path

from chatterbox_vllm import ChatterboxTTSAsync

# Test prompts
PROMPTS = [
    "Hello world!",
    "The quick brown fox.",
    "Testing CUDA MPS parallelism.",
    "Multi-process GPU sharing.",
    "S3Gen with software GPU virtualization.",
    "Parallel inference without MIG.",
    "Single GPU, multiple instances.",
    "CUDA MPS demonstration.",
]


def check_mps_status():
    """Check if CUDA MPS is running."""
    try:
        result = subprocess.run(
            ['echo', 'get_state', '|', 'nvidia-cuda-mps-control'],
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'ready' in result.stdout.lower() or result.returncode == 0:
            return True
    except:
        pass
    return False


def start_mps():
    """Start CUDA MPS daemon."""
    print("=" * 80)
    print("Starting CUDA MPS Daemon")
    print("=" * 80)

    try:
        # Set GPU to exclusive mode
        subprocess.run(['nvidia-smi', '-i', '0', '-c', 'EXCLUSIVE_PROCESS'], check=True)
        print("✓ GPU set to EXCLUSIVE_PROCESS mode")

        # Start MPS daemon
        subprocess.run(['nvidia-cuda-mps-control', '-d'], check=True)
        print("✓ MPS daemon started")

        # Set environment variables
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-mps'
        os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-mps-log'

        time.sleep(1)

        # Verify
        result = subprocess.run(
            ['echo', 'get_state', '|', 'nvidia-cuda-mps-control'],
            shell=True,
            capture_output=True,
            text=True
        )
        print(f"✓ MPS status: {result.stdout.strip()}")

        print("\n✅ CUDA MPS is now running!")
        print("=" * 80)
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start MPS: {e}")
        print("\nTo manually start MPS, run:")
        print("  sudo ./start_mps.sh")
        return False


async def test_with_mps():
    """Test S3Gen with CUDA MPS enabled."""
    print("\n" + "=" * 80)
    print("Testing S3Gen with CUDA MPS")
    print("=" * 80)

    # Check if MPS is enabled
    mps_enabled = os.environ.get('CUDA_MPS_PIPE_DIRECTORY') is not None
    print(f"\nCUDA MPS enabled: {mps_enabled}")

    # Initialize model
    print("\n[1/3] Loading model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=False,
    )

    # Prepare prompts
    print("\n[2/3] Generating audio...")
    request_ids = [f"mps_test_{i:03d}" for i in range(len(PROMPTS))]

    # Measure time
    start_time = time.time()

    # Generate audio
    results = await model.generate(
        prompts=PROMPTS,
        request_ids=request_ids,
        ref_audio=None,
        ref_sr=24000,
        temperature=0.8,
        exaggeration=0.5,
    )

    total_time = time.time() - start_time

    # Report results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"Total requests:      {len(results)}")
    print(f"Total time:          {total_time:.2f}s")
    print(f"Average per request: {total_time / len(results):.3f}s")
    print(f"Throughput:          {len(results) / total_time:.2f} req/s")

    # Save samples
    print("\n[3/3] Saving audio samples...")
    output_dir = Path(__file__).parent / "test_mps_output"
    output_dir.mkdir(exist_ok=True)

    import torchaudio
    for i, (wav, prompt) in enumerate(zip(results, PROMPTS)):
        output_path = output_dir / f"mps_test_{i:03d}.wav"
        if wav.dim() == 1:
            wav_to_save = wav.unsqueeze(0)
        else:
            wav_to_save = wav
        torchaudio.save(str(output_path), wav_to_save, 24000)
        print(f"  Saved: {output_path.name} ({prompt[:30]}...)")

    print(f"\n✅ All audio saved to: {output_dir}")
    print("\n" + "=" * 80)

    # Cleanup
    await model.shutdown()

    return total_time, len(results)


async def compare_mps_vs_no_mps():
    """Compare performance with and without MPS."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: MPS vs No-MPS")
    print("=" * 80)

    results = {}

    # Test WITHOUT MPS
    print("\n[1/2] Testing WITHOUT MPS...")
    mps_env = os.environ.pop('CUDA_MPS_PIPE_DIRECTORY', None)
    os.environ.pop('CUDA_MPS_LOG_DIRECTORY', None)

    time_no_mps, count = await test_with_mps()
    results['no_mps'] = {'time': time_no_mps, 'count': count}

    # Restore MPS env if it was set
    if mps_env:
        os.environ['CUDA_MPS_PIPE_DIRECTORY'] = mps_env

    # Test WITH MPS
    print("\n" + "=" * 80)
    print("[2/2] Testing WITH MPS...")
    print("=" * 80)

    # Start MPS if not running
    if not check_mps_status():
        if not start_mps():
            print("\n⚠️  Could not start MPS, skipping MPS test")
            return

    time_with_mps, count = await test_with_mps()
    results['with_mps'] = {'time': time_with_mps, 'count': count}

    # Calculate speedup
    speedup = time_no_mps / time_with_mps

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS:")
    print("=" * 80)
    print(f"Without MPS: {time_no_mps:.2f}s ({count} requests)")
    print(f"With MPS:    {time_with_mps:.2f}s ({count} requests)")
    print(f"\nSpeedup:     {speedup:.2f}x")
    print(f"Improvement: {(1 - time_with_mps/time_no_mps) * 100:.1f}% faster")
    print("=" * 80)


async def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("CUDA MPS S3Gen Test")
    print("=" * 80)
    print("\nThis test demonstrates CUDA MPS for parallel S3Gen inference.")
    print("MPS allows multiple processes to share GPU resources (VRAM + tensor cores).")

    # Check if MPS is already running
    mps_running = check_mps_status()
    print(f"\nCUDA MPS status: {'Running ✅' if mps_running else 'Not running ❌'}")

    # Run comparison test
    await compare_mps_vs_no_mps()

    print("\n✅ All tests completed!")
    print("\nTo stop MPS later, run:")
    print("  ./stop_mps.sh")
    print("  or")
    print("  echo quit | nvidia-cuda-mps-control && nvidia-smi -i 0 -c DEFAULT")


if __name__ == "__main__":
    # Set multiprocessing start method
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    asyncio.run(main())
