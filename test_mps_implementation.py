#!/usr/bin/env python3
"""
Test script for CUDA MPS parallel S3Gen implementation.

This script tests:
1. Worker module imports and initialization
2. Basic worker functionality
3. Integration with ChatterboxTTSAsync
"""
import os
import sys
import asyncio
from pathlib import Path


def test_worker_imports():
    """Test that worker module can be imported."""
    print("[Test 1] Testing worker module imports...")
    try:
        from chatterbox_vllm.s3gen_mps_worker import _init_worker, _run_s3gen_worker, _get_worker_status
        print("✓ Worker module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import worker module: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_status():
    """Test worker status function."""
    print("\n[Test 2] Testing worker status...")
    try:
        from chatterbox_vllm.s3gen_mps_worker import _get_worker_status
        status = _get_worker_status()
        print(f"Worker status: {status}")
        assert 'worker_id' in status
        assert 'model_loaded' in status
        print("✓ Worker status function works")
        return True
    except Exception as e:
        print(f"✗ Failed to get worker status: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_async_imports():
    """Test that tts_async can be imported with MPS support."""
    print("\n[Test 3] Testing tts_async imports...")
    try:
        from chatterbox_vllm.tts_async import ChatterboxTTSAsync, MPS_AVAILABLE, Pool
        print(f"MPS available: {MPS_AVAILABLE}")
        print(f"Pool type: {Pool}")
        print("✓ tts_async imported successfully with MPS support")
        return True
    except Exception as e:
        print(f"✗ Failed to import tts_async: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_initialization():
    """Test that ChatterboxTTSAsync initializes with ckpt_dir."""
    print("\n[Test 4] Testing model initialization with ckpt_dir...")

    # Check if checkpoint directory exists
    ckpt_dir = os.environ.get('CHATTERBOX_CKPT', './models/chatterbox')

    if not Path(ckpt_dir).exists():
        print(f"⚠ Checkpoint directory not found: {ckpt_dir}")
        print("Skipping model initialization test")
        return True  # Don't fail, just skip

    try:
        from chatterbox_vllm.tts_async import ChatterboxTTSAsync

        print(f"Loading model from {ckpt_dir}...")
        model = await ChatterboxTTSAsync.from_local(
            ckpt_dir,
            target_device="cuda:0",
            max_model_len=1000,
            variant="english",
            s3gen_use_fp16=False,
            s3gen_compile_model=False,
        )

        # Check that ckpt_dir is set
        assert model.ckpt_dir is not None, "ckpt_dir should be set"
        print(f"✓ Model initialized with ckpt_dir: {model.ckpt_dir}")

        # Check that S3Gen config is stored
        assert hasattr(model, 's3gen_use_fp16'), "Model should have s3gen_use_fp16"
        assert hasattr(model, 's3gen_compile_model'), "Model should have s3gen_compile_model"
        print(f"✓ S3Gen config stored: fp16={model.s3gen_use_fp16}, compile={model.s3gen_compile_model}")

        await model.shutdown()
        return True

    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mps_environment():
    """Test MPS environment setup."""
    print("\n[Test 5] Testing MPS environment...")

    mps_dir = os.environ.get('CUDA_MPS_PIPE_DIRECTORY')
    if mps_dir:
        print(f"✓ CUDA_MPS_PIPE_DIRECTORY set: {mps_dir}")
    else:
        print("⚠ CUDA_MPS_PIPE_DIRECTORY not set (MPS not enabled)")

    # Check if MPS daemon is running
    try:
        import subprocess
        result = subprocess.run(
            ['pgrep', '-f', 'nvidia-cuda-mps-control'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✓ MPS daemon (nvidia-cuda-mps-control) is running")
        else:
            print("⚠ MPS daemon not running")
    except Exception as e:
        print(f"⚠ Could not check MPS daemon: {e}")

    return True


async def main():
    """Run all tests."""
    print("=" * 70)
    print("CUDA MPS Parallel S3Gen Implementation Tests")
    print("=" * 70)

    results = []

    # Run synchronous tests
    results.append(("Worker imports", test_worker_imports()))
    results.append(("Worker status", test_worker_status()))
    results.append(("TTS async imports", test_tts_async_imports()))
    results.append(("MPS environment", test_mps_environment()))

    # Run async tests
    results.append(("Model initialization", await test_model_initialization()))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
