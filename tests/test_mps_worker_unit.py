#!/usr/bin/env python3
"""
Unit tests for CUDA MPS worker module (without full ChatterboxTTS import).
"""
import os
import sys


def test_worker_module_syntax():
    """Test that worker module has correct syntax."""
    print("[Test 1] Testing worker module syntax...")
    try:
        import py_compile
        py_compile.compile('src/chatterbox_vllm/s3gen_mps_worker.py', doraise=True)
        print("✓ Worker module syntax is valid")
        return True
    except Exception as e:
        print(f"✗ Syntax error in worker module: {e}")
        return False


def test_worker_functions_exist():
    """Test that worker functions are defined."""
    print("\n[Test 2] Testing worker functions exist...")
    try:
        # Import the module directly
        sys.path.insert(0, 'src')
        import chatterbox_vllm.s3gen_mps_worker as worker

        # Check functions exist
        assert hasattr(worker, '_init_worker'), "_init_worker not found"
        assert hasattr(worker, '_run_s3gen_worker'), "_run_s3gen_worker not found"
        assert hasattr(worker, '_get_worker_status'), "_get_worker_status not found"

        print("✓ All worker functions defined")
        return True
    except Exception as e:
        print(f"✗ Failed to import worker module: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_worker_status_uninitialized():
    """Test worker status before initialization."""
    print("\n[Test 3] Testing worker status (uninitialized)...")
    try:
        sys.path.insert(0, 'src')
        import chatterbox_vllm.s3gen_mps_worker as worker

        status = worker._get_worker_status()

        assert status['worker_id'] is None, "worker_id should be None"
        assert status['model_loaded'] is False, "model_loaded should be False"
        assert status['device'] is None, "device should be None"

        print(f"✓ Worker status correct (uninitialized): {status}")
        return True
    except Exception as e:
        print(f"✗ Failed to get worker status: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_async_syntax():
    """Test that tts_async has correct syntax."""
    print("\n[Test 4] Testing tts_async syntax...")
    try:
        import py_compile
        py_compile.compile('src/chatterbox_vllm/tts_async.py', doraise=True)
        print("✓ tts_async syntax is valid")
        return True
    except Exception as e:
        print(f"✗ Syntax error in tts_async: {e}")
        return False


def test_mps_import():
    """Test that multiprocessing can be imported."""
    print("\n[Test 5] Testing multiprocessing import...")
    try:
        from multiprocessing import Pool
        print("✓ multiprocessing.Pool imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import multiprocessing: {e}")
        return False


def test_mps_environment():
    """Test MPS environment setup."""
    print("\n[Test 6] Testing MPS environment...")

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


def main():
    """Run all tests."""
    print("=" * 70)
    print("CUDA MPS Worker Module - Unit Tests")
    print("=" * 70)

    results = []

    # Run tests
    results.append(("Worker syntax", test_worker_module_syntax()))
    results.append(("Worker functions", test_worker_functions_exist()))
    results.append(("Worker status", test_worker_status_uninitialized()))
    results.append(("tts_async syntax", test_tts_async_syntax()))
    results.append(("Multiprocessing import", test_mps_import()))
    results.append(("MPS environment", test_mps_environment()))

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
    sys.exit(main())
