#!/usr/bin/env python3
"""
TensorRT vs PyTorch S3Gen Benchmark

Compares performance of:
1. PyTorch S3Gen (baseline)
2. TensorRT-optimized S3Gen

Note: Requires TensorRT engine to be built first using build_s3gen_tensorrt.py
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

from chatterbox_vllm import ChatterboxTTSAsync


def check_tensorrt_requirements():
    """Check if TensorRT requirements are met."""
    errors = []

    try:
        import tensorrt as trt
        print(f"✓ TensorRT {trt.__version__} found")
    except ImportError:
        errors.append("TensorRT not installed. Install with: pip install tensorrt")

    return errors


async def benchmark_mode(
    use_tensorrt: bool,
    engine_path: str = None,
    num_runs: int = 5,
    texts: List[str] = None,
) -> Dict:
    """Benchmark a specific mode."""
    mode_name = "TensorRT" if use_tensorrt else "PyTorch"

    print(f"\n{'='*80}")
    print(f"BENCHMARKING {mode_name.upper()}")
    print(f"{'='*80}\n")

    # Default test texts if not provided
    if texts is None:
        texts = [
            "Hello, this is a test.",
            "This is a medium length text for testing the text to speech synthesis system.",
            "This is a longer text passage designed to test the upper limits with more content to process through the pipeline.",
        ]

    try:
        # Initialize model
        print(f"Initializing model...")
        if use_tensorrt:
            model = await ChatterboxTTSAsync.from_pretrained(
                max_batch_size=16,
                max_model_len=1000,
                s3gen_use_fp16=True,
                s3gen_use_tensorrt=True,
                s3gen_tensorrt_engine_path=engine_path,
            )
        else:
            model = await ChatterboxTTSAsync.from_pretrained(
                max_batch_size=16,
                max_model_len=1000,
                s3gen_use_fp16=True,
            )

        print(f"✓ Model initialized")

        results = []

        for i, text in enumerate(texts):
            category = "short" if len(text.split()) <= 10 else "medium" if len(text.split()) <= 25 else "long"

            for run in range(num_runs):
                start = time.time()

                try:
                    audio = await model.generate(
                        prompts=[text],
                        temperature=0.8,
                        exaggeration=0.5,
                    )

                    if audio and len(audio) > 0:
                        elapsed = time.time() - start
                        results.append({
                            "run": i * num_runs + run,
                            "text": text[:50],
                            "category": category,
                            "word_count": len(text.split()),
                            "time": elapsed,
                            "success": True,
                        })
                        print(f"  Run {run+1}/{num_runs} ({category}): {elapsed:.3f}s")
                    else:
                        print(f"  Run {run+1}/{num_runs}: FAILED (no output)")

                except Exception as e:
                    print(f"  Run {run+1}/{num_runs}: ERROR - {e}")
                    results.append({
                        "run": i * num_runs + run,
                        "text": text[:50],
                        "category": category,
                        "time": -1,
                        "success": False,
                        "error": str(e),
                    })

        # Calculate statistics
        successful = [r for r in results if r["success"]]

        if successful:
            by_category = {}
            for r in successful:
                cat = r["category"]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(r["time"])

            stats = {
                "mode": mode_name,
                "total_runs": len(results),
                "successful": len(successful),
                "failed": len(results) - len(successful),
                "by_category": {}
            }

            for cat, times in by_category.items():
                import statistics
                stats["by_category"][cat] = {
                    "count": len(times),
                    "mean": statistics.mean(times),
                    "min": min(times),
                    "max": max(times),
                    "median": statistics.median(times),
                }

            return stats

        else:
            print(f"\n✗ All runs failed!")
            return {"mode": mode_name, "successful": 0, "failed": len(results)}

    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return {"mode": mode_name, "error": str(e)}


def print_comparison(pytorch_stats: Dict, tensorrt_stats: Dict):
    """Print comparison results."""
    print("\n" + "="*100)
    print(f"{'TENSORRT VS PYTORCH COMPARISON':^100}")
    print("="*100 + "\n")

    if "error" in pytorch_stats:
        print(f"PyTorch mode failed: {pytorch_stats['error']}")
        return

    if "error" in tensorrt_stats:
        print(f"TensorRT mode failed: {tensorrt_stats['error']}")
        return

    print(f"{'Category':<12} {'PyTorch':<12} {'TensorRT':<12} {'Speedup':<12} {'Improvement':<15}")
    print("-"*100)

    categories = set(list(pytorch_stats.get("by_category", {}).keys()) +
                     list(tensorrt_stats.get("by_category", {}).keys()))

    for cat in sorted(categories):
        py_stats = pytorch_stats["by_category"].get(cat, {})
        trt_stats = tensorrt_stats["by_category"].get(cat, {})

        if not py_stats or not trt_stats:
            continue

        py_mean = py_stats["mean"]
        trt_mean = trt_stats["mean"]
        speedup = py_mean / trt_mean
        improvement = ((py_mean - trt_mean) / py_mean) * 100

        if speedup > 1:
            speedup_str = f"{speedup:.2f}x ✅"
            impr_str = f"{improvement:+.1f}%"
        else:
            speedup_str = f"{speedup:.2f}x ⚠️"
            impr_str = f"{improvement:+.1f}%"

        print(f"{cat:<12} {py_mean:<12.3f} {trt_mean:<12.3f} {speedup_str:<12} {impr_str:<15}")

    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)

    # Calculate overall speedup
    py_overall = []
    trt_overall = []

    for cat in categories:
        py_stats = pytorch_stats["by_category"].get(cat, {})
        trt_stats = tensorrt_stats["by_category"].get(cat, {})

        if py_stats and trt_stats:
            py_overall.extend([py_stats["mean"]] * py_stats["count"])
            trt_overall.extend([trt_stats["mean"]] * trt_stats["count"])

    if py_overall and trt_overall:
        import statistics
        overall_speedup = statistics.mean(py_overall) / statistics.mean(trt_overall)
        overall_improvement = ((statistics.mean(py_overall) - statistics.mean(trt_overall)) / statistics.mean(py_overall)) * 100

        print(f"\n{'='*100}")

        if overall_speedup > 1.0:
            print(f"✅ TensorRT is {overall_speedup:.2f}x FASTER on average")
            print(f"   Average latency improvement: {overall_improvement:.1f}%")
        else:
            print(f"⚠️ TensorRT is {1/overall_speedup:.2f}x SLOWER on average")
            print(f"   This might be due to:")
            print(f"   - Overhead of dynamic shape handling")
            print(f"   - Batch size not optimized for TensorRT")
            print(f"   - Warmup needed (first run is slower)")

    print("\n" + "="*100)


async def main():
    """Run TensorRT benchmark."""
    print("\n" + "="*100)
    print(f"{'S3GEN TENSORRT BENCHMARK':^100}")
    print("="*100)

    # Check requirements
    errors = check_tensorrt_requirements()
    if errors:
        print("\n" + "\n".join(errors))
        print("\nPlease install required dependencies and try again.")
        return 1

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    # Test texts
    texts = [
        "Hello, this is a test.",
        "This is a medium length text for testing the text to speech synthesis system.",
        "This is a longer text passage designed to test the upper limits.",
    ]

    num_runs = 3
    engine_path = "trt_engines/s3gen_decoder.engine"

    # Check if engine exists
    if not Path(engine_path).exists():
        print(f"\n⚠️ TensorRT engine not found: {engine_path}")
        print("\nTo build the TensorRT engine:")
        print("  1. Install TensorRT: pip install tensorrt")
        print("  2. Export model to ONNX:")
        print("     python build_s3gen_tensorrt.py --export-onnx")
        print("  3. Build TensorRT engine with trtexec:")
        print("     trtexec --onnx=trt_engines/s3gen_decoder.onnx \\")
        print("            --saveEngine=trt_engines/s3gen_decoder.engine \\")
        print("            --fp16 \\")
        print("            --workspace=1024MB")
        print("\nFor now, benchmarking PyTorch only...")
        tensorrt_stats = None
    else:
        print(f"\n✓ TensorRT engine found: {engine_path}")

        # Benchmark PyTorch baseline
        print("\n" + "="*100)
        print("STEP 1: BENCHMARK PYTORCH (BASELINE)")
        print("="*100)

        pytorch_stats = await benchmark_mode(
            use_tensorrt=False,
            num_runs=num_runs,
            texts=texts,
        )

        # Benchmark TensorRT
        print("\n" + "="*100)
        print("STEP 2: BENCHMARK TENSORRT (OPTIMIZED)")
        print("="*100)

        tensorrt_stats = await benchmark_mode(
            use_tensorrt=True,
            engine_path=engine_path,
            num_runs=num_runs,
            texts=texts,
        )

        # Print comparison
        print_comparison(pytorch_stats, tensorrt_stats)

        return 0

    # If no TensorRT engine, just benchmark PyTorch
    print("\n" + "="*100)
    print("BENCHMARKING PYTORCH ONLY")
    print("="*100)

    pytorch_stats = await benchmark_mode(
        use_tensorrt=False,
        num_runs=num_runs,
        texts=texts,
    )

    print(f"\n✓ PyTorch baseline established")
    print(f"  Build TensorRT engine to see potential speedup")
    print(f"  Expected: 2-3x speedup with TensorRT")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
