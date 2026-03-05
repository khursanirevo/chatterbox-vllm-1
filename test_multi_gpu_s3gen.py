#!/usr/bin/env python3
"""
Test multi-GPU S3Gen processing.

This script tests parallel S3Gen inference across multiple GPUs.
"""
import asyncio
import time
import torch
from pathlib import Path

from chatterbox_vllm import ChatterboxTTSAsync

# Test prompts of varying lengths
PROMPTS = [
    ("Hello world!", "short"),
    ("The quick brown fox jumps over the lazy dog.", "short"),
    ("This is a test of the emergency broadcast system.", "medium"),
    ("In a hole in the ground there lived a hobbit.", "medium"),
    ("It was the best of times, it was the worst of times.", "long"),
    ("To be or not to be, that is the question.", "short"),
    ("All that is gold does not glitter.", "medium"),
    ("The story so far: in the beginning, the Universe was created.", "short"),
]

# Reference audio file (use a sample from the repository)
REF_AUDIO = Path(__file__).parent / "test_reference.wav"


async def test_multi_gpu_s3gen():
    """Test multi-GPU S3Gen processing."""
    print("=" * 80)
    print("MULTI-GPU S3Gen TEST")
    print("=" * 80)

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

    if num_gpus < 2:
        print("\n⚠️  WARNING: Multi-GPU test requires at least 2 GPUs")
        print("   Falling back to single-GPU mode")

    # Initialize model
    print("\n[1/3] Loading model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=False,
    )

    # Load reference audio
    print("\n[2/3] Loading reference audio...")
    if not REF_AUDIO.exists():
        print(f"Warning: Reference audio not found at {REF_AUDIO}")
        print("Using default reference...")
        ref_audio = None
    else:
        import torchaudio
        ref_audio, ref_sr = torchaudio.load(str(REF_AUDIO))
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio[0:1, :]
        ref_audio = ref_audio / ref_audio.abs().max()

    # Prepare prompts and request IDs
    print("\n[3/3] Generating audio with multi-GPU S3Gen...")
    request_ids = [f"test_req_{i:03d}" for i in range(len(PROMPTS))]
    prompts = [p[0] for p in PROMPTS]

    # Measure time
    start_time = time.time()

    # Generate audio
    results = await model.generate(
        prompts=prompts,
        request_ids=request_ids,
        ref_audio=ref_audio,
        ref_sr=24000 if ref_audio is not None else None,
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

    # Calculate expected vs actual
    if num_gpus >= 2:
        expected_sequential = len(results) * 0.6  # Estimated
        speedup = expected_sequential / total_time
        print(f"\nEstimated speedup:  {speedup:.2f}x")
        print(f"(Using {num_gpus} GPUs in parallel)")

    # Save samples
    print("\n[4/3] Saving audio samples...")
    output_dir = Path(__file__).parent / "test_multi_gpu_output"
    output_dir.mkdir(exist_ok=True)

    import torchaudio
    for i, (wav, (prompt, category)) in enumerate(zip(results, PROMPTS)):
        output_path = output_dir / f"test_{i:03d}_{category}.wav"
        if wav.dim() == 1:
            wav_to_save = wav.unsqueeze(0)
        else:
            wav_to_save = wav
        torchaudio.save(str(output_path), wav_to_save, 24000)
        print(f"  Saved: {output_path.name} ({prompt[:30]}...)")

    print(f"\n✅ All audio saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)

    # Cleanup
    await model.shutdown()


async def test_gpu_utilization():
    """Test GPU utilization during multi-GPU S3Gen."""
    print("\n" + "=" * 80)
    print("GPU UTILIZATION TEST")
    print("=" * 80)

    import subprocess

    # Get initial GPU utilization
    def get_gpu_stats():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        stats = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_id = int(parts[0])
                stats[gpu_id] = {
                    'util': float(parts[1]),
                    'mem': float(parts[2]),
                }
        return stats

    print("\nInitial GPU state:")
    initial_stats = get_gpu_stats()
    for gpu_id, stats in initial_stats.items():
        print(f"  GPU {gpu_id}: {stats['util']}% util, {stats['mem']} MiB memory")

    # Run a quick test
    print("\nRunning multi-GPU S3Gen test...")
    await test_multi_gpu_s3gen()

    print("\nFinal GPU state:")
    final_stats = get_gpu_stats()
    for gpu_id, stats in final_stats.items():
        print(f"  GPU {gpu_id}: {stats['util']}% util, {stats['mem']} MiB memory")

    print("\n✅ GPU utilization test completed!")


if __name__ == "__main__":
    print("Starting multi-GPU S3Gen tests...\n")

    # Test 1: Basic multi-GPU processing
    asyncio.run(test_multi_gpu_s3gen())

    # Test 2: GPU utilization
    print("\n" + "=" * 80)
    asyncio.run(test_gpu_utilization())

    print("\n✅ All tests completed successfully!")
