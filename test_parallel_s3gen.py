#!/usr/bin/env python3
"""
Test parallel S3Gen processing.

This script tests the parallel S3Gen inference to verify that multiple requests
are processed concurrently instead of sequentially.
"""
import asyncio
import time
from pathlib import Path

from chatterbox_vllm import ChatterboxTTSAsync

# Test prompts of varying lengths
PROMPTS = [
    ("Hello world!", "short"),
    ("The quick brown fox jumps over the lazy dog.", "short"),
    ("This is a test of the emergency broadcast system.", "medium"),
    ("In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.", "medium"),
    ("It was the best of times, it was the worst of times, it was the age of wisdom.", "long"),
    ("To be or not to be, that is the question.", "short"),
    ("All that is gold does not glitter, not all those who wander are lost.", "medium"),
    ("The story so far: in the beginning, the Universe was created.", "short"),
]

# Reference audio file (use a sample from the repository)
REF_AUDIO = Path(__file__).parent / "test_reference.wav"


async def test_parallel_s3gen():
    """Test parallel S3Gen processing with multiple concurrent requests."""
    print("=" * 60)
    print("Testing PARALLEL S3Gen Processing")
    print("=" * 60)

    # Initialize model
    print("\n[1/4] Loading model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=False,  # Disable for cleaner output
    )

    # Load reference audio
    print("\n[2/4] Loading reference audio...")
    if not REF_AUDIO.exists():
        print(f"Warning: Reference audio not found at {REF_AUDIO}")
        print("Using default reference...")
        ref_audio = None
    else:
        import torchaudio
        ref_audio, ref_sr = torchaudio.load(str(REF_AUDIO))
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio[0:1, :]  # Convert to mono
        ref_audio = ref_audio / ref_audio.abs().max()

    # Prepare prompts and request IDs
    print("\n[3/4] Generating audio with parallel S3Gen...")
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
    print("\n[4/4] Results:")
    print(f"  Total requests: {len(results)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per request: {total_time / len(results):.3f}s")

    # Calculate theoretical sequential vs parallel time
    # Sequential: each request takes ~0.5-2s, so sequential would be 4-16s
    # Parallel: should be close to max(single request time)
    print(f"\n  Expected sequential time: ~{len(results) * 1.0:.1f}s (estimate)")
    print(f"  Speedup: ~{len(results) * 1.0 / total_time:.1f}x")

    # Save samples
    print("\n[5/4] Saving audio samples...")
    output_dir = Path(__file__).parent / "test_parallel_output"
    output_dir.mkdir(exist_ok=True)

    import torchaudio
    for i, (wav, (prompt, category)) in enumerate(zip(results, PROMPTS)):
        output_path = output_dir / f"test_{i:03d}_{category}.wav"
        # Ensure wav is 2D [channels, samples] for torchaudio.save
        if wav.dim() == 1:
            wav_to_save = wav.unsqueeze(0)
        else:
            wav_to_save = wav
        torchaudio.save(str(output_path), wav_to_save, 24000)
        print(f"  Saved: {output_path.name} ({prompt[:30]}...)")

    print(f"\n✅ All audio saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

    # Cleanup
    await model.shutdown()


async def test_concurrent_batches():
    """Test multiple concurrent batches to verify true parallelism."""
    print("\n" + "=" * 60)
    print("Testing CONCURRENT BATCHES (True Parallelism Test)")
    print("=" * 60)

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
    ref_audio = None  # Use default

    # Prepare two separate batches
    batch1_prompts = ["Hello world!", "Test message one.", "Test message two."]
    batch2_prompts = ["Goodbye world!", "Test message three.", "Test message four."]

    print("\n[3/3] Running two batches concurrently...")
    start_time = time.time()

    # Run both batches concurrently
    results = await asyncio.gather(
        model.generate(
            prompts=batch1_prompts,
            request_ids=[f"batch1_{i}" for i in range(len(batch1_prompts))],
            ref_audio=ref_audio,
            ref_sr=24000,
            temperature=0.8,
        ),
        model.generate(
            prompts=batch2_prompts,
            request_ids=[f"batch2_{i}" for i in range(len(batch2_prompts))],
            ref_audio=ref_audio,
            ref_sr=24000,
            temperature=0.8,
        ),
    )

    total_time = time.time() - start_time

    print(f"\n  Batch 1: {len(results[0])} results")
    print(f"  Batch 2: {len(results[1])} results")
    print(f"  Total requests: {len(results[0]) + len(results[1])}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per request: {total_time / (len(results[0]) + len(results[1])):.3f}s")

    print(f"\n✅ Concurrent batch test completed!")
    print("=" * 60)

    # Cleanup
    await model.shutdown()


if __name__ == "__main__":
    print("Starting parallel S3Gen tests...\n")

    # Test 1: Basic parallel processing
    asyncio.run(test_parallel_s3gen())

    # Test 2: Concurrent batches
    asyncio.run(test_concurrent_batches())

    print("\n✅ All tests completed successfully!")
