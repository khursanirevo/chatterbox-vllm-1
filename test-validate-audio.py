#!/usr/bin/env python3
"""
Validate that audio generation produces correct speech from input text.

This script:
1. Generates audio using the existing ChatterboxTTS
2. Saves output to file
3. Provides instructions for validation

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test-validate-audio.py
"""

import os
import time
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS
from chatterbox_vllm.models.s3tokenizer import S3_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def validate_audio_generation():
    """Generate audio and provide validation instructions."""
    print("="*70)
    print("AUDIO VALIDATION TEST")
    print("="*70)

    # Test cases with different texts
    test_cases = [
        {
            "text": "Hello world, this is a test of the text to speech system.",
            "filename": "test-validation-hello.wav",
            "max_tokens": 300,
        },
        {
            "text": "The quick brown fox jumps over the lazy dog.",
            "filename": "test-validation-fox.wav",
            "max_tokens": 200,
        },
        {
            "text": "This is a longer sentence to test the quality and continuity of the speech generation across multiple chunks.",
            "filename": "test-validation-long.wav",
            "max_tokens": 500,
        },
    ]

    print(f"\nRunning {len(test_cases)} validation tests...\n")

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_cases)}")
        print(f"{'='*70}\n")

        text = test["text"]
        output_file = test["filename"]
        max_tokens = test["max_tokens"]

        print(f"Text:     '{text}'")
        print(f"Output:   {output_file}")
        print(f"Max tokens: {max_tokens}\n")

        # Initialize TTS model
        print("Initializing ChatterboxTTS...")
        start_init = time.time()

        model = ChatterboxTTS.from_pretrained(
            max_model_len=max_tokens,
            gpu_memory_utilization=0.90,
        )

        init_time = time.time() - start_init
        print(f"✓ Model initialized in {init_time:.2f}s\n")

        # Generate audio
        print("Generating audio...")
        start_gen = time.time()

        audio_chunks = []
        metrics = None

        for audio_chunk, chunk_metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=True,
        ):
            audio_chunks.append(audio_chunk)
            metrics = chunk_metrics

        gen_time = time.time() - start_gen

        # Concatenate audio chunks
        audio = torch.cat(audio_chunks, dim=-1)
        duration = audio.shape[-1] / S3_SR

        # Save audio (torchaudio.save takes filepath, tensor, sample_rate)
        ta.save(str(output_file), audio, S3_SR)
        file_size = Path(output_file).stat().st_size / 1024

        print(f"\n✓ Audio saved: {output_file}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Generation time: {gen_time:.2f}s")
        print(f"  RTF: {gen_time/duration:.3f}")

        # Store results
        results.append({
            "test": i,
            "text": text,
            "file": output_file,
            "duration": duration,
            "size_kb": file_size,
            "gen_time": gen_time,
            "rtf": gen_time/duration,
        })

        # Cleanup for next test
        model.shutdown()
        print(f"\n✓ Test {i} complete")

        # Brief pause between tests
        if i < len(test_cases):
            time.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    print(f"\n✅ All {len(test_cases)} audio files generated successfully!\n")

    print("Generated files:")
    for r in results:
        print(f"  {r['test']}. {r['file']}")
        print(f"     Text: '{r['text']}'")
        print(f"     Duration: {r['duration']:.2f}s, RTF: {r['rtf']:.3f}")
        print()

    print("="*70)
    print("VALIDATION INSTRUCTIONS")
    print("="*70)
    print("\n1. Play each audio file and verify the spoken text matches:")
    print()

    for r in results:
        print(f"\n📁 File {r['test']}: {r['file']}")
        print(f"   🎵 Play: ffplay {r['file']}")
        print(f"   📝 Expected text: '{r['text']}'")

    print("\n" + "="*70)
    print("VALIDATION CHECKLIST")
    print("="*70)
    print("""
For each audio file, verify:
  ☐ The spoken words match the input text
  ☐ The voice sounds natural and clear
  ☐ There are no glitches or pauses between chunks
  ☐ The intonation and rhythm sound appropriate
  ☐ No background noise or artifacts

If all items pass, the audio generation is working correctly!
    """)

    print("="*70)
    print("PERFORMANCE METRICS")
    print("="*70)

    avg_rtf = sum(r["rtf"] for r in results) / len(results)
    avg_gen_time = sum(r["gen_time"] for r in results) / len(results)

    print(f"\nAverage RTF: {avg_rtf:.3f} (lower is better, <1.0 = faster than real-time)")
    print(f"Average generation time: {avg_gen_time:.2f}s")

    print("\nNote: This is the SYNCHRONOUS implementation.")
    print("      The ASYNC version would achieve ~767ms first chunk latency.")
    print("      (Current: ~3.4s first chunk)")

    return results


if __name__ == "__main__":
    try:
        results = validate_audio_generation()

        print("\n" + "="*70)
        print("✅ VALIDATION COMPLETE")
        print("="*70)
        print("\nPlease play the audio files to validate the output is correct.")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
