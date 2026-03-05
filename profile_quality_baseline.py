#!/usr/bin/env python3
"""
Quality Baseline Profiler with n_timesteps=10

Generates audio samples with n_timesteps=10 (highest quality) for comparison.
Saves audio files for listening tests.

Use this to compare against n_timesteps=5 for quality degradation.
"""

import asyncio
import time
import torch
import torchaudio as ta
from pathlib import Path
from typing import List

from chatterbox_vllm import ChatterboxTTSAsync


# Test texts with varying complexity
TEST_CASES = {
    "short": "Hello, how are you today?",
    "medium": "The weather today is quite nice, with clear skies and mild temperatures expected throughout the day.",
    "long": "This is a significantly longer text passage designed to test the text to speech synthesis pipeline with more content to process through multiple stages including tokenization, language model inference, and audio decoding.",
    "punctuation": "Hello! How are you? I'm doing great, thanks for asking. What about you?",
    "numbers": "The price is $19.99, and it will arrive in 3 to 5 business days.",
    "emotional": "I am so excited to share this wonderful news with everyone! This is absolutely amazing!",
    "question": "Really? That's quite surprising, isn't it?",
}


async def profile_quality_baseline(
    n_timesteps: int = 10,
    output_dir: str = "quality_baseline_samples",
) -> dict:
    """
    Generate audio samples for quality comparison.

    Args:
        n_timesteps: Number of diffusion steps (10 for baseline, 5 for comparison)
        output_dir: Directory to save audio samples

    Returns:
        Dictionary with generation times and file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_name = f"n_timesteps_{n_timesteps}"

    print(f"\n{'='*80}")
    print(f"QUALITY BASELINE PROFILING: {config_name}")
    print(f"{'='*80}")
    print(f"\nOutput directory: {output_path.absolute()}")

    # Initialize model
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=False,
    )

    results = {}

    try:
        for name, text in TEST_CASES.items():
            print(f"\n[{name.upper()}]")
            print(f"  Text: {text[:60]}{'...' if len(text) > 60 else ''}")

            start_time = time.time()

            result = await model.generate(
                prompts=[text],
                temperature=0.8,
                exaggeration=0.5,
                max_tokens=1000,
                diffusion_steps=n_timesteps,
            )

            end_time = time.time()
            latency = end_time - start_time

            print(f"  Latency: {latency:.3f}s")

            # Save audio
            audio = result[0]
            filename = output_path / f"{config_name}_{name}.wav"

            # Convert to proper format
            audio_data = audio.cpu().squeeze().numpy()
            sample_rate = 24000  # S3GEN_SR

            ta.save(
                str(filename),
                torch.from_numpy(audio_data).unsqueeze(0),
                sample_rate,
            )

            print(f"  Saved: {filename.name}")

            # Store metadata
            results[name] = {
                "text": text,
                "latency": latency,
                "audio_file": str(filename),
                "sample_rate": sample_rate,
                "duration_samples": audio.shape[1],
                "duration_seconds": audio.shape[1] / sample_rate,
            }

    finally:
        await model.shutdown()

    # Summary
    print(f"\n{'='*80}")
    print(f"PROFILING SUMMARY: {config_name}")
    print(f"{'='*80}")

    total_latency = sum(r["latency"] for r in results.values())
    avg_latency = total_latency / len(results)

    print(f"\nGenerated {len(results)} audio samples")
    print(f"Total time: {total_latency:.2f}s")
    print(f"Average latency: {avg_latency:.3f}s")

    print(f"\n{'Test Case':<15} {'Latency':<10} {'Duration':<10} {'File'}")
    print("-"*80)

    for name, data in results.items():
        print(f"{name:<15} {data['latency']:.3f}s     {data['duration_seconds']:.2f}s      {data['audio_file']}")

    print(f"\n{'='*80}")
    print(f"All samples saved to: {output_path.absolute()}")
    print(f"{'='*80}\n")

    # Save metadata
    import json
    metadata_file = output_path / f"{config_name}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Metadata saved to: {metadata_file}")

    return results


async def main():
    """Run quality baseline profiling."""
    # Test with n_timesteps=10 (baseline)
    results_10 = await profile_quality_baseline(n_timesteps=10)

    # Optionally test with n_timesteps=5 for comparison
    print("\n" + "="*80)
    print("Would you like to generate comparison samples with n_timesteps=5?")
    print("This will allow you to listen and compare quality.")
    print("="*80)
    print("\nTo generate comparison samples later, run:")
    print("  uv run python profile_quality_baseline.py --compare")
    print("\nFor now, only generating n_timesteps=10 baseline.\n")

    return results_10


if __name__ == "__main__":
    import sys

    # Check for --compare flag
    if "--compare" in sys.argv:
        async def compare_both():
            print("\n" + "="*80)
            print("GENERATING COMPARISON SAMPLES")
            print("="*80)
            print("\nGenerating samples with both n_timesteps=10 and n_timesteps=5")
            print("for quality comparison.\n")

            await profile_quality_baseline(n_timesteps=10)
            print("\n" + "-"*80 + "\n")
            await profile_quality_baseline(n_timesteps=5, output_dir="quality_baseline_samples_comparison")

        asyncio.run(compare_both())
    else:
        results = asyncio.run(main())
