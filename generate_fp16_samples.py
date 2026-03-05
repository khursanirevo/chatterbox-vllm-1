#!/usr/bin/env python3
"""
Generate audio samples for FP16 mode quality testing.

This script creates short, medium, and long audio samples using FP16 mode
so you can listen to the output quality.
"""

import asyncio
import torch
import torchaudio as ta
from chatterbox_vllm import ChatterboxTTSAsync
from pathlib import Path


async def generate_samples():
    """Generate audio samples with FP16 mode."""

    print("="*60)
    print("Generating FP16 Audio Samples")
    print("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA required for FP16 mode")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    # Initialize model with FP16
    print("\nInitializing model with FP16 enabled...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,  # Enable FP16
        s3gen_compile_model=False,
    )

    # Verify FP16 is enabled
    flow = model.s3gen.flow
    affine_dtype = flow.spk_embed_affine_layer.weight.dtype
    print(f"✓ Model precision: {affine_dtype}")

    if affine_dtype != torch.float16:
        print(f"✗ ERROR: Expected FP16 but got {affine_dtype}")
        return

    # Create output directory
    output_dir = Path("fp16_samples")
    output_dir.mkdir(exist_ok=True)

    # Test prompts
    prompts = {
        "short": "Hello, this is a test of the FP16 mode for Chatterbox text-to-speech synthesis.",
        "medium": "This is a medium length text that will help us evaluate the quality and performance of the FP16 optimization. The speech should sound natural and clear, with no artifacts or distortions that might indicate precision issues.",
        "long": "This is a much longer text designed to thoroughly test the FP16 implementation across extended speech synthesis. When we optimize machine learning models for lower precision like FP16, we need to ensure that the audio quality remains high throughout the entire generation process. This longer sample will help us verify that there are no cumulative precision errors or quality degradation that might only become apparent with longer texts. The speech should maintain consistent quality, clarity, and naturalness from beginning to end, demonstrating that the FP16 optimization is working correctly without compromising the output quality.",
    }

    print("\n" + "="*60)
    print("Generating Audio Samples")
    print("="*60)

    for category, text in prompts.items():
        print(f"\n{category.upper()} PROMPT:")
        print(f"  {text[:80]}...")
        print(f"\nGenerating...")

        try:
            audio = await model.generate(
                prompts=[text],
                audio_prompt_path=None,
                temperature=0.8,
                exaggeration=0.5,
            )

            if audio and len(audio) > 0:
                # Save audio
                output_path = output_dir / f"fp16_{category}.wav"
                ta.save(str(output_path), audio[0], model.sr)

                duration = audio[0].shape[1] / model.sr
                print(f"  ✓ Saved to: {output_path}")
                print(f"  ✓ Duration: {duration:.2f}s")
                print(f"  ✓ Samples: {audio[0].shape[1]}")
                print(f"  ✓ Shape: {audio[0].shape}")
            else:
                print(f"  ✗ FAILED: No audio generated")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("✅ All samples generated successfully!")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.wav")):
        print(f"  - {f.name}")

    print("\nYou can now listen to the samples to verify FP16 quality.")
    print(f"\nPlay with: ffplay {output_dir}/*.wav")


if __name__ == "__main__":
    asyncio.run(generate_samples())
