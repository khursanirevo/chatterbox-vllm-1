#!/usr/bin/env python3
"""
Test script to verify FP16 dtype mismatch fix.

This script tests that the S3Gen model can run with use_fp16=True
without encountering dtype mismatch errors.
"""

import asyncio
import torch
from chatterbox_vllm import ChatterboxTTSAsync
from pathlib import Path


async def test_fp16():
    """Test that FP16 mode works without dtype errors."""

    print("Testing FP16 dtype mismatch fix...")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. FP16 testing requires GPU.")
        return False

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA capability: {torch.cuda.get_device_capability(0)}")

    # Test with FP16 enabled
    print("\nInitializing model with use_fp16=True...")

    try:
        model = await ChatterboxTTSAsync.from_pretrained(
            max_batch_size=16,
            max_model_len=1000,
            s3gen_use_fp16=True,  # Enable FP16
            s3gen_compile_model=False,
        )
        print("✓ Model initialized successfully with FP16")

        # Check that layers are in fp16
        s3gen = model.s3gen
        flow = s3gen.flow

        affine_dtype = flow.spk_embed_affine_layer.weight.dtype
        proj_dtype = flow.encoder_proj.weight.dtype
        emb_dtype = flow.input_embedding.weight.dtype

        print(f"\nLayer dtypes:")
        print(f"  spk_embed_affine_layer: {affine_dtype}")
        print(f"  encoder_proj: {proj_dtype}")
        print(f"  input_embedding: {emb_dtype}")

        if affine_dtype == torch.float16:
            print("✓ Affine layers correctly converted to FP16")
        else:
            print(f"✗ ERROR: Expected float16, got {affine_dtype}")
            return False

        # Test a simple generation
        print("\nTesting generation with FP16...")
        text = "Hello, this is a test of the FP16 mode."

        # Use default reference audio for simplicity
        results = await model.generate(
            prompts=[text],
            audio_prompt_path=None,  # Use default reference
            temperature=0.8,
            exaggeration=0.5,
        )

        if results and len(results) > 0:
            print("✓ Generation successful with FP16!")
            print(f"  Generated audio shape: {results[0].shape}")
            print(f"  Generated audio dtype: {results[0].dtype}")
            return True
        else:
            print("✗ ERROR: Generation returned no results")
            return False

    except RuntimeError as e:
        if "dtype" in str(e) or "Half and Float" in str(e):
            print(f"\n✗ FP16 dtype mismatch ERROR:")
            print(f"  {e}")
            print("\nThe FP16 fix did NOT work correctly.")
            return False
        else:
            print(f"\n✗ Runtime error (not dtype related):")
            print(f"  {e}")
            raise
    except Exception as e:
        print(f"\n✗ Unexpected error:")
        print(f"  {type(e).__name__}: {e}")
        raise


async def main():
    success = await test_fp16()

    print("\n" + "=" * 60)
    if success:
        print("✅ FP16 TEST PASSED!")
        print("\nThe dtype mismatch has been fixed.")
        print("You can now use s3gen_use_fp16=True for 20-30% speedup.")
    else:
        print("❌ FP16 TEST FAILED!")
        print("\nThe dtype mismatch fix needs further investigation.")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
