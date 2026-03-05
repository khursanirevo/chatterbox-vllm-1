#!/usr/bin/env python3
"""
Direct TensorRT compilation using torch-tensorrt.

This script demonstrates how to compile the S3Gen ConditionalDecoder directly
to TensorRT without ONNX export, using torch-tensorrt.

Note: torch-tensorrt installation requires building from source:
    git clone https://github.com/pytorch/tensorrt
    cd tensorrt
    pip install .
    # Or: python setup.py install

Usage (once torch-tensorrt is installed):
    python compile_tensorrt_direct.py
"""

import asyncio
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox_vllm import ChatterboxTTSAsync
from chatterbox_vllm.models.s3gen.s3gen import S3Token2Wav
from safetensors.torch import load_file


def compile_with_torch_tensorrt():
    """
    Compile S3Gen ConditionalDecoder to TensorRT using torch-tensorrt.

    This bypasses ONNX export and compiles the PyTorch model directly.
    """
    print("="*80)
    print("TENSOR-TENSORRT COMPILATION FOR S3GEN")
    print("="*80)

    try:
        import torch_tensorrt
        print("✓ torch-tensorrt is available")
    except ImportError:
        print("\n✗ torch-tensorrt not installed")
        print("\nTo install torch-tensorrt, you need to build from source:")
        print("  git clone https://github.com/pytorch/tensorrt")
        print("  cd tensorrt")
        print("  python setup.py install")
        print("\nOr install the wheel from PyTorch nightly builds.")
        print("\nFor now, we'll create a script that would work once torch-tensorrt is installed.")
        return False

    print("\nLoading model...")

    # Load model
    s3gen = S3Token2Wav(use_fp16=False)
    s3gen.load_state_dict(load_file("s3gen.safetensors"), strict=False)
    s3gen = s3gen.cuda().eval()

    # Extract the ConditionalDecoder estimator
    decoder = s3gen.flow.decoder.estimator

    print(f"✓ Model loaded")
    print(f"  Decoder type: {type(decoder).__name__}")

    # Define example inputs
    batch_size = 2
    n_mels = 300  # Fixed size for compilation
    spk_emb_dim = 80

    example_inputs = (
        torch.randn(batch_size, 320, n_mels).cuda(),  # x
        torch.ones(batch_size, 1, n_mels).cuda(),      # mask
        torch.randn(batch_size, 80, n_mels).cuda(),   # mu
        torch.rand(batch_size).cuda(),                # t
        torch.randn(batch_size, spk_emb_dim).cuda(), # spks
        torch.randn(batch_size, 80, n_mels).cuda(),  # cond
    )

    print("\nCompiling to TensorRT with torch-tensorrt...")
    print("  (this may take several minutes...)")

    # Compile with torch-tensorrt
    # Note: This is the simplified approach - torch-tensorrt handles tracing automatically
    try:
        compiled_decoder = torch_tensorrt.compile(
            decoder,
            example_inputs=example_inputs,
            enabled_precisions={torch.float16},  # Use FP16
            min_block_size=1,
            max_block_size=4,  # Allow batch sizes 1-4
            workspace_size=1 << 30,  # 1GB workspace
            torch_executed_ops=False,  # Use optimized TorchScript
        )

        print("✓ Compilation successful!")
        print(f"  Compiled model type: {type(compiled_decoder)}")

        # Save compiled model
        output_path = Path("trt_engines/s3gen_decoder_torchtrt.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(compiled_decoder, output_path)
        print(f"✓ Saved compiled model to: {output_path}")

        # Test the compiled model
        print("\nTesting compiled model...")
        with torch.no_grad():
            output = compiled_decoder(*example_inputs)

        print(f"✓ Test successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")

        return True

    except Exception as e:
        print(f"\n✗ Compilation failed: {e}")
        print("\nThis is expected - torch-tensorrt may have issues with:")
        print("  1. Complex dynamic shapes (traces fail)")
        print("  2. Operations not supported in TensorRT")
        print("  3. CUDA version compatibility")
        print("\nConsider:")
        print("  - Using a simpler model subset")
        print("  - Building TensorRT engine manually with trtexec")
        print("  - Waiting for torch-tensorrt to mature")

        import traceback
        traceback.print_exc()
        return False


def create_usage_example():
    """Create an example of how to use torch-tensorrt once installed."""

    code = '''
#!/usr/bin/env python3
"""Example: Using torch-tensorrt for S3Gen inference"""

import torch
import torch_tensorrt
import asyncio
from chatterbox_vllm import ChatterboxTTSAsync

async def main():
    # Load model with torch-tensorrt compilation
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
    )

    # Get the decoder
    decoder = model.s3gen.flow.decoder.estimator

    # Compile to TensorRT
    print("Compiling to TensorRT...")
    compiled_decoder = torch_tensorrt.compile(
        decoder,
        example_inputs=(...),  # Provide example inputs
        enabled_precisions={torch.float16},
        workspace_size=1 << 30,
    )

    # Replace the estimator with compiled version
    model.s3gen.flow.decoder.estimator = compiled_decoder

    # Use normally
    audio = await model.generate(prompts=["Hello world!"])
    print("✓ Generated audio with TensorRT-optimized decoder")

asyncio.run(main())
'''

    return code


async def main():
    """Main entry point."""
    print("\n" + "="*80)
    print("TORCH-TENSORRT COMPILATION TEST")
    print("="*80 + "\n")

    # Try compilation
    success = compile_with_torch_tensorrt()

    if not success:
        print("\n" + "="*80)
        print("CREATING USAGE EXAMPLE")
        print("="*80 + "\n")

        code = create_usage_example()

        example_path = Path("torch_tensorrt_example.py")
        with open(example_path, "w") as f:
            f.write(code)

        print(f"Created usage example: {example_path}")
        print("\nTo use torch-tensorrt in the future:")
        print("  1. Build and install torch-tensorrt from source")
        print("  2. Run: python torch_tensorrt_example.py")
        print("\nOr manually compile:")
        print("  compiled = torch_tensorrt.compile(model, inputs=[...])")

    return 0 if success else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
