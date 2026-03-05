
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
