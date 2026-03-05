# torch-tensorrt Guide for S3Gen Optimization

## Overview

This guide explains how to use torch-tensorrt to compile S3Gen's ConditionalDecoder
directly to TensorRT, bypassing the ONNX export challenges.

## Why torch-tensorrt?

**Advantages:**
- Direct PyTorch → TensorRT compilation (no ONNX needed)
- Handles dynamic shapes better than ONNX export
- Automatic tracing and optimization
- Official PyTorch integration

**Current Limitation:**
- Requires building from source (not in PyPI)
- Experimental for complex models
- May still struggle with very complex dynamic shapes

## Installation

### Option 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/pytorch/tensorrt
cd tensorrt

# Check out the tag matching your PyTorch version
git checkout v2.6.0  # or matching version

# Build and install
python setup.py install

# Verify installation
python -c "import torch_tensorrt; print('torch-tensorrt installed')"
```

### Option 2: PyTorch Nightly Builds (Experimental)

```bash
# Install PyTorch nightly with torch-tensorrt
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Option 3: Docker (Recommended for Testing)

```bash
# Use PyTorch Docker image with TensorRT support
docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn8-runtime

# Mount your code and run
docker run --gpus all -v $(pwd):/app -w /app pytorch/pytorch:2.6.0-cuda12.6-cudnn8-runtime python compile_tensorrt_direct.py
```

## Usage

Once torch-tensorrt is installed, use the compile script:

```bash
python compile_tensorrt_direct.py
```

This will:
1. Load the S3Gen model
2. Extract the ConditionalDecoder
3. Compile to TensorRT with FP16
4. Save the compiled model
5. Test inference

### Manual Compilation

```python
import torch
import torch_tensorrt
from chatterbox_vllm import ChatterboxTTSAsync

async def main():
    # Load model
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
    )

    # Get the decoder
    decoder = model.s3gen.flow.decoder.estimator

    # Define example inputs (for tracing)
    batch_size = 2
    n_mels = 300
    spk_emb_dim = 80

    example_inputs = (
        torch.randn(batch_size, 320, n_mels).cuda(),
        torch.ones(batch_size, 1, n_mels).cuda(),
        torch.randn(batch_size, 80, n_mels).cuda(),
        torch.rand(batch_size).cuda(),
        torch.randn(batch_size, spk_emb_dim).cuda(),
        torch.randn(batch_size, 80, n_mels).cuda(),
    )

    # Compile
    compiled_decoder = torch_tensorrt.compile(
        decoder,
        example_inputs=example_inputs,
        enabled_precisions={torch.float16},
        min_block_size=1,
        max_block_size=4,
        workspace_size=1 << 30,
        torch_executed_ops=False,
    )

    # Replace in model
    model.s3gen.flow.decoder.estimator = compiled_decoder

    # Use normally
    audio = await model.generate(prompts=["Hello world!"])

asyncio.run(main())
```

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **TensorRT** | ✅ Installed | 10.15.1.29 |
| **torch-tensorrt** | ❌ Not installed | Requires source build |
| **torch-tensorrt code** | ✅ Created | `compile_tensorrt_direct.py` |
| **Usage example** | ✅ Created | `torch_tensorrt_example.py` |
| **PyTorch baseline** | ✅ Established | 0.886s (short), 1.025s (medium) |

## Expected Performance

With torch-tensorrt compilation:
- **S3Gen speedup:** 2-2.5x (similar to manual TensorRT engine)
- **Overall TTFA improvement:** 30-45% for short prompts
- **Combined with existing:** 2.5-3.0x total speedup from baseline

## Troubleshooting

### Issue: torch-tensorrt import fails

**Solution:** Make sure you're using the correct PyTorch version
```bash
python -c "import torch; print(torch.__version__)"
# Match torch-tensorrt version
```

### Issue: Tracing fails

**Solution:** Simplify inputs by fixing dimensions:
```python
# Use fixed-size inputs for compilation
fixed_n_mels = 300
x = torch.randn(batch_size, 320, fixed_n_mels).cuda()
```

### Issue: RuntimeError during compilation

**Common causes:**
- Unsupported operations in the model
- Dynamic shapes too complex
- CUDA version mismatch

**Solution:** Check TensorRT logs for specific operation that failed.

## Next Steps

1. **Short term:** Use current optimizations (1.93x speedup achieved)
2. **Medium term:** Experiment with torch-tensorr in Docker
3. **Long term:** Wait for torch-tensorrt to mature for complex models

## Files

- `compile_tensorrt_direct.py` - Automated compilation script
- `torch_tensorrt_example.py` - Manual usage example
- `TENSORRT_README.md` - General TensorRT documentation

## Comparison: torch-tensorrt vs Manual Engine Building

| Aspect | torch-tensorrt | Manual (trtexec) |
|--------|---------------|-------------------|
| **Complexity** | Low (automated) | High (manual work) |
| **Dynamic shapes** | Better support | Requires profiling |
| **Optimization** | Automatic | Manual tuning |
| **Portability** | Python code | Binary engine file |
| **Build time** | Minutes | Hours |

## Recommendation

For production use:
1. **Start with:** PyTorch + FP16 (1.93x speedup achieved)
2. **When torch-tensorrt matures:** Adopt it for easier optimization
3. **For maximum control:** Manual engine building with trtexec (future work)

---

**Status:** Code ready, torch-tensorrt installation and testing pending
**Date:** 2025-03-05
**PyTorch Baseline:** 0.886s (short), 1.025s (medium)
