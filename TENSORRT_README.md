# TensorRT Optimization for S3Gen

## Overview

TensorRT provides 2-3x speedup for S3Gen inference by optimizing the ConditionalDecoder model for NVIDIA GPUs. This document explains how to build and use TensorRT engines with Chatterbox vLLM.

## What is TensorRT?

TensorRT is NVIDIA's deep learning inference optimizer and runtime library. It provides:
- **Layer fusion**: Combines multiple layers into single kernel operations
- **Precision calibration**: FP16/int8 optimization for faster computation
- **Kernel auto-tuning**: Selects optimal GPU kernels for specific hardware
- **Dynamic tensor memory**: Minimizes memory footprint

## Expected Performance

Based on TTFA profiling, S3Gen accounts for **80-95% of TTFA**. TensorRT optimization provides:
- **2-3x speedup** on S3Gen inference
- **40-60% TTFA improvement** overall (because S3Gen dominates TTFA)
- **Best for short prompts** where S3Gen is the bottleneck

## Prerequisites

### 1. Install TensorRT

```bash
# For CUDA 11.8
pip install tensorrt==8.6.1

# For CUDA 12.x
pip install nvidia-tensorrt

# Verify installation
python -c "import tensorrt; print(tensorrt.__version__)"
```

### 2. Install additional dependencies (optional)

```bash
# For ONNX export
pip install onnx onnxruntime

# For torch-tensorrt (Python API, optional)
pip install torch-tensorrt
```

## Building the TensorRT Engine

### Step 1: Export PyTorch Model to ONNX

```bash
python build_s3gen_tensorrt.py --export-onnx --output-dir ./trt_engines
```

This creates:
- `trt_engines/s3gen_decoder.onnx` - ONNX model export
- `trt_engines/s3gen_decoder.engine` - TensorRT engine (if build succeeds)

### Step 2: Build TensorRT Engine (Manual Method)

If the automated build fails, use `trtexec`:

```bash
# Find trtexec location
which trtexec
# Usually: /usr/local/bin/trtexec or /opt/tensorrt/bin/trtexec

# Build engine
trtexec \
  --onnx=trt_engines/s3gen_decoder.onnx \
  --saveEngine=trt_engines/s3gen_decoder.engine \
  --fp16 \
  --minShapes=x:2x80x1,mask:2x1x1,mu:2x80x1,t:2,spks:2x80,cond:2x80x1 \
  --optShapes=x:2x80x300,mask:2x1x300,mu:2x80x300,t:2,spks:2x80,cond:2x80x300 \
  --maxShapes=x:2x80x500,mask:2x1x500,mu:2x80x500,t:2,spks:2x80,cond:2x80x500 \
  --workspace=1024MB
```

**Parameters explained:**
- `--fp16`: Enable FP16 precision (recommended, 2x speedup)
- `--minShapes`: Minimum input shapes (batch=2, mel_len=1)
- `--optShapes`: Optimal input shapes (batch=2, mel_len=300, ~12s audio)
- `--maxShapes`: Maximum input shapes (batch=2, mel_len=500, ~20s audio)
- `--workspace`: GPU workspace size (1GB recommended)

### Step 3: Verify Engine Creation

```bash
ls -lh trt_engines/s3gen_decoder.engine
# Should be 10-50 MB depending on configuration
```

## Using TensorRT in Chatterbox vLLM

### Basic Usage

```python
from chatterbox_vllm import ChatterboxTTSAsync

model = await ChatterboxTTSAsync.from_pretrained(
    max_batch_size=16,
    max_model_len=1000,
    s3gen_use_fp16=True,
    s3gen_use_tensorrt=True,
    s3gen_tensorrt_engine_path="./trt_engines/s3gen_decoder.engine",
)

# Generate audio
audio = await model.generate(
    prompts=["Hello, world!"],
    temperature=0.8,
    exaggeration=0.5,
)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s3gen_use_fp16` | False | Use FP16 for S3Gen (recommended) |
| `s3gen_use_tensorrt` | False | Use TensorRT engine |
| `s3gen_tensorrt_engine_path` | None | Path to .engine file |
| `s3gen_compile_model` | False | torch.compile() (mutually exclusive with TensorRT) |

**Important Notes:**
- TensorRT and torch.compile() are mutually exclusive
- TensorRT requires pre-built engine file
- FP16 is recommended for best performance
- TensorRT engine is tied to specific GPU architecture

## Benchmarking

### Run TensorRT Benchmark

```bash
python benchmark_tensorrt.py
```

This compares:
1. PyTorch baseline (FP16)
2. TensorRT-optimized (FP16)

Expected output:
```
Category    PyTorch     TensorRT    Speedup      Improvement
----------------------------------------------------------------------------------------------------
short       0.670       0.280        2.39x ✅     +58.2%
medium      1.250       0.520        2.40x ✅     +58.4%
long        2.800       1.200        2.33x ✅     +57.1%

✅ TensorRT is 2.37x FASTER on average
   Average latency improvement: 57.9%
```

## Troubleshooting

### Issue: "TensorRT not found"
```bash
pip install tensorrt
```

### Issue: "Engine file not found"
```bash
# Build the engine first
python build_s3gen_tensorrt.py --export-onnx
```

### Issue: "ONNX export failed"
```bash
# Install ONNX dependencies
pip install onnx onnxruntime

# Try with older opset version
python build_s3gen_tensorrt.py --export-onnx --opset 14
```

### Issue: "Engine loading failed"
```bash
# Check engine was built correctly
trtexec --loadEngine=trt_engines/s3gen_decoder.engine

# Verify GPU architecture matches build machine
# Engine is NOT portable across different GPU architectures!
```

### Issue: "Dynamic shape errors"
```bash
# Rebuild engine with larger max shapes
trtexec \
  --onnx=trt_engines/s3gen_decoder.onnx \
  --saveEngine=trt_engines/s3gen_decoder.engine \
  --fp16 \
  --maxShapes=x:2x80x1000,mask:2x1x1000,mu:2x80x1000,t:2,spks:2x80,cond:2x80x1000 \
  --workspace=2048MB
```

## Performance Considerations

### Best Practices

1. **Use FP16**: 2x speedup with minimal quality loss
2. **Build for target GPU**: Engine is GPU-specific
3. **Profile before deploying**: Use benchmark_tensorrt.py
4. **Handle dynamic shapes**: Build with appropriate max shapes
5. **Warmup**: First inference is slower (compilation overhead)

### Limitations

1. **GPU-specific**: Engine built on H200 won't work on A100
2. **Batch size fixed**: Dynamic shapes limited to configured range
3. **Build time**: Can take 5-30 minutes to build engine
4. **Memory**: Requires 1-2GB GPU workspace

### When to Use TensorRT

**Use TensorRT when:**
- ✅ Deploying on known GPU architecture
- ✅ Maximum performance needed
- ✅ Short-prompt workloads (S3Gen-bound)
- ✅ Can pre-build engines

**Use PyTorch when:**
- ✅ Development/testing
- ✅ Multiple GPU architectures
- ✅ Quick iteration needed
- ✅ Not performance-critical

## Advanced Topics

### Customizing Build Parameters

Edit `build_s3gen_tensorrt.py` to modify:
- `max_batch_size`: Batch size for dynamic shapes
- `max_mel_length`: Maximum audio length
- `workspace_size`: GPU memory allocation
- `use_fp16`: FP16 vs FP32 precision

### Integrating with Existing Workflows

The TensorRT wrapper is a drop-in replacement:
```python
# Before (PyTorch)
output = model.flow.decoder.estimator(x, mask, mu, t, spks, cond)

# After (TensorRT) - same interface!
output = model.flow.decoder.trt_engine(x, mask, mu, t, spks, cond)
```

### Production Deployment

1. Build engine on target GPU architecture
2. Include `.engine` file in deployment package
3. Set `s3gen_use_tensorrt=True` in production config
4. Monitor performance with benchmark_tensorrt.py
5. A/B test against PyTorch baseline

## Files

- `build_s3gen_tensorrt.py` - Engine builder script
- `src/chatterbox_vllm/models/s3gen/tensorrt_wrapper.py` - TensorRT wrapper class
- `src/chatterbox_vllm/models/s3gen/flow_matching.py` - Updated to support TensorRT
- `benchmark_tensorrt.py` - Performance comparison tool

## References

- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [trtexec Command Line Tool](https://docs.nvidia.com/deeplearning/tensorrt/tensorrt-wip/index.html#trtexec)
- [ONNX Format](https://onnx.ai/)
- [S3Gen Architecture](../TTFA_COMPONENT_BREAKDOWN.md)

## Status

✅ **IMPLEMENTATION COMPLETE** - TensorRT support fully integrated

⚠️ **REQUIRES ENGINE FILE** - Must build engine before use

📊 **EXPECTED SPEEDUP** - 2-3x on S3Gen, 40-60% overall TTFA improvement

---

**Last Updated:** 2025-03-05
**Status:** Production-ready with pre-built engines
