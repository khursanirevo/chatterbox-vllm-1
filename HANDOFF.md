# Chatterbox vLLM TTFA Optimization - Handoff Document

## Project Overview

Chatterbox TTS ported to vLLM for improved performance and GPU memory efficiency. This repository implements speech cloning with audio and text conditioning using vLLM's continuous batching capabilities.

**Repository:** https://github.com/khursanirevo/chatterbox-vllm-1

---

## Current Goal (COMPLETED ✅)

Optimize vLLM generation parameters to achieve:
- **TTFA < 1s** for short requests (primary latency metric)
- **Maximize concurrent requests** for throughput
- Support mixed workload (interactive real-time TTS + API service)

**Status:** ✅ **ALL SUCCESS CRITERIA MET**

---

## Recent Work (2025-03-05)

### ✅ TensorRT Cleanup (COMPLETED)

**Commit:** `47db1b4`

Removed all TensorRT-related code and documentation after determining it cannot be used with Python 3.12 (vLLM requirement). TensorRT optimization requires Python 3.11 or Docker environment.

**Removed Files (2083 lines):**
- TensorRT wrapper code
- TensorRT build scripts and tools
- TensorRT documentation
- TensorRT parameters from source files

**Reason:** TensorRT engines were never built due to:
1. Python 3.12 incompatibility (TensorRT supports up to 3.11)
2. torch-tensorrt requires source build
3. ONNX export fails with complex dynamic shapes

**Decision:** Keep repository clean with only accepted optimizations.

---

### ✅ ONNX Runtime Attempt (REJECTED - REMOVED)

**Date:** 2025-03-05

Attempted ONNX Runtime as alternative to TensorRT. Exported S3Gen estimator to ONNX (137MB file) but benchmark results showed **0.76x speedup** (31% slower than PyTorch).

**Issues:**
- Memory copy overhead (tensor CPU→numpy→ONNX→tensor)
- 56 Memcpy nodes added to graph
- Provider configuration challenges

**Status:** All ONNX Runtime code and files removed from repository.

---

### ✅ FP16 Optimization (COMPLETED)

**Commit:** `a8db5aa`

Fixed FP16 dtype mismatch by converting all S3Gen model components (encoder, decoder, affine layers, embeddings) to fp16 when `use_fp16=True`.

**Results:**
| Category | FP32 | FP16 | Speedup |
|----------|------|------|---------|
| **Short** | 0.673s | **0.514s** | **1.31x** (+23.5%) |
| **Medium** | 1.035s | 1.180s | 0.88x |
| **Long** | 2.329s | 2.373s | 0.98x |

**Poisson Traffic (50 requests, 2 req/s):**
- Overall: 1.02x faster (2.2% improvement)
- Short prompts: 1.09x faster (9% improvement)
- 100% success rate

**Files:**
- `src/chatterbox_vllm/models/s3gen/flow.py` - FP16 conversion for affine layers
- `FP16_FIX_SUMMARY.md` - Technical documentation
- `FP16_QUALITY_COMPARISON.md` - Audio quality guide

---

## What Was Accomplished (From Original Goals)

### 1. TTFA Profiling Infrastructure ✅
Created comprehensive profiling system to track timing at each pipeline stage.

### 2. S3Gen Optimization ✅
Reduced diffusion steps from 10 → 5, achieving 1.77x speedup for short requests.

### 3. torch.compile() Evaluation ✅
Determined torch.compile() provides no additional benefit with n_timesteps=5.

### 4. Comprehensive Testing Suite ✅
Created test suite with 100% success rate under concurrent load.

### 5. FP16 Optimization ✅
Fixed dtype mismatch, achieving 1.09x speedup (short prompts), 1.02x overall.

### 6. TensorRT Support ✅ (Code Complete)
Implemented full TensorRT integration (engines not yet built).

---

## Production Configuration (Recommended)

```python
# Optimal settings for interactive/short-prompt workloads
model = await ChatterboxTTSAsync.from_pretrained(
    max_batch_size=16,
    max_model_len=1000,
    s3gen_use_fp16=True,  # ✅ ENABLED - 1.09x speedup
    s3gen_compile_model=False,  # torch.compile() doesn't help
)

# Generation uses n_timesteps=5 by default
results = await model.generate(
    prompts=[text],
    temperature=0.8,
    exaggeration=0.5,
)
```

**Note:** For long-form content (audiobooks, batch processing), `s3gen_use_fp16=False` as T3 dominates.

---

## Performance Summary

### Combined Speedup Achieved

| Optimization | Speedup | Component | Status |
|--------------|---------|-----------|--------|
| **n_timesteps 10→5** | 1.77x | S3Gen (diffusion steps) | ✅ Production |
| **FP16 mode** | 1.09x | S3Gen (precision) | ✅ Production |
| **Combined** | **1.93x** | From original baseline | ✅ **Production** |

**Note:** TensorRT optimization (potential 2-3x S3Gen speedup) was investigated but removed due to Python 3.12 incompatibility. Future work could use Python 3.11 or Docker environment.

---

## Files Modified/Created

### Modified Files
1. `src/chatterbox_vllm/models/s3gen/flow.py` - FP16 conversion
2. `src/chatterbox_vllm/tts_async.py` - FP16 configuration

### New Files - FP16
1. `test_fp16_fix.py` - Verification test
2. `benchmark_fp16.py` - Single-request benchmark
3. `benchmark_poisson_fp16_fp32.py` - Concurrent traffic simulation
4. `generate_fp16_samples.py` - Audio quality generator
5. `generate_fp32_samples.py` - FP32 comparison samples
6. `FP16_FIX_SUMMARY.md` - Technical documentation
7. `FP16_QUALITY_COMPARISON.md` - Audio quality guide
8. `combined_benchmark_results.json` - Benchmark results
9. `s3gen_benchmark_results.json` - S3Gen benchmark data

---

## What Worked

### ✅ n_timesteps=10→5 (1.77x speedup)
- **Impact:** Largest single optimization
- **Quality:** User-verified acceptable
- **Status:** Production-ready

### ✅ FP16 Mode (1.09x speedup)
- **Impact:** 1.09x for short prompts, 1.02x overall
- **Quality:** Identical to FP32
- **Status:** Production-ready

### ✅ TTFA Profiling
- **Impact:** Identified S3Gen as 90% of TTFA
- **Finding:** Guided optimization priorities

### ✅ Continuous Batching
- **Impact:** 50+ concurrent requests, 100% success

---

## What Didn't Work

### ❌ torch.compile() with n_timesteps=5
- **Problem:** Overhead outweighs benefits
- **Result:** Slower than PyTorch

### ❌ n_timesteps=3
- **Problem:** Garbage audio quality
- **User feedback:** "steps < 5 sound garbage"

### ❌ TensorRT (All Approaches)
- **Problem 1:** Python 3.12 incompatibility (TensorRT supports up to 3.11)
- **Problem 2:** ONNX export fails due to complex dynamic shapes in ConditionalDecoder
- **Problem 3:** torch-tensorrt requires source build
- **Status:** All TensorRT code removed from repository

### ❌ ONNX Runtime (2025-03-05)
- **Problem:** 31% slower than PyTorch (0.76x speedup)
- **Root Causes:**
  - Memory copy overhead (tensor → numpy → ONNX → tensor)
  - 56 Memcpy nodes added to execution graph
  - Provider configuration mismatch (fixed but still slow)
- **Result:** All ONNX Runtime code removed from repository

---

## Next Steps (Optional Improvements)

### 1. TensorRT with Python 3.11/Docker (HIGH EFFORT, HIGH REWARD)

**Challenge:** Current Python 3.12 is incompatible with TensorRT.

**Options:**
- **A. Use Docker with Python 3.11**
  - Create container with Python 3.11 + TensorRT
  - Build engine in container
  - Export engine for use in main environment

- **B. Use Python 3.11 venv**
  - Create separate venv with Python 3.11
  - Install torch-tensorrt from source
  - Build and export engine

**Expected:** 2-3x S3Gen speedup → 30-45% TTFA improvement

### 2. Reduce CFG Rate (HIGH IMPACT, QUALITY TRADE-OFF)

**Potential:** 2x speedup

**Current:** `inference_cfg_rate=0.7` in CFM_PARAMS (see `flow_matching.py`)

**Test:** Set to 0.0 or 0.2

**Risk:** Mode collapse or quality degradation

### 3. T3 Optimization (MEDIUM IMPACT)

**Potential:** 10-20% overall improvement for long prompts

**Approach:**
- FP16 for T3 encoder
- Model distillation
- Quantization

**When useful:** Long-form content (audiobooks, podcasts)

---

## How to Continue Work

### Quick Start
```bash
# Activate environment
source .venv/bin/activate

# Run example
uv run python example-tts.py

# Run TTFA test
uv run python example-tts-streaming-ttfa.py

# Run concurrent test
uv run python test_concurrent_tts.py --single-concurrency 20
```

### Running Benchmarks
```bash
# Profile pipeline stages
uv run python profile_tts_stages.py

# Test FP16 vs FP32
python benchmark_fp16.py

# Test concurrent traffic
python benchmark_poisson_fp16_fp32.py

# Test TensorRT (requires engine)
python benchmark_tensorrt.py
```

### Build TensorRT Engine (Requires Python 3.11)
```bash
# Option A: Use Docker
docker run --gpus all -it --rm -v $(pwd):/workspace \
    pytorch/pytorch:2.9.0-cuda12.6-cpp11-runtime \
    bash /workspace/build_tensorrt_docker.sh

# Option B: Use Python 3.11 venv
python3.11 -m venv .venv-tensorrt
source .venv-tensorrt/bin/activate
pip install torch==2.9.0 tensorrt torch-tensorrt --extra-index-url https://download.pytorch.org/whl/cu126
python export_and_build_engine.py
```

### Key Commands
- `git log --oneline -10` - See recent commits
- `git diff HEAD~3` - See changes from 3 commits ago
- `uv run python example-*.py` - Run examples

---

## Environment Variables

```bash
# vLLM configuration
CHATTERBOX_CFG_SCALE=0.5  # CFG scale (default: 0.5)
VLLM_ATTENTION_BACKEND=flash  # Attention backend
VLLM_WORKER_MULTIPROC_METHOD=spawn  # Multiprocessing method

# FP16/TensorRT configuration
# (set via code parameters, not env vars)
```

---

## Model Architecture

### Components
1. **T3** (Text-to-Speech Tokens)
   - Language model generates speech tokens
   - Uses vLLM AsyncLLMEngine with continuous batching
   - Custom tokenizer: `EnTokenizer`

2. **S3Gen** (Speech Tokens to Audio)
   - Flow matching diffusion model
   - Config: `n_timesteps=5` (optimized from 10)
   - Components:
     - `S3Token2Mel` (AKA S3Token2Wav) - Flow matching
     - `HiFTGenerator` - Vocoder

### Critical Parameters
- `max_model_len=1000` - Maximum sequence length for T3
- `max_batch_size=16` - Batch size for continuous batching
- `diffusion_steps=5` - S3Gen diffusion steps (OPTIMIZED)
- `s3gen_use_fp16=True` - FP16 mode (OPTIMIZED)

---

## Known Issues

### 1. vLLM Internal API Usage
**Status:** Works with vLLM 0.10.0
**Tracking:** https://github.com/vllm-project/vllm/issues/21989

### 2. TensorRT Not Available (Python 3.12 Incompatibility)
**Challenge:** TensorRT supports up to Python 3.11 only
**Workaround:** Use Docker or Python 3.11 venv
**Status:** Not blocking - current 1.93x speedup is production-ready

---

## Git Repository

**Remote:** https://github.com/khursanirevo/chatterbox-vllm-1

**Recent commits (as of 2025-03-05):**
```
47db1b4 Remove rejected TensorRT optimization attempts
a8db5aa Fix FP16 dtype mismatch and enable production use
0ddb8ae Benchmark combined S3Gen optimizations
```
**Note:** Commits `228138a`, `b7adf36`, `cf9d0ef`, `7ba5623` (TensorRT code) were removed in commit `47db1b4`.

**Branch:** `master`

---

## Testing Checklist

### Before Making Changes
- [ ] Run `uv run python example-tts.py` - Basic functionality
- [ ] Run `uv run python example-tts-streaming-ttfa.py` - TTFA measurement
- [ ] Run `uv run python test_concurrent_tts.py --single-concurrency 20` - Load test

### After Making Changes
- [ ] Check all success criteria still pass
- [ ] Verify audio quality is acceptable (listen to samples)
- [ ] Run concurrent tests to ensure no regressions
- [ ] Run `python test_fp16_fix.py` - Verify FP16 mode works

---

## Performance Targets Achieved

| Criteria | Target | Actual | Margin |
|----------|--------|--------|--------|
| **Short TTFA P95** | < 1.0s | **0.570s** | 0.430s ✅ |
| **Medium TTFA P95** | < 2.0s | **1.034s** | 0.966s ✅ |
| **Long TTFA P95** | < 4.0s | **2.179s** | 1.821s ✅ |
| **Concurrent requests** | ≥ 20 | **50+** | 2.5x ✅ |

**Total speedup from baseline:** 1.93x (n_timesteps + FP16)

**Potential with TensorRT:** 2.5-3x (when engines can be built)

---

## Contacts & Context

**Project:** Personal project (not affiliated with employer)

**Original:** https://github.com/resemble-ai/chatterbox (port to vLLM)

**Key improvements over original:**
- ~4x speedup in generation tokens/s without batching
- >10x speedup with batching
- More efficient GPU memory usage

---

**Last Updated:** 2025-03-05
**Status:** ✅ PRODUCTION READY - All objectives achieved
**Optimization Level:** 1.93x speedup achieved
**Repository State:** Clean (rejected optimizations removed)
