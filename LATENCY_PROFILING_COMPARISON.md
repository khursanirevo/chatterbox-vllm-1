# Latency Profiling Comparison: Cold-Start vs Steady-State

**Date**: 2026-03-07

## Executive Summary

Two profiling approaches were used to measure first chunk latency:

1. **Cold-Start Profiling** (`test-profiling-first-chunk.py`)
   - Creates NEW model for each iteration
   - Includes initialization overhead
   - Measures "real-world" cold-start performance

2. **Steady-State Profiling** (`test-profiling-steady-state.py`)
   - Uses SAME model for all iterations
   - Excludes initialization overhead
   - Measures optimal runtime performance

## Results Comparison

| Metric | Cold-Start | Steady-State | Difference |
|--------|------------|--------------|------------|
| **Iterations** | 10 | 20 | - |
| **Warmup** | 2 runs | 3 runs | - |
| **Average** | 932.76ms | **750.77ms** | **-181ms (-19.4%)** |
| **Min** | 737.45ms | **611.77ms** | **-126ms (-17.1%)** |
| **Max** | 1149.82ms | **904.02ms** | **-246ms (-21.4%)** |
| **Median** | 923.22ms | **750.51ms** | **-173ms (-18.7%)** |
| **Std Dev** | 107.29ms | **91.16ms** | **-16ms (-15.0%)** |
| **CV** | ~11.5% | **12.1%** | Similar consistency |

## Key Findings

### 1. Initialization Overhead
- **Cold-start adds ~180ms** per iteration
- This includes vLLM engine initialization, profiling, and cache setup
- Significant for single-request scenarios

### 2. Steady-State Performance
- **Average: 750.77ms** - True runtime performance
- **Best case: 611.77ms** - Excellent!
- **Consistency: CV 12.1%** - Good predictability

### 3. Real-World Implications

**Cold-Start (first request after restart):**
- Expect ~933ms latency
- Important for serverless/lambda deployments
- Model loading + generation time

**Steady-State (subsequent requests):**
- Expect ~751ms latency
- Important for long-running servers
- Generation time only

## Latency Breakdown Comparison

### Cold-Start
```
First Chunk: 932.76ms
├── T3 generation:  460.31ms (49.3%)
├── S3Gen:          461.12ms (49.4%)
└── Overhead:        11.34ms (1.2%)
```

### Steady-State
```
First Chunk: 750.77ms
├── T3 generation:  360.54ms (48.0%) ← Faster!
├── S3Gen:          386.66ms (51.5%)
└── Overhead:         3.57ms (0.5%)
```

**Note**: T3 generation is ~100ms faster in steady-state, likely due to:
- Cached CUDA kernels
- Warm KV cache
- Optimized memory allocation

## Distribution Analysis

### Cold-Start (10 iterations)
```
Min:     737ms
Max:     1150ms
Range:   413ms
```

### Steady-State (20 iterations)
```
< 700ms:   7/20 (35%)
< 800ms:  14/20 (70%)
< 900ms:  19/20 (95%)
<1000ms:  20/20 (100%)  ← All under 1s!
```

## AsyncLLMEngine Projections

### Based on Cold-Start
```
Current:    ~933ms first chunk
Async:      ~511ms first chunk (50ms + 461ms)
Speedup:    1.82x
Status:     ✅ Under 1s
```

### Based on Steady-State (More Accurate)
```
Current:    ~751ms first chunk
Async:      ~437ms first chunk (50ms + 387ms)
Speedup:    1.72x
Status:     ✅ Well under 1s!
```

## Recommendations

### For Production Deployments

**Serverless/Lambda (cold-start heavy):**
- Expect ~933ms first chunk latency
- Consider model preloading/warming
- AsyncLLMEngine would achieve ~511ms

**Long-running servers (steady-state):**
- Expect ~751ms first chunk latency
- Consistent performance after warmup
- AsyncLLMEngine would achieve ~437ms ✅

### For Benchmarking

**Use cold-start metrics when:**
- Comparing different models/frameworks
- Measuring real-world first-request latency
- Evaluating serverless deployments

**Use steady-state metrics when:**
- Optimizing generation pipeline
- Measuring peak performance
- Evaluating long-running services

## Test Scripts

### Cold-Start Profiling
```bash
CUDA_VISIBLE_DEVICES=0 uv run python test-profiling-first-chunk.py
```

### Steady-State Profiling
```bash
CUDA_VISIBLE_DEVICES=0 uv run python test-profiling-steady-state.py
```

## Conclusion

Both profiling methods provide valuable insights:

- **Cold-start**: Realistic for first-request scenarios
- **Steady-state**: Shows optimal runtime performance

The ~180ms difference represents initialization overhead that can be eliminated with:
- Model preloading
- Warmup requests
- Long-running server architecture

**AsyncLLMEngine would achieve ~437-511ms first chunk latency**, consistently meeting the <1s target! 🎯
