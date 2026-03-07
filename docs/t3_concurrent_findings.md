# T3 AsyncLLMEngine Concurrent Behavior Findings

**Date:** 2026-03-08
**Test:** `profile_t3_concurrent.py`
**Objective:** Investigate why continuous batching isn't achieving <1s first chunk with 8-16 concurrent requests

## Executive Summary

The profiling reveals a **critical bottleneck in T3 AsyncLLMEngine's scheduling** under concurrent load. Performance degrades severely as concurrency increases:

- **1 concurrent:** 633ms ✅
- **2 concurrent:** 911ms ✅
- **4 concurrent:** 2,207ms ❌ (2.3x degradation vs 2 concurrent)

**Root Cause:** AsyncLLMEngine is NOT processing concurrent requests in parallel batches. Instead, it appears to be serializing request processing or has a scheduling bottleneck.

## Detailed Results

### Timeline Breakdown (4 concurrent)

| Metric | Value | Status |
|--------|-------|--------|
| Time to First Token (TTFT) | 2,207ms | ❌ 2.3x slower than 2 concurrent |
| Time to 25 Tokens | 2,207ms | ❌ Same as TTFT (batching issue) |
| Time to First Audio | 2,207ms | ❌ Exceeds 1s target |
| Token Generation Rate | 0 tok/s | ❌ Cannot measure |

### Key Observations

1. **TTFT = 25 Token Time**
   - For all concurrency levels, TTFT equals time to 25 tokens
   - This indicates the entire batch is delivered at once, not streamed
   - Suggests AsyncLLMEngine is NOT using continuous batching

2. **Non-linear Performance Degradation**
   ```
   1 concurrent:  633ms  (baseline)
   2 concurrent:  911ms  (1.44x slower, 1.8x throughput)
   4 concurrent:  2207ms (2.42x slower, 1.8x throughput)
   ```
   - If properly batched, 4 concurrent should be ~1000ms (not 2207ms)
   - Current scaling is linear O(n), not logarithmic O(log n)

3. **S3Gen Time = 0ms**
   - The profiler shows `t3_generation_time = 0` and `s3gen_time = 0`
   - This is because the entire 25-token chunk arrives in a single event
   - The test breaks after first chunk, so we don't see incremental tokens

## Root Cause Analysis

### Hypothesis 1: Request Serialization (MOST LIKELY)

**Evidence:**
- 4 concurrent requests take 2.2 seconds total
- If batched properly, should complete in ~1 second (4 requests × 633ms / 4 concurrent)
- The logs show requests are added within 30ms of each other
- But all 4 complete at roughly the same time (~2.2s)

**Likely Issue:**
AsyncLLMEngine is processing requests sequentially instead of in parallel batches. Possible causes:
- Scheduler is not properly configured for continuous batching
- Max_num_batched_tokens constraint (5120) might be forcing serialization
- Chunked prefill might be disabled or not working
- Single scheduler step (num_scheduler_steps=1) might be bottleneck

### Hypothesis 2: Batch Size Too Small

**Evidence:**
- Max model length: 2000 tokens
- Max num batched tokens: 5120
- Test uses ~25 tokens per request

**Analysis:**
- 4 requests × 25 tokens = 100 tokens (well under 5120 limit)
- Should be able to batch all 4 requests together
- But performance suggests they're being serialized

### Hypothesis 3: CFG Scale Overhead

**Evidence:**
- Test applies CFG scale: 0.5 (conditional generation)
- This requires 2 forward passes per token (unconditional + conditional)

**Impact:**
- Could explain 633ms baseline for 1 concurrent
- But doesn't explain why 4 concurrent is 3.5x slower

## Recommendations

### Immediate Actions (Priority 1)

1. **Verify AsyncLLMEngine Configuration**
   ```python
   # Check these parameters in AsyncLLMEngine initialization:
   - max_num_batched_tokens: 5120 ✅ (sufficient)
   - num_scheduler_steps: 1 (try increasing to 4-8)
   - chunked_prefill: enabled ✅
   - enable_prefix_caching: true ✅
   ```

2. **Add Detailed Logging**
   - Log scheduler decisions for each request
   - Track batch size at each iteration
   - Monitor queue length and wait times
   - Profile scheduler step duration

3. **Test with Different Scheduler Configurations**
   - Increase `num_scheduler_steps` to 4-8
   - Disable chunked prefill to see if it's causing issues
   - Try different `max_num_batched_tokens` values

### Further Investigation (Priority 2)

1. **Profile vLLM Scheduler**
   - Add metrics to track:
     - Number of requests in each batch
     - Time between scheduler steps
     - Prefill vs decode batch sizes
     - KV cache utilization

2. **Test with Simpler Workload**
   - Use a standard LLM (not T3) to isolate if issue is T3-specific
   - Compare with vLLM's own benchmarking tools
   - Test with different token lengths (shorter/longer)

3. **Check vLLM Version Compatibility**
   - Current version: v0.10.0
   - Verify T3 model is compatible with continuous batching
   - Check if there are known issues with this configuration

## Expected Behavior (Target)

With proper continuous batching:

```
1 concurrent:  633ms   (baseline)
2 concurrent:  ~700ms  (batch 2 requests together)
4 concurrent:  ~800ms  (batch 4 requests together)
8 concurrent:  ~1000ms (batch 8 requests together)
```

Instead of current linear degradation:
```
1 concurrent:  633ms
2 concurrent:  911ms   (1.44x slower)
4 concurrent:  2207ms  (3.5x slower)
```

## Next Steps

1. **Implement enhanced logging** in AsyncLLMEngine wrapper
2. **Test with increased `num_scheduler_steps`**
3. **Compare with vLLM offline API** (non-streaming)
4. **File bug report** if issue persists (vLLM or T3-specific)

## Appendix: Test Configuration

```python
Model: t3-model (Chatterbox T3)
Platform: CUDA
Max model length: 2000 tokens
Max num batched tokens: 5120
Chunked prefill: enabled
Num scheduler steps: 1
Enable prefix caching: true
GPU memory utilization: 0.5
KV cache: 598,192 tokens
Max concurrency: 299x (theoretical)

Test parameters:
- Chunk size: 25 tokens
- Text: ~30 tokens per request
- Concurrent levels: 1, 2, 4, 8, 16
- Stopped at: 4 concurrent (severe degradation)
```

## Conclusion

The T3 AsyncLLMEngine is **NOT achieving continuous batching** under concurrent load. The root cause appears to be request serialization rather than parallel batch processing. This is a critical issue preventing <1s first chunk targets at 8-16 concurrent requests.

**Estimated impact:** Without fixing this, the system can handle at most 2-3 concurrent requests while meeting latency targets. To achieve 8-16 concurrent with <1s latency, the batching issue must be resolved.
