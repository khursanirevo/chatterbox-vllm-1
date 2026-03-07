# T3 AsyncLLMEngine Concurrent Behavior Findings

**Date:** 2026-03-08
**Test:** `profile_t3_concurrent.py`
**Objective:** Profile T3 AsyncLLMEngine behavior with concurrent requests

## Test Configuration

```
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
```

## Observed Results

### Performance Metrics by Concurrency Level

| Concurrent | First Token | 25 Tokens | First Audio | GPU Util | GPU Memory (MB) |
|------------|-------------|-----------|-------------|----------|-----------------|
| 1          | 645ms       | 645ms     | 645ms       | 48.9%    | 75,184          |
| 2          | 1,640ms     | 1,640ms   | 1,640ms     | 77.4%    | 75,894          |
| 4          | 2,500ms     | 2,500ms   | 2,500ms     | 82.5%    | 76,346          |
| 8          | 6,964ms     | 6,964ms   | 6,964ms     | 64.7%    | 77,068          |
| 16         | 19,759ms    | 19,759ms  | 19,759ms    | 69.9%    | 78,879          |

### Per-Request Timeline Observations

**1 Concurrent Request:**
- Queue Position: 0
- Time to First Token: 645ms
- Time to 25 Tokens: 645ms (same as TTFT)
- Time to First Audio: 645ms
- Estimated Batch Size: 1
- Token Generation Rate: N/A (cannot measure from chunked delivery)
- GPU Utilization: 48.9%
- GPU Memory: 75,184 MB

**2 Concurrent Requests:**
- Queue Positions: 0, 1
- Time to First Token: 1,603ms, 1,676ms (avg: 1,640ms)
- Time to 25 Tokens: 1,603ms, 1,676ms (avg: 1,640ms)
- Time to First Audio: 1,603ms, 1,676ms (avg: 1,640ms)
- Estimated Batch Size: 2
- Token Generation Rate: N/A (cannot measure from chunked delivery)
- GPU Utilization: 77.4%
- GPU Memory: 75,894 MB

**4 Concurrent Requests:**
- Queue Positions: 0, 1, 2, 3
- Time to First Token: 2,526ms, 2,445ms, 2,520ms, 2,508ms (avg: 2,500ms)
- Time to 25 Tokens: 2,526ms, 2,445ms, 2,520ms, 2,508ms (avg: 2,500ms)
- Time to First Audio: 2,526ms, 2,445ms, 2,520ms, 2,508ms (avg: 2,500ms)
- Estimated Batch Size: 4
- Token Generation Rate: N/A (cannot measure from chunked delivery)
- GPU Utilization: 82.5%
- GPU Memory: 76,346 MB

**8 Concurrent Requests:**
- Queue Positions: 0-7
- Time to First Token: 6,862ms - 7,117ms (avg: 6,964ms)
- Time to 25 Tokens: 6,862ms - 7,117ms (avg: 6,964ms)
- Time to First Audio: 6,862ms - 7,117ms (avg: 6,964ms)
- Estimated Batch Size: 8
- Token Generation Rate: N/A (cannot measure from chunked delivery)
- GPU Utilization: 64.7%
- GPU Memory: 77,068 MB

**16 Concurrent Requests:**
- Queue Positions: 0-15
- Time to First Token: 6,897ms - 63,624ms (avg: 19,759ms)
- Time to 25 Tokens: 6,897ms - 63,624ms (avg: 19,759ms)
- Time to First Audio: 6,897ms - 63,624ms (avg: 19,759ms)
- Estimated Batch Size: 16
- Token Generation Rate: N/A (cannot measure from chunked delivery)
- GPU Utilization: 69.9%
- GPU Memory: 78,879 MB
- Note: 4 requests (positions 12-15) showed extreme outliers (48-64 seconds)

### Key Observations

1. **Time to First Token Equals Time to 25 Tokens**
   - For all concurrency levels, TTFT equals time to 25 tokens
   - This indicates the entire batch is delivered at once, not streamed incrementally

2. **Non-linear Performance Degradation**
   - 1→2 concurrent: 2.54x increase (645ms → 1,640ms)
   - 2→4 concurrent: 1.52x increase (1,640ms → 2,500ms)
   - 4→8 concurrent: 2.79x increase (2,500ms → 6,964ms)
   - 8→16 concurrent: 2.84x increase (6,964ms → 19,759ms)
   - Scaling is roughly O(n log n) rather than O(log n)

3. **All Requests Complete Simultaneously**
   - At each concurrency level (except 16), all requests finish at approximately the same time
   - Time differences between requests in same batch are minimal (<5% variance)

4. **Estimated Batch Size**
   - Estimated batch size equals concurrent level
   - Note: vLLM doesn't expose per-request batch info, so concurrent level is used as estimate

5. **Queue Position Impact**
   - Requests with higher queue positions show slightly longer TTFT
   - Difference is minimal (<5% variance within same batch)
   - At 16 concurrent, 4 requests showed extreme outliers (48-64 seconds)

6. **Token Generation Rate**
   - Cannot measure because tokens arrive in chunks, not incrementally
   - Returns 0.0 tok/s as tokens are delivered in batches

7. **GPU Utilization Patterns**
   - 1 concurrent: 48.9% (underutilized)
   - 2 concurrent: 77.4% (better utilization)
   - 4 concurrent: 82.5% (near peak)
   - 8 concurrent: 64.7% (dropped, possibly due to scheduling overhead)
   - 16 concurrent: 69.9% (similar to 8 concurrent)

8. **GPU Memory Usage**
   - Linear increase with concurrency: 75GB → 79GB
   - Memory pressure is not the bottleneck

## Measured Metrics Summary

### What Was Measured

- **Per-request timeline breakdown:** Submit → first token → 25 tokens → first audio
- **Queue position:** Position in request queue when submitted
- **Batch size:** Approximate batch size at first token (concurrent level)
- **GPU utilization:** Average GPU utilization during generation (%)
- **GPU memory:** Average GPU memory usage during generation (MB)
- **Token generation rate:** Tokens per second during generation phase

### What Was Not Measured

- Actual vLLM batch size (not exposed by AsyncLLMEngine API)
- Per-scheduler-step batch composition
- Queue wait time vs processing time (not separated in current metrics)
- Incremental token arrival times (tokens delivered in chunks)
- S3Gen processing time (overlaps with token generation)

## Data Collection Notes

1. **Token Tracking:** Current implementation tracks token counts in chunks, not individual token arrivals
2. **Batch Size:** vLLM AsyncLLMEngine doesn't expose batch information, so concurrent level is used as approximation
3. **GPU Metrics:** Collected via pynvml in background thread during request generation
4. **Queue Position:** Assigned based on submission order within concurrent batch
5. **Outliers at 16 Concurrent:** 4 requests took 48-64 seconds, indicating possible resource contention or scheduling issues

## Raw Data

All test runs produce detailed per-request metrics including:
- Request ID
- Submit timestamp
- First token timestamp
- Chunk ready timestamp
- First chunk timestamp
- Complete timestamp
- Token count and arrivals
- Queue position
- Batch size estimates
- GPU metrics

Data is available in the test output for further analysis.
