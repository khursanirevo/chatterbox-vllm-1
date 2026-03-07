# T3 Parallel Processing Investigation & Optimization

**Date:** 2025-03-08
**Status:** Investigation Phase
**Goal:** Achieve 8-16 concurrent requests with <1s first chunk latency

## Problem Statement

Current bottleneck with 8-16 concurrent requests:
- **Target:** <1s first chunk
- **Actual:** ~2-4s first chunk at 8-16 concurrent
- **Root cause:** T3 token generation is sequential despite AsyncLLMEngine's continuous batching

## Current Architecture Analysis

### Request Flow
```
1. Client submits text request
2. AsyncChatterboxTTS.generate_stream() submits to AsyncLLMEngine
3. AsyncLLMEngine streams tokens incrementally
4. Accumulate 25 tokens (chunk_size)
5. Process through S3Gen (parallel via stream pool)
6. Return first audio chunk
```

### Current Performance (Steady State)
```
1 concurrent:  ~580ms (✅ <1s)
2 concurrent:  ~1040ms (❌ 1.8x slowdown)
4 concurrent:  ~2392ms (❌ 4.1x slowdown)
8 concurrent:  ~3000ms+ (❌ >5x slowdown)
```

### Key Mystery
- **First token:** 19-67ms (excellent - continuous batching works!)
- **25 tokens (first chunk):** ~400ms+ (why so slow?)
- **S3Gen:** ~135ms (stream pool working perfectly)

## Investigation Plan

### Phase 1: Deep Profiling (Current Task)

Create `profile_t3_concurrent.py` to measure:

1. **Per-Request Timeline:**
   - Request submission time
   - Time in AsyncLLMEngine queue
   - First token time
   - Token generation rate (tokens/second)
   - Time to accumulate 25 tokens
   - S3Gen processing time
   - Total first chunk time

2. **AsyncLLMEngine Behavior:**
   - How many requests processed per batch?
   - Batch size utilization
   - Inter-token timing (is it really batched?)
   - GPU utilization during concurrent requests

3. **Queue Dynamics:**
   - Request arrival pattern
   - Queue position vs completion order
   - Waiting time vs processing time

4. **Concurrency Scaling:**
   - Test at 1, 2, 4, 8, 16 concurrent
   - Identify where performance degrades
   - Find the inflection point

### Phase 2: Root Cause Analysis

Based on profiling results, determine:

**Scenario A:** AsyncLLMEngine is batching correctly
- Tokens are generated in batches
- But each request still waits its turn
- **Solution:** Optimize request submission or reduce chunk_size

**Scenario B:** AsyncLLMEngine is not batching efficiently
- Each request gets separate forward pass
- Batch size is 1 despite multiple requests
- **Solution:** Investigate AsyncLLMEngine configuration, adjust parameters

**Scenario C:** Token streaming creates serialization
- Generator yields tokens one-by-one
- Cannot batch across different requests' token generation
- **Solution:** Implement incremental S3Gen processing

### Phase 3: Solution Implementation

Based on root cause, implement:

**Option 1: Reduce Chunk Size**
- Lower chunk_size from 25 to 10-15 tokens
- Trade: More S3Gen calls, faster first chunk
- Target: First S3Gen call at ~150-200ms

**Option 2: Incremental S3Gen Processing**
- Process S3Gen as tokens arrive, don't wait for full chunk
- More complex implementation
- Better audio continuity due to smaller updates

**Option 3: AsyncLLMEngine Optimization**
- Adjust engine parameters for better batching
- Pre-batch requests before submission
- Tune max_num_batched_tokens, scheduling policy

**Option 4: Request Queue Coordinator**
- Implement intelligent request queue
- Batch similar requests together
- Prioritize based on queue position and tokens generated

**Option 5: Multiple T3 Instances**
- Run 2-4 AsyncLLMEngine instances
- Split requests between instances
- True parallelism at cost of GPU memory

## Success Criteria

- [ ] 8 concurrent requests: <1s first chunk (avg)
- [ ] 16 concurrent requests: <1s first chunk (avg)
- [ ] Stream pool continues working (0.01ms queue wait)
- [ ] Audio quality maintained
- [ ] GPU memory usage <90%

## Deliverables

1. **Investigation Phase:**
   - [ ] `profile_t3_concurrent.py` - Detailed profiling test
   - [ ] Profiling results analysis
   - [ ] Root cause identification

2. **Implementation Phase:**
   - [ ] Chosen solution implementation
   - [ ] Integration with existing code
   - [ ] Testing and validation

3. **Documentation:**
   - [ ] Findings documented in MEMORY.md
   - [ ] Solution explained in comments
   - [ ] Performance comparison (before/after)

## Technical Notes

### AsyncLLMEngine Configuration (Current)
```python
AsyncEngineArgs(
    model="./t3-model",
    tokenizer="EnTokenizer",
    tokenizer_mode="custom",
    gpu_memory_utilization=0.90,
    max_model_len=2000,
    enforce_eager=True,
    tensor_parallel_size=1,
)
```

### Relevant Metrics
- **TTFA (Time To First Audio):** Main metric to optimize
- **TTFT (Time To First Token):** Should be <100ms
- **Token Generation Rate:** Should be >100 tokens/sec
- **Batch Utilization:** Should be >70%

### Known Constraints
- GPU memory: Limited (typically 24GB)
- T3 model size: Large (requires significant memory)
- S3Gen: Already parallelized via stream pool
- Chunk size: Affects latency vs quality tradeoff

## Next Steps

1. ✅ Create design document
2. ⏭️ Create detailed profiling test
3. ⏭️ Run profiling at various concurrency levels
4. ⏭️ Analyze results and identify root cause
5. ⏭️ Implement solution based on findings
6. ⏭️ Validate against success criteria
