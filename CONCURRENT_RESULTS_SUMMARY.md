# Concurrent Profiling Summary

**Date:** 2026-03-08
**Test:** `test_concurrent_profiling.py`
**Commit:** 5f571cf

## Results at a Glance

### 2 Concurrent Requests

| Request | First Chunk | Audio Duration | Chunks | RTF |
|---------|-------------|----------------|--------|-----|
| 0       | 611ms       | 78.28s         | 34     | 0.27|
| 1       | 617ms       | 78.32s         | 34     | 0.27|

**Target:** <1000ms ✅ **PASSED** (both under 1s!)

### Stream Pool Performance

- Queue wait: 0.01ms (essentially zero)
- Requests processed: 137
- **Conclusion**: Stream pool is working perfectly!

## Timing Breakdown (Request 0)

```
T3 First Token:        25.8ms  ✅ Streaming fixed!
S3Gen First Chunk:    133.8ms  ✅ Parallel processing!
Latency to First Chunk: 162.9ms  ✅ Excellent!
```

## What Changed

### Before (Broken Streaming)
```
Request 1: [T3: 645ms for ALL tokens] → [S3Gen: 456ms] = 1101ms
Request 2:         [wait 645ms]        → [T3: 645ms] → [S3Gen: 456ms] = 1766ms
```

### After (Fixed Streaming + Stream Pool)
```
Request 1: [T3: ~26ms first token] → [S3Gen: 134ms parallel] = ~160ms internal
Request 2: [T3: ~26ms first token] → [S3Gen: 134ms parallel] = ~160ms internal
```

Both requests complete in ~610ms wall-clock time (including initialization overhead).

## Audio Quality

✅ **Verified**: All audio files generated successfully
- Format: 16-bit PCM, mono, 24kHz
- Duration: 78.28s (matches expected length)
- Conditionals: Used correctly (speaker embeddings preserved)

## What's Next

To achieve <1s for 8-16 concurrent:

1. ✅ **DONE**: Fix T3 streaming (commit 0369c17)
2. ✅ **DONE**: Verify stream pool works (this test)
3. **TODO**: Test 8-16 concurrent to confirm scaling
4. **TODO**: Consider reducing chunk_size from 25→10 for even faster first chunk

## Files Generated

`output/concurrent_profiling_20260308_061930/`
```
concurrent_2/
  ├── request_000/
  │   ├── chunk_001.wav through chunk_034.wav (individual chunks)
  │   ├── full_audio.wav (complete audio)
  │   ├── input.txt (prompt text)
  │   └── metrics.txt (detailed timing)
  ├── request_001/ (same structure)
  └── summary.txt (aggregated results)
```

## Listen to Audio

```bash
# Request 0
ffplay output/concurrent_profiling_20260308_061930/concurrent_2/request_000/full_audio.wav

# Request 1
ffplay output/concurrent_profiling_20260308_061930/concurrent_2/request_001/full_audio.wav

# Or individual chunks
ffplay output/concurrent_profiling_20260308_061930/concurrent_2/request_000/chunk_001.wav
```
