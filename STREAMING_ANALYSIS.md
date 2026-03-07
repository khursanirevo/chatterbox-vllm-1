# Streaming Analysis - Why Removing Conditionals Breaks Audio

**Date:** 2026-03-08
**Issue:** User reported audio became garbage after removing multi_modal_data

## Root Cause Analysis

### The Critical Insight: T3 and S3Gen Must Match

**T3 (text-to-speech-tokens)** generates speech tokens based on:
- Input text
- Voice conditioning (speaker embedding)

**S3Gen (speech-tokens-to-audio)** converts tokens to audio based on:
- Speech tokens from T3
- Voice conditioning (speaker embedding)

**If T3 and S3Gen use different voices → GARBAGE AUDIO!**

### What Happened with My Fix

My fix (commit 0369c17, now reverted):
```python
# WRONG: Removed multi_modal_data
async for request_output in self.engine.generate(
    prompt=text_normalized,  # No conditionals!
    ...
):
```

Result:
- **T3**: No conditionals → generates tokens for DEFAULT voice
- **S3Gen**: `get_audio_conditionals()` → uses SPECIFIC voice
- **MISMATCH** → garbage audio!

### Why Old Code (8c72f6a) Seemed to Work

The old version used:
```python
# Old code (8c72f6a)
async for request_output in self.engine.generate(
    prompt=prompt,  # Plain string, no conditionals
    ...
):
    audio_chunk = await self._process_token_chunk_async(
        conditionals=self.default_conds,  # Default voice
        ...
    )
```

Result:
- **T3**: No conditionals → generates tokens for DEFAULT voice
- **S3Gen**: `default_conds` → uses DEFAULT voice
- **MATCH** → good audio!

## The Real Status: Streaming IS Working!

### Debug Output Shows Incremental Token Arrival

```
[DEBUG] First token received at 24.01ms  ✅
[DEBUG] Yield: +1 tokens (total: 1) at 24.01ms
[DEBUG] Yield: +2 tokens (total: 2) at 6.89ms
[DEBUG] Yield: +3 tokens (total: 3) at 6.52ms
...
[DEBUG] Yield: +25 tokens (total: 25) at 6.84ms
```

This shows **T3 IS streaming tokens incrementally**! Each yield brings more tokens.

### First Chunk Breakdown

```
T3 (25 tokens):    ~187ms  (24ms first + 24 tokens × ~7ms)
S3Gen inference:    433ms  (main bottleneck!)
────────────────────────────────
First chunk:        ~620ms
```

The bottleneck is **S3Gen**, not T3 streaming!

## Conclusion

1. ✅ **T3 streaming is working** with multi_modal_data
2. ✅ **Audio quality is good** with multi_modal_data (T3 and S3Gen use same voice)
3. ❌ **Removing conditionals breaks audio** (voice mismatch)
4. ✅ **Current code (after revert) is correct**

## Audio Files for Comparison

- `test_with_conditionals.wav` - Generated WITH multi_modal_data (GOOD quality)
- `test_streaming_fix_audio.wav` - Generated WITHOUT multi_modal_data (GARBAGE quality)

## Next Steps for Performance

To improve first chunk time:
1. ✅ T3 streaming is already working (~187ms for 25 tokens)
2. ⚠️ S3Gen is the bottleneck (~433ms)
3. 💡 Reduce S3Gen diffusion steps (already done in WebSocket API)
4. 💡 Consider S3Gen optimization or model tuning
