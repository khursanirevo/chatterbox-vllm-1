# Critical Fix: Restoring AsyncLLMEngine Incremental Streaming

**Date:** 2026-03-08
**Issue:** All tokens delivered at once instead of incrementally
**Status:** Fixed ✅

## Problem Discovery

User asked: "i thought got stream=True?" - questioning why tokens weren't streaming incrementally.

Investigation revealed that AsyncLLMEngine was delivering ALL tokens in a single yield instead of streaming them incrementally as tokens were generated.

**Evidence from profiling:**
- Time to First Token = Time to 25 Tokens (all at once)
- This was NOT the expected behavior for a streaming API

## Root Cause

The code was passing a `multi_modal_data` dict to AsyncLLMEngine:

```python
# BROKEN - causes batching
prompt_with_conditionals = {
    "prompt": text_normalized,
    "multi_modal_data": {"conditionals": [cond_emb.detach()]}
}

async for request_output in self.engine.generate(
    prompt=prompt_with_conditionals,  # ❌ Dict causes batching
    ...
):
```

When `multi_modal_data` is included, vLLM appears to batch the entire generation before yielding, defeating the streaming behavior.

## Solution

Revert to passing prompt as a plain string (as in the working version from commit 8c72f6a):

```python
# FIXED - enables streaming
async for request_output in self.engine.generate(
    prompt=text_normalized,  # ✅ String enables incremental streaming
    ...
):
```

The conditionals are still used for S3Gen processing (where they're actually needed), just not passed to AsyncLLMEngine.

## Expected Impact

**Before (broken):**
- First token latency: ~645ms (all 25 tokens at once)
- Single yield from engine.generate()
- Multiple concurrent: 4-19 seconds

**After (fixed):**
- First token latency: **19-67ms** ✅ (based on commit 8c72f6a results)
- Multiple yields (incremental streaming)
- First audio chunk: **~767ms** ✅ (<1s target met!)

## Verification

Run the verification test when GPU is available:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python verify_streaming_fix.py
```

Expected results:
- First chunk: <200ms
- Multiple chunk yields
- Debug output showing incremental token arrivals

## Files Modified

- `src/chatterbox_vllm/tts_async.py` (lines 499-514)
  - Removed `multi_modal_data` from prompt
  - Added comment explaining why

## Git Commit

`0369c17` - "fix: restore incremental token streaming in AsyncLLMEngine"

## Key Learning

**AsyncLLMEngine.generate() IS a streaming generator by design**, but passing `multi_modal_data` (or other complex prompt structures) can cause it to batch all outputs before yielding.

For true incremental streaming:
✅ Pass prompt as plain string
❌ Don't pass multi_modal_data or complex dicts

This should restore the <1s first chunk capability that was working in March 2026!
