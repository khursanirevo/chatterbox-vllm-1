# AsyncLLMEngine Custom Tokenizer Fix

## Problem

When using `ChatterboxTTSAsync` or `ChatterboxTTSStreaming` (which use `AsyncLLMEngine`), the following error occurred:

```
ValueError: Tokenizer EnTokenizer not found.
```

## Root Cause

`AsyncLLMEngine` spawns worker processes using Python's `multiprocessing` module. When CUDA is initialized, vLLM uses the `spawn` method (not `fork`), which creates fresh Python interpreters that:

1. Don't inherit the parent process's imports
2. Don't have the custom tokenizer registrations from `chatterbox_vllm/__init__.py`
3. Fail when trying to look up `EnTokenizer` or `MtlTokenizer` in the `TokenizerRegistry`

## Solution

The fix uses Python's `sitecustomize.py` mechanism to automatically register tokenizers in spawned worker processes.

### Implementation

1. **Created `src/chatterbox_vllm/sitecustomize.py`**
   - Automatically imported by Python when starting up
   - Registers custom tokenizers before any vLLM code runs
   - Uses try/except to handle cases where vLLM isn't installed

2. **Created `src/chatterbox_vllm/vllm_worker_patch.py`**
   - `apply_worker_patch()` function adds the chatterbox_vllm directory to `PYTHONPATH`
   - Spawned processes inherit `PYTHONPATH` and automatically import `sitecustomize.py`
   - Also adds the path to `sys.path` for the current process

3. **Modified `src/chatterbox_vllm/tts_async.py`**
   - Calls `apply_worker_patch()` before creating `AsyncLLMEngine`
   - Ensures `PYTHONPATH` is set before worker processes are spawned

4. **Modified `src/chatterbox_vllm/__init__.py`**
   - Exports `apply_worker_patch` for external use
   - Updated documentation to reflect the fix

### How It Works

1. User calls `ChatterboxTTSStreaming.from_pretrained()`
2. `apply_worker_patch()` adds `src/chatterbox_vllm` to `PYTHONPATH`
3. `AsyncLLMEngine` spawns worker processes via `multiprocessing.spawn`
4. Each spawned process:
   - Inherits the `PYTHONPATH` environment variable
   - Automatically imports `sitecustomize.py` from that path
   - `sitecustomize.py` registers the custom tokenizers
   - vLLM code runs and can find the registered tokenizers

## Testing

### Test 1: AsyncLLMEngine Initialization
```bash
uv run python -c "
import asyncio
import sys
sys.path.insert(0, 'src')

from chatterbox_vllm import ChatterboxTTSStreaming

async def test():
    tts = await ChatterboxTTSStreaming.from_pretrained(
        max_batch_size=4,
        max_model_len=500,
    )
    print(f'SUCCESS: AsyncLLMEngine initialized! SR={tts.sr}')
    tts.shutdown()

asyncio.run(test())
"
```

Expected output: `SUCCESS: AsyncLLMEngine initialized! SR=24000`

### Test 2: Token-Level Streaming
```bash
uv run python example-token-streaming-ttfa.py
```

Expected output: TTFA measurements showing streaming performance

### Test 3: WebSocket Service
```bash
uv run uvicorn websocket_service:app --host 127.0.0.1 --port 8002
```

Then test with client:
```bash
python test_websocket_client.py
```

## Benefits

- ✅ AsyncLLMEngine works with CUDA (no need for fork-based multiprocessing)
- ✅ Continuous batching support for improved throughput
- ✅ Token-level streaming for improved TTFA
- ✅ No modification to vLLM installation required
- ✅ Works with both v0 and v1 engine architectures

## Files Modified

1. **NEW**: `src/chatterbox_vllm/sitecustomize.py` - Auto-imported by spawned processes
2. **NEW**: `src/chatterbox_vllm/vllm_worker_patch.py` - Worker patch utility
3. **MODIFY**: `src/chatterbox_vllm/tts_async.py` - Apply patch before engine creation
4. **MODIFY**: `src/chatterbox_vllm/__init__.py` - Export patch utility and update docs

## Rollback

If the fix causes issues, you can:
1. Remove the `apply_worker_patch()` call from `tts_async.py`
2. Delete `src/chatterbox_vllm/vllm_worker_patch.py`
3. Delete `src/chatterbox_vllm/sitecustomize.py`
4. Remove `apply_worker_patch` from `__init__.py` exports
5. Use `ChatterboxTTSAsyncWrapper` as the fallback async implementation
