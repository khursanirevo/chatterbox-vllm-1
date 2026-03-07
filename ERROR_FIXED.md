# Streaming TTS Implementation - Issues and Fixes

This document documents the issues encountered during implementation and testing of the streaming TTS functionality, along with their solutions.

## Issues Encountered

### 1. GPU Memory Configuration Issue

**Error**:
```
ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`
Available KV cache memory: -0.26 GiB  # Negative memory!
```

**Root Cause**:
- The memory calculation heuristic in `ChatterboxTTS.from_local()` calculates `unused_gpu_memory` based on all GPUs visible to CUDA
- When restricting to a single GPU via `CUDA_VISIBLE_DEVICES=0`, the calculation became invalid
- The heuristic uses: `unused_gpu_memory = total_gpu_memory - torch.cuda.memory_allocated()`
- This calculation happens before vLLM is fully initialized, leading to incorrect values

**Solution**:
Set `CUDA_VISIBLE_DEVICES` at the beginning of the Python script (before any imports) and explicitly specify GPU memory utilization:

```python
import os
# Set GPU before any CUDA operations
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = ChatterboxTTS.from_pretrained(
    max_batch_size=3,
    max_model_len=1000,
    gpu_memory_utilization=0.90,  # Use 90% of GPU 0's memory
)
```

**DO NOT** use: `CUDA_VISIBLE_DEVICES=0 uv run python script.py` (command prefix)
**DO** set it inside the script before any CUDA operations.

---

### 2. Tensor Dimension Mismatch in Streaming

**Error**:
```
RuntimeError: Tensors must have same number of dimensions: got 1 and 2
  File "/mnt/data/work/chatterbox-vllm/src/chatterbox_vllm/tts.py", line 278, in _process_token_chunk
    tokens_to_process = torch.cat([context_tokens, token_chunk], dim=-1)
```

**Root Cause**:
In `generate_stream()` method:
- `all_tokens_processed` was initialized as a 1D tensor: `torch.tensor([], device="cuda", dtype=torch.long)`
- `token_chunk` was 2D: `speech_tokens[i:i + chunk_size].unsqueeze(0)` → shape `(1, T_new)`
- When updating processed tokens: `all_tokens_processed = torch.cat([all_tokens_processed, chunk.squeeze(0)], dim=0)` kept it 1D
- `_process_token_chunk()` tried to concatenate 1D context with 2D chunk, causing dimension mismatch

**Solution**:
Updated `_process_token_chunk()` to handle dimension mismatches by normalizing to 1D before concatenation:

```python
# Build tokens with context window
if len(all_tokens_so_far) > 0:
    # Ensure all_tokens_so_far is 1D for slicing, then reshape
    if all_tokens_so_far.dim() > 1:
        all_tokens_so_far = all_tokens_so_far.squeeze(0)
    context_tokens = (
        all_tokens_so_far[-context_window:]
        if len(all_tokens_so_far) > context_window
        else all_tokens_so_far
    )
    # Ensure token_chunk is 1D for concatenation
    token_chunk_1d = token_chunk.squeeze(0) if token_chunk.dim() > 1 else token_chunk
    tokens_to_process = torch.cat([context_tokens, token_chunk_1d], dim=-1).unsqueeze(0)
    context_length = len(context_tokens)
else:
    tokens_to_process = token_chunk
    context_length = 0
```

---

### 3. Torchaudio API Incompatibility

**Error**:
```
AttributeError: module 'torchaudio' has no attribute 'concatenate'
  File "/mnt/data/work/chatterbox-vllm/example-tts-stream.py", line 41, in <module>
    full_audio = ta.concatenate(audio_chunks)
```

**Root Cause**:
- `torchaudio.concatenate()` doesn't exist in older versions of torchaudio
- The API changed between versions, with concatenation moving to PyTorch core

**Solution**:
Use PyTorch's native `torch.cat()` instead:

```python
import torch

# Instead of: full_audio = ta.concatenate(audio_chunks)
full_audio = torch.cat(audio_chunks, dim=-1)
```

---

### 4. Misleading "Tokenizer Not Found" Error

**Error** (when running with incorrect GPU configuration):
```
ValueError: Tokenizer EnTokenizer not found.
```

**Important**: This was NOT actually a tokenizer issue!

**Root Cause**:
- When GPU memory allocation fails during vLLM engine initialization, the initialization fails prematurely
- The vLLM worker process runs in a separate multiprocessing spawn
- The failure occurs before the tokenizer registry is fully set up
- This results in a confusing "tokenizer not found" error instead of the actual memory error

**How to Identify the Real Issue**:
1. Try running the existing `example-tts.py` (non-streaming version)
2. If it also fails, the issue is likely environment/configuration, not the new code
3. Check for "Available KV cache memory" messages in the logs

**Solution**:
Fix the underlying GPU memory configuration issue (see Issue #1 above). Once the GPU settings are correct, the tokenizer loads fine without any changes.

---

## Test Results

After fixing all issues, the streaming implementation works correctly:

```
Generating streaming audio for: This is a streaming demo...

[T3] Speech token generation: 1.45s
Latency to first chunk: 2.354s
Received chunk 1: shape=torch.Size([1, 24000]), duration=1.000s
Received chunk 2: shape=torch.Size([1, 24000]), duration=1.000s
... (7 more chunks) ...
Received chunk 9: shape=torch.Size([1, 14400]), duration=0.600s
[S3Gen] Streaming complete: 9 chunks
Total time: 6.16s, Audio: 8.60s, RTF: 0.716

Saved streaming audio to test-streaming-vllm.wav
Total chunks: 9
Final duration: 8.60s
```

**Performance Metrics**:
- **RTF (Real-Time Factor)**: 0.716 (generates ~40% faster than real-time)
- **Latency to first chunk**: ~2.3s (T3 generation + first S3Gen chunk)
- **Total duration**: 8.60 seconds of audio
- **Chunk size**: 1 second chunks (24000 samples at 24kHz)

---

### 5. T3 Model Configuration Issues

**Error**:
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1024 (input tensor's size at dimension 1), but got split_sizes=[1024, 1024]

ValueError: Invalid repository ID or local directory specified: './t3-model'
```

**Root Cause**: Attempting to "fix" `t3-model/config.json` without understanding the T3 model's unique architecture.

**The Misunderstanding**:
1. Saw `vocab_size: 8` in the config and thought it was wrong (tokenizer has 704 tokens)
2. Changed `hidden_size` from 2048 to 1024, breaking CFG (Classifier-Free Guidance)
3. Changed `vocab_size` to match tokenizer size, breaking LLaMA backbone loading

**The Reality: T3 Has THREE Separate Embeddings**

The T3 model architecture uses **three different token embeddings**, not one:

| Embedding | Vocab Size | Purpose |
|-----------|-----------|---------|
| `text_emb` | 704 (En) / 2454 (Multilingual) | **Input text tokens** from EnTokenizer |
| `speech_emb` | 8194 | **Speech codec tokens** (EnCodec) for S3Gen |
| `tfmr.embed_tokens` (LLaMA backbone) | 8 | Pre-trained LLaMA-520M base model |

**Why vocab_size=8 is Correct**:
- The `tfmr.embed_tokens` with vocab_size=8 is the **LLaMA backbone only**
- T3 doesn't actually use this for text/speech
- T3 uses custom embeddings (`text_emb` and `speech_emb`) instead
- The config's vocab_size only describes the LLaMA backbone, NOT the working vocabulary

**Why hidden_size=2048 is Critical**:
```python
# In t3.py forward method (line 691):
cond_embeds, uncond_embeds = inputs_embeds.split([self.dim, self.dim], dim=1)
```

This requires 2048 dims total for CFG:
- 1024 dims for conditional embeddings (speaker + audio prompt)
- 1024 dims for unconditional embeddings (negative prompt)
- CFG scale combines them: `cond_logits + cfg_scale * (cond_logits - uncond_logits)`

**The SPEECH_TOKEN_OFFSET Trick**:
```python
SPEECH_TOKEN_OFFSET = 2500  # In t3.py line 49
```

- Tokens < 2500: Prefill tokens (conditionals, speaker embedding)
- Tokens >= 2500: Speech generation tokens (2500 + actual token_id)
- This allows distinguishing prefill from decode within the same sequence

**Effective vocab_size for logits**:
```python
# In t3.py compute_logits (line 663-665):
logits = torch.cat([
    torch.zeros(logits.shape[0], SPEECH_TOKEN_OFFSET).fill_(-inf),  # 2500 fake dims
    logits,  # 8194 speech tokens
], dim=1)
# Total: 2500 + 8194 = 10694 effective vocab
```

**Solution**:
NEVER modify `t3-model/config.json` without understanding the full T3 architecture. The correct config is:

```json
{
  "architectures": ["ChatterboxT3"],     // Custom T3 model class
  "model_type": "llama",                // Extends LLaMA
  "hidden_size": 2048,                   // CRITICAL: for CFG split
  "vocab_size": 8                        // LLaMA backbone vocab only
}
```

**History**:
- **Commit 337730b** (Jun 2025): Originally a symlink to `../src/chatterbox_vllm/models/t3/config.json`
- **Commit d93dabc** (Aug 2025): Converted to regular file (fix issue #12)
- **This session**: Accidentally broke it by trying to "fix" it
- **Fix**: `git checkout t3-model/config.json` to restore working version

**Key Lesson**:
> The T3 model's config.json describes the LLaMA backbone parameters, NOT the actual working vocabulary. The model uses custom embeddings and CFG which require specific values. Never modify it without deep understanding of the architecture.

**Related**:
- T3 model source: `src/chatterbox_vllm/models/t3/t3.py`
- T3 config: `src/chatterbox_vllm/models/t3/modules/t3_config.py`
- Design doc: `docs/plans/2026-03-07-s3gen-stream-pool-design.md`

---

## Key Takeaways

1. **Always set GPU visibility before CUDA operations**: When using `CUDA_VISIBLE_DEVICES`, set it at the very beginning of your Python script, not as a command prefix.

2. **Dimension consistency is critical**: When working with tensors in streaming contexts, be explicit about dimensions and handle both 1D and 2D cases.

3. **API compatibility matters**: Use PyTorch core functions when possible, as they're more stable across versions than library-specific APIs.

4. **Error messages can be misleading**: When debugging, run known-working code to isolate whether the issue is with your changes or the environment configuration.

5. **T3 model architecture is complex**: The config.json describes the LLaMA backbone (vocab_size=8, hidden_size=2048), but the actual model uses custom embeddings with much larger effective vocabulary. Don't modify config without understanding the full architecture.

---

## Related Files

- **Implementation**: `src/chatterbox_vllm/tts.py`
- **Example**: `example-tts-stream.py`
- **Streaming method**: `generate_stream()` (line ~450)
- **Token processing**: `_process_token_chunk()` (line ~250)
