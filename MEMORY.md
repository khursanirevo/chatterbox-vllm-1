# Chatterbox vLLM - Project Memory

This document tracks the complete history of work done on the Chatterbox vLLM TTS streaming implementation.

## Project Overview

**Repository**: chatterbox-vllm
**Branch**: simple
**Main Branch**: master
**Goal**: Add streaming TTS capability to the vLLM Chatterbox implementation while maintaining batch processing efficiency.

---

## Session: 2026-03-07 - Streaming TTS Implementation

### Objective
Implement `generate_stream()` method for the vLLM Chatterbox TTS to enable real-time audio playback with incremental chunk delivery, matching functionality from the non-vLLM version.

### Implementation Approach

**Strategy**: Two-stage streaming
1. **Stage 1**: Generate all speech tokens using vLLM (fast batch processing)
2. **Stage 2**: Stream token chunks through S3Gen incrementally for real-time audio

This approach:
- ✅ Maintains vLLM's batch processing efficiency for token generation
- ✅ Enables streaming for real-time audio playback
- ✅ Minimal code changes (~200 lines added)
- ✅ Backward compatible (existing `generate()` method unchanged)

### Files Modified

#### 1. `src/chatterbox_vllm/tts.py`
**Changes**:
- Added `StreamingMetrics` dataclass (line ~29-36)
- Added `_process_token_chunk()` method (line ~250-330)
- Added `generate_stream()` method (line ~450-545)
- Updated imports to include `Generator` type

**Key Features**:
- Configurable chunk size (default: 25 tokens)
- Context window for audio continuity (default: 50 tokens)
- Fade-in between chunks (default: 0.02s)
- Comprehensive metrics tracking (latency, RTF, chunk count)
- Full compatibility with existing vLLM sampling parameters

#### 2. `example-tts-stream.py` (NEW)
Demonstrates streaming usage:
- Sets GPU visibility before CUDA operations
- Generates streaming audio chunk by chunk
- Collects and saves final audio output
- Prints progress metrics

#### 3. `test-generate-sizes.py` (NEW)
Tests streaming with different text lengths:
- Short text (6 words, ~2s audio, max_tokens=500)
- Medium text (32 words, ~10s audio, max_tokens=1000)
- Long text (136 words, ~46s audio, max_tokens=2000)

#### 4. `ERROR_FIXED.md` (NEW)
Documents all issues encountered and solutions:
1. GPU Memory Configuration Issue
2. Tensor Dimension Mismatch in Streaming
3. Torchaudio API Incompatibility
4. Misleading "Tokenizer Not Found" Error

### Issues Encountered and Resolved

#### Issue 1: GPU Memory Configuration
**Problem**: `ValueError: No available memory for the cache blocks` with negative memory

**Solution**:
- Set `CUDA_VISIBLE_DEVICES` at script start (not as command prefix)
- Explicitly set `gpu_memory_utilization=0.90`

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = ChatterboxTTS.from_pretrained(
    gpu_memory_utilization=0.90,
)
```

#### Issue 2: Tensor Dimension Mismatch
**Problem**: `RuntimeError: Tensors must have same number of dimensions: got 1 and 2`

**Root Cause**: `all_tokens_processed` was 1D, `token_chunk` was 2D

**Solution**: Added dimension normalization in `_process_token_chunk()`:
```python
# Ensure all_tokens_so_far is 1D for slicing
if all_tokens_so_far.dim() > 1:
    all_tokens_so_far = all_tokens_so_far.squeeze(0)
# Ensure token_chunk is 1D for concatenation
token_chunk_1d = token_chunk.squeeze(0) if token_chunk.dim() > 1 else token_chunk
tokens_to_process = torch.cat([context_tokens, token_chunk_1d], dim=-1).unsqueeze(0)
```

#### Issue 3: Torchaudio API Incompatibility
**Problem**: `AttributeError: module 'torchaudio' has no attribute 'concatenate'`

**Solution**: Use PyTorch's native concatenation:
```python
# Instead of: ta.concatenate(audio_chunks)
full_audio = torch.cat(audio_chunks, dim=-1)
```

#### Issue 4: Misleading Tokenizer Error
**Problem**: "Tokenizer EnTokenizer not found" when GPU memory failed

**Resolution**: This was a red herring - the real issue was GPU memory. Fixing memory configuration resolved the tokenizer loading automatically.

### Performance Results

Test configuration: `max_model_len=2000`, `gpu_memory_utilization=0.90`, GPU 0 only

| Text Length | Duration | Chunks | RTF | Latency | File |
|-------------|----------|--------|-----|---------|------|
| Short (6 words) | 2.00s | 2 | 1.028 | 1.68s | test-short.wav (94KB) |
| Medium (32 words) | 9.88s | 10 | 0.576 | 2.30s | test-medium.wav (464KB) |
| Long (136 words) | 46.36s | 47 | 0.551 | 8.85s | test-long.wav (2.2MB) |

**Key Observations**:
- RTF of 0.55 means audio generates ~45% faster than real-time
- Longer texts benefit more from streaming (better RTF)
- Latency scales sub-linearly with text length
- All files are standard 24kHz mono WAV format

### API Usage

#### Basic Streaming
```python
from chatterbox_vllm.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(
    max_model_len=2000,
    gpu_memory_utilization=0.90,
)

for audio_chunk, metrics in model.generate_stream(
    text="Your text here",
    chunk_size=25,
    context_window=50,
    print_metrics=True,
):
    # Process audio_chunk (1, T) tensor
    # Play or save incrementally
    pass

model.shutdown()
```

#### Parameters
- `text`: Input text to synthesize
- `audio_prompt_path`: Optional reference audio for voice cloning
- `language_id`: Language code (multilingual variant only)
- `exaggeration`: Emotion exaggeration (0.0 to 1.0)
- `temperature`: Sampling temperature
- `max_tokens`: Maximum tokens to generate
- `chunk_size`: Speech tokens per audio chunk (default: 25)
- `context_window`: Context tokens for continuity (default: 50)
- `fade_duration`: Fade-in duration in seconds (default: 0.02)
- `diffusion_steps`: S3Gen diffusion steps (default: 10)
- `top_p`: Top-p sampling parameter
- `repetition_penalty`: Repetition penalty

### Generated Test Files

- `test-short.wav` - 2 seconds, short text
- `test-medium.wav` - 10 seconds, medium text
- `test-long.wav` - 46 seconds, long text
- `test-streaming-vllm.wav` - Original streaming demo

### Technical Architecture

```
Input Text
    ↓
[Stage 1: vLLM T3 Generation]
    Generate all speech tokens (batch-optimized)
    ↓
Speech Tokens (all at once)
    ↓
[Stage 2: S3Gen Streaming]
    Split into chunks → Process with context window
    ↓
Audio Chunks (yielded incrementally)
    ↓
Real-time playback or saving
```

**Context Window**: Each chunk includes previous tokens for continuity
**Fade-in**: Smooth transitions between chunks
**Metrics**: Tracks latency, RTF, chunk count

### Future Enhancements

Potential improvements identified:

1. **AsyncLLMEngine Integration**: True streaming during token generation for lower latency
2. **Token-level streaming**: Stream tokens during T3 generation (requires vLLM engine changes)
3. **Batch streaming**: Support multiple concurrent streams
4. **Backpressure handling**: Pause generation if consumer is slow
5. **Real-time playback integration**: Add audio playback threading for immediate output

### Related Files

- **Main implementation**: `src/chatterbox_vllm/tts.py`
- **Streaming examples**: `example-tts-stream.py`, `test-generate-sizes.py`
- **Error documentation**: `ERROR_FIXED.md`
- **This memory**: `MEMORY.md`

### Commit Information

**Branch**: simple
**Status**: Ready for commit
**Changes**:
- Modified: `src/chatterbox_vllm/tts.py` (~200 lines added)
- New: `example-tts-stream.py`
- New: `test-generate-sizes.py`
- New: `ERROR_FIXED.md`
- New: `MEMORY.md` (this file)

**Generated artifacts** (not in git):
- `test-short.wav`, `test-medium.wav`, `test-long.wav`
- `test-streaming-vllm.wav`

---

## Next Session Setup

To continue development, load this repository and review:
1. This `MEMORY.md` file
2. `ERROR_FIXED.md` for known issues
3. `test-generate-sizes.py` for usage examples
4. Current implementation in `src/chatterbox_vllm/tts.py`

**Quick test**:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python example-tts-stream.py
```

**Key files to understand**:
- Lines 29-36: `StreamingMetrics` dataclass
- Lines 250-330: `_process_token_chunk()` method
- Lines 450-545: `generate_stream()` method

**Remember**: Always set `CUDA_VISIBLE_DEVICES` before CUDA operations and use `gpu_memory_utilization=0.90` for optimal performance.
