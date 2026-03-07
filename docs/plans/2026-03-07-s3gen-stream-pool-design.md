# S3Gen Stream Pool Design

## Objective

Enable concurrent S3Gen inference using CUDA streams to eliminate the 12x concurrent slowdown bottleneck.

## Problem Statement

From current profiling data:
- **T3 (vLLM)**: Scales well - only 3.4x slowdown with 8x concurrency
- **S3Gen**: 12x slowdown under concurrent load - processes requests sequentially

Current architecture blocks all S3Gen operations:
```
Request 1: ━━━━━━━━━━━━━━━━ (S3Gen blocks for 300ms)
Request 2:   ━━━━━━━━━━━━━━━━ (waits, then blocks)
Request 3:     ━━━━━━━━━━━━━━━━ (waits longer)
...
Request 8:       ━━━━━━━━━━━━━━━━ (massive queue buildup)
```

## Requirements

- **Hardware**: RTX 4090/5090 or H200
- **Concurrency**: 8-16 concurrent requests
- **Quality**: Maintain FP32 (no FP16)
- **Goal**: Reduce 12x concurrent slowdown to ~2-3x

## Solution: CUDA Stream Pool

### Architecture Overview

```
AsyncChatterboxTTS (current generate_stream - unchanged)
                        |
                        v
            S3GenStreamPool (NEW)
  ┌───────────────────────────────────────────────────┐
  │  Stream Pool (12 CUDA streams by default)         │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ... ┌──────┐        │
  │  │stream│ │stream│ │stream│     │stream│        │
  │  │  0   │ │  1   │ │  2   │     │  11  │        │
  │  └──┬───┘ └──┬───┘ └──┬───┘     └───┬──┘        │
  └─────┼────────┼────────┼─────────────┼────────────┘
        │        │        │             │
        v        v        v             v
     Single S3Gen Model (shared across streams)
                        |
                        v
              Concurrent GPU Execution
```

### Key Design Decisions

1. **Single S3Gen Instance**: Model parameters are read-only during inference - safe to share across streams
2. **Stream Pool Size**: Default 12 streams (optimal for 8-16 concurrent requests)
3. **Fair Distribution**: `asyncio.Queue` provides FIFO ordering for stream acquisition
4. **Natural Backpressure**: Queue blocks when all streams busy - prevents overflow

## Components

### 1. S3GenStreamPool Class

**Location**: `src/chatterbox_vllm/s3gen_stream_pool.py` (NEW)

```python
class S3GenStreamPool:
    """Manages a pool of CUDA streams for concurrent S3Gen inference."""

    def __init__(
        self,
        s3gen_model: S3Token2Wav,
        num_streams: int = 12,
        device: str = "cuda",
    ):
        self.s3gen = s3gen_model
        self.device = device
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]

    async def initialize(self) -> None:
        """Initialize the stream queue (must be called in async context)."""
        self.stream_queue = asyncio.Queue()
        for stream in self.streams:
            await self.stream_queue.put(stream)

    async def process_async(
        self,
        token_chunk: torch.Tensor,
        context_tokens: Optional[torch.Tensor],
        s3gen_ref: dict[str, Any],
        context_window: int = 50,
        fade_duration: float = 0.02,
        diffusion_steps: int = 10,
    ) -> Optional[torch.Tensor]:
        """
        Process tokens through S3Gen on an available CUDA stream.

        Flow:
        1. Acquire stream from queue (blocks if none available)
        2. Run synchronous inference in thread pool with assigned stream
        3. Return stream to queue for reuse
        4. Return audio result
        """
        queue_start = time.time()
        stream = await self.stream_queue.get()
        queue_wait_ms = (time.time() - queue_start) * 1000

        self.metrics.active_streams += 1
        self.metrics.queue_depth = self.stream_queue.qsize()

        try:
            loop = asyncio.get_event_loop()

            def _inference_on_stream():
                with torch.cuda.stream(stream):
                    tokens_to_process = self._build_token_context(
                        token_chunk, context_tokens, context_window
                    )
                    audio = self.s3gen.inference(
                        speech_tokens=tokens_to_process,
                        ref_dict=s3gen_ref,
                        finalize=False,
                        n_timesteps=diffusion_steps,
                    )
                    audio = self._post_process_audio(audio, fade_duration)
                    return audio

            audio_chunk = await loop.run_in_executor(None, _inference_on_stream)
            return audio_chunk

        finally:
            self.metrics.active_streams -= 1
            await self.stream_queue.put(stream)
            self.metrics.total_requests += 1
            self.metrics.avg_queue_wait_ms = (
                (self.metrics.avg_queue_wait_ms * (self.metrics.total_requests - 1) + queue_wait_ms)
                / self.metrics.total_requests
            )

    async def shutdown(self) -> None:
        """Gracefully shutdown - wait for active requests to complete."""
        while self.metrics.active_streams > 0:
            await asyncio.sleep(0.1)
        while not self.stream_queue.empty():
            self.stream_queue.get_nowait()
```

### 2. StreamPoolMetrics Dataclass

```python
@dataclass
class StreamPoolMetrics:
    """Metrics for stream pool performance monitoring."""
    total_requests: int = 0
    active_streams: int = 0
    queue_depth: int = 0
    avg_queue_wait_ms: float = 0.0
    stream_utilization: List[float] = field(default_factory=list)
```

### 3. Modified AsyncChatterboxTTS

**Changes in `src/chatterbox_vllm/tts_async.py`**:

```python
class AsyncChatterboxTTS:
    def __init__(
        self,
        # ... existing params ...
        s3gen_stream_pool: Optional[S3GenStreamPool] = None,  # NEW
    ):
        # ... existing ...
        self.s3gen_stream_pool = s3gen_stream_pool

    @classmethod
    async def from_pretrained(
        cls,
        # ... existing params ...
        enable_stream_pool: bool = True,  # NEW
        num_s3gen_streams: int = 12,  # NEW
        **kwargs
    ) -> "AsyncChatterboxTTS":
        # ... existing model loading ...

        s3gen_stream_pool = None
        if enable_stream_pool:
            s3gen_stream_pool = S3GenStreamPool(
                s3gen_model=s3gen,
                num_streams=num_s3gen_streams,
                device=device,
            )
            await s3gen_stream_pool.initialize()

        return cls(
            # ... existing ...
            s3gen_stream_pool=s3gen_stream_pool,
        )
```

### 4. Integration Point

Replace `_process_token_chunk_async` calls in `generate_stream()`:

```python
# OLD:
audio_chunk = await self._process_token_chunk_async(...)

# NEW:
if self.s3gen_stream_pool:
    audio_chunk = await self.s3gen_stream_pool.process_async(...)
else:
    audio_chunk = await self._process_token_chunk_async(...)  # Fallback
```

## Data Flow

### Concurrent Request Flow

```
Time →

Request A:  [Get Stream]──[S3Gen on Stream 3]──[Return Stream]──→ Audio A
Request B:    [Get Stream]──[S3Gen on Stream 7]──[Return Stream]──→ Audio B
Request C:      [Get Stream]──[S3Gen on Stream 2]──[Return Stream]──→ Audio C
Request D:        [Get Stream]──[S3Gen on Stream 11]──[Return Stream]──→ Audio D
               ↓ queue wait   ↓ concurrent GPU execution
```

### CUDA Stream Synchronization

PyTorch handles concurrent streams automatically:

```python
# When you execute:
with torch.cuda.stream(stream_3):
    output = s3gen.inference(tokens)  # Launch kernel on stream 3

with torch.cuda.stream(stream_7):
    output = s3gen.inference(tokens)  # Launch kernel on stream 7

# PyTorch/CUDA runtime handles:
# 1. Each stream maintains its own command queue
# 2. GPU scheduler executes kernels from all streams concurrently
# 3. No explicit synchronization needed for independent operations
```

### Memory Safety

Single model, multiple streams is safe because:
1. S3Gen parameters are read-only during inference (`@torch.inference_mode()`)
2. Each stream has its own workspace memory (activations)
3. No race conditions - only reads from model weights

## Error Handling

### Key Guarantees

1. **Stream Always Returned**: `finally` block ensures pool integrity
2. **CUDA OOM Isolation**: One OOM doesn't crash other requests
3. **Metric Accuracy**: Metrics updated correctly even on failure

### Error Handling Pattern

```python
try:
    # ... inference code ...
    return audio_chunk

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logger.warning(f"CUDA OOM on stream, request failed: {e}")
        return None
    else:
        logger.error(f"CUDA runtime error: {e}")
        raise

except ValueError as e:
    logger.warning(f"Invalid input to S3Gen: {e}")
    return None

finally:
    # CRITICAL: Always return stream to pool
    await self.stream_queue.put(stream)
    self.metrics.active_streams -= 1
```

## Testing Strategy

### Unit Tests

**File**: `tests/test_s3gen_stream_pool.py`

- Test stream initialization
- Test single request processing
- Test concurrent requests
- Test stream reuse
- Test empty token handling
- Test metrics tracking

### Integration Test

**File**: `tests/test_stream_pool_concurrent.py`

- Test 8 concurrent requests complete successfully
- Verify first chunk latency < 1s with stream pool
- Verify stream pool metrics are accurate

### Benchmark

**File**: `benchmarks/benchmark_stream_pool.py`

Compare performance across concurrency levels (1, 4, 8, 12, 16):
- With stream pool: ~500-800ms first chunk @ 8 concurrent
- Without stream pool: ~3000ms first chunk @ 8 concurrent

### Expected Results

| Metric | Without Pool | With Pool | Improvement |
|--------|--------------|-----------|-------------|
| 1 concurrent first chunk | ~485ms | ~500ms | ~0% (baseline) |
| 8 concurrent first chunk | ~3138ms | ~700ms | **~4.5x faster** |
| RTF @ 8 concurrent | 5.2 | ~1.2 | **~4x better** |
| S3Gen throughput | Sequential | Parallel | **~8x concurrent** |

## Implementation Checklist

- [ ] Create `src/chatterbox_vllm/s3gen_stream_pool.py`
- [ ] Implement `S3GenStreamPool` class
- [ ] Implement `StreamPoolMetrics` dataclass
- [ ] Modify `AsyncChatterboxTTS.__init__()` to accept stream pool
- [ ] Modify `AsyncChatterboxTTS.from_pretrained()` to create stream pool
- [ ] Update `generate_stream()` to use stream pool
- [ ] Add `--enable-stream-pool` flag to websocket_api.py
- [ ] Add `--num-s3gen-streams` flag to websocket_api.py
- [ ] Create unit tests
- [ ] Create integration tests
- [ ] Create benchmark script
- [ ] Create verification script
- [ ] Update MEMORY.md with stream pool documentation
- [ ] Test with concurrent load

## Future Enhancements

1. **Dynamic Stream Sizing**: Auto-tune stream count based on load
2. **Priority Queues**: Support different priority levels
3. **Stream Health Monitoring**: Detect and restart unhealthy streams
4. **Per-Stream Metrics**: Track utilization per stream for optimization
5. **Pipeline Parallelism**: Use separate streams for prep/infer/postprocess stages

## References

- PyTorch CUDA Streams Documentation: https://pytorch.org/docs/stable/cuda.html#torch.cuda.Stream
- CUDA C Programming Guide: Streams and Events
- Existing profiling data in MEMORY.md (Session: 2026-03-07)
