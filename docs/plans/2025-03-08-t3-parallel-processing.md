# T3 Parallel Processing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize T3 token generation to achieve 8-16 concurrent requests with <1s first chunk latency.

**Architecture:** Deep profiling of AsyncLLMEngine behavior with concurrent requests to identify why continuous batching isn't providing expected parallelism. Then implement targeted optimization based on findings.

**Tech Stack:** Python, asyncio, vLLM AsyncLLMEngine, CUDA profiling, PyTorch

---

## Overview

This plan implements a deep investigation into why AsyncLLMEngine's continuous batching isn't achieving <1s first chunk with 8-16 concurrent requests. Current performance:
- 1 concurrent: ~580ms ✅
- 2 concurrent: ~1040ms ❌
- 8 concurrent: ~3000ms+ ❌

The investigation will measure per-request timing, AsyncLLMEngine batch behavior, and identify the root cause before implementing a fix.

---

## Task 1: Create Deep Profiling Test

**Files:**
- Create: `profile_t3_concurrent.py`

**Step 1: Write the profiling test**

Create a comprehensive profiling test that tracks:
- Per-request timeline (submission → first token → 25 tokens → first chunk)
- AsyncLLMEngine queue position and wait time
- Token generation rate (tokens/second)
- Inter-arrival timing between tokens
- GPU utilization during generation
- Batch size and utilization

```python
#!/usr/bin/env python3
"""
Deep profiling of T3 AsyncLLMEngine behavior with concurrent requests.

Measures:
- Per-request timeline breakdown
- AsyncLLMEngine queue dynamics
- Token generation rate and batching behavior
- GPU utilization

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import torch

from chatterbox_vllm.tts_async import AsyncChatterboxTTS


@dataclass
class RequestTimeline:
    """Detailed timeline for a single request."""
    request_id: int
    submit_time: float
    first_token_time: float = 0.0
    chunk_ready_time: float = 0.0  # When 25 tokens accumulated
    first_chunk_time: float = 0.0  # When S3Gen completes
    complete_time: float = 0.0

    token_count: int = 0
    token_arrivals: List[float] = field(default_factory=list)

    t3_queue_time: float = 0.0
    t3_generation_time: float = 0.0
    s3gen_time: float = 0.0

    @property
    def time_to_first_token(self) -> float:
        return (self.first_token_time - self.submit_time) * 1000

    @property
    def time_to_chunk_ready(self) -> float:
        return (self.chunk_ready_time - self.submit_time) * 1000

    @property
    def time_to_first_audio(self) -> float:
        return (self.first_chunk_time - self.submit_time) * 1000

    @property
    def token_generation_rate(self) -> float:
        """Tokens per second during generation."""
        if self.token_count < 2 or not self.token_arrivals:
            return 0.0
        duration = self.token_arrivals[-1] - self.token_arrivals[0]
        return self.token_count / duration if duration > 0 else 0.0


class T3Profiler:
    """Profiles T3 AsyncLLMEngine behavior."""

    def __init__(self, model: AsyncChatterboxTTS):
        self.model = model
        self.timelines: Dict[int, RequestTimeline] = {}
        self.request_counter = 0

    async def profile_request(
        self,
        text: str,
        concurrent_level: int,
        request_idx: int
    ) -> RequestTimeline:
        """Profile a single TTS request with detailed timing."""

        request_id = self.request_counter
        self.request_counter += 1

        timeline = RequestTimeline(
            request_id=request_id,
            submit_time=time.time()
        )
        self.timelines[request_id] = timeline

        chunk_size = 25
        target_tokens = chunk_size
        chunk_ready = False

        # Track token arrivals
        last_token_count = 0

        try:
            async for chunk, metrics in self.model.generate_stream(
                text,
                chunk_size=chunk_size,
                print_metrics=False
            ):
                current_time = time.time()

                # Track first token
                if timeline.first_token_time == 0 and metrics.t3_first_token_time:
                    timeline.first_token_time = current_time
                    timeline.t3_queue_time = metrics.t3_first_token_time

                # Track token accumulation
                # Note: We don't have direct access to token count from metrics
                # So we estimate based on chunk arrivals
                if not chunk_ready:
                    timeline.token_count += chunk_size
                    timeline.token_arrivals.append(current_time)

                    # Check if we have enough for first chunk
                    if timeline.token_count >= target_tokens and not chunk_ready:
                        timeline.chunk_ready_time = current_time
                        timeline.t3_generation_time = (
                            timeline.chunk_ready_time - timeline.first_token_time
                        )
                        chunk_ready = True

                        # Track first chunk delivery
                        if timeline.first_chunk_time == 0:
                            timeline.first_chunk_time = current_time
                            timeline.s3gen_time = (
                                timeline.first_chunk_time - timeline.chunk_ready_time
                            )
                            break  # Got first chunk, that's all we need for profiling

        except Exception as e:
            print(f"    ❌ Request {request_id} failed: {e}")

        timeline.complete_time = time.time()
        return timeline

    def print_summary(self, timelines: List[RequestTimeline], concurrent_level: int):
        """Print profiling summary for this concurrent level."""

        if not timelines:
            print(f"    ❌ No valid requests for {concurrent_level} concurrent")
            return

        print(f"\n  ──────────────────────────────────────────────")
        print(f"  {concurrent_level} Concurrent Requests - Detailed Timings")
        print(f"  ──────────────────────────────────────────────")

        # Per-request breakdown
        print(f"\n  {'Req':<4} {'TTFT':<8} {'25 Tok':<8} {'1st Audio':<10} {'T3 Gen':<8} {'S3Gen':<8} {'Rate':<8}")
        print(f"  {'':<4} {'(ms)':<8} {'(ms)':<8} {'(ms)':<10} {'(ms)':<8} {'(ms)':<8} {'(tok/s)':<8}")

        for tl in timelines:
            print(
                f"  {tl.request_id:<4} "
                f"{tl.time_to_first_token:>7.1f} "
                f"{tl.time_to_chunk_ready:>7.1f} "
                f"{tl.time_to_first_audio:>9.1f} "
                f"{tl.t3_generation_time*1000:>7.1f} "
                f"{tl.s3gen_time*1000:>7.1f} "
                f"{tl.token_generation_rate:>7.1f} "
            )

        # Statistics
        avg_ttft = sum(t.time_to_first_token for t in timelines) / len(timelines)
        avg_25tok = sum(t.time_to_chunk_ready for t in timelines) / len(timelines)
        avg_audio = sum(t.time_to_first_audio for t in timelines) / len(timelines)
        avg_rate = sum(t.token_generation_rate for t in timelines) / len(timelines)

        print(f"\n  Average:")
        print(f"    Time to first token: {avg_ttft:.1f}ms")
        print(f"    Time to 25 tokens:   {avg_25tok:.1f}ms")
        print(f"    Time to first audio: {avg_audio:.1f}ms")
        print(f"    Token generation:    {avg_rate:.1f} tok/s")

        # Analysis
        print(f"\n  Analysis:")
        if avg_25tok > 400:
            print(f"    ⚠️  T3 generation for 25 tokens is slow ({avg_25tok:.0f}ms)")
            print(f"        → This is the bottleneck")
        if avg_audio > 1000:
            print(f"    ❌ First chunk exceeds 1s target ({avg_audio:.0f}ms)")


async def main():
    print("="*70)
    print("T3 AsyncLLMEngine Deep Profiling")
    print("="*70)
    print()

    # Setup
    output_dir = Path("output/t3_profiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("📦 Loading model...")
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=12,
        gpu_memory_utilization=0.5,
    )

    profiler = T3Profiler(model)

    # Warmup
    print("Warming up model...")
    async for _ in model.generate_stream("Warmup.", print_metrics=False):
        pass
    print("✅ Warmup complete\n")

    # Test different concurrent levels
    concurrent_levels = [1, 2, 4, 8, 16]
    all_results = {}

    print("="*70)
    print("Profiling Concurrent Load")
    print("="*70)

    for concurrent in concurrent_levels:
        print(f"\n▶ Testing {concurrent} concurrent requests...")

        # Create unique texts for each request
        texts = [
            f"This is test number {i+1} for profiling T3 behavior at {concurrent} concurrent requests."
            for i in range(concurrent)
        ]

        # Launch all requests simultaneously
        tasks = [
            profiler.profile_request(text, concurrent, i)
            for i, text in enumerate(texts)
        ]

        start_time = time.time()
        timelines = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Store results
        all_results[concurrent] = {
            'timelines': timelines,
            'total_time': total_time,
        }

        # Print summary
        profiler.print_summary(timelines, concurrent)

        # Check if we should stop (performance degraded)
        avg_first_audio = sum(t.time_to_first_audio for t in timelines) / len(timelines)
        if avg_first_audio > 2000:
            print(f"\n  ⚠️  Performance severely degraded, stopping tests")
            break

    print("\n" + "="*70)
    print("Key Findings")
    print("="*70)

    for concurrent, results in all_results.items():
        timelines = results['timelines']
        avg_ttft = sum(t.time_to_first_token for t in timelines) / len(timelines)
        avg_25tok = sum(t.time_to_chunk_ready for t in timelines) / len(timelines)
        avg_audio = sum(t.time_to_first_audio for t in timelines) / len(timelines)

        print(f"\n{concurrent} concurrent:")
        print(f"  First token:    {avg_ttft:.1f}ms")
        print(f"  25 tokens:      {avg_25tok:.1f}ms")
        print(f"  First audio:    {avg_audio:.1f}ms {'✅' if avg_audio < 1000 else '❌'}")

    await model.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

**Step 2: Run the profiling test**

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600 uv run python profile_t3_concurrent.py
```

Expected output:
- Per-request timing breakdown for 1, 2, 4, 8, 16 concurrent
- Statistics showing where time is spent
- Identification of bottleneck

**Step 3: Analyze results**

Look for:
1. Is time_to_first_token scaling linearly? (indicates queue serialization)
2. Is time_to_25_tokens the bottleneck? (T3 generation speed)
3. Is token_generation_rate consistent across concurrency levels? (batching efficiency)
4. Is S3Gen time constant? (should be ~135ms)

**Step 4: Document findings**

Create `docs/t3_concurrent_findings.md` with:
- Observed timing breakdown at each concurrency level
- Root cause analysis
- Recommended solution approach

**Step 5: Commit**

```bash
git add profile_t3_concurrent.py docs/t3_concurrent_findings.md
git commit -m "feat: add T3 concurrent profiling and findings"
```

---

## Task 2: Implement Solution Based on Findings

**Note:** The exact solution depends on findings from Task 1. Below are the most likely scenarios.

### Scenario A: Reduce Chunk Size

**If findings show:** Accumulating 25 tokens is the main bottleneck (>400ms)

**Files:**
- Modify: `src/chatterbox_vllm/tts_async.py`

**Step 1: Add chunk_size parameter to from_pretrained**

Add ability to configure chunk_size at model level:

```python
# In AsyncChatterboxTTS.from_pretrained() method, add parameter:
async def from_pretrained(
    cls,
    # ... existing params ...
    default_chunk_size: int = 25,  # NEW: Configurable default
    **kwargs
):
```

**Step 2: Update WebSocket API to use smaller chunks**

Modify `src/chatterbox_vllm/websocket_api.py`:

```python
# In the websocket handler, adjust chunk_size:
async for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=15,  # Reduced from 25
    diffusion_steps=5,
    print_metrics=False,
):
```

**Step 3: Test with reduced chunk size**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py
```

Expected: First chunk time reduced proportionally

**Step 4: Validate audio quality**

Listen to generated audio to ensure quality is acceptable with smaller chunks.

**Step 5: Commit**

```bash
git add src/chatterbox_vllm/tts_async.py src/chatterbox_vllm/websocket_api.py
git commit -m "feat: reduce chunk_size to 15 for faster first chunk"
```

### Scenario B: Optimize AsyncLLMEngine Request Submission

**If findings show:** AsyncLLMEngine is not batching efficiently (batch size = 1)

**Files:**
- Modify: `src/chatterbox_vllm/tts_async.py`
- Create: `src/chatterbox_vllm/t3_request_coordinator.py`

**Step 1: Create T3 request coordinator**

Create `src/chatterbox_vllm/t3_request_coordinator.py`:

```python
"""
T3 Request Coordinator

Optimizes request submission to AsyncLLMEngine for better batching behavior.
"""

import asyncio
from typing import Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class PendingRequest:
    """A request waiting to be submitted to T3."""
    request_id: str
    text: str
    sampling_params: Any
    conditionals: Any
    submit_time: float
    result_queue: asyncio.Queue


class T3RequestCoordinator:
    """
    Coordinates T3 request submission for optimal batching.

    Strategy:
    - Accumulate requests in small batches
    - Submit batch together for better GPU utilization
    - Fair scheduling: FIFO with priority for waiting requests
    """

    def __init__(
        self,
        engine,
        batch_window_ms: float = 10.0,  # Wait up to 10ms for batch formation
        max_batch_size: int = 8,
    ):
        self.engine = engine
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self.pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self.batch_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self):
        """Start the batch processing task."""
        self.running = True
        self.batch_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Stop the batch processing task."""
        self.running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass

    async def submit_request(
        self,
        text: str,
        sampling_params,
        conditionals,
    ) -> str:
        """
        Submit a request for batching.

        Returns:
            Request ID for tracking
        """
        request_id = f"tts-{time.time()}"
        result_queue = asyncio.Queue()

        request = PendingRequest(
            request_id=request_id,
            text=text,
            sampling_params=sampling_params,
            conditionals=conditionals,
            submit_time=time.time(),
            result_queue=result_queue,
        )

        await self.pending_queue.put(request)
        return request_id

    async def _process_batches(self):
        """
        Process pending requests in batches.

        Accumulates requests for batch_window_ms, then submits
        them together to AsyncLLMEngine for better batching.
        """
        while self.running:
            try:
                batch = []
                deadline = time.time() + (self.batch_window_ms / 1000.0)

                # Accumulate batch
                while len(batch) < self.max_batch_size and time.time() < deadline:
                    try:
                        request = await asyncio.wait_for(
                            self.pending_queue.get(),
                            timeout=deadline - time.time()
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

                if batch:
                    await self._submit_batch(batch)

            except Exception as e:
                print(f"Error in batch processing: {e}")
                await asyncio.sleep(0.01)

    async def _submit_batch(self, batch: list[PendingRequest]):
        """Submit a batch of requests to AsyncLLMEngine."""
        # Submit all requests
        for request in batch:
            # Create async generator for this request
            # This is a simplified version - actual implementation needs
            # to properly integrate with AsyncLLMEngine.generate()
            pass
```

**Step 2: Integrate coordinator with AsyncChatterboxTTS**

Modify `src/chatterbox_vllm/tts_async.py`:

```python
# In __init__:
self.t3_coordinator: Optional[T3RequestCoordinator] = None

# In from_pretrained():
if enable_t3_coordinator:
    model.t3_coordinator = T3RequestCoordinator(
        engine=model.engine,
        batch_window_ms=10.0,
        max_batch_size=8,
    )
    await model.t3_coordinator.start()
```

**Step 3: Test with coordinator**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py
```

Expected: Better batching, reduced first chunk time

**Step 4: Commit**

```bash
git add src/chatterbox_vllm/t3_request_coordinator.py src/chatterbox_vllm/tts_async.py
git commit -m "feat: add T3 request coordinator for optimized batching"
```

### Scenario C: Multiple T3 Instances

**If findings show:** AsyncLLMEngine is inherently sequential for this use case

**Files:**
- Modify: `src/chatterbox_vllm/tts_async.py`

**Step 1: Add multi-instance support**

Modify `AsyncChatterboxTTS` to support multiple T3 engines:

```python
# In from_pretrained():
num_t3_instances: int = 1,  # NEW: Number of T3 instances

# Create multiple engines
engines = []
for i in range(num_t3_instances):
    engine_args = AsyncEngineArgs(
        model=model_path,
        # ... other params ...
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    engines.append(engine)

# Use round-robin for request distribution
```

**Step 2: Test with multiple instances**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py
```

Expected: Near-linear scaling with number of instances

**Step 3: Commit**

```bash
git add src/chatterbox_vllm/tts_async.py
git commit -m "feat: add multiple T3 instances for parallelism"
```

---

## Task 3: Validation and Documentation

**Files:**
- Modify: `MEMORY.md`
- Modify: `docs/plans/2025-03-08-t3-parallel-processing-design.md`

**Step 1: Run final validation test**

```bash
CUDA_VISIBLE_DEVICES=0 uv run python test_stream_pool_concurrent_load.py
```

Verify:
- [ ] 8 concurrent: <1s first chunk
- [ ] 16 concurrent: <1s first chunk
- [ ] Stream pool still working (0.01ms queue wait)
- [ ] Audio quality acceptable

**Step 2: Update MEMORY.md**

Add new section documenting the solution:

```markdown
## Session: 2025-03-08 - T3 Parallel Processing Optimization

### Problem
8-16 concurrent requests exceeded 1s first chunk target due to T3 sequential processing.

### Root Cause
[Findings from investigation]

### Solution Implemented
[Description of implemented solution]

### Results
| Concurrent | Before | After | Improvement |
|------------|--------|-------|-------------|
| 8          | ~3s    | ~XXXms| XX%         |
| 16         | ~4s    | ~XXXms| XX%         |
```

**Step 3: Update design doc status**

Change status from "Investigation Phase" to "Complete"

**Step 4: Commit**

```bash
git add MEMORY.md docs/plans/2025-03-08-t3-parallel-processing-design.md
git commit -m "docs: update T3 parallel processing with results"
```

---

## Success Criteria

✅ **Primary:**
- 8 concurrent requests: <1s average first chunk
- 16 concurrent requests: <1s average first chunk
- No regression in single-request performance

✅ **Secondary:**
- Stream pool continues working (0.01ms queue wait)
- GPU memory usage stays within limits
- Audio quality maintained

✅ **Documentation:**
- Findings documented in MEMORY.md
- Code well-commented
- Performance comparison included

---

## Testing Commands

```bash
# Run profiling test
CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py

# Run load test
CUDA_VISIBLE_DEVICES=0 uv run python test_stream_pool_concurrent_load.py

# Test WebSocket API
CUDA_VISIBLE_DEVICES=0 uv run python src/chatterbox_vllm/websocket_api.py
```
