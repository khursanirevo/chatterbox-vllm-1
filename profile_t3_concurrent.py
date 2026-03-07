#!/usr/bin/env python3
"""
Deep profiling of T3 AsyncLLMEngine behavior with concurrent requests.

Measures:
- Per-request timeline breakdown
- AsyncLLMEngine queue dynamics and position
- Token generation rate and batching behavior
- GPU utilization during generation

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python profile_t3_concurrent.py
"""

import asyncio
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import pynvml

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

    # Token tracking
    token_count: int = 0
    token_arrivals: List[int] = field(default_factory=list)  # Actual token counts

    # Queue and batch tracking
    queue_position: int = -1
    estimated_batch_size: int = -1  # Estimated based on concurrent level (vLLM doesn't expose actual batch size)

    # Timing breakdown
    t3_queue_time: float = 0.0
    t3_generation_time: float = 0.0
    s3gen_time: float = 0.0

    # GPU utilization
    gpu_utilization_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0

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
        """Tokens per second during generation.

        Note: Returns 0.0 because tokens arrive in chunks, not incrementally.
        Cannot measure true token generation rate from chunked delivery.
        """
        return 0.0


class GPUMonitor:
    """Monitor GPU utilization and memory in background."""

    def __init__(self, device_id: int = 0, sample_interval: float = 0.1):
        self.device_id = device_id
        self.sample_interval = sample_interval
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.utilization_samples: List[float] = []
        self.memory_samples: List[float] = []

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.available = True
        except Exception as e:
            print(f"Warning: GPU monitoring not available: {e}")
            self.available = False

    def _monitor_loop(self):
        """Background thread to collect GPU metrics."""
        while self.monitoring:
            try:
                if self.available:
                    # Get GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                    self.utilization_samples.append(util.gpu)

                    # Get memory usage
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                    memory_mb = mem_info.used / 1024 / 1024
                    self.memory_samples.append(memory_mb)
            except Exception:
                pass

            time.sleep(self.sample_interval)

    def start(self):
        """Start monitoring in background."""
        if not self.available:
            return

        self.monitoring = True
        self.utilization_samples = []
        self.memory_samples = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop monitoring and return averages."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)

        avg_util = sum(self.utilization_samples) / len(self.utilization_samples) if self.utilization_samples else 0.0
        avg_mem = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0.0
        return avg_util, avg_mem


class T3Profiler:
    """Profiles T3 AsyncLLMEngine behavior."""

    def __init__(self, model: AsyncChatterboxTTS, gpu_monitor: GPUMonitor):
        self.model = model
        self.gpu_monitor = gpu_monitor
        self.timelines: Dict[int, RequestTimeline] = {}
        self.request_counter = 0
        self.request_queue: Dict[int, int] = {}  # Track queue position

    async def profile_request(
        self,
        text: str,
        concurrent_level: int,
        request_idx: int
    ) -> RequestTimeline:
        """Profile a single TTS request with detailed timing."""

        request_id = self.request_counter
        self.request_counter += 1

        # Record queue position
        self.request_queue[request_id] = request_idx

        timeline = RequestTimeline(
            request_id=request_id,
            submit_time=time.time(),
            queue_position=request_idx
        )
        self.timelines[request_id] = timeline

        chunk_size = 25
        target_tokens = chunk_size
        chunk_ready = False

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

                # Track actual token accumulation
                # The chunk contains audio, but we can infer tokens from chunk_size
                if not chunk_ready:
                    timeline.token_count += chunk_size
                    timeline.token_arrivals.append(timeline.token_count)

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
                            # Got first chunk, continue to collect batch info
                            # but don't break yet - wait to see more batches

            # After generation completes, estimate batch size
            # Note: vLLM doesn't expose actual batch size, so we estimate from concurrent level
            timeline.estimated_batch_size = concurrent_level

        except Exception as e:
            print(f"    ❌ Request {request_id} failed: {e}")

        timeline.complete_time = time.time()
        return timeline

    def print_summary(self, timelines: List[RequestTimeline], concurrent_level: int):
        """Print profiling summary for this concurrent level."""

        if not timelines:
            print(f"    ❌ No valid requests for {concurrent_level} concurrent")
            return

        print(f"\n  ─────────────────────────────────────────────────────────────")
        print(f"  {concurrent_level} Concurrent Requests - Detailed Timings")
        print(f"  ─────────────────────────────────────────────────────────────")

        # Per-request breakdown
        print(f"\n  {'Req':<4} {'Queue':<6} {'TTFT':<8} {'25 Tok':<8} {'1st Audio':<10} {'Batch':<6} {'GPU%':<6} {'GPUmem':<8}")
        print(f"  {'':<4} {'Pos':<6} {'(ms)':<8} {'(ms)':<8} {'(ms)':<10} {'Size':<6} {'':<6} {'(MB)':<8}")

        for tl in timelines:
            print(
                f"  {tl.request_id:<4} "
                f"{tl.queue_position:<6} "
                f"{tl.time_to_first_token:>7.1f} "
                f"{tl.time_to_chunk_ready:>7.1f} "
                f"{tl.time_to_first_audio:>9.1f} "
                f"{tl.estimated_batch_size:<6} "
                f"{tl.gpu_utilization_percent:>5.1f} "
                f"{tl.gpu_memory_used_mb:>7.1f} "
            )

        # Statistics
        avg_ttft = sum(t.time_to_first_token for t in timelines) / len(timelines)
        avg_25tok = sum(t.time_to_chunk_ready for t in timelines) / len(timelines)
        avg_audio = sum(t.time_to_first_audio for t in timelines) / len(timelines)
        avg_rate = sum(t.token_generation_rate for t in timelines) / len(timelines)
        avg_gpu = sum(t.gpu_utilization_percent for t in timelines) / len(timelines)
        avg_mem = sum(t.gpu_memory_used_mb for t in timelines) / len(timelines)

        print(f"\n  Average:")
        print(f"    Queue position:     {sum(t.queue_position for t in timelines) / len(timelines):.1f}")
        print(f"    Time to first token: {avg_ttft:.1f}ms")
        print(f"    Time to 25 tokens:   {avg_25tok:.1f}ms")
        print(f"    Time to first audio: {avg_audio:.1f}ms")
        print(f"    Token generation:    {avg_rate:.1f} tok/s")
        print(f"    GPU utilization:     {avg_gpu:.1f}%")
        print(f"    GPU memory:          {avg_mem:.1f} MB")


async def main():
    print("="*70)
    print("T3 AsyncLLMEngine Deep Profiling")
    print("="*70)
    print()

    # Setup
    output_dir = Path("output/t3_profiling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_id=0)

    # Load model
    print("📦 Loading model...")
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=12,
        gpu_memory_utilization=0.5,
    )

    profiler = T3Profiler(model, gpu_monitor)

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

        # Start GPU monitoring before launching requests
        gpu_monitor.start()

        # Launch all requests simultaneously
        tasks = [
            profiler.profile_request(text, concurrent, i)
            for i, text in enumerate(texts)
        ]

        start_time = time.time()
        timelines = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Stop GPU monitoring and assign metrics to all timelines
        avg_util, avg_mem = gpu_monitor.stop()
        for timeline in timelines:
            timeline.gpu_utilization_percent = avg_util
            timeline.gpu_memory_used_mb = avg_mem

        # Store results
        all_results[concurrent] = {
            'timelines': timelines,
            'total_time': total_time,
        }

        # Print summary
        profiler.print_summary(timelines, concurrent)

    print("\n" + "="*70)
    print("Summary of Findings")
    print("="*70)

    for concurrent, results in all_results.items():
        timelines = results['timelines']
        avg_ttft = sum(t.time_to_first_token for t in timelines) / len(timelines)
        avg_25tok = sum(t.time_to_chunk_ready for t in timelines) / len(timelines)
        avg_audio = sum(t.time_to_first_audio for t in timelines) / len(timelines)
        avg_gpu = sum(t.gpu_utilization_percent for t in timelines) / len(timelines)

        print(f"\n{concurrent} concurrent:")
        print(f"  First token:    {avg_ttft:.1f}ms")
        print(f"  25 tokens:      {avg_25tok:.1f}ms")
        print(f"  First audio:    {avg_audio:.1f}ms")
        print(f"  GPU utilization: {avg_gpu:.1f}%")

    await model.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
