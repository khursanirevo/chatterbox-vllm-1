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
