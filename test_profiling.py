#!/usr/bin/env python3
"""
Chatterbox vLLM Profiling Tool

Unified profiling script with multiple modes:
- simple: Single-run profiling with detailed metrics
- first-chunk: Multi-iteration profiling with cold starts
- steady-state: Steady-state profiling (same model, multiple runs)

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test_profiling.py simple
    CUDA_VISIBLE_DEVICES=0 uv run python test_profiling.py first-chunk --iterations 10
    CUDA_VISIBLE_DEVICES=0 uv run python test_profiling.py steady-state --iterations 20
"""

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS, StreamingMetrics
from chatterbox_vllm.models.s3gen import S3GEN_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class LatencyStats:
    """Statistics for latency measurements."""
    values: List[float] = field(default_factory=list)

    def add(self, value: float):
        self.values.append(value)

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def min(self) -> float:
        return min(self.values) if self.values else 0.0

    @property
    def max(self) -> float:
        return max(self.values) if self.values else 0.0

    @property
    def avg(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def percentile_95(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * 0.95)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def percentile_99(self) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * 0.99)
        return sorted_values[min(index, len(sorted_values) - 1)]


@dataclass
class DetailedMetrics:
    """Detailed metrics for a single generation."""
    iteration: int
    first_chunk_latency: float
    t3_token_generation: float
    s3gen_first_chunk: float
    total_generation_time: float
    audio_duration: float
    token_count: int
    chunk_count: int


def print_stats_table(stats: LatencyStats, label: str, unit: str = "ms"):
    """Print a statistics table."""
    print(f"\n{label}:")
    print(f"  {'Count:':<20} {stats.count}")
    print(f"  {'Min:':<20} {stats.min*1000:.2f} {unit}")
    print(f"  {'Max:':<20} {stats.max*1000:.2f} {unit}")
    print(f"  {'Average:':<20} {stats.avg*1000:.2f} {unit}")
    print(f"  {'Median:':<20} {stats.median*1000:.2f} {unit}")
    print(f"  {'Std Dev:':<20} {stats.stdev*1000:.2f} {unit}")
    if hasattr(stats, 'percentile_95'):
        print(f"  {'95th Percentile:':<20} {stats.percentile_95*1000:.2f} {unit}")
        print(f"  {'99th Percentile:':<20} {stats.percentile_99*1000:.2f} {unit}")


def mode_simple(args):
    """Simple single-run profiling with detailed metrics."""
    print("=" * 70)
    print("SIMPLE PROFILING MODE")
    print("=" * 70)

    print("\nLoading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory,
    )
    print("Model loaded!\n")

    text = args.text or (
        "This is a profiling test for the streaming TTS implementation. "
        "We will measure exactly how much time each stage takes, from text "
        "tokenization through T3 speech token generation to the first S3Gen "
        "audio chunk. This helps identify bottlenecks and optimize the pipeline."
    )

    print(f"Text: {text}\n")
    print("=" * 70)
    print("GENERATING WITH DETAILED PROFILING")
    print("=" * 70)

    audio_chunks = []
    for audio_chunk, metrics in model.generate_stream(
        text=text,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        context_window=args.context_window,
        print_metrics=True,
    ):
        audio_chunks.append(audio_chunk)
        if metrics.chunk_count == 1:
            print(f"\n[PROGRESS] Received chunk 1: shape={audio_chunk.shape}, "
                  f"duration={audio_chunk.shape[-1]/model.sr:.3f}s")
        elif metrics.chunk_count % 5 == 0:
            print(f"[PROGRESS] Received chunk {metrics.chunk_count}: "
                  f"last_chunk={metrics.last_chunk_time*1000:.1f}ms, "
                  f"avg_chunk={metrics.avg_chunk_time*1000:.1f}ms")

    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=-1)
        output_path = args.output or "test-profiling.wav"
        ta.save(output_path, full_audio, model.sr)

        print(f"\n{'='*70}")
        print(f"SAVED: {output_path}")
        print(f"{'='*70}")
        print(f"Duration: {full_audio.shape[-1]/model.sr:.2f}s")
        print(f"Chunks: {len(audio_chunks)}")
        print(f"File size: {os.path.getsize(output_path)/1024:.1f}KB")

    model.shutdown()
    print("\nDone!")


def mode_first_chunk(args):
    """Multi-iteration profiling with cold starts (new model each iteration)."""
    print("=" * 70)
    print("FIRST CHUNK LATENCY PROFILING (COLD START)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Text: '{args.text}'")
    print(f"  Iterations: {args.iterations}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Max tokens: {args.max_tokens}")

    first_chunk_latency = LatencyStats()
    t3_generation_time = LatencyStats()
    s3gen_first_chunk = LatencyStats()
    total_generation_time = LatencyStats()
    audio_duration = LatencyStats()
    all_metrics: List[DetailedMetrics] = []

    print("\n" + "=" * 70)
    print(f"WARMUP ({args.warmup} iterations)")
    print("=" * 70)

    for i in range(args.warmup):
        print(f"\nWarmup {i+1}/{args.warmup}...", end="", flush=True)
        model = ChatterboxTTS.from_pretrained(
            max_model_len=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory,
        )
        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break
        model.shutdown()
        del model
        torch.cuda.empty_cache()
        print(" ✓")

    print("\n" + "=" * 70)
    print(f"PROFILING ({args.iterations} iterations)")
    print("=" * 70)

    for i in range(args.iterations):
        print(f"\nIteration {i+1}/{args.iterations}...", end="", flush=True)
        iter_start = time.time()

        init_start = time.time()
        model = ChatterboxTTS.from_pretrained(
            max_model_len=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory,
        )
        init_time = time.time() - init_start

        iter_first_chunk_latency = None
        iter_t3_time = None
        iter_s3gen_time = None
        iter_audio_duration = 0.0
        audio_chunks = []

        gen_start = time.time()
        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            print_metrics=False,
        ):
            audio_chunks.append(audio_chunk)
            if metrics.chunk_count == 1:
                iter_first_chunk_latency = metrics.latency_to_first_chunk
                iter_t3_time = metrics.t3_token_generation_time
                iter_s3gen_time = metrics.s3gen_first_chunk_time
            if metrics.chunk_count == 1:
                break

        gen_time = time.time() - gen_start

        if audio_chunks:
            audio = torch.cat(audio_chunks, dim=-1)
            iter_audio_duration = audio.shape[-1] / S3GEN_SR

        if iter_first_chunk_latency is not None:
            first_chunk_latency.add(iter_first_chunk_latency)
            t3_generation_time.add(iter_t3_time)
            s3gen_first_chunk.add(iter_s3gen_time)
            total_generation_time.add(gen_time)
            audio_duration.add(iter_audio_duration)

            detailed = DetailedMetrics(
                iteration=i + 1,
                first_chunk_latency=iter_first_chunk_latency,
                t3_token_generation=iter_t3_time,
                s3gen_first_chunk=iter_s3gen_time,
                total_generation_time=gen_time,
                audio_duration=iter_audio_duration,
                token_count=args.max_tokens,
                chunk_count=1,
            )
            all_metrics.append(detailed)

        model.shutdown()
        del model
        torch.cuda.empty_cache()

        iter_time = time.time() - iter_start
        print(f" ✓ (First chunk: {iter_first_chunk_latency*1000:.1f}ms, Total: {iter_time:.2f}s)")

    print("\n" + "=" * 70)
    print("LATENCY STATISTICS")
    print("=" * 70)

    print_stats_table(first_chunk_latency, "⚡ FIRST CHUNK LATENCY", "ms")
    print_stats_table(t3_generation_time, "📝 T3 TOKEN GENERATION", "ms")
    print_stats_table(s3gen_first_chunk, "🎵 S3GEN FIRST CHUNK", "ms")
    print_stats_table(total_generation_time, "⏱️  TOTAL GENERATION TIME", "ms")

    print("\n" + "=" * 70)
    print("LATENCY BREAKDOWN")
    print("=" * 70)

    avg_first_chunk = first_chunk_latency.avg * 1000
    avg_t3 = t3_generation_time.avg * 1000
    avg_s3gen = s3gen_first_chunk.avg * 1000

    print(f"\nAverage first chunk latency: {avg_first_chunk:.2f}ms")
    print(f"  - T3 generation:           {avg_t3:.2f}ms ({avg_t3/avg_first_chunk*100:.1f}%)")
    print(f"  - S3Gen first chunk:       {avg_s3gen:.2f}ms ({avg_s3gen/avg_first_chunk*100:.1f}%)")
    print(f"  - Other overhead:          {avg_first_chunk - avg_t3 - avg_s3gen:.2f}ms "
          f"({(avg_first_chunk - avg_t3 - avg_s3gen)/avg_first_chunk*100:.1f}%)")

    print("\n" + "=" * 70)
    print("COMPARISON: SYNC vs ASYNC")
    print("=" * 70)

    estimated_async_first_token = 0.050
    estimated_async_s3gen = avg_s3gen / 1000
    estimated_async_total = estimated_async_first_token + estimated_async_s3gen

    print(f"\nCurrent (Sync vLLM):")
    print(f"  First chunk latency: {avg_first_chunk:.0f}ms")
    print(f"  - T3 (all tokens):   {avg_t3:.0f}ms")
    print(f"  - S3Gen:             {avg_s3gen:.0f}ms")

    print(f"\nAsyncLLMEngine (Estimated):")
    print(f"  First chunk latency: {estimated_async_total*1000:.0f}ms")
    print(f"  - T3 (first token):  {estimated_async_first_token*1000:.0f}ms")
    print(f"  - S3Gen:             {estimated_async_s3gen*1000:.0f}ms")

    speedup = avg_first_chunk / (estimated_async_total * 1000)
    print(f"\nEstimated speedup:   {speedup:.2f}x faster")


def mode_steady_state(args):
    """Steady-state profiling (same model, multiple runs)."""
    print("=" * 70)
    print("STEADY-STATE FIRST CHUNK LATENCY PROFILING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Text: '{args.text}'")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Measurement iterations: {args.iterations}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"\n⚠️  This measures STEADY-STATE performance")
    print(f"    (same model, multiple runs - no initialization overhead)")

    first_chunk_latency = LatencyStats()
    t3_generation_time = LatencyStats()
    s3gen_first_chunk = LatencyStats()

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    init_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_model_len=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory,
    )
    init_time = time.time() - init_start
    print(f"\n✓ Model initialized in {init_time:.2f}s")

    print("\n" + "=" * 70)
    print(f"WARMUP PHASE ({args.warmup} iterations)")
    print("=" * 70)

    for i in range(args.warmup):
        print(f"\nWarmup {i+1}/{args.warmup}...", end="", flush=True)
        warmup_start = time.time()
        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break
        warmup_time = time.time() - warmup_start
        print(f" ✓ ({warmup_time*1000:.1f}ms)")

    print("\n" + "=" * 70)
    print(f"MEASUREMENT PHASE ({args.iterations} iterations)")
    print("=" * 70)
    print("\nMeasuring steady-state latency (same model, multiple runs)...")

    for i in range(args.iterations):
        iter_start = time.time()
        iter_first_chunk_latency = None
        iter_t3_time = None
        iter_s3gen_time = None

        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                iter_first_chunk_latency = metrics.latency_to_first_chunk
                iter_t3_time = metrics.t3_token_generation_time
                iter_s3gen_time = metrics.s3gen_first_chunk_time
                break

        if iter_first_chunk_latency is not None:
            first_chunk_latency.add(iter_first_chunk_latency)
            t3_generation_time.add(iter_t3_time)
            s3gen_first_chunk.add(iter_s3gen_time)

        iter_time = time.time() - iter_start

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Iteration {i+1:3d}/{args.iterations}: "
                  f"{iter_first_chunk_latency*1000:7.2f}ms  (T3: {iter_t3_time*1000:.1f}ms, "
                  f"S3Gen: {iter_s3gen_time*1000:.1f}ms)")

    print("\n" + "=" * 70)
    print("STEADY-STATE LATENCY STATISTICS")
    print("=" * 70)

    print_stats_table(first_chunk_latency, "⚡ FIRST CHUNK LATENCY", "ms")
    print_stats_table(t3_generation_time, "📝 T3 TOKEN GENERATION", "ms")
    print_stats_table(s3gen_first_chunk, "🎵 S3GEN FIRST CHUNK", "ms")

    avg_first_chunk = first_chunk_latency.avg * 1000
    avg_t3 = t3_generation_time.avg * 1000
    avg_s3gen = s3gen_first_chunk.avg * 1000

    print("\n" + "=" * 70)
    print("LATENCY BREAKDOWN")
    print("=" * 70)
    print(f"\nAverage first chunk latency: {avg_first_chunk:.2f}ms")
    print(f"  - T3 generation:           {avg_t3:.2f}ms ({avg_t3/avg_first_chunk*100:.1f}%)")
    print(f"  - S3Gen first chunk:       {avg_s3gen:.2f}ms ({avg_s3gen/avg_first_chunk*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)

    cv = (first_chunk_latency.stdev / first_chunk_latency.avg) * 100
    print(f"\nCoefficient of Variation: {cv:.2f}%")
    if cv < 10:
        print(f"  ✅ Excellent consistency!")
    elif cv < 20:
        print(f"  ✓ Good consistency")
    else:
        print(f"  ⚠️  High variance detected")

    range_ms = (first_chunk_latency.max - first_chunk_latency.min) * 1000
    print(f"\nRange: {range_ms:.2f}ms")
    print(f"  Best:  {first_chunk_latency.min*1000:.2f}ms")
    print(f"  Worst: {first_chunk_latency.max*1000:.2f}ms")

    model.shutdown()
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox vLLM Profiling Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple profiling
  uv run python test_profiling.py simple

  # First chunk latency with cold starts
  uv run python test_profiling.py first-chunk --iterations 10

  # Steady-state profiling
  uv run python test_profiling.py steady-state --iterations 20
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Profiling mode")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--max-tokens", type=int, default=200, help="Maximum tokens")
    common_parser.add_argument("--chunk-size", type=int, default=25, help="Chunk size")
    common_parser.add_argument("--gpu-memory", type=float, default=0.90, help="GPU memory utilization")
    common_parser.add_argument("--text", default="Hello world, this is a test.", help="Input text")

    # Simple mode
    simple_parser = subparsers.add_parser("simple", help="Simple single-run profiling", parents=[common_parser])
    simple_parser.add_argument("--max-batch-size", type=int, default=3, help="Max batch size")
    simple_parser.add_argument("--context-window", type=int, default=50, help="Context window")
    simple_parser.add_argument("--output", "-o", help="Output WAV file path")

    # First chunk mode
    first_chunk_parser = subparsers.add_parser("first-chunk", help="Multi-iteration cold start profiling", parents=[common_parser])
    first_chunk_parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    first_chunk_parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")

    # Steady-state mode
    steady_parser = subparsers.add_parser("steady-state", help="Steady-state profiling", parents=[common_parser])
    steady_parser.add_argument("--iterations", type=int, default=20, help="Number of iterations")
    steady_parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("CHATTERBOX vLLM TTS PROFILING")
    print("=" * 70)

    if args.mode == "simple":
        mode_simple(args)
    elif args.mode == "first-chunk":
        mode_first_chunk(args)
    elif args.mode == "steady-state":
        mode_steady_state(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
