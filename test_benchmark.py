#!/usr/bin/env python3
"""
Chatterbox vLLM Benchmark Tool

Unified benchmark script with multiple modes:
- generate: Generate audio for different text lengths
- async: Test AsyncLLMEngine streaming
- concurrent: Test concurrent request handling
- validate: Validate audio output

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python test_benchmark.py generate
    CUDA_VISIBLE_DEVICES=0 uv run python test_benchmark.py async
    CUDA_VISIBLE_DEVICES=0 uv run python test_benchmark.py concurrent --burst-size 16
"""

import argparse
import asyncio
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torchaudio as ta

# Import ChatterboxTTS components
from chatterbox_vllm.tts import ChatterboxTTS, StreamingMetrics
from chatterbox_vllm.models.s3gen import S3GEN_SR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============== DATA CLASSES ==============

@dataclass
class RequestResult:
    """Result for a single request."""
    request_id: int
    start_time: float
    end_time: float
    first_chunk_time: float = 0.0
    total_time: float = 0.0
    t3_time: float = 0.0
    s3gen_time: float = 0.0
    audio_duration: float = 0.0
    success: bool = True
    error: str = None

@dataclass
class BurstTestResults:
    """Results for a burst test."""
    burst_size: int
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successful_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def first_chunk_latencies(self) -> List[float]:
        return [r.first_chunk_time for r in self.results if r.success]

    @property
    def avg_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.mean(vals) if vals else 0.0

    @property
    def min_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return min(vals) if vals else 0.0

    @property
    def max_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return max(vals) if vals else 0.0

    @property
    def median_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.median(vals) if vals else 0.0

    @property
    def stdev_first_chunk(self) -> float:
        vals = self.first_chunk_latencies
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    @property
    def p95_first_chunk(self) -> float:
        vals = sorted(self.first_chunk_latencies)
        if not vals:
            return 0.0
        idx = int(len(vals) * 0.95)
        return vals[min(idx, len(vals) - 1)]

    @property
    def under_1s_count(self) -> int:
        return sum(1 for t in self.first_chunk_latencies if t < 1.0)

    @property
    def under_1s_pct(self) -> float:
        total = len(self.first_chunk_latencies)
        if total == 0:
            return 0.0
        return (self.under_1s_count / total) * 100


# ============== TEST TEXTS ==============

TEST_CASES = {
    "short": {
        "text": "Hello, this is a short test.",
        "output": "test-short.wav",
        "max_tokens": 500
    },
    "medium": {
        "text": (
            "This is a medium-length text to test the streaming capabilities. "
            "It contains multiple sentences and should take a few seconds to generate. "
            "The streaming feature allows audio chunks to be produced incrementally."
        ),
        "output": "test-medium.wav",
        "max_tokens": 1000
    },
    "long": {
        "text": (
            "This is a longer text designed to thoroughly test the streaming TTS implementation. "
            "When we have substantially more content, it allows us to observe how the system handles "
            "multiple audio chunks over an extended period. The streaming approach is particularly "
            "useful for real-time applications where users don't want to wait for the entire generation "
            "to complete before hearing the first audio. With this implementation, we use a two-stage "
            "process: first, vLLM rapidly generates all the speech tokens, and then we stream those "
            "tokens through the S3Gen model in chunks. This provides an excellent balance between "
            "the batch processing efficiency of vLLM and the real-time playback capabilities needed "
            "for interactive applications. The result is an RTF of approximately 0.7, meaning the "
            "audio generates faster than real-time playback speed."
        ),
        "output": "test-long.wav",
        "max_tokens": 2000
    }
}

BURST_TEXTS = [
    "Hello world, this is a test of concurrent text to speech.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a longer sentence to test the system under load.",
    "Testing continuous batching with multiple concurrent requests.",
]

# 32 UNIQUE texts to avoid prefix caching
UNIQUE_TEXTS = [
    "The weather is beautiful today with clear blue skies.",
    "Machine learning models require large amounts of training data.",
    "The conference room was packed with enthusiastic attendees.",
    "Quantum computing leverages superposition for parallel processing.",
    "The ancient library contained thousands of rare manuscripts.",
    "Renewable energy sources include solar, wind, and hydroelectric power.",
    "The mountain trail offered breathtaking views of the valley below.",
    "Software engineering best practices emphasize code quality and testing.",
    "The marathon runner trained for months to prepare for competition.",
    "Neural networks consist of interconnected layers of artificial neurons.",
    "The culinary festival featured dishes from around the world.",
    "Climate change affects ecosystems and weather patterns globally.",
    "The new smartphone model features advanced camera capabilities.",
    "Acoustic guitars produce sound through vibrating strings and resonance.",
    "The space mission will explore distant galaxies and nebulae.",
    "Financial markets experience volatility during uncertain times.",
    "The art gallery showcased contemporary paintings and sculptures.",
    "Biology students study cell structures and genetic inheritance.",
    "The urban transportation system includes buses, trains, and subways.",
    "Cloud computing provides scalable infrastructure for applications.",
    "The archaeological discovery revealed ancient civilization artifacts.",
    "Chemical reactions involve breaking and forming molecular bonds.",
    "The music concert featured a symphony orchestra performance.",
    "Data visualization helps interpret complex information graphically.",
    "The ocean contains diverse marine ecosystems and coral reefs.",
    "Robotic automation transforms manufacturing and logistics processes.",
    "Philosophy examines fundamental questions about existence and knowledge.",
    "The wildlife documentary captured stunning footage of nature.",
    "Cryptocurrency uses blockchain technology for secure transactions.",
    "The sailing ship navigated through stormy waters to reach port.",
    "Psychological research explores human behavior and mental processes.",
    "The fashion industry trends change rapidly with seasonal collections.",
    "Geologists study rock formations to understand Earth's history.",
]


# ============== MODE: GENERATE ==============

def mode_generate(args):
    """Generate audio for different text lengths."""
    print("=" * 70)
    print("GENERATE MODE - Audio for Different Text Lengths")
    print("=" * 70)

    print("\nLoading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=2000,
        gpu_memory_utilization=0.90,
    )
    print("Model loaded!\n")

    results = {}
    sizes = args.sizes or ["short", "medium", "long"]

    for size in sizes:
        if size not in TEST_CASES:
            print(f"Warning: Unknown size '{size}', skipping")
            continue

        config = TEST_CASES[size]
        text = config["text"]
        output_path = config["output"]
        max_tokens = config["max_tokens"]

        print(f"\n{'='*60}")
        print(f"Generating {size.upper()} audio (max_tokens={max_tokens})")
        print(f"{'='*60}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n")

        audio_chunks = []
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            context_window=50,
            print_metrics=True,
        ):
            audio_chunks.append(audio_chunk)

        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=-1)
            ta.save(output_path, full_audio, model.sr)
            duration = full_audio.shape[-1] / model.sr

            results[size] = {
                "text": text,
                "output_path": output_path,
                "duration": duration,
                "chunks": len(audio_chunks),
                "rtf": metrics.rtf,
                "latency": metrics.latency_to_first_chunk
            }

            print(f"\n✓ Saved to: {output_path}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Chunks: {len(audio_chunks)}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    for size in sizes:
        if size in results:
            r = results[size]
            print(f"[{size.upper()}]")
            print(f"  Audio: {r['output_path']}")
            print(f"  Duration: {r['duration']:.2f}s | Chunks: {r['chunks']} | RTF: {r['rtf']:.3f}")
            print()

    model.shutdown()
    print("Done!")


# ============== MODE: ASYNC ==============

async def mode_async(args):
    """Test AsyncLLMEngine streaming."""
    # Import vLLM async components
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from chatterbox_vllm.models.t3 import T3VllmModel

    print("=" * 70)
    print("ASYNC MODE - AsyncLLMEngine Token Streaming")
    print("=" * 70)

    print("\nInitializing AsyncLLMEngine...")
    engine_args = AsyncEngineArgs(
        model="./t3-model",
        tokenizer="EnTokenizer",
        tokenizer_mode="custom",
        gpu_memory_utilization=0.90,
        max_model_len=2000,
        enforce_eager=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("✓ Engine ready\n")

    tests = {
        "short": ("Hello world, this is a test.", 100),
        "medium": ("This is a longer text that will generate more tokens.", 200),
    }

    for name, (text, max_tokens) in tests.items():
        print(f"Text ({name}): {text}")
        print("-" * 70)

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        prompt = f"[START]{text}[STOP]"
        all_tokens = []
        first_token_time = None
        start_time = time.time()

        async for output in engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=f"{name}_test",
        ):
            if output.outputs:
                tokens = output.outputs[0].token_ids
                all_tokens = list(tokens)

                if first_token_time is None and len(all_tokens) > 0:
                    first_token_time = time.time()
                    print(f"  ⚡ First token: {(first_token_time - start_time)*1000:.1f}ms")

                elapsed = time.time() - start_time
                if len(all_tokens) % 25 == 0:
                    print(f"  Progress: {len(all_tokens)} tokens in {elapsed*1000:.1f}ms", end="\r")

            if output.finished:
                break

        total_time = time.time() - start_time
        print(f"\n  ✓ Complete: {len(all_tokens)} tokens in {total_time*1000:.1f}ms\n")

    del engine
    print("\nDone!")


# ============== MODE: CONCURRENT ==============

def process_single_request(request_id: int, text: str, model: ChatterboxTTS, max_tokens: int = 200) -> RequestResult:
    """Process a single TTS request."""
    start_time = time.time()
    first_chunk_time = None
    t3_time = None
    s3gen_time = None
    audio_duration = 0.0
    success = True
    error = None

    try:
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            if first_chunk_time is None:
                first_chunk_time = metrics.latency_to_first_chunk
                t3_time = metrics.t3_token_generation_time
                s3gen_time = metrics.s3gen_first_chunk_time

            if metrics.chunk_count == 1:
                audio_duration = audio_chunk.shape[-1] / 24000
                break

        end_time = time.time()
        total_time = end_time - start_time
        success = True

    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        error = str(e)
        success = False

    return RequestResult(
        request_id=request_id,
        start_time=start_time,
        end_time=end_time,
        first_chunk_time=first_chunk_time or 0.0,
        total_time=total_time,
        t3_time=t3_time or 0.0,
        s3gen_time=s3gen_time or 0.0,
        audio_duration=audio_duration,
        success=success,
        error=error,
    )


def mode_concurrent(args):
    """Test concurrent request handling."""
    print("=" * 70)
    print("CONCURRENT MODE - Burst Testing")
    print("=" * 70)

    burst_sizes = args.burst_sizes or [1, 4, 8]
    texts = UNIQUE_TEXTS if args.unique else BURST_TEXTS

    print(f"\nInitializing model...")
    init_start = time.time()
    model = ChatterboxTTS.from_pretrained(
        max_model_len=200,
        gpu_memory_utilization=0.90,
    )
    init_time = time.time() - init_start
    print(f"✓ Model initialized in {init_time:.2f}s")

    # Warmup
    print("\nWarming up model...")
    for i in range(2):
        for audio_chunk, metrics in model.generate_stream(
            text=texts[0],
            max_tokens=200,
            chunk_size=25,
            print_metrics=False,
        ):
            if metrics.chunk_count == 1:
                break
    print("✓ Warmup complete")

    all_results = {}

    for burst_size in burst_sizes:
        print(f"\n{'#'*70}")
        print(f"# BURST SIZE: {burst_size} CONCURRENT REQUESTS")
        print(f"{'#'*70}")

        results = BurstTestResults(burst_size=burst_size)
        burst_start = time.time()

        with ThreadPoolExecutor(max_workers=burst_size) as executor:
            futures = []
            for i in range(burst_size):
                text = texts[i % len(texts)]
                future = executor.submit(
                    process_single_request,
                    request_id=i,
                    text=text,
                    model=model,
                    max_tokens=200,
                )
                futures.append(future)

            for i, future in enumerate(futures):
                result = future.result()
                results.results.append(result)

                elapsed = result.end_time - burst_start
                print(f"  Request {result.request_id+1:2d}: "
                      f"TTFA={result.first_chunk_time*1000:7.2f}ms, "
                      f"Total={result.total_time*1000:7.2f}ms")

        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS: {burst_size} REQUESTS")
        print(f"{'='*70}")
        print(f"\nSuccess: {results.successful_count}/{results.burst_size}")

        if results.successful_count > 0:
            print(f"\n⚡ FIRST CHUNK LATENCY:")
            print(f"  Average:   {results.avg_first_chunk*1000:7.2f}ms")
            print(f"  Min:       {results.min_first_chunk*1000:7.2f}ms")
            print(f"  Max:       {results.max_first_chunk*1000:7.2f}ms")
            print(f"  Median:    {results.median_first_chunk*1000:7.2f}ms")
            print(f"  95th pctl: {results.p95_first_chunk*1000:7.2f}ms")

            print(f"\n🎯 <1s TARGET:")
            print(f"  Under 1s:  {results.under_1s_count}/{results.successful_count} ({results.under_1s_pct:.1f}%)")

        all_results[burst_size] = results
        time.sleep(2)

    # Summary comparison
    print("\n" + "=" * 70)
    print("BURST SIZE COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Burst':<10} {'Avg TTFA':<12} {'Median':<12} {'95th':<12} {'<1s':<10}")
    print("-" * 60)

    for burst_size in burst_sizes:
        if burst_size in all_results:
            r = all_results[burst_size]
            print(f"{burst_size:<10} {r.avg_first_chunk*1000:<12.1f} "
                  f"{r.median_first_chunk*1000:<12.1f} {r.p95_first_chunk*1000:<12.1f} "
                  f"{r.under_1s_pct:<10.0f}%")

    model.shutdown()
    print("\nDone!")


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description="Chatterbox vLLM Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate audio for different text lengths
  uv run python test_benchmark.py generate

  # Test async token streaming
  uv run python test_benchmark.py async

  # Test concurrent requests
  uv run python test_benchmark.py concurrent --burst-sizes 4 8 16

  # Test with unique texts (no prefix caching)
  uv run python test_benchmark.py concurrent --unique --burst-sizes 8 16 32
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Test mode")

    # Generate mode
    gen_parser = subparsers.add_parser("generate", help="Generate audio for different text lengths")
    gen_parser.add_argument("--sizes", nargs="+", choices=["short", "medium", "long"],
                           help="Text sizes to generate")

    # Async mode
    async_parser = subparsers.add_parser("async", help="Test AsyncLLMEngine streaming")

    # Concurrent mode
    concurrent_parser = subparsers.add_parser("concurrent", help="Test concurrent request handling")
    concurrent_parser.add_argument("--burst-sizes", type=int, nargs="+", default=[1, 4, 8],
                                  help="Burst sizes to test")
    concurrent_parser.add_argument("--unique", action="store_true",
                                  help="Use unique texts (no prefix caching)")

    args = parser.parse_args()

    if not args.mode:
        parser.print_help()
        return

    print("\n" + "=" * 70)
    print("CHATTERBOX vLLM BENCHMARK")
    print("=" * 70)

    if args.mode == "generate":
        mode_generate(args)
    elif args.mode == "async":
        asyncio.run(mode_async(args))
    elif args.mode == "concurrent":
        mode_concurrent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
