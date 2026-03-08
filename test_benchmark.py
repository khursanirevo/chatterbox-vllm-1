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
import json
import os
import statistics
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
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


# ============== SAVE HELPERS ==============

def bytes_to_tensor(audio_bytes: bytes, sample_rate: int = 24000) -> torch.Tensor:
    """
    Convert raw audio bytes (16-bit PCM) to torch tensor.

    Args:
        audio_bytes: Raw audio bytes (16-bit PCM)
        sample_rate: Sample rate (for shape info)

    Returns:
        Audio tensor of shape (1, num_samples)
    """
    # Convert bytes to numpy array (16-bit PCM)
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

    # Convert to float32 and normalize to [-1, 1]
    audio_np = audio_np.astype(np.float32) / 32768.0

    # Convert to torch tensor and add channel dimension
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)

    return audio_tensor


def save_audio_outputs(
    output_dir: Path,
    request_id: int,
    text: str,
    audio_chunks: List[bytes],
    full_audio_bytes: bytes,
    sample_rate: int,
    metrics: Optional[StreamingMetrics] = None,
    prefix: str = "request",
) -> None:
    """
    Save audio chunks, full audio, and input text to disk.

    Args:
        output_dir: Base output directory
        request_id: Request identifier
        text: Input text
        audio_chunks: List of audio chunk bytes (16-bit PCM)
        full_audio_bytes: Complete audio bytes (16-bit PCM)
        sample_rate: Sample rate
        metrics: Optional StreamingMetrics with timing information
        prefix: Output file prefix
    """
    # Create request-specific directory
    request_dir = output_dir / f"{prefix}_{request_id:04d}"
    request_dir.mkdir(parents=True, exist_ok=True)

    # Save input text
    text_file = request_dir / "input.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    # Save full audio as WAV using bytes directly
    full_audio_path = request_dir / "full_audio.wav"
    with wave.open(str(full_audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(full_audio_bytes)

    # Save individual chunks as WAV using bytes directly
    chunks_dir = request_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    for i, chunk_bytes in enumerate(audio_chunks):
        chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
        with wave.open(str(chunk_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(chunk_bytes)

    # Save metadata JSON
    metadata = {
        "request_id": request_id,
        "text": text,
        "num_chunks": len(audio_chunks),
        "sample_rate": sample_rate,
        "duration_seconds": len(full_audio_bytes) / 2 / sample_rate,  # 2 bytes per sample
        "full_audio_path": str(full_audio_path),
        "chunks_dir": str(chunks_dir),
    }

    # Add timing metrics if available
    if metrics is not None:
        metadata["latency_to_first_chunk_ms"] = round(metrics.latency_to_first_chunk * 1000, 2)
        metadata["rtf"] = round(metrics.rtf, 3)
        metadata["total_generation_time_ms"] = round(metrics.total_generation_time * 1000, 2)
        metadata["t3_token_generation_time_ms"] = round(metrics.t3_token_generation_time * 1000, 2)
        metadata["s3gen_first_chunk_time_ms"] = round(metrics.s3gen_first_chunk_time * 1000, 2)
        metadata["first_s3gen_inference_time_ms"] = round(metrics.first_s3gen_inference_time * 1000, 2)
        metadata["first_serialization_time_ms"] = round(metrics.first_serialization_time * 1000, 2)
        metadata["avg_chunk_time_ms"] = round(metrics.avg_chunk_time * 1000, 2) if metrics.avg_chunk_time > 0 else None

    metadata_file = request_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ============== MODE: GENERATE ==============

def mode_generate(args):
    """Generate audio for different text lengths."""
    print("=" * 70)
    print("GENERATE MODE - Audio for Different Text Lengths")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)

    print("\nLoading model...")
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=3,
        max_model_len=2000,
        gpu_memory_utilization=0.90,
    )
    print("Model loaded!\n")

    results = {}
    sizes = args.sizes or ["short", "medium", "long"]

    for idx, size in enumerate(sizes):
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

        audio_chunks = []  # List of bytes
        final_metrics = None
        for audio_bytes, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            context_window=50,
            print_metrics=True,
        ):
            audio_chunks.append(audio_bytes)
            final_metrics = metrics  # Keep track of the final metrics

        if audio_chunks:
            # Concatenate all bytes
            full_audio_bytes = b"".join(audio_chunks)

            # Save WAV using bytes directly
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                wav_file.setframerate(model.sr)
                wav_file.writeframes(full_audio_bytes)

            # Calculate duration from bytes
            duration = len(full_audio_bytes) / 2 / model.sr

            results[size] = {
                "text": text,
                "output_path": output_path,
                "duration": duration,
                "chunks": len(audio_chunks),
                "rtf": final_metrics.rtf,
                "latency": final_metrics.latency_to_first_chunk
            }

            print(f"\n✓ Saved to: {output_path}")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Chunks: {len(audio_chunks)}")

            # Save detailed outputs (chunks, full audio, text, metrics)
            save_audio_outputs(
                output_dir=output_dir,
                request_id=idx,
                text=text,
                audio_chunks=audio_chunks,
                full_audio_bytes=full_audio_bytes,
                sample_rate=model.sr,
                metrics=final_metrics,
                prefix=size,
            )
            print(f"  Saved detailed outputs to: {output_dir / size}_{idx:04d}")

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

def process_single_request(
    request_id: int,
    text: str,
    model: ChatterboxTTS,
    max_tokens: int = 200,
    save_audio: bool = False,
    output_dir: Optional[Path] = None,
) -> RequestResult:
    """Process a single TTS request."""
    start_time = time.time()
    first_chunk_time = None
    t3_time = None
    s3gen_time = None
    audio_duration = 0.0
    success = True
    error = None
    audio_chunks = []  # List of bytes
    final_metrics = None

    try:
        for audio_bytes, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=25,
            print_metrics=False,
        ):
            if first_chunk_time is None:
                first_chunk_time = metrics.latency_to_first_chunk
                t3_time = metrics.t3_token_generation_time
                s3gen_time = metrics.s3gen_first_chunk_time

            audio_chunks.append(audio_bytes)
            final_metrics = metrics  # Keep track of the final metrics

            if metrics.chunk_count == 1:
                # Calculate duration: bytes / 2 (16-bit) / sample_rate
                audio_duration = len(audio_bytes) / 2 / model.sr
                if not save_audio:
                    break

        end_time = time.time()
        total_time = end_time - start_time
        success = True

        # Save audio outputs if requested
        if save_audio and output_dir is not None and audio_chunks and final_metrics is not None:
            full_audio_bytes = b"".join(audio_chunks)
            save_audio_outputs(
                output_dir=output_dir,
                request_id=request_id,
                text=text,
                audio_chunks=audio_chunks,
                full_audio_bytes=full_audio_bytes,
                sample_rate=model.sr,
                metrics=final_metrics,
                prefix="concurrent",
            )

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


def mode_burst(args):
    """Test sequential burst request handling."""
    print("=" * 70)
    print("BURST MODE - Sequential Burst Testing")
    print("=" * 70)

    burst_sizes = args.burst_sizes or [1, 4, 8]
    texts = UNIQUE_TEXTS if args.unique else BURST_TEXTS
    save_audio = args.save_audio
    output_dir = Path(args.output_dir) if save_audio else None

    if save_audio:
        print(f"Audio saving enabled. Output directory: {output_dir}")

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
        print(f"# BURST SIZE: {burst_size} REQUESTS (SEQUENTIAL PROCESSING)")
        print(f"{'#'*70}")

        results = BurstTestResults(burst_size=burst_size)
        burst_start = time.time()

        # Process requests sequentially
        for i in range(burst_size):
            text = texts[i % len(texts)]
            result = process_single_request(
                request_id=i,
                text=text,
                model=model,
                max_tokens=200,
                save_audio=save_audio,
                output_dir=output_dir,
            )
            results.results.append(result)

            elapsed = result.end_time - burst_start
            print(f"  Request {result.request_id+1:2d}: "
                  f"TTFA={result.first_chunk_time*1000:7.2f}ms, "
                  f"Total={result.total_time*1000:7.2f}ms")

        # Calculate burst metrics
        burst_end = time.time()
        burst_duration = burst_end - burst_start
        throughput = burst_size / burst_duration

        # Print results
        print(f"\n{'='*70}")
        print(f"RESULTS: {burst_size} REQUESTS")
        print(f"{'='*70}")
        print(f"\nSuccess: {results.successful_count}/{results.burst_size}")

        if results.successful_count > 0:
            print(f"\n⏱️  BURST METRICS:")
            print(f"  Burst duration:      {burst_duration:.2f}s")
            print(f"  Throughput:          {throughput:.2f} requests/second")
            print(f"  Avg request time:    {burst_duration/burst_size:.2f}s")

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

    print(f"\n{'Burst':<10} {'Avg TTFA':<12} {'Throughput':<15} {'Duration':<12} {'<1s':<10}")
    print("-" * 65)

    for burst_size in burst_sizes:
        if burst_size in all_results:
            r = all_results[burst_size]
            # Calculate approximate throughput from results
            first_start = r.results[0].start_time if r.results else 0
            last_end = r.results[-1].end_time if r.results else 0
            duration = last_end - first_start
            throughput = burst_size / duration if duration > 0 else 0

            print(f"{burst_size:<10} {r.avg_first_chunk*1000:<12.1f} "
                  f"{throughput:<15.2f} {duration:<12.2f} {r.under_1s_pct:<10.0f}%")

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

  # Generate with custom output directory
  uv run python test_benchmark.py generate --output-dir output/my_test

  # Test async token streaming
  uv run python test_benchmark.py async

  # Test sequential burst requests
  uv run python test_benchmark.py burst --burst-sizes 4 8 16

  # Test burst and save audio chunks, full audio, and text
  uv run python test_benchmark.py burst --burst-sizes 4 --save-audio

  # Test with unique texts (no prefix caching) and save audio
  uv run python test_benchmark.py burst --unique --burst-sizes 8 16 --save-audio
        """
    )

    subparsers = parser.add_subparsers(dest="mode", help="Test mode")

    # Generate mode
    gen_parser = subparsers.add_parser("generate", help="Generate audio for different text lengths")
    gen_parser.add_argument("--sizes", nargs="+", choices=["short", "medium", "long"],
                           help="Text sizes to generate")
    gen_parser.add_argument("--output-dir", type=str, default="output/generate",
                           help="Output directory for audio and text")

    # Async mode
    async_parser = subparsers.add_parser("async", help="Test AsyncLLMEngine streaming")

    # Burst mode (sequential processing)
    burst_parser = subparsers.add_parser("burst", help="Test sequential burst request handling")
    burst_parser.add_argument("--burst-sizes", type=int, nargs="+", default=[1, 4, 8],
                              help="Burst sizes to test")
    burst_parser.add_argument("--unique", action="store_true",
                              help="Use unique texts (no prefix caching)")
    burst_parser.add_argument("--output-dir", type=str, default="output/burst",
                              help="Output directory for audio and text")
    burst_parser.add_argument("--save-audio", action="store_true",
                              help="Save audio chunks, full audio, and text")

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
    elif args.mode == "burst":
        mode_burst(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
