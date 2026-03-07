#!/usr/bin/env python3
"""
Benchmark first audio chunk latency with detailed output.

Organizes output as:
    output/01_{text_input}/chunks/chunk_000.wav
    output/01_{text_input}/full.mp3

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python benchmark_first_chunk.py
"""

import os
import shutil
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio as ta

from chatterbox_vllm.tts import ChatterboxTTS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Test texts with different characteristics
TEST_CASES = [
    {
        "name": "short_hello",
        "text": "Hello world, this is a test.",
        "max_tokens": 200,
    },
    {
        "name": "medium_sentence",
        "text": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "max_tokens": 400,
    },
    {
        "name": "long_paragraph",
        "text": "Artificial intelligence has revolutionized how we interact with technology in recent years. From virtual assistants that can understand natural language to autonomous vehicles that navigate complex environments, AI is becoming increasingly sophisticated.",
        "max_tokens": 600,
    },
    {
        "name": "very_long_text",
        "text": "The streaming text-to-speech system represents a significant advancement in real-time audio generation. By leveraging continuous batching and efficient token streaming, we can achieve sub-second latency for the first audio chunk. This is particularly important for interactive applications where users expect immediate feedback. The system processes text incrementally, generating speech tokens that are then converted to audio chunks in real-time.",
        "max_tokens": 1000,
    },
]


def sanitize_filename(name: str) -> str:
    """Sanitize name for use as filename."""
    return name.lower().replace(" ", "_").replace("-", "_")[:50]


def ensure_output_dir(base_dir: Path, test_name: str) -> Path:
    """Create output directory structure."""
    output_dir = base_dir / test_name
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_chunk(audio_chunk: torch.Tensor, chunk_dir: Path, index: int, sample_rate: int):
    """Save individual audio chunk."""
    chunk_path = chunk_dir / f"chunk_{index:03d}.wav"
    ta.save(str(chunk_path), audio_chunk, sample_rate)
    return chunk_path


def save_full_audio(audio_chunks: List[torch.Tensor], output_path: Path, sample_rate: int):
    """Concatenate and save full audio."""
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=-1)

        # Save as WAV first
        wav_path = output_path.with_suffix(".wav")
        ta.save(str(wav_path), full_audio, sample_rate)

        # Convert to MP3 if ffmpeg is available
        try:
            import subprocess
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), str(output_path)],
                check=True,
                capture_output=True,
            )
            # Remove WAV after successful MP3 conversion
            wav_path.unlink()
        except (FileNotFoundError, subprocess.CalledProcessError):
            # ffmpeg not available, keep WAV
            if output_path.suffix == ".mp3":
                output_path = wav_path

        return full_audio.shape[-1] / sample_rate
    return 0.0


def run_benchmark(
    model: ChatterboxTTS,
    test_cases: List[dict],
    output_base: Path,
    chunk_size: int = 25,
):
    """Run benchmark on all test cases."""

    results = []

    for idx, test_case in enumerate(test_cases, 1):
        name = test_case["name"]
        text = test_case["text"]
        max_tokens = test_case["max_tokens"]

        # Create sanitized folder name
        folder_name = f"{idx:02d}_{sanitize_filename(name)}"
        output_dir = ensure_output_dir(output_base, folder_name)
        chunks_dir = output_dir / "chunks"

        print(f"\n{'='*70}")
        print(f"Test {idx}/{len(test_cases)}: {name}")
        print(f"{'='*70}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Output: {output_dir}")
        print(f"\nGenerating...")

        # Save text to file for reference
        (output_dir / "text.txt").write_text(text)

        # Generate streaming audio
        start_time = time.time()
        first_chunk_time = None
        first_chunk_latency = None
        audio_chunks = []

        chunk_index = 0
        for audio_chunk, metrics in model.generate_stream(
            text=text,
            max_tokens=max_tokens,
            chunk_size=chunk_size,
            print_metrics=True,
        ):
            # Capture first chunk latency
            if first_chunk_time is None:
                first_chunk_time = time.time()
                first_chunk_latency = first_chunk_time - start_time
                print(f"\n⚡ First chunk latency: {first_chunk_latency*1000:.1f}ms")

            # Save individual chunk
            save_chunk(audio_chunk, chunks_dir, chunk_index, model.sr)
            chunk_index += 1

            audio_chunks.append(audio_chunk)

        total_time = time.time() - start_time

        # Save full audio
        full_path = output_dir / "full.mp3"
        duration = save_full_audio(audio_chunks, full_path, model.sr)

        # Calculate stats
        full_audio = torch.cat(audio_chunks, dim=-1)

        result = {
            "index": idx,
            "name": name,
            "text": text,
            "output_dir": output_dir,
            "first_chunk_latency_ms": first_chunk_latency * 1000 if first_chunk_latency else 0,
            "total_time_s": total_time,
            "duration_s": duration,
            "chunks": len(audio_chunks),
            "rtf": total_time / duration if duration > 0 else 0,
        }
        results.append(result)

        print(f"\n✓ Complete!")
        print(f"  Output: {output_dir}")
        print(f"  First chunk: {result['first_chunk_latency_ms']:.1f}ms")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Chunks: {len(audio_chunks)}")
        print(f"  RTF: {result['rtf']:.3f}")

    return results


def print_summary(results: List[dict], output_path: Path):
    """Print benchmark summary."""

    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Print table
    print(f"\n{'#':<4} {'Test':<25} {'1st Chunk':<12} {'Total':<10} {'Duration':<10} {'RTF':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['index']:<4} {r['name']:<25} "
              f"{r['first_chunk_latency_ms']:>8.1f}ms  "
              f"{r['total_time_s']:>8.2f}s  "
              f"{r['duration_s']:>8.2f}s  "
              f"{r['rtf']:>6.3f}")

    # Statistics
    first_chunk_latencies = [r["first_chunk_latency_ms"] for r in results]
    avg_latency = sum(first_chunk_latencies) / len(first_chunk_latencies)
    min_latency = min(first_chunk_latencies)
    max_latency = max(first_chunk_latencies)

    print(f"\n{'='*70}")
    print("FIRST CHUNK LATENCY STATISTICS")
    print(f"{'='*70}")
    print(f"  Average: {avg_latency:.1f}ms")
    print(f"  Min:     {min_latency:.1f}ms")
    print(f"  Max:     {max_latency:.1f}ms")
    print(f"  Range:   {max_latency - min_latency:.1f}ms")

    # Target check
    under_1s = sum(1 for l in first_chunk_latencies if l < 1000)
    pct_under_1s = (under_1s / len(first_chunk_latencies)) * 100

    print(f"\n🎯 <1s TARGET:")
    print(f"  Under 1s: {under_1s}/{len(results)} ({pct_under_1s:.1f}%)")

    if pct_under_1s == 100:
        print(f"  ✅ ALL TESTS PASSED!")
    elif pct_under_1s >= 75:
        print(f"  ✓ Good performance")
    else:
        print(f"  ⚠️  Some tests above target")

    # Save summary to file
    summary_path = output_path / "benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Total tests: {len(results)}\n\n")

        f.write(f"{'#':<4} {'Test':<25} {'1st Chunk':<12} {'Total':<10} {'Duration':<10} {'RTF':<8}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['index']:<4} {r['name']:<25} "
                   f"{r['first_chunk_latency_ms']:>8.1f}ms  "
                   f"{r['total_time_s']:>8.2f}s  "
                   f"{r['duration_s']:>8.2f}s  "
                   f"{r['rtf']:>6.3f}\n")

        f.write(f"\nFIRST CHUNK LATENCY:\n")
        f.write(f"  Average: {avg_latency:.1f}ms\n")
        f.write(f"  Min:     {min_latency:.1f}ms\n")
        f.write(f"  Max:     {max_latency:.1f}ms\n")
        f.write(f"\nUnder 1s: {under_1s}/{len(results)} ({pct_under_1s:.1f}%)\n")

    print(f"\n📄 Summary saved to: {summary_path}")


def main():
    print("=" * 70)
    print("FIRST AUDIO CHUNK LATENCY BENCHMARK")
    print("=" * 70)

    # Setup output directory
    output_base = Path("output")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_output = output_base / f"benchmark_{timestamp}"

    print(f"\nOutput directory: {run_output}")

    # Initialize model
    print("\nInitializing model...")
    model = ChatterboxTTS.from_pretrained(
        max_model_len=2000,
        gpu_memory_utilization=0.90,
    )
    print("✓ Model ready")

    # Warmup model to measure steady-state performance
    print("\nWarming up model...")
    warmup_text = "The quick brown fox jumps over the lazy dog."
    for i in range(3):
        for audio_chunk, _ in model.generate_stream(
            text=warmup_text,
            max_tokens=200,
            chunk_size=25,
            print_metrics=False,
        ):
            break
    print("✓ Warmup complete (measuring steady-state performance)")

    # Run benchmark
    results = run_benchmark(
        model=model,
        test_cases=TEST_CASES,
        output_base=run_output,
        chunk_size=25,
    )

    # Print summary
    print_summary(results, run_output)

    # Cleanup
    model.shutdown()

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 All outputs saved to: {run_output.absolute()}")
    print("\nDirectory structure:")
    for r in results:
        print(f"  {r['output_dir'].relative_to(run_output)}/")
        print(f"    ├── text.txt")
        print(f"    ├── chunks/")
        print(f"    │   ├── chunk_000.wav")
        print(f"    │   ├── chunk_001.wav")
        print(f"    │   └── ...")
        print(f"    └── full.mp3")


if __name__ == "__main__":
    main()
