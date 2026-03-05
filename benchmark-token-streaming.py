#!/usr/bin/env python3
"""
Token-Level Streaming TTFA Measurement

Demonstrates the improved Time To First Audio (TTFA) with token-level streaming
compared to the standard batch generation approach.
"""

import asyncio
import time
import torch
from typing import List, Dict
import statistics

from chatterbox_vllm.tts_streaming import ChatterboxTTSStreaming


# Test prompts of varying lengths
TEST_PROMPTS = {
    "short": [
        "Hello.",
        "Yes, please.",
        "Thank you.",
        "Good morning.",
        "Perfect.",
    ],
    "medium": [
        "This is a medium length text that will take some time to process.",
        "The weather today is quite nice with clear skies.",
        "I would like to order a large pizza with pepperoni.",
    ],
    "long": [
        "This is a significantly longer text passage that will require more processing time through the text to speech synthesis pipeline, including tokenization, language model inference, and audio decoding.",
        "The history of artificial intelligence dates back to ancient times, but the modern field of AI research was founded in 1956 at a conference held at Dartmouth College.",
    ],
}


async def measure_streaming_ttfa(
    model: ChatterboxTTSStreaming,
    prompt: str,
    category: str,
    run_id: int,
) -> Dict:
    """
    Measure TTFA with token-level streaming.

    Returns:
        Dictionary with TTFA metrics
    """
    print(f"[{category}] Running: {prompt[:50]}...")

    start_time = time.time()
    chunks = []
    first_chunk_time = None
    last_chunk_time = None

    # Stream audio with token-level generation
    async for chunk in model.stream_audio_tokens(
        prompt=prompt,
        temperature=0.8,
        exaggeration=0.5,
        max_tokens=1000,
    ):
        if first_chunk_time is None:
            first_chunk_time = time.time()
            # TTFA = time to first chunk
            ttfa = first_chunk_time - start_time
            print(f"  ✓ First chunk received in {ttfa:.2f}s")

        chunks.append(chunk.cpu())
        last_chunk_time = time.time()

    total_time = time.time() - start_time

    result = {
        "run_id": run_id,
        "category": category,
        "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
        "word_count": len(prompt.split()),
        "ttfa": first_chunk_time - start_time if first_chunk_time else None,
        "total_time": total_time,
        "num_chunks": len(chunks),
        "total_samples": sum(c.shape[1] for c in chunks),
        "audio_duration": sum(c.shape[1] for c in chunks) / model.sr,
        "success": True,
    }

    print(f"  ✓ Complete: {result['num_chunks']} chunks, {result['audio_duration']:.2f}s audio, "
          f"TTFA: {result['ttfa']:.2f}s, Total: {result['total_time']:.2f}s")

    return result


async def measure_standard_ttfa(
    model: ChatterboxTTSStreaming,
    prompt: str,
    category: str,
    run_id: int,
) -> Dict:
    """
    Measure TTFA with standard batch generation (baseline for comparison).

    Returns:
        Dictionary with TTFA metrics
    """
    print(f"[{category}] Running (standard): {prompt[:50]}...")

    start_time = time.time()

    # Use standard generate method (non-streaming)
    results = await model.generate(
        prompts=[prompt],
        temperature=0.8,
        exaggeration=0.5,
        max_tokens=1000,
    )

    first_audio_time = time.time()
    total_time = first_audio_time - start_time

    if results and results[0] is not None:
        audio = results[0]
        result = {
            "run_id": run_id,
            "category": category,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "word_count": len(prompt.split()),
            "ttfa": total_time,  # For standard, TTFA = total time (audio available at end)
            "total_time": total_time,
            "num_chunks": 1,
            "total_samples": audio.shape[1],
            "audio_duration": audio.shape[1] / model.sr,
            "success": True,
        }
        print(f"  ✓ Complete: {result['audio_duration']:.2f}s audio, "
              f"TTFA: {result['ttfa']:.2f}s")
        return result

    return {
        "run_id": run_id,
        "category": category,
        "prompt": prompt[:50],
        "ttfa": None,
        "total_time": total_time,
        "success": False,
    }


def print_comparison_report(streaming_results: List[Dict], standard_results: List[Dict]):
    """Print comprehensive comparison report."""
    print("\n" + "="*100)
    print("TOKEN-LEVEL STREAMING VS STANDARD GENERATION - TTFA COMPARISON")
    print("="*100)

    streaming_by_category = {}
    standard_by_category = {}

    for r in streaming_results:
        cat = r["category"]
        if cat not in streaming_by_category:
            streaming_by_category[cat] = []
        streaming_by_category[cat].append(r)

    for r in standard_results:
        cat = r["category"]
        if cat not in standard_by_category:
            standard_by_category[cat] = []
        standard_by_category[cat].append(r)

    print(f"\n{'Category':<15} {'Method':<25} {'Count':<8} {'Min TTFA':<12} {'Mean TTFA':<12} "
          f"{'Median TTFA':<12} {'Max TTFA':<12}")
    print("-" * 110)

    for category in ["short", "medium", "long"]:
        streaming_cats = streaming_by_category.get(category, [])
        standard_cats = standard_by_category.get(category, [])

        if streaming_cats:
            stream_ttfas = [r["ttfa"] for r in streaming_cats if r["ttfa"]]
            print(f"{category.capitalize():<15} {'Streaming':<25} {len(stream_ttfas):<8} "
                  f"{min(stream_ttfas):<12.2f} {statistics.mean(stream_ttfas):<12.2f} "
                  f"{statistics.median(stream_ttfas):<12.2f} {max(stream_ttfas):<12.2f}")

        if standard_cats:
            standard_ttfas = [r["ttfa"] for r in standard_cats if r["ttfa"]]
            print(f"{category.capitalize():<15} {'Standard (Baseline)':<25} {len(standard_ttfas):<8} "
                  f"{min(standard_ttfas):<12.2f} {statistics.mean(standard_ttfas):<12.2f} "
                  f"{statistics.median(standard_ttfas):<12.2f} {max(standard_ttfas):<12.2f}")

        print()  # Blank line between categories

    # Overall comparison
    print("="*100)
    print("OVERALL IMPROVEMENT")
    print("="*100)

    all_streaming_ttfa = [r["ttfa"] for r in streaming_results if r["ttfa"]]
    all_standard_ttfa = [r["ttfa"] for r in standard_results if r["ttfa"]]

    print(f"\nAll Requests Combined:")
    print(f"  Streaming TTFA:   Mean={statistics.mean(all_streaming_ttfa):.2f}s, "
          f"Median={statistics.median(all_streaming_ttfa):.2f}s")
    print(f"  Standard TTFA:    Mean={statistics.mean(all_standard_ttfa):.2f}s, "
          f"Median={statistics.median(all_standard_ttfa):.2f}s")

    if statistics.mean(all_streaming_ttfa) > 0:
        improvement = ((statistics.mean(all_standard_ttfa) - statistics.mean(all_streaming_ttfa)) /
                      statistics.mean(all_standard_ttfa) * 100)
        print(f"\n  ✅ TTFA Improvement: {improvement:.1f}% faster with streaming!")

    # Per-category improvement
    print(f"\nImprovement by Category:")
    for category in ["short", "medium", "long"]:
        stream_ttfas = [r["ttfa"] for r in streaming_by_category.get(category, []) if r["ttfa"]]
        standard_ttfas = [r["ttfa"] for r in standard_by_category.get(category, []) if r["ttfa"]]

        if stream_ttfas and standard_ttfas:
            improvement = ((statistics.mean(standard_ttfas) - statistics.mean(stream_ttfas)) /
                          statistics.mean(standard_ttfas) * 100)
            print(f"  {category.capitalize():<10}: {improvement:>+6.1f}% "
                  f"(Stream: {statistics.mean(stream_ttfas):.2f}s vs "
                  f"Standard: {statistics.mean(standard_ttfas):.2f}s)")

    print("\n" + "="*100)


async def main():
    """Run token-level streaming TTFA comparison."""
    print("\n" + "="*100)
    print("CHATTERBOX VLLM - TOKEN-LEVEL STREAMING TTFA MEASUREMENT")
    print("="*100)

    print("\nThis example compares Time To First Audio (TTFA) between:")
    print("  1. Token-level streaming - audio chunks as tokens arrive")
    print("  2. Standard generation - audio after full completion")

    # Initialize streaming model
    print(f"\nInitializing ChatterboxTTSStreaming...")
    model = await ChatterboxTTSStreaming.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
    )

    # Prepare test cases
    test_cases = []
    run_id = 0

    for category, prompts in TEST_PROMPTS.items():
        for prompt in prompts:
            test_cases.append((category, prompt, run_id))
            run_id += 1

    print(f"\nRunning {len(test_cases)} test cases...")

    # Measure streaming TTFA
    print("\n" + "="*100)
    print("PART 1: Token-Level Streaming")
    print("="*100)

    streaming_results = []
    for category, prompt, run_id in test_cases:
        result = await measure_streaming_ttfa(model, prompt, category, run_id)
        streaming_results.append(result)
        await asyncio.sleep(0.1)  # Small delay between requests

    # Measure standard TTFA (baseline)
    print("\n" + "="*100)
    print("PART 2: Standard Generation (Baseline)")
    print("="*100)

    standard_results = []
    for category, prompt, run_id in test_cases:
        result = await measure_standard_ttfa(model, prompt, category, run_id)
        standard_results.append(result)
        await asyncio.sleep(0.1)

    # Print comparison report
    print_comparison_report(streaming_results, standard_results)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("  1. Token-level streaming yields first audio chunk as soon as tokens arrive")
    print("  2. Standard generation must wait for complete token generation")
    print("  3. TTFA improvement is most significant for longer texts")
    print("  4. User experience: streaming feels much more responsive")

    print("\nRECOMMENDATIONS:")
    print("  ✅ Use token-level streaming for interactive applications")
    print("  ✅ Use standard generation for batch/audio book processing")
    print("  ✅ Configure min_tokens_for_audio based on your use case")
    print("  ✅ Adjust stream_chunk_samples for desired chunk size")

    # Cleanup
    await model.shutdown()

    print("\n" + "="*100)


if __name__ == "__main__":
    asyncio.run(main())
