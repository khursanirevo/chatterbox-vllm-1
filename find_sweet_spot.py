#!/usr/bin/env python3
"""
Find the sweet spot for text length to achieve <1s first chunk latency.

Tests various text lengths to find the maximum text that stays under 1s.
"""

import os
import time

from chatterbox_vllm.tts import ChatterboxTTS

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_text_length(model: ChatterboxTTS, text: str) -> dict:
    """Test a single text and return first chunk latency."""
    start = time.time()
    t3_time = None
    s3gen_time = None

    for audio_chunk, metrics in model.generate_stream(
        text=text,
        max_tokens=500,
        chunk_size=25,
        print_metrics=False,
    ):
        t3_time = metrics.t3_token_generation_time
        s3gen_time = metrics.s3gen_first_chunk_time
        break

    total = time.time() - start

    return {
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "total_ms": total * 1000,
        "t3_ms": t3_time * 1000,
        "s3gen_ms": s3gen_time * 1000,
    }


def main():
    print("="*70)
    print("FINDING SWEET SPOT FOR <1s FIRST CHUNK")
    print("="*70)

    # Test texts of increasing length
    test_texts = [
        ("Hello world.", 2),
        ("Hello world, this is a test.", 8),
        ("The quick brown fox jumps over the lazy dog.", 9),
        ("This is a medium length text for testing the TTS system performance.", 13),
        ("The quick brown fox jumps over the lazy dog. This sentence contains every letter.", 16),
        ("Artificial intelligence has revolutionized how we interact with technology in recent years.", 14),
        ("This is a longer text designed to test the streaming capabilities with more content to process.", 19),
        ("The streaming TTS system represents a significant advancement in real-time audio generation using neural networks.", 18),
        ("Machine learning models require large amounts of training data to achieve good performance on various tasks and applications.", 20),
        ("Climate change affects ecosystems globally. Rising temperatures, melting glaciers, and extreme weather events are becoming more common. This requires urgent action.", 24),
    ]

    print("\nInitializing model...")
    model = ChatterboxTTS.from_pretrained(
        max_model_len=2000,
        gpu_memory_utilization=0.90,
    )
    print("✓ Model ready")

    # Warmup
    print("\nWarming up...")
    for _ in range(2):
        for audio_chunk, _ in model.generate_stream(
            text="Hello world.",
            max_tokens=200,
            chunk_size=25,
            print_metrics=False,
        ):
            break
    print("✓ Warmup complete\n")

    results = []

    print(f"{'Words':<8} {'Chars':<8} {'Total':<10} {'T3':<10} {'S3Gen':<10} {'Status'}")
    print("-" * 70)

    for text, expected_words in test_texts:
        result = test_text_length(model, text)
        results.append(result)

        status = "✅ PASS" if result["total_ms"] < 1000 else "❌ FAIL"
        print(f"{result['word_count']:<8} {result['char_count']:<8} "
              f"{result['total_ms']:>7.0f}ms    {result['t3_ms']:>7.0f}ms    "
              f"{result['s3gen_ms']:>7.0f}ms    {status}")

    # Analysis
    print("\n" + "="*70)
    print("SWEET SPOT ANALYSIS")
    print("="*70)

    passing = [r for r in results if r["total_ms"] < 1000]
    failing = [r for r in results if r["total_ms"] >= 1000]

    if passing:
        max_passing = max(passing, key=lambda x: x["word_count"])
        print(f"\n✅ Maximum passing text:")
        print(f"   Words: {max_passing['word_count']}")
        print(f"   Chars: {max_passing['char_count']}")
        print(f"   Total: {max_passing['total_ms']:.0f}ms")
        print(f"   Text: '{max_passing['text']}'")

    if failing:
        min_failing = min(failing, key=lambda x: x["word_count"])
        print(f"\n❌ Minimum failing text:")
        print(f"   Words: {min_failing['word_count']}")
        print(f"   Chars: {min_failing['char_count']}")
        print(f"   Total: {min_failing['total_ms']:.0f}ms")
        print(f"   Text: '{min_failing['text']}'")

    # Calculate T3 generation rate
    print("\n" + "="*70)
    print("T3 GENERATION RATE")
    print("="*70)

    rates = []
    for r in results:
        rate = r["word_count"] / (r["t3_ms"] / 1000)  # words per second
        rates.append(rate)

    avg_rate = sum(rates) / len(rates)
    print(f"\nAverage T3 rate: {avg_rate:.1f} words/second")
    print(f"Or: {1000/avg_rate:.0f} ms per word")

    # Sweet spot calculation
    # We need: T3 + 300ms (S3Gen) < 1000ms
    # T3 < 700ms
    t3_budget_ms = 700
    max_words = int((t3_budget_ms / 1000) * avg_rate)

    print(f"\n🎯 SWEET SPOT:")
    print(f"   S3Gen constant: ~300ms")
    print(f"   T3 budget: {t3_budget_ms}ms (to stay under 1s)")
    print(f"   Max words: ~{max_words} words")
    print(f"   Max chars: ~{max_words * 5} chars (assuming avg 5 chars/word)")

    print(f"\n💡 RECOMMENDATION:")
    print(f"   For best results, keep texts under {max_words} words")
    print(f"   For longer texts, use AsyncLLMEngine (can handle any length)")

    model.shutdown()


if __name__ == "__main__":
    main()
