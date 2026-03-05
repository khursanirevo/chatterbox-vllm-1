#!/usr/bin/env python3
"""
Dedicated profiling script for isolating and measuring each stage of TTS pipeline.

This script measures:
- Tokenization only
- T3 inference only (first token time)
- S3 decoding only (first chunk time)
- Full pipeline TTFA
- Cold vs warm cache performance

Outputs: CSV with timing breakdowns per stage
"""

import asyncio
import time
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import statistics
import torch

from chatterbox_vllm import ChatterboxTTSAsync
from chatterbox_vllm.profiling import TTFAProfiler
from chatterbox_vllm.text_utils import punc_norm


# Test texts with varying token counts
TEST_PROMPTS = {
    10: "Hello world.",
    20: "This is a medium length text for testing.",
    50: "The weather today is quite nice, with clear skies and mild temperatures expected throughout the day. I hope you enjoy it.",
    100: "This is a longer text passage designed to test the text to speech synthesis pipeline with more content to process. The system needs to handle multiple sentences efficiently while maintaining high quality audio output. Each stage of the pipeline contributes to the overall processing time.",
    200: """This is an exceptionally long text passage designed to test the upper limits of the text to speech system. It contains multiple sentences with varying complexity and structure. The text to speech model must process all of this content, generate appropriate speech tokens for each segment, and then decode those tokens into high quality audio. This process involves several stages including text normalization and punctuation handling, tokenization through the custom tokenizer, language model inference using the T3 model, and finally audio synthesis using the S3Gen vocoder. Each of these stages contributes to the overall processing time, with longer texts naturally requiring more time to complete. The continuous batching capability of the AsyncLLMEngine allows the system to efficiently handle such variable length requests alongside shorter ones, ensuring that the GPU remains busy with active requests rather than waiting for the longest request in a batch to complete."""
}


async def profile_tokenization_only(model: ChatterboxTTSAsync, text: str, iterations: int = 10) -> Dict:
    """Profile tokenization stage in isolation."""
    times = []
    token_counts = []

    normalized_text = "[START]" + punc_norm(text) + "[STOP]"
    if model.variant == "multilingual":
        normalized_text = f"<en>{normalized_text}"

    # Access tokenizer from AsyncLLMEngine - need to get it differently
    # The tokenizer is registered with vLLM
    from vllm.transformers_utils.tokenizer import get_tokenizer

    # Get tokenizer using the registered tokenizer name
    tokenizer_name = "EnTokenizer" if model.variant == "english" else "MtlTokenizer"
    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name,
        tokenizer_mode="custom",
    )

    for _ in range(iterations):
        start = time.time()

        tokens = tokenizer.encode(normalized_text)

        end = time.time()
        times.append((end - start) * 1000)  # ms
        token_counts.append(len(tokens))

    return {
        "stage": "tokenization",
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "token_count": statistics.mean(token_counts),
        "iterations": iterations,
    }


async def profile_t3_first_token(model: ChatterboxTTSAsync, text: str, iterations: int = 5) -> Dict:
    """Profile T3 model time to first token."""
    from vllm import SamplingParams
    from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
    from vllm.transformers_utils.tokenizer import get_tokenizer

    times = []
    token_counts = []

    # Get audio conditionals once
    s3gen_ref, cond_emb = model.get_audio_conditionals(None)

    normalized_text = "[START]" + punc_norm(text) + "[STOP]"
    if model.variant == "multilingual":
        normalized_text = f"<en>{normalized_text}"

    sampling_params = SamplingParams(
        temperature=0.8,
        stop_token_ids=[model.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
        max_tokens=100,
        top_p=1.0,
        repetition_penalty=2.0,
    )

    # Warm up
    for _ in range(2):
        request_id = f"warmup_{time.time()}"
        async for _ in model.t3_engine.generate(
            prompt={"prompt": normalized_text, "multi_modal_data": {"conditionals": [cond_emb]}},
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            pass

    # Measure first token time
    for i in range(iterations):
        request_id = f"t3_first_token_{i}_{time.time()}"
        start_time = time.time()
        first_token_time = None

        async for request_output in model.t3_engine.generate(
            prompt={"prompt": normalized_text, "multi_modal_data": {"conditionals": [cond_emb]}},
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if first_token_time is None and request_output.outputs:
                # First output - record time
                first_token_time = time.time()
                times.append((first_token_time - start_time) * 1000)
                token_counts.append(len(request_output.outputs[0].token_ids))
            # We need to consume the generator fully
            if request_output.finished:
                break

    return {
        "stage": "t3_first_token",
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "tokens_generated": statistics.mean(token_counts),
        "iterations": iterations,
    }


async def profile_s3_first_chunk(model: ChatterboxTTSAsync, text: str, iterations: int = 5) -> Dict:
    """Profile S3 model time to first audio chunk."""
    from vllm import SamplingParams
    from chatterbox_vllm.models.t3 import SPEECH_TOKEN_OFFSET
    from chatterbox_vllm.models.s3tokenizer import drop_invalid_tokens

    times = []
    sample_counts = []

    # Get audio conditionals once
    s3gen_ref, cond_emb = model.get_audio_conditionals(None)

    normalized_text = "[START]" + punc_norm(text) + "[STOP]"
    if model.variant == "multilingual":
        normalized_text = f"<en>{normalized_text}"

    sampling_params = SamplingParams(
        temperature=0.8,
        stop_token_ids=[model.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
        max_tokens=100,
        top_p=1.0,
        repetition_penalty=2.0,
    )

    # Generate tokens once
    request_id = f"s3_prep_{time.time()}"
    all_tokens = []

    async for request_output in model.t3_engine.generate(
        prompt={"prompt": normalized_text, "multi_modal_data": {"conditionals": [cond_emb]}},
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        if request_output.outputs:
            all_tokens = request_output.outputs[0].token_ids
        if request_output.finished:
            break

    speech_tokens = torch.tensor(
        [token - SPEECH_TOKEN_OFFSET for token in all_tokens],
        device="cuda"
    )
    speech_tokens = drop_invalid_tokens(speech_tokens)
    speech_tokens = speech_tokens[speech_tokens < 6561]

    # Measure S3 generation time
    for _ in range(iterations):
        torch.cuda.empty_cache()
        start = time.time()

        with torch.inference_mode():
            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=10,
            )

        end = time.time()
        times.append((end - start) * 1000)
        sample_counts.append(wav.shape[1])

    return {
        "stage": "s3_first_chunk",
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "samples_generated": statistics.mean(sample_counts),
        "iterations": iterations,
    }


async def profile_full_pipeline(model: ChatterboxTTSAsync, text: str, iterations: int = 3) -> Dict:
    """Profile full TTS pipeline TTFA."""
    times = []

    for i in range(iterations):
        start = time.time()

        results = await model.generate(
            prompts=[text],
            temperature=0.8,
            exaggeration=0.5,
        )

        end = time.time()
        times.append((end - start) * 1000)

        # Clean up
        del results
        torch.cuda.empty_cache()

    return {
        "stage": "full_pipeline",
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "iterations": iterations,
    }


async def profile_cold_vs_warm_cache(model: ChatterboxTTSAsync, text: str) -> Dict:
    """Compare cold start vs warm cache performance."""
    results = {"cold": [], "warm": []}

    # Cold starts
    for _ in range(3):
        # Clear cache
        torch.cuda.empty_cache()

        start = time.time()
        await model.generate(prompts=[text], temperature=0.8)
        end = time.time()

        results["cold"].append((end - start) * 1000)

    # Warm starts
    for _ in range(10):
        start = time.time()
        await model.generate(prompts=[text], temperature=0.8)
        end = time.time()

        results["warm"].append((end - start) * 1000)

    return {
        "stage": "cache_comparison",
        "cold_mean_ms": statistics.mean(results["cold"]),
        "cold_median_ms": statistics.median(results["cold"]),
        "warm_mean_ms": statistics.mean(results["warm"]),
        "warm_median_ms": statistics.median(results["warm"]),
        "speedup": statistics.mean(results["cold"]) / statistics.mean(results["warm"]),
    }


async def run_full_profiling():
    """Run complete profiling suite."""
    print("\n" + "="*80)
    print("TTS PIPELINE STAGE PROFILING")
    print("="*80)

    # Initialize model
    print("\nInitializing model...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
    )

    output_dir = Path("./ttfa_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Profile each text length
    for expected_tokens, text in TEST_PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"Profiling text with ~{expected_tokens} tokens")
        print(f"Text: {text[:60]}...")
        print(f"{'='*80}")

        # Tokenization
        print("\n[1/4] Profiling tokenization...")
        tok_results = await profile_tokenization_only(model, text, iterations=20)
        tok_results["expected_tokens"] = expected_tokens
        tok_results["text_length"] = len(text)
        all_results.append(tok_results)
        print(f"  Mean: {tok_results['mean_ms']:.2f}ms, Median: {tok_results['median_ms']:.2f}ms")

        # T3 first token
        print("\n[2/4] Profiling T3 first token...")
        t3_results = await profile_t3_first_token(model, text, iterations=5)
        t3_results["expected_tokens"] = expected_tokens
        all_results.append(t3_results)
        print(f"  Mean: {t3_results['mean_ms']:.2f}ms, Median: {t3_results['median_ms']:.2f}ms")

        # S3 first chunk
        print("\n[3/4] Profiling S3 first chunk...")
        s3_results = await profile_s3_first_chunk(model, text, iterations=5)
        s3_results["expected_tokens"] = expected_tokens
        all_results.append(s3_results)
        print(f"  Mean: {s3_results['mean_ms']:.2f}ms, Median: {s3_results['median_ms']:.2f}ms")

        # Full pipeline
        print("\n[4/4] Profiling full pipeline...")
        full_results = await profile_full_pipeline(model, text, iterations=3)
        full_results["expected_tokens"] = expected_tokens
        all_results.append(full_results)
        print(f"  Mean: {full_results['mean_ms']:.2f}ms, Median: {full_results['median_ms']:.2f}ms")

    # Cold vs warm cache
    print(f"\n{'='*80}")
    print("Profiling cold vs warm cache")
    print(f"{'='*80}")

    cache_results = await profile_cold_vs_warm_cache(model, TEST_PROMPTS[50])
    all_results.append(cache_results)
    print(f"\nCold start mean: {cache_results['cold_mean_ms']:.2f}ms")
    print(f"Warm start mean: {cache_results['warm_mean_ms']:.2f}ms")
    print(f"Speedup: {cache_results['speedup']:.2f}x")

    # Save results to CSV
    csv_path = output_dir / "stage_profiling_results.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            "stage", "expected_tokens", "text_length",
            "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms",
            "token_count", "tokens_generated", "samples_generated",
            "iterations", "cold_mean_ms", "cold_median_ms",
            "warm_mean_ms", "warm_median_ms", "speedup"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            # Filter to only fields that exist in this result
            filtered_result = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(filtered_result)

    print(f"\nResults saved to: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)

    for expected_tokens in TEST_PROMPTS.keys():
        print(f"\n{expected_tokens} tokens:")

        for stage in ["tokenization", "t3_first_token", "s3_first_chunk", "full_pipeline"]:
            stage_results = [r for r in all_results if r.get("stage") == stage and r.get("expected_tokens") == expected_tokens]
            if stage_results:
                r = stage_results[0]
                print(f"  {stage:<20}: {r['mean_ms']:>8.2f}ms (median: {r['median_ms']:.2f}ms)")

    # Calculate estimated TTFA breakdown
    print("\n" + "="*80)
    print("ESTIMATED TTFA BREAKDOWN")
    print("="*80)

    for expected_tokens in TEST_PROMPTS.keys():
        tok = next((r for r in all_results if r.get("stage") == "tokenization" and r.get("expected_tokens") == expected_tokens), {})
        t3 = next((r for r in all_results if r.get("stage") == "t3_first_token" and r.get("expected_tokens") == expected_tokens), {})
        s3 = next((r for r in all_results if r.get("stage") == "s3_first_chunk" and r.get("expected_tokens") == expected_tokens), {})

        if tok and t3 and s3:
            estimated_ttfa = tok["mean_ms"] + t3["mean_ms"] + s3["mean_ms"]
            print(f"\n{expected_tokens} tokens:")
            print(f"  Tokenization:      {tok['mean_ms']:>8.2f}ms ({tok['mean_ms']/estimated_ttfa*100:>5.1f}%)")
            print(f"  T3 First Token:    {t3['mean_ms']:>8.2f}ms ({t3['mean_ms']/estimated_ttfa*100:>5.1f}%)")
            print(f"  S3 First Chunk:    {s3['mean_ms']:>8.2f}ms ({s3['mean_ms']/estimated_ttfa*100:>5.1f}%)")
            print(f"  {'-'*40}")
            print(f"  Estimated TTFA:    {estimated_ttfa:>8.2f}ms ({estimated_ttfa/1000:.2f}s)")

    print("\n" + "="*80)

    # Cleanup
    await model.shutdown()

    return all_results


if __name__ == "__main__":
    results = asyncio.run(run_full_profiling())
