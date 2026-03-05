#!/usr/bin/env python3
"""
Detailed component profiling for Chatterbox TTS pipeline.

This script profiles each component of the TTS pipeline using the built-in
TTFA tracking system.

Usage:
    uv run python profile_detailed_components.py
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox_vllm import ChatterboxTTSAsync


async def profile_components():
    """Profile each TTS component in detail."""
    print("=" * 100)
    print("DETAILED COMPONENT PROFILING FOR CHATTERBOX TTS")
    print("=" * 100)

    # Get GPU info
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("\nRunning on CPU")

    # Initialize model with TTFA tracking enabled
    print("\nInitializing model with TTFA tracking...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
        s3gen_use_fp16=True,
        enable_ttfa_tracking=True,
    )

    # Test cases with different lengths
    test_cases = [
        ("Short", "Hello world.", 5),
        ("Medium", "This is a medium length text for testing the speech synthesis system with more content.", 5),
        ("Long", "This is a longer text that should take more time to process through the text to speech pipeline. It contains multiple sentences and should help us understand how the system scales with input length. The goal is to identify which components become bottlenecks as the input size increases.", 3),
    ]

    # Collect all timing data
    all_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            's3gen_use_fp16': True,
        },
        'results': []
    }

    for category, text, num_runs in test_cases:
        print(f"\n{'=' * 100}")
        print(f"Profiling: {category.upper()} - '{text[:50]}...'")
        print(f"{'=' * 100}")
        print(f"Runs: {num_runs}")

        category_data = {
            'category': category,
            'text': text,
            'text_length': len(text),
            'num_runs': num_runs,
            'runs': []
        }

        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")

            # Generate audio and get TTFA tracking
            start_total = time.time()
            audio = await model.generate(
                prompts=[text],
                temperature=0.8,
                exaggeration=0.5,
            )
            total_time = time.time() - start_total

            # Extract TTFA data
            ttfa = model.ttfa_tracking.get_last_ttfa()

            if audio and len(audio) > 0 and ttfa:
                audio_duration = len(audio[0]) / 24000
                rtlf = total_time / audio_duration

                run_data = {
                    'run': run + 1,
                    'total_time_ms': total_time * 1000,
                    'audio_duration_s': audio_duration,
                    'rtlf': rtlf,
                    'components': {
                        'tokenization_ms': ttfa.get('tokenization_ms', 0),
                        't3_first_token_ms': ttfa.get('t3_first_token_ms', 0),
                        't3_decoding_ms': ttfa.get('t3_decoding_ms', 0),
                        't3_total_ms': ttfa.get('t3_first_token_ms', 0) + ttfa.get('t3_decoding_ms', 0),
                        's3gen_speaker_encoder_ms': ttfa.get('s3gen_speaker_encoder_ms', 0),
                        's3gen_mel_ms': ttfa.get('s3gen_mel_ms', 0),
                        's3gen_waveform_ms': ttfa.get('s3gen_waveform_ms', 0),
                        's3gen_total_ms': ttfa.get('s3gen_total_ms', 0),
                        'postprocess_ms': ttfa.get('postprocess_ms', 0),
                    }
                }

                category_data['runs'].append(run_data)

                # Print run results
                print(f"  Tokenization:          {run_data['components']['tokenization_ms']:7.2f}ms")
                print(f"  T3 First Token:         {run_data['components']['t3_first_token_ms']:7.2f}ms")
                print(f"  T3 Decoding:            {run_data['components']['t3_decoding_ms']:7.2f}ms")
                print(f"  T3 Total:               {run_data['components']['t3_total_ms']:7.2f}ms")
                print(f"  S3Gen Speaker Encoder:  {run_data['components']['s3gen_speaker_encoder_ms']:7.2f}ms")
                print(f"  S3Gen Mel Generation:   {run_data['components']['s3gen_mel_ms']:7.2f}ms")
                print(f"  S3Gen Waveform:         {run_data['components']['s3gen_waveform_ms']:7.2f}ms")
                print(f"  S3Gen Total:            {run_data['components']['s3gen_total_ms']:7.2f}ms")
                print(f"  Post-processing:        {run_data['components']['postprocess_ms']:7.2f}ms")
                print(f"  ──")
                print(f"  TOTAL (TTFA):          {run_data['total_time_ms']:7.2f}ms (Audio: {audio_duration:.2f}s, RTLF: {rtlf:.2f}x)")
            else:
                print(f"  ✗ Failed to generate audio")

        all_data['results'].append(category_data)

    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    for cat_data in all_data['results']:
        category = cat_data['category']
        text_length = cat_data['text_length']
        runs = cat_data['runs']

        print(f"\n{category.upper()} (text length: {text_length} chars):")
        print("-" * 100)

        # Calculate statistics for each component
        components = ['tokenization', 't3_first_token', 't3_decoding', 't3_total',
                      's3gen_speaker_encoder', 's3gen_mel', 's3gen_waveform', 's3gen_total',
                      'postprocess', 'total']

        stats = {}
        for comp in components:
            key = f'{comp}_ms'
            values = [r['components'][key] for r in runs]
            stats[comp] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
            }

        # Print table
        print(f"{'Component':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Median':>10}")
        print("-" * 100)

        component_names = {
            'tokenization': 'Tokenization',
            't3_first_token': 'T3 First Token',
            't3_decoding': 'T3 Decoding',
            't3_total': 'T3 Total',
            's3gen_speaker_encoder': 'S3Gen Speaker Enc',
            's3gen_mel': 'S3Gen Mel Gen',
            's3gen_waveform': 'S3Gen Waveform',
            's3gen_total': 'S3Gen Total',
            'postprocess': 'Post-process',
            'total': 'TOTAL (TTFA)',
        }

        for comp in components:
            s = stats[comp]
            name = component_names[comp]
            print(f"{name:<25} {s['mean']:10.2f} {s['std']:10.2f} {s['min']:10.2f} {s['max']:10.2f} {s['median']:10.2f}")

    # Print percentage breakdown
    print("\n" + "=" * 100)
    print("PERCENTAGE BREAKDOWN (averaged across all runs)")
    print("=" * 100)

    # Calculate overall means
    overall_means = {}
    for comp in components:
        key = f'{comp}_ms'
        all_values = []
        for cat_data in all_data['results']:
            all_values.extend([r['components'][key] for r in cat_data['runs']])
        overall_means[comp] = np.mean(all_values)

    total_mean = overall_means['total']

    for comp in ['t3_total', 's3gen_speaker_encoder', 's3gen_mel', 's3gen_waveform', 'tokenization', 'postprocess']:
        if comp in overall_means:
            pct = (overall_means[comp] / total_mean) * 100
            bar_length = int(pct / 2)
            bar = "█" * bar_length
            name = component_names[comp]
            print(f"{name:<25} {pct:6.2f}% {overall_means[comp]:8.2f}ms {bar}")

    # Save results to JSON
    output_file = Path('detailed_component_profiling_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Create a summary markdown file
    create_summary_markdown(all_data)

    return all_data


def create_summary_markdown(data):
    """Create a markdown summary of the profiling results."""
    output_file = Path('COMPONENT_PROFILING_SUMMARY.md')

    with open(output_file, 'w') as f:
        f.write("# Component Profiling Summary\n\n")
        f.write(f"**Timestamp:** {data['metadata']['timestamp']}\n")
        f.write(f"**GPU:** {data['metadata']['gpu']}\n")
        f.write(f"**CUDA:** {data['metadata']['cuda_version']}\n")
        f.write(f"**FP16:** {data['metadata']['s3gen_use_fp16']}\n\n")

        f.write("## Component Breakdown (Averaged Across All Runs)\n\n")

        # Calculate overall means
        components = ['t3_total', 's3gen_speaker_encoder', 's3gen_mel', 's3gen_waveform', 'tokenization', 'postprocess']
        component_names = {
            't3_total': 'T3 Total (Text → Speech Tokens)',
            's3gen_speaker_encoder': 'S3Gen Speaker Encoder',
            's3gen_mel': 'S3Gen Mel Generation',
            's3gen_waveform': 'S3Gen Waveform Generation (HiFT Vocoder)',
            'tokenization': 'Tokenization',
            'postprocess': 'Post-processing',
        }

        # Get all values
        all_values = {}
        for comp in components:
            key = f'{comp}_ms'
            values = []
            for cat_data in data['results']:
                values.extend([r['components'][key] for r in cat_data['runs']])
            all_values[comp] = values

        total_mean = np.mean([v for vals in all_values.values() for v in vals])

        f.write("| Component | Mean (ms) | Std (ms) | Percentage |\n")
        f.write("|-----------|-----------|-----------|------------|\n")

        for comp in components:
            values = all_values[comp]
            mean = np.mean(values)
            std = np.std(values)
            pct = (mean / total_mean) * 100
            name = component_names[comp]
            f.write(f"| **{name}** | {mean:.2f} | {std:.2f} | {pct:.1f}% |\n")

        f.write(f"\n**Total (TTFA):** {total_mean:.2f}ms\n\n")

        # Per-category breakdown
        f.write("## Per-Category Breakdown\n\n")

        for cat_data in data['results']:
            category = cat_data['category']
            text_length = cat_data['text_length']
            runs = cat_data['runs']

            f.write(f"### {category} ({text_length} chars)\n\n")

            # Calculate stats for this category
            f.write("| Component | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |\n")
            f.write("|-----------|-----------|-------------|----------|----------|\n")

            for comp in components:
                key = f'{comp}_ms'
                values = [r['components'][key] for r in runs]
                mean = np.mean(values)
                median = np.median(values)
                min_val = np.min(values)
                max_val = np.max(values)
                name = component_names[comp]
                f.write(f"| {name} | {mean:.2f} | {median:.2f} | {min_val:.2f} | {max_val:.2f} |\n")

            # TTFA total
            total_values = [r['total_time_ms'] for r in runs]
            f.write(f"| **TOTAL (TTFA)** | {np.mean(total_values):.2f} | {np.median(total_values):.2f} | {np.min(total_values):.2f} | {np.max(total_values):.2f} |\n\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Find dominant component
        dominant_comp = max(components, key=lambda c: np.mean(all_values[c]))
        dominant_pct = (np.mean(all_values[dominant_comp]) / total_mean) * 100
        f.write(f"- **Dominant Component:** {component_names[dominant_comp]} ({dominant_pct:.1f}% of TTFA)\n\n")

        f.write("## Optimization Recommendations\n\n")
        f.write("Based on the profiling data, here are the components with the most optimization potential:\n\n")

        # Sort by percentage
        sorted_comp = sorted(components, key=lambda c: np.mean(all_values[c]), reverse=True)
        for i, comp in enumerate(sorted_comp[:5], 1):
            mean = np.mean(all_values[comp])
            pct = (mean / total_mean) * 100
            name = component_names[comp]
            f.write(f"{i}. **{name}** - {pct:.1f}% of TTFA ({mean:.2f}ms)\n")

    print(f"✓ Summary saved to {output_file}")


async def main():
    """Main entry point."""
    import time
    data = await profile_components()

    print("\n" + "=" * 100)
    print("PROFILING COMPLETE")
    print("=" * 100)
    print("\nResults saved to:")
    print("  - detailed_component_profiling_results.json (raw data)")
    print("  - COMPONENT_PROFILING_SUMMARY.md (summary report)")


if __name__ == "__main__":
    asyncio.run(main())
