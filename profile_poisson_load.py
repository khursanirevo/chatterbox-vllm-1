#!/usr/bin/env python3
"""
Profile Chatterbox TTS under Poisson traffic load.

This script sends requests at a specified rate (Poisson distribution) and tracks:
- Queue time: Time from submission to start of processing
- Processing time: Time from start to completion
- Total latency: Queue time + Processing time

This gives a realistic picture of system performance under load.

Usage:
    uv run python profile_poisson_load.py --rate 2 --requests 50
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox_vllm import ChatterboxTTSAsync


class PoissonTrafficProfiler:
    """Profile TTS system under Poisson traffic load."""

    def __init__(self, rate: float, total_requests: int):
        """
        Initialize profiler.

        Args:
            rate: Average requests per second
            total_requests: Total number of requests to send
        """
        self.rate = rate
        self.total_requests = total_requests
        self.results = []

        # Test texts of different lengths
        self.test_texts = [
            "Hello world.",
            "This is a medium length text.",
            "This is a longer text passage that should take more time to process through the speech synthesis pipeline. It contains multiple sentences.",
            "Short.",
            "Medium length text for testing the system.",
            "The quick brown fox jumps over the lazy dog. This classic sentence contains all the letters of the alphabet and is commonly used for testing purposes.",
        ]

    def _get_next_inter_arrival_time(self) -> float:
        """Generate next inter-arrival time using exponential distribution (Poisson process)."""
        return np.random.exponential(1.0 / self.rate)

    async def submit_request(
        self,
        request_id: int,
        model: 'ChatterboxTTSAsync',
        submit_time: float,
        text: str,
    ) -> Dict:
        """
        Submit a single request and track timing.

        Args:
            request_id: Unique request identifier
            model: ChatterboxTTSAsync instance
            submit_time: Time when request was submitted
            text: Text to synthesize

        Returns:
            Dictionary with timing results
        """
        # Record submission time
        result = {
            'request_id': request_id,
            'text': text,
            'text_length': len(text),
            'submit_time': submit_time,
            'start_time': None,
            'complete_time': None,
            'queue_time_ms': None,
            'processing_time_ms': None,
            'total_latency_ms': None,
            'audio_duration_s': None,
            'success': False,
            'error': None,
        }

        try:
            # Mark start time (just before generation)
            start_time = time.time()

            # Generate audio
            audio = await model.generate(
                prompts=[text],
                temperature=0.8,
                exaggeration=0.5,
            )

            # Mark complete time
            complete_time = time.time()

            # Calculate durations
            queue_time = start_time - submit_time
            processing_time = complete_time - start_time
            total_latency = complete_time - submit_time

            # Get audio duration
            audio_duration = len(audio[0]) / 24000 if audio and len(audio) > 0 else 0

            result['start_time'] = start_time
            result['complete_time'] = complete_time
            result['queue_time_ms'] = queue_time * 1000
            result['processing_time_ms'] = processing_time * 1000
            result['total_latency_ms'] = total_latency * 1000
            result['audio_duration_s'] = audio_duration
            result['success'] = True

        except Exception as e:
            result['complete_time'] = time.time()
            result['error'] = str(e)
            result['success'] = False

        return result

    async def run(self):
        """Run the Poisson traffic test."""
        print("=" * 100)
        print("POISSON TRAFFIC LOAD PROFILING")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Rate: {self.rate} requests/second")
        print(f"  Total requests: {self.total_requests}")
        print(f"  Expected duration: ~{self.total_requests / self.rate:.1f}s")
        print(f"  Test texts: {len(self.test_texts)} (randomly selected)")

        # Initialize model
        print("\nInitializing model...")
        model = await ChatterboxTTSAsync.from_pretrained(
            max_batch_size=16,
            max_model_len=1000,
            s3gen_use_fp16=True,
            enable_ttfa_tracking=False,  # Disable for load testing
        )

        print("✓ Model ready\n")

        # Track active requests
        active_tasks = []
        completed_requests = 0
        submit_time_accumulator = 0

        # Start time
        test_start_time = time.time()
        last_print_time = test_start_time

        for request_id in range(self.total_requests):
            # Select random text
            text = self.test_texts[np.random.randint(0, len(self.test_texts))]

            # Submit request
            submit_time = time.time()
            submit_time_accumulator += self._get_next_inter_arrival_time()

            # Wait for inter-arrival time
            while time.time() - submit_time_accumulator < 0:
                await asyncio.sleep(0.001)

            # Submit request
            task = asyncio.create_task(
                self.submit_request(request_id, model, submit_time, text)
            )
            active_tasks.append(task)

            # Clean up completed tasks
            active_tasks = [t for t in active_tasks if not t.done()]
            completed_count = self.total_requests - len(active_tasks)

            # Print progress every second or every 5 requests
            current_time = time.time()
            if current_time - last_print_time >= 1.0 or completed_count > completed_requests:
                print(f"\r[{completed_count}/{self.total_requests}] requests completed", end='', flush=True)
                last_print_time = current_time

            # Yield to allow other tasks to complete
            await asyncio.sleep(0.001)

        # Wait for all requests to complete
        print(f"\n\nWaiting for {len(active_tasks)} remaining requests...")
        for task in active_tasks:
            result = await task
            self.results.append(result)

        test_end_time = time.time()

        # Analyze results
        self.analyze_results(test_end_time - test_start_time)

    def analyze_results(self, total_duration: float):
        """Analyze and print results."""
        print("\n" + "=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)

        # Filter successful results
        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]

        print(f"\nTest Duration: {total_duration:.2f}s")
        print(f"Requests: {len(self.results)}")
        print(f"  Success: {len(successful_results)} ({len(successful_results)/len(self.results)*100:.1f}%)")
        print(f"  Failed: {len(failed_results)} ({len(failed_results)/len(self.results)*100:.1f}%)")

        if len(successful_results) == 0:
            print("\n✗ No successful requests to analyze")
            return

        # Calculate statistics
        queue_times = [r['queue_time_ms'] for r in successful_results]
        processing_times = [r['processing_time_ms'] for r in successful_results]
        total_latencies = [r['total_latency_ms'] for r in successful_results]
        audio_durations = [r['audio_duration_s'] for r in successful_results]

        import pandas as pd

        # Overall statistics
        print("\n" + "-" * 100)
        print("OVERALL STATISTICS")
        print("-" * 100)

        stats = [
            ('Queue Time', queue_times),
            ('Processing Time', processing_times),
            ('Total Latency', total_latencies),
        ]

        for name, values in stats:
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            p50 = np.percentile(values, 50)
            p95 = np.percentile(values, 95)
            p99 = np.percentile(values, 99)
            min_val = np.min(values)

            print(f"\n{name}:")
            print(f"  Mean:    {mean:8.2f}ms")
            print(f"  Median:  {median:8.2f}ms")
            print(f"  Std:     {std:8.2f}ms")
            print(f"  P50:     {p50:8.2f}ms")
            print(f"  P95:     {p95:8.2f}ms")
            print(f"  P99:     {p99:8.8f}f")
            print(f"  Min:     {min_val:8.2f}ms")

        # Request distribution by text length
        print("\n" + "-" * 100)
        print("BREAKDOWN BY TEXT LENGTH")
        print("-" * 100)

        df = pd.DataFrame(successful_results)
        df['length_category'] = pd.cut(
            df['text_length'],
            bins=[0, 20, 50, 100, float('inf')],
            labels=['Short (≤20)', 'Medium (21-50)', 'Long (51-100)', 'Very Long (100+)']
        )

        for category in ['Short (≤20)', 'Medium (21-50)', 'Long (51-100)', 'Very Long (100+)']:
            cat_df = df[df['length_category'] == category]
            if len(cat_df) == 0:
                continue

            print(f"\n{category}: (n={len(cat_df)})")
            print(f"  Queue Time:      {cat_df['queue_time_ms'].mean():8.2f}ms mean")
            print(f"  Processing Time:  {cat_df['processing_time_ms'].mean():8.2f}ms mean")
            print(f"  Total Latency:    {cat_df['total_latency_ms'].mean():8.2f}ms mean")
            print(f"  P95 Total:        {cat_df['total_latency_ms'].quantile(0.95):8.2f}ms")

        # Throughput analysis
        print("\n" + "-" * 100)
        print("THROUGHPUT ANALYSIS")
        print("-" * 100)

        throughput = len(successful_results) / total_duration
        print(f"\nThroughput: {throughput:.2f} requests/second")
        print(f"Target rate: {self.rate} requests/second")
        print(f"Achieved: {(throughput/self.rate)*100:.1f}% of target")

        # Real-Time Factor (RTLF) analysis
        rtlfs = [r['total_latency_ms'] / 1000 / r['audio_duration_s'] for r in successful_results]
        print(f"\nReal-Time Latency Factor (RTLF):")
        print(f"  Mean:    {np.mean(rtlfs):.2f}x")
        print(f"  Median: {np.median(rtlfs):.2f}x")
        print(f"  P95:     {np.percentile(rtlfs, 95):.2f}x")
        print(f"  Target:  <1.0x for real-time")

        realtime_count = sum(1 for r in rtlfs if r < 1.0)
        realtime_pct = (realtime_count / len(rtlfs)) * 100
        print(f"  Real-time requests: {realtime_count}/{len(rtlfs)} ({realtime_pct:.1f}%)")

        # Percentage breakdown
        print("\n" + "=" * 100)
        print("PERCENTAGE BREAKDOWN (Averaged Across All Requests)")
        print("=" * 100)

        total_mean = np.mean(total_latencies)
        queue_mean = np.mean(queue_times)
        processing_mean = np.mean(processing_times)

        queue_pct = (queue_mean / total_mean) * 100
        processing_pct = (processing_mean / total_mean) * 100

        print(f"Queue Time:      {queue_pct:6.2f}% ({queue_mean:.2f}ms)")
        print(f"Processing Time: {processing_pct:6.2f}% ({processing_mean:.2f}ms)")

        bar_length = 30
        queue_bar = "█" * int(queue_pct / 100 * bar_length)
        process_bar = "█" * int(processing_pct / 100 * bar_length)

        print(f"\nQueue:      [{queue_bar:<30}]")
        print(f"Processing: [{process_bar:<30}]")

        # Save results
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'rate': self.rate,
                'total_requests': self.total_requests,
                'test_duration_s': total_duration,
            },
            'results': self.results
        }

        output_file = Path('poisson_load_profiling_results.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")

        # Create summary markdown
        self.create_markdown_summary(output_data)


    def create_markdown_summary(self, data: dict):
        """Create markdown summary of results."""
        output_file = Path('POISSON_LOAD_PROFILING_SUMMARY.md')

        with open(output_file, 'w') as f:
            f.write("# Poisson Load Profiling Summary\n\n")
            f.write(f"**Timestamp:** {data['metadata']['timestamp']}\n")
            f.write(f"**GPU:** {data['metadata']['gpu']}\n")
            f.write(f"**Rate:** {data['metadata']['rate']} requests/second\n")
            f.write(f"**Total Requests:** {data['metadata']['total_requests']}\n")
            f.write(f"**Test Duration:** {data['metadata']['test_duration_s']:.2f}s\n\n")

            # Get successful results
            successful = [r for r in data['results'] if r['success']]
            if not successful:
                f.write("No successful requests to analyze.\n")
                return

            import pandas as pd
            import numpy as np

            df = pd.DataFrame(successful)

            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write("| Metric | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |\n")
            f.write("|--------|-----------|-------------|----------|----------|\n")

            for metric, key, name in [
                ('Queue Time', 'queue_time_ms', 'Queue Time (wait before processing)'),
                ('Processing Time', 'processing_time_ms', 'Processing Time (T3 + S3Gen)'),
                ('Total Latency', 'total_latency_ms', 'Total Latency (queue + processing)'),
            ]:
                values = df[key].values
                f.write(f"| {name} | {np.mean(values):.2f} | {np.median(values):.2f} | {np.percentile(values, 95):.2f} | {np.percentile(values, 99):.2f} |\n")

            # Throughput
            throughput = len(successful) / data['metadata']['test_duration_s']
            f.write(f"\n**Throughput:** {throughput:.2f} requests/second\n")
            f.write(f"**Target Rate:** {data['metadata']['rate']} requests/second\n")
            f.write(f"**Achieved:** {(throughput/data['metadata']['rate'])*100:.1f}% of target\n\n")

            # Breakdown by text length
            f.write("## Breakdown by Text Length\n\n")

            df['length_category'] = pd.cut(
                df['text_length'],
                bins=[0, 20, 50, 100, float('inf')],
                labels=['Short (≤20)', 'Medium (21-50)', 'Long (51-100)', 'Very Long (100+)']
            )

            f.write("| Category | Count | Queue (ms) | Processing (ms) | Total (ms) |\n")
            f.write("|----------|-------|-----------|----------------|----------|\n")

            for category in ['Short (≤20)', 'Medium (21-50)', 'Long (51-100)', 'Very Long (100+)']:
                cat_df = df[df['length_category'] == category]
                if len(cat_df) == 0:
                    continue

                f.write(f"| {category} | {len(cat_df)} | {cat_df['queue_time_ms'].mean():.2f} | {cat_df['processing_time_ms'].mean():.2f} | {cat_df['total_latency_ms'].mean():.2f} |\n")

        print(f"✓ Summary saved to {output_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Chatterbox TTS under Poisson traffic load")
    parser.add_argument('--rate', type=float, default=2.0, help='Requests per second')
    parser.add_argument('--requests', type=int, default=50, help='Total number of requests')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 mode')

    args = parser.parse_args()

    profiler = PoissonTrafficProfiler(rate=args.rate, total_requests=args.requests)
    await profiler.run()


if __name__ == "__main__":
    asyncio.run(main())
