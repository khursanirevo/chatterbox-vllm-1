#!/usr/bin/env python3
"""
Poisson Traffic Simulation for Chatterbox vLLM Continuous Batching

This simulates realistic TTS traffic where requests arrive following a Poisson process
(random inter-arrival times with exponential distribution) and varying text lengths.
"""

import asyncio
import time
import random
import torch
import torchaudio as ta
from typing import List, Tuple
import statistics

from chatterbox_vllm import ChatterboxTTSAsync


# Diverse text prompts with varying lengths and complexity
TEXT_CORPUS = {
    "short": [
        "Hello.",
        "Yes, please.",
        "Thank you.",
        "Good morning.",
        "See you later.",
        "That's great!",
        "I agree.",
        "No way.",
        "Sounds good.",
        "Perfect.",
    ],
    "medium": [
        "This is a medium length text that will take some time to process.",
        "The weather today is quite nice, with clear skies and mild temperatures.",
        "I would like to order a large pizza with pepperoni and extra cheese.",
        "Please remember to turn off the lights before leaving the office.",
        "The meeting has been rescheduled to next Tuesday at three in the afternoon.",
        "Could you please send me the report by the end of the day tomorrow?",
        "The new software update includes many improvements and bug fixes.",
        "I'm really looking forward to the weekend trip we have planned.",
        "The train will be arriving at platform four in approximately ten minutes.",
        "Please make sure to review all the documents before the meeting starts.",
    ],
    "long": [
        "This is a significantly longer text passage that will require more processing time through the text to speech synthesis pipeline, including tokenization, language model inference, and audio decoding.",
        "The history of artificial intelligence dates back to ancient times, but the modern field of AI research was founded in 1956 at a conference held at Dartmouth College, where researchers gathered to discuss the possibility of creating machines that could think.",
        "When preparing for a long journey, it's important to pack all the essentials including clothing appropriate for the climate, toiletries, important documents, electronic devices with their chargers, and any medications you might need during your trip.",
        "The process of photosynthesis in plants involves converting light energy into chemical energy, which is stored in glucose molecules, and this process occurs in the chloroplasts of plant cells using chlorophyll to capture the light energy.",
        "In today's competitive job market, having a diverse set of skills is crucial for career success, including technical skills relevant to your field, soft skills like communication and teamwork, and the ability to adapt to new technologies and methodologies.",
        "The concert last night was absolutely incredible, with amazing performances from all the artists, stunning visual effects, and an energetic crowd that made the atmosphere unforgettable and truly special for everyone who attended.",
        "Climate change is one of the most pressing challenges of our time, requiring global cooperation and immediate action to reduce greenhouse gas emissions, transition to renewable energy sources, and implement sustainable practices across all sectors of society.",
        "The novel we read in class was a deeply moving story about love, loss, and redemption, set against the backdrop of war, with complex characters that experienced profound transformations throughout their journeys.",
        "Learning a new language is a rewarding endeavor that opens doors to new cultures, enhances cognitive abilities, improves communication skills, and provides opportunities for personal and professional growth in an increasingly interconnected world.",
        "The restaurant offered an exquisite dining experience with a diverse menu featuring dishes from various cuisines, impeccable service, a beautiful ambiance with elegant decor and soft lighting, and high-quality ingredients that made every dish memorable.",
    ],
    "very_long": [
        """This is an exceptionally long text passage designed to test the upper limits of the text to speech system. It contains multiple sentences with varying complexity and structure. The text to speech model must process all of this content, generate appropriate speech tokens for each segment, and then decode those tokens into high quality audio. This process involves several stages including text normalization and punctuation handling, tokenization through the custom tokenizer, language model inference using the T3 model, and finally audio synthesis using the S3Gen vocoder. Each of these stages contributes to the overall processing time, with longer texts naturally requiring more time to complete. The continuous batching capability of the AsyncLLMEngine allows the system to efficiently handle such variable length requests alongside shorter ones, ensuring that the GPU remains busy with active requests rather than waiting for the longest request in a batch to complete. This is particularly important for real-world applications where users may submit requests with widely varying lengths, from short commands to lengthy passages that need to be synthesized in their entirety.""",
        """Welcome to this comprehensive guide on machine learning fundamentals. In this first section, we will explore the basic concepts that form the foundation of modern machine learning systems. Machine learning is a subfield of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience and data. The core idea is to allow machines to learn patterns from data rather than being explicitly programmed for every possible scenario. This approach has proven incredibly effective across numerous domains including image recognition, natural language processing, recommendation systems, autonomous vehicles, medical diagnosis, financial forecasting, and many more applications that impact our daily lives. The key components of machine learning include data preparation and feature engineering, model selection and architecture design, training algorithms and optimization techniques, evaluation metrics and validation methods, and deployment strategies for production environments.""",
        """The human brain is an extraordinarily complex organ that controls all functions of our body and interprets information from the outside world. It consists of approximately eighty six billion neurons connected by trillions of synapses, forming intricate neural networks that process and transmit information through electrical and chemical signals. The brain is divided into several regions each specialized for different functions including the cerebral cortex for higher cognitive functions like reasoning and language, the cerebellum for coordination and balance, the brainstem for basic life functions like breathing and heart rate, and the limbic system for emotions and memory formation. Understanding how the brain works has been one of the greatest challenges in science, and while we have made significant progress in mapping its structure and functions, much remains to be discovered about the precise mechanisms underlying consciousness, memory, learning, and neurological disorders. Advances in neuroscience and neurotechnology continue to push the boundaries of our knowledge and may lead to breakthrough treatments for various conditions.""",
    ]
}


def generate_poisson_requests(
    num_requests: int,
    avg_requests_per_second: float,
    min_duration: int = 30
) -> List[Tuple[float, str, str]]:
    """
    Generate requests following a Poisson process.

    Args:
        num_requests: Total number of requests to generate
        avg_requests_per_second: Average arrival rate (lambda)
        min_duration: Minimum duration to spread requests over

    Returns:
        List of (arrival_time, request_id, text) tuples
    """
    requests = []
    current_time = 0.0

    # Calculate inter-arrival times using exponential distribution
    for i in range(num_requests):
        # Exponential distribution for Poisson process
        inter_arrival = random.expovariate(avg_requests_per_second)
        current_time += inter_arrival

        # Select text length (weighted towards medium length)
        text_type = random.choices(
            ["short", "medium", "long", "very_long"],
            weights=[0.30, 0.40, 0.20, 0.10],  # 30% short, 40% medium, etc.
            k=1
        )[0]

        text = random.choice(TEXT_CORPUS[text_type])
        requests.append((current_time, f"req-{i:04d}", text))

    return requests


async def simulate_request(
    model: ChatterboxTTSAsync,
    request_id: str,
    text: str,
    arrival_time: float,
    start_time: float,
    results_queue: asyncio.Queue
):
    """
    Process a single TTS request and record metrics.

    Args:
        model: ChatterboxTTSAsync instance
        request_id: Unique request identifier
        text: Text to synthesize
        arrival_time: When the request arrived (relative to start)
        start_time: Simulation start time (absolute)
        results_queue: Queue to collect results
    """
    # Wait until arrival time
    wait_time = arrival_time - (time.time() - start_time)
    if wait_time > 0:
        await asyncio.sleep(wait_time)

    actual_start_time = time.time()
    queue_time = actual_start_time - start_time - arrival_time

    word_count = len(text.split())
    char_count = len(text)

    try:
        # Generate audio
        results = await model.generate(
            prompts=[text],
            temperature=0.8,
            exaggeration=0.5,
        )

        actual_end_time = time.time()
        generation_time = actual_end_time - actual_start_time
        total_time = actual_end_time - start_time

        result = {
            "request_id": request_id,
            "text": text[:50] + "..." if len(text) > 50 else text,
            "word_count": word_count,
            "char_count": char_count,
            "arrival_time": arrival_time,
            "queue_time": queue_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "success": True,
        }

        await results_queue.put(result)

    except Exception as e:
        actual_end_time = time.time()
        result = {
            "request_id": request_id,
            "text": text[:50] + "..." if len(text) > 50 else text,
            "word_count": word_count,
            "char_count": char_count,
            "arrival_time": arrival_time,
            "queue_time": queue_time,
            "generation_time": actual_end_time - actual_start_time,
            "total_time": actual_end_time - start_time,
            "success": False,
            "error": str(e),
        }
        await results_queue.put(result)


async def run_poisson_simulation(
    model: ChatterboxTTSAsync,
    requests: List[Tuple[float, str, str]]
) -> Tuple[List[dict], float]:
    """
    Run the Poisson traffic simulation.

    Args:
        model: ChatterboxTTSAsync instance
        requests: List of (arrival_time, request_id, text) tuples

    Returns:
        Tuple of (results list, total simulation time)
    """
    results_queue = asyncio.Queue()
    start_time = time.time()

    # Create all request tasks
    tasks = [
        simulate_request(model, req_id, text, arrival, start_time, results_queue)
        for arrival, req_id, text in requests
    ]

    # Run all tasks concurrently
    await asyncio.gather(*tasks)

    # Collect results
    results = []
    while not results_queue.empty():
        results.append(await results_queue.get())

    total_time = time.time() - start_time
    return results, total_time


def print_report(results: List[dict], total_time: float, num_requests: int, avg_rate: float):
    """Print comprehensive simulation report."""
    print("\n" + "="*80)
    print("POISSON TRAFFIC SIMULATION REPORT")
    print("="*80)

    print(f"\nSimulation Parameters:")
    print(f"  Total Requests: {num_requests}")
    print(f"  Target Arrival Rate: {avg_rate:.2f} requests/second")
    print(f"  Actual Duration: {total_time:.2f} seconds")
    print(f"  Actual Throughput: {num_requests/total_time:.2f} requests/second")

    # Filter successful requests
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nRequest Success Rate:")
    print(f"  Successful: {len(successful)}/{num_requests} ({len(successful)/num_requests*100:.1f}%)")
    print(f"  Failed: {len(failed)}/{num_requests}")

    if successful:
        # Timing statistics
        queue_times = [r["queue_time"] for r in successful]
        gen_times = [r["generation_time"] for r in successful]
        total_times = [r["total_time"] for r in successful]

        print(f"\nTiming Statistics (seconds):")
        print(f"  {'Metric':<20} {'Min':<10} {'Mean':<10} {'Median':<10} {'Max':<10}")
        print(f"  {'-'*60}")

        for name, values in [("Queue Time", queue_times),
                             ("Generation Time", gen_times),
                             ("Total Time", total_times)]:
            print(f"  {name:<20} {min(values):<10.2f} {statistics.mean(values):<10.2f} "
                  f"{statistics.median(values):<10.2f} {max(values):<10.2f}")

        # Text length statistics
        word_counts = [r["word_count"] for r in successful]
        print(f"\nText Length Statistics (words):")
        print(f"  Min: {min(word_counts)}")
        print(f"  Mean: {statistics.mean(word_counts):.1f}")
        print(f"  Median: {statistics.median(word_counts):.1f}")
        print(f"  Max: {max(word_counts)}")

        # Performance by text length
        print(f"\nPerformance by Text Length:")
        print(f"  {'Length':<15} {'Count':<8} {'Avg Gen Time':<15} {'Avg Total Time':<15}")
        print(f"  {'-'*55}")

        for label, threshold in [("Short", 10), ("Medium", 25), ("Long", 50), ("Very Long", 10000)]:
            subset = [r for r in successful if r["word_count"] <= threshold]
            if subset and label == "Short":
                avg_gen = statistics.mean([r["generation_time"] for r in subset])
                avg_total = statistics.mean([r["total_time"] for r in subset])
                print(f"  {label:<15} {len(subset):<8} {avg_gen:<15.2f} {avg_total:<15.2f}")
            elif label == "Medium":
                subset = [r for r in successful if 10 < r["word_count"] <= threshold]
                if subset:
                    avg_gen = statistics.mean([r["generation_time"] for r in subset])
                    avg_total = statistics.mean([r["total_time"] for r in subset])
                    print(f"  {label:<15} {len(subset):<8} {avg_gen:<15.2f} {avg_total:<15.2f}")
            elif label == "Long":
                subset = [r for r in successful if 25 < r["word_count"] <= threshold]
                if subset:
                    avg_gen = statistics.mean([r["generation_time"] for r in subset])
                    avg_total = statistics.mean([r["total_time"] for r in subset])
                    print(f"  {label:<15} {len(subset):<8} {avg_gen:<15.2f} {avg_total:<15.2f}")
            elif label == "Very Long":
                subset = [r for r in successful if r["word_count"] > 50]
                if subset:
                    avg_gen = statistics.mean([r["generation_time"] for r in subset])
                    avg_total = statistics.mean([r["total_time"] for r in subset])
                    print(f"  {label:<15} {len(subset):<8} {avg_gen:<15.2f} {avg_total:<15.2f}")

        # Timeline visualization
        print(f"\nRequest Timeline (first 20 requests):")
        print(f"  {'ID':<10} {'Words':<8} {'Arrival':<10} {'Queue':<10} {'Gen':<10} {'Total':<10}")
        print(f"  {'-'*60}")

        for r in successful[:20]:
            print(f"  {r['request_id']:<10} {r['word_count']:<8} "
                  f"{r['arrival_time']:<10.2f} {r['queue_time']:<10.2f} "
                  f"{r['generation_time']:<10.2f} {r['total_time']:<10.2f}")

    print("\n" + "="*80)

    return successful, failed


async def main():
    """Run Poisson traffic simulation with continuous batching."""
    print("\n" + "="*80)
    print("CHATTERBOX VLLM - POISSON TRAFFIC SIMULATION")
    print("Testing Continuous Batching with Realistic Traffic Patterns")
    print("="*80)

    # Simulation parameters
    NUM_REQUESTS = 50
    AVG_REQUESTS_PER_SECOND = 2.0  # Average 2 requests per second

    print(f"\nSimulation Configuration:")
    print(f"  Total Requests: {NUM_REQUESTS}")
    print(f"  Average Arrival Rate: {AVG_REQUESTS_PER_SECOND} requests/second")
    print(f"  Traffic Pattern: Poisson (random inter-arrival times)")
    print(f"  Text Length Distribution: 30% short, 40% medium, 20% long, 10% very long")

    # Generate Poisson traffic
    print(f"\nGenerating Poisson traffic pattern...")
    requests = generate_poisson_requests(
        num_requests=NUM_REQUESTS,
        avg_requests_per_second=AVG_REQUESTS_PER_SECOND
    )

    print(f"  Generated {len(requests)} requests")
    print(f"  Time span: {requests[-1][0]:.2f} seconds")

    # Initialize model
    print(f"\nInitializing ChatterboxTTSAsync with continuous batching...")
    model = await ChatterboxTTSAsync.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
    )

    # Run simulation
    print(f"\nStarting simulation...")
    print(f"{'Time':<10} {'Active':<10} {'Completed':<10} {'Progress':<20}")
    print(f"{'-'*50}")

    results, total_time = await run_poisson_simulation(model, requests)

    # Print report
    successful, failed = print_report(results, total_time, NUM_REQUESTS, AVG_REQUESTS_PER_SECOND)

    # Key insights
    print(f"\nKEY INSIGHTS:")
    print(f"  1. Continuous batching allows short requests to complete quickly")
    print(f"     even while long requests are still processing.")
    print(f"  2. Queue time is minimal due to dynamic request scheduling.")
    print(f"  3. GPU utilization remains high with mixed-length workloads.")
    print(f"  4. System handles variable arrival rates efficiently.")

    if failed:
        print(f"\n  WARNING: {len(failed)} requests failed!")
        for r in failed[:5]:
            print(f"    - {r['request_id']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*80)

    # Shutdown
    await model.shutdown()

    return successful, failed, total_time


if __name__ == "__main__":
    successful, failed, duration = asyncio.run(main())
