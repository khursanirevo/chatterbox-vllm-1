"""
Integration test for S3Gen stream pool with actual TTS generation.

This test requires GPU and model files - mark as slow.
"""

import pytest
import asyncio
import time
from chatterbox_vllm.tts_async import AsyncChatterboxTTS

@pytest.mark.slow
@pytest.mark.asyncio
async def test_stream_pool_with_real_model():
    """Test stream pool with real TTS model."""
    # Create model with stream pool
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=4,
        gpu_memory_utilization=0.3,  # Lower for testing
    )

    try:
        # Verify stream pool exists
        assert model.s3gen_stream_pool is not None
        assert model.s3gen_stream_pool.num_streams == 4

        # Test single request
        chunks = []
        async for chunk, metrics in model.generate_stream("Hello world", print_metrics=False):
            chunks.append(chunk)
            if len(chunks) >= 2:
                break

        assert len(chunks) >= 2
        assert model.s3gen_stream_pool.metrics.total_requests > 0

        print(f"Stream pool metrics: {model.s3gen_stream_pool.metrics}")

    finally:
        await model.shutdown()

@pytest.mark.slow
@pytest.mark.asyncio
async def test_stream_pool_concurrent_performance():
    """Test that stream pool improves concurrent performance."""
    # Create model with stream pool
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=8,
        gpu_memory_utilization=0.3,
    )

    try:
        # Test 4 concurrent requests
        texts = [
            "This is test number one.",
            "This is test number two.",
            "This is test number three.",
            "This is test number four.",
        ]

        start_time = time.time()
        tasks = [model.generate_stream(t, print_metrics=False) for t in texts]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Collect first chunk latencies
        first_chunk_latencies = [
            list(r)[0][1].latency_to_first_chunk * 1000
            for r in results
        ]
        avg_first_chunk = sum(first_chunk_latencies) / len(first_chunk_latencies)

        print(f"\n📊 Concurrent Test Results (4 requests):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg first chunk: {avg_first_chunk:.0f}ms")
        print(f"  Stream pool: {model.s3gen_stream_pool.metrics}")

        # With stream pool, should be reasonable (< 2s)
        assert avg_first_chunk < 2000, f"First chunk too slow: {avg_first_chunk}ms"

    finally:
        await model.shutdown()
