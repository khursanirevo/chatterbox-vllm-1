#!/usr/bin/env python3
"""
Quick verification that S3Gen stream pool is working correctly.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python verify_stream_pool.py
"""

import asyncio
import sys
from chatterbox_vllm.tts_async import AsyncChatterboxTTS

async def main():
    print("🔍 Verifying S3Gen Stream Pool Implementation\n")

    # Create model with stream pool
    print("📦 Loading model with stream pool...")
    model = await AsyncChatterboxTTS.from_pretrained(
        model_path="./t3-model",
        enable_stream_pool=True,
        num_s3gen_streams=4,
        gpu_memory_utilization=0.3,
    )

    try:
        # Check stream pool exists
        assert model.s3gen_stream_pool is not None, "Stream pool not initialized"
        print(f"✅ Stream pool created: {model.s3gen_stream_pool.num_streams} streams")

        # Test single request
        print("\n📝 Test 1: Single request")
        chunks = []
        chunk_count = 0
        async for chunk, metrics in model.generate_stream("Hello world", print_metrics=True):
            chunks.append(chunk)
            chunk_count += 1
            if chunk_count >= 3:
                break
        print(f"   Received {len(chunks)} chunks ✅")

        # Test concurrent requests
        print("\n📝 Test 2: 3 concurrent requests")
        texts = [
            "This is the first test.",
            "This is the second test.",
            "This is the third test.",
        ]

        start = asyncio.get_event_loop().time()
        tasks = [model.generate_stream(t, print_metrics=False) for t in texts]
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start

        print(f"   Completed 3 requests in {elapsed:.2f}s ✅")

        # Print metrics
        print(f"\n📊 Stream Pool Metrics:")
        print(f"   Total requests: {model.s3gen_stream_pool.metrics.total_requests}")
        print(f"   Active streams: {model.s3gen_stream_pool.metrics.active_streams}")
        print(f"   Avg queue wait: {model.s3gen_stream_pool.metrics.avg_queue_wait_ms:.2f}ms")
        print(f"   Queue depth: {model.s3gen_stream_pool.metrics.queue_depth}")

        print("\n✅ All verification tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await model.shutdown()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
