#!/usr/bin/env python3
"""
WebSocket Streaming API for Chatterbox TTS

Provides real-time audio streaming via WebSocket for text-to-speech.
Client sends plain text, server streams binary PCM audio chunks + statistics.
"""

import os
import asyncio
import time
import argparse
import logging
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatterbox_vllm.tts_async import AsyncChatterboxTTS
from chatterbox_vllm.models.s3gen import S3GEN_SR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set GPU before importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global model instance
async_model: Optional[AsyncChatterboxTTS] = None
warmed_up = False
model_lock = asyncio.Lock()

# Global args (set by parse_args)
args = None

# Hardcoded generation parameters (matching simple_stream.py)
DEFAULT_PARAMS = {
    "temperature": 0.8,
    "max_tokens": 500,
    "chunk_size": 25,
    "context_window": 50,
    "diffusion_steps": 5,  # Faster generation
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chatterbox WebSocket Streaming TTS Server with Stream Pool Support"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (default: uses model default)",
    )
    parser.add_argument(
        "--audio-prompt-path",
        type=str,
        default=None,
        help="Path to audio prompt for TTS (default: uses model default)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model variant (default: small)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2000,
        help="Maximum model length (default: 2000)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (default: 0.7)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=True,
        help="Disable CUDA graph and use eager execution (default: True)",
    )
    parser.add_argument(
        "--s3gen-use-fp16",
        action="store_true",
        default=True,
        help="Use FP16 for S3Gen (default: True)",
    )
    parser.add_argument(
        "--enable-stream-pool",
        action="store_true",
        default=True,
        help="Enable CUDA stream pool for concurrent S3Gen inference (default: True)",
    )
    parser.add_argument(
        "--disable-stream-pool",
        action="store_true",
        help="Disable stream pool (use sequential processing)",
    )
    parser.add_argument(
        "--num-s3gen-streams",
        type=int,
        default=12,
        help="Number of CUDA streams in S3Gen pool (default: 12)",
    )

    return parser.parse_args()


class HealthResponse(BaseModel):
    status: str
    service: str


# Initialize FastAPI app
app = FastAPI(title="Chatterbox WebSocket Streaming TTS")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_model() -> AsyncChatterboxTTS:
    """Get or initialize the global async model instance."""
    global async_model, warmed_up, args

    async with model_lock:
        if async_model is None:
            logger.info("="*70)
            logger.info("Initializing AsyncChatterboxTTS model")
            logger.info("="*70)
            async_model = await AsyncChatterboxTTS.from_pretrained(
                model_path=args.model_path,
                audio_prompt_path=args.audio_prompt_path,
                variant=args.variant,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                s3gen_use_fp16=args.s3gen_use_fp16,
                enable_stream_pool=not args.disable_stream_pool,
                num_s3gen_streams=args.num_s3gen_streams,
            )

            # Log stream pool status
            if async_model.s3gen_stream_pool:
                logger.info(f"S3Gen Stream Pool enabled: {async_model.s3gen_stream_pool.num_streams} streams")
            else:
                logger.info("S3Gen Stream Pool disabled (sequential processing)")

            # Warmup
            logger.info("Warming up model...")
            async for _ in async_model.generate_stream("Warmup.", print_metrics=False):
                pass
            warmed_up = True
            logger.info("✓ AsyncChatterboxTTS ready!")
            logger.info("="*70)

    return async_model


@app.get("/health")
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", service="chatterbox-websocket-tts")


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Protocol:
    - Client sends: plain text string
    - Server sends: binary PCM float32 audio chunks (24000 Hz, mono)
    - Server sends: JSON statistics at end
    """
    await websocket.accept()

    try:
        # Receive plain text from client
        recv_start = time.time()
        text = await websocket.receive_text()
        recv_time = (time.time() - recv_start) * 1000

        if not text or not text.strip():
            await websocket.send_json({
                "type": "error",
                "error": "Empty text received"
            })
            await websocket.close()
            return

        logger.info(f"Received text: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Track model setup timing
        model_setup_start = time.time()
        model = await get_model()
        model_setup_time = (time.time() - model_setup_start) * 1000

        # Generate and stream audio with granular timing
        request_start = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_samples = 0
        final_metrics = None

        # Track time until generate_stream starts
        gen_stream_start = time.time()

        async for audio_chunk, metrics in model.generate_stream(
            text=text,
            **DEFAULT_PARAMS
        ):
            # Update final_metrics with latest metrics
            final_metrics = metrics

            # Time until first chunk is yielded from generate_stream
            if first_chunk_time is None:
                gen_stream_first_chunk = (time.time() - gen_stream_start) * 1000
                first_chunk_time = time.time()
                first_chunk_ms = (first_chunk_time - request_start) * 1000

                # Time from start to T3 first token (from metrics)
                t3_first_token_ms = final_metrics.t3_first_token_time * 1000 if final_metrics else 0

                # GPU to CPU transfer timing
                gpu_start = time.time()
                audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                gpu_time = (time.time() - gpu_start) * 1000

                # Serialization timing
                ser_start = time.time()
                audio_bytes = audio_np.tobytes()
                ser_time = (time.time() - ser_start) * 1000

                # WebSocket send timing
                send_start = time.time()
                await websocket.send_bytes(audio_bytes)
                send_time = (time.time() - send_start) * 1000

                # Store first chunk timing for later printing (metrics are populated async)
                if first_chunk_time is None:
                    first_chunk_gpu_transfer = gpu_time
                    first_chunk_serialization = ser_time
                    first_chunk_send = send_time
                    first_chunk_overhead = other_overhead

                # Calculate queue/setup time (time before T3 starts)
                queue_time = max(0, gen_stream_first_chunk - (final_metrics.s3gen_first_chunk_time * 1000 if final_metrics else 0))

                # Calculate "other overhead" more precisely
                accounted = (
                    model_setup_time +
                    (final_metrics.t3_token_generation_time * 1000 if final_metrics else 0) +
                    (final_metrics.first_s3gen_inference_time * 1000 if final_metrics else 0) +
                    gpu_time +
                    ser_time +
                    send_time
                )
                other_overhead = max(0, first_chunk_ms - accounted)

                logger.info(f"⚡ First audio chunk: {first_chunk_ms:.1f}ms")
                logger.info(f"   ├─ Model setup (get_model): {model_setup_time:.1f}ms")
                logger.info(f"   ├─ Queue/AsyncLLMEngine setup: {queue_time:.1f}ms")
                logger.info(f"   ├─ S3Gen inference (from metrics): {final_metrics.first_s3gen_inference_time * 1000 if final_metrics else 0:.1f}ms")
                logger.info(f"   ├─ GPU → CPU transfer: {gpu_time:.1f}ms")
                logger.info(f"   ├─ Serialization: {ser_time:.1f}ms")
                logger.info(f"   ├─ WebSocket send: {send_time:.1f}ms")
                logger.info(f"   └─ Other overhead: {other_overhead:.1f}ms")
                logger.info(f"   (Detailed breakdown will be shown after completion)")

            total_samples += audio_chunk.shape[-1]
            chunk_count += 1  # Increment chunk counter
            logger.info(f"Sent chunk {chunk_count}: {audio_chunk.shape[-1] / S3GEN_SR:.3f}s")

        # After async loop completes, print final populated metrics
        if final_metrics and final_metrics.conditionals_prep_ms > 0:
            logger.info(f"\n📊 Final Granular Breakdown (after async completion):")
            logger.info(f"   Conditionals prep: {final_metrics.conditionals_prep_ms:.1f}ms")
            logger.info(f"   Text prep: {final_metrics.text_prep_ms:.1f}ms")
            logger.info(f"   Token conversion: {final_metrics.token_conversion_ms:.1f}ms")
            logger.info(f"   Context prep: {final_metrics.context_prep_ms:.1f}ms")
            logger.info(f"   Chunk prep overhead: {final_metrics.chunk_prep_overhead_ms:.1f}ms")
            logger.info(f"   S3Gen inference: {final_metrics.first_s3gen_inference_time * 1000:.1f}ms")
            logger.info(f"   T3 generation: {final_metrics.t3_token_generation_time * 1000:.1f}ms")
            logger.info(f"   T3 first token: {final_metrics.t3_first_token_time * 1000:.1f}ms")

        # Calculate and send statistics
        total_time = time.time() - request_start
        duration = total_samples / S3GEN_SR
        rtf = total_time / duration if duration > 0 else 0

        # Include granular profiling
        stats = {
            "type": "complete",
            "first_chunk_ms": round((first_chunk_time - request_start) * 1000, 2) if first_chunk_time else 0,
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "chunks": chunk_count,
            "total_time_s": round(total_time, 2),
            # T3 and S3Gen profiling
            "profiling": {
                "t3_token_generation_ms": round(final_metrics.t3_token_generation_time * 1000, 2) if final_metrics else 0,
                "t3_first_token_ms": round(final_metrics.t3_first_token_time * 1000, 2) if final_metrics else 0,
                "s3gen_first_chunk_ms": round(final_metrics.s3gen_first_chunk_time * 1000, 2) if final_metrics else 0,
                "s3gen_first_inference_ms": round(final_metrics.first_s3gen_inference_time * 1000, 2) if final_metrics else 0,
                # Granular internal timing
                "conditionals_prep_ms": round(final_metrics.conditionals_prep_ms, 2) if final_metrics else 0,
                "text_prep_ms": round(final_metrics.text_prep_ms, 2) if final_metrics else 0,
                "token_conversion_ms": round(final_metrics.token_conversion_ms, 2) if final_metrics else 0,
                "context_prep_ms": round(final_metrics.context_prep_ms, 2) if final_metrics else 0,
                "chunk_prep_overhead_ms": round(final_metrics.chunk_prep_overhead_ms, 2) if final_metrics else 0,
                # Per-component overhead
                "model_setup_ms": round(model_setup_time, 2),
                "queue_setup_ms": round(queue_time, 2),
                "websocket_recv_ms": round(recv_time, 2),
                "gpu_cpu_transfer_ms": round(first_chunk_gpu_transfer, 2),
                "serialization_ms": round(first_chunk_serialization, 2),
                "websocket_send_ms": round(first_chunk_send, 2),
                "other_overhead_ms": round(first_chunk_overhead, 2),
            } if final_metrics else None,
        }

        logger.info(f"Complete: {stats}")
        await websocket.send_json(stats)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        import traceback
        logger.error(f"Error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        await websocket.close()


async def main():
    global args
    args = parse_args()

    logger.info("="*70)
    logger.info("Chatterbox WebSocket Streaming TTS Server")
    logger.info("Using AsyncChatterboxTTS with AsyncLLMEngine")
    logger.info("="*70)
    logger.info("\nInitializing model...")
    await get_model()
    logger.info("\nServer starting on http://0.0.0.0:8000")
    logger.info("WebSocket endpoint: ws://localhost:8000/ws/tts")
    logger.info("\nPress Ctrl+C to stop\n")

    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
