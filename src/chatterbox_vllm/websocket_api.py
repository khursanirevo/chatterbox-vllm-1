#!/usr/bin/env python3
"""
WebSocket Streaming API for Chatterbox TTS

Provides real-time audio streaming via WebSocket for text-to-speech.
Client sends plain text, server streams binary PCM audio chunks + statistics.
"""

import os
import asyncio
import time
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatterbox_vllm.tts_async import AsyncChatterboxTTS
from chatterbox_vllm.models.s3gen import S3GEN_SR

# Set GPU before importing
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global model instance
async_model: Optional[AsyncChatterboxTTS] = None
warmed_up = False
model_lock = asyncio.Lock()

# Hardcoded generation parameters (matching simple_stream.py)
DEFAULT_PARAMS = {
    "temperature": 0.8,
    "max_tokens": 500,
    "chunk_size": 25,
    "context_window": 50,
    "diffusion_steps": 5,  # Faster generation
}


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
    global async_model, warmed_up

    async with model_lock:
        if async_model is None:
            print("="*70)
            print("Initializing AsyncChatterboxTTS model")
            print("="*70)
            async_model = await AsyncChatterboxTTS.from_pretrained(
                max_model_len=2000,
                gpu_memory_utilization=0.90,
                enforce_eager=True,
            )
            # Warmup
            print("Warming up model...")
            async for _ in async_model.generate_stream("Warmup.", print_metrics=False):
                pass
            warmed_up = True
            print("✓ AsyncChatterboxTTS ready!")
            print("="*70)

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
        text = await websocket.receive_text()

        if not text or not text.strip():
            await websocket.send_json({
                "type": "error",
                "error": "Empty text received"
            })
            await websocket.close()
            return

        print(f"Received text: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Get model
        model = await get_model()

        # Generate and stream audio
        start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_samples = 0

        async for audio_chunk, metrics in model.generate_stream(
            text=text,
            **DEFAULT_PARAMS
        ):
            chunk_count += 1

            if first_chunk_time is None:
                first_chunk_time = time.time()
                first_chunk_ms = (first_chunk_time - start_time) * 1000
                print(f"⚡ First audio chunk: {first_chunk_ms:.1f}ms")

            # Convert audio to binary PCM (float32)
            audio_np = audio_chunk.cpu().numpy().astype(np.float32)
            audio_bytes = audio_np.tobytes()

            # Send binary audio
            await websocket.send_bytes(audio_bytes)

            total_samples += audio_chunk.shape[-1]
            print(f"Sent chunk {chunk_count}: {audio_chunk.shape[-1] / S3GEN_SR:.3f}s")

        # Calculate and send statistics
        total_time = time.time() - start_time
        duration = total_samples / S3GEN_SR
        rtf = total_time / duration if duration > 0 else 0

        stats = {
            "type": "complete",
            "first_chunk_ms": round((first_chunk_time - start_time) * 1000, 2) if first_chunk_time else 0,
            "duration_s": round(duration, 2),
            "rtf": round(rtf, 3),
            "chunks": chunk_count,
            "total_time_s": round(total_time, 2),
        }

        print(f"Complete: {stats}")
        await websocket.send_json(stats)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        await websocket.close()


async def main():
    print("="*70)
    print("Chatterbox WebSocket Streaming TTS Server")
    print("Using AsyncChatterboxTTS with AsyncLLMEngine")
    print("="*70)
    print("\nInitializing model...")
    await get_model()
    print("\nServer starting on http://0.0.0.0:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/tts")
    print("\nPress Ctrl+C to stop\n")

    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
