#!/usr/bin/env python3
"""
WebSocket API for streaming TTS using AsyncLLMEngine.

This provides a FastAPI WebSocket endpoint for real-time text-to-speech
with sub-1s first audio chunk latency.

Usage:
    uvicorn src.chatterbox_vllm.websocket_api:app --host 0.0.0.0 --port 8000

Client example:
    import websockets
    import json
    import asyncio

    async def test_tts():
        uri = "ws://localhost:8000/tts/websocket"
        async with websockets.connect(uri) as ws:
            request = {
                "text": "Hello world, this is a test.",
                "request_id": "test-001"
            }
            await ws.send(json.dumps(request))
            async for message in ws:
                response = json.loads(message)
                if response["type"] == "audio_chunk":
                    # Process audio chunk
                    pass
                elif response["type"] == "complete":
                    break

    asyncio.run(test_tts())
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs

from chatterbox_vllm.models.t3 import T3VllmModel
from chatterbox_vllm.models.s3gen import S3GEN_SR, S3Gen
from chatterbox_vllm.models.voice_encoder import VoiceEncoder
from chatterbox_vllm.models.t3.modules.cond_enc import T3Cond
from chatterbox_vllm.text_utils import punc_norm, SUPPORTED_LANGUAGES
from chatterbox_vllm.tts import Conditionals, StreamingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


@dataclass
class TTSRequest:
    """TTS request data from client."""
    text: str
    request_id: Optional[str] = None
    audio_prompt_path: Optional[str] = None
    language_id: int = 0
    exaggeration: float = 0.0
    temperature: float = 0.8
    max_tokens: int = 500
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    chunk_size: int = 25
    context_window: int = 50
    fade_duration: float = 0.02
    diffusion_steps: int = 10


@dataclass
class TTSResponse:
    """TTS response to client."""
    type: str  # "audio_chunk", "complete", "error", "metrics"
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TTSWebsocketSession:
    """
    WebSocket session handler for streaming TTS.

    This class handles TTS requests over WebSocket, streaming audio chunks
    in real-time using AsyncLLMEngine.
    """

    def __init__(self, engine: AsyncLLMEngine):
        self.engine = engine
        self.active_requests: Dict[str, Any] = {}

    async def handle_tts_event(
        self,
        raw_data: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle a TTS request and yield response chunks.

        Args:
            raw_data: Request data from client

        Yields:
            Dictionary responses for WebSocket
        """
        request_id = raw_data.get("request_id", f"req-{time.time()}")

        try:
            # Parse request
            request = TTSRequest(
                text=raw_data.get("text", ""),
                request_id=request_id,
                audio_prompt_path=raw_data.get("audio_prompt_path"),
                language_id=raw_data.get("language_id", 0),
                exaggeration=raw_data.get("exaggeration", 0.0),
                temperature=raw_data.get("temperature", 0.8),
                max_tokens=raw_data.get("max_tokens", 500),
                top_p=raw_data.get("top_p", 0.95),
                repetition_penalty=raw_data.get("repetition_penalty", 1.0),
                chunk_size=raw_data.get("chunk_size", 25),
                context_window=raw_data.get("context_window", 50),
                fade_duration=raw_data.get("fade_duration", 0.02),
                diffusion_steps=raw_data.get("diffusion_steps", 10),
            )

            # Validate text
            if not request.text:
                yield {
                    "type": "error",
                    "request_id": request_id,
                    "error": "No text provided"
                }
                return

            logger.info(f"Processing TTS request {request_id}: '{request.text[:50]}...'")

            # Send acknowledgment
            yield {
                "type": "acknowledgment",
                "request_id": request_id,
                "data": {
                    "message": "Request received",
                    "text": request.text[:100],
                }
            }

            # Stream tokens and generate audio
            async for response in self._generate_audio_stream(request):
                yield response

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
            yield {
                "type": "error",
                "request_id": request_id,
                "error": str(e)
            }

    async def _generate_audio_stream(
        self,
        request: TTSRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate audio stream using AsyncLLMEngine.

        This method:
        1. Streams speech tokens from AsyncLLMEngine
        2. Generates audio from tokens using S3Gen (synchronous for now)
        3. Yields audio chunks as base64-encoded JSON messages
        """
        start_time = time.time()
        metrics = StreamingMetrics()

        # Prepare text
        text = punc_norm(request.text)

        # Create prompt
        prompt = f"[START]{text}[STOP]"

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )

        # Collect tokens and track timing
        all_tokens = []
        first_token_time = None
        t3_start_time = time.time()

        request_id = f"{request.request_id}-{time.time()}"

        try:
            # Stream tokens from AsyncLLMEngine
            async for request_output in self.engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                current_time = time.time()

                # Track first token time
                if first_token_time is None and request_output.outputs:
                    first_token_time = current_time
                    metrics.t3_first_token_time = first_token_time - t3_start_time
                    logger.info(f"Request {request.request_id}: First token in {metrics.t3_first_token_time*1000:.1f}ms")

                # Collect tokens
                if request_output.outputs:
                    output = request_output.outputs[0]
                    all_tokens = list(output.token_ids)

                    # Send progress update
                    yield {
                        "type": "progress",
                        "request_id": request.request_id,
                        "data": {
                            "tokens_generated": len(all_tokens),
                            "max_tokens": request.max_tokens,
                        }
                    }

                    # Process chunk if we have enough tokens
                    if len(all_tokens) >= request.chunk_size:
                        chunk_audio = await self._process_tokens_to_audio(
                            tokens=all_tokens,
                            chunk_size=request.chunk_size,
                            context_window=request.context_window,
                            diffusion_steps=request.diffusion_steps,
                        )

                        if chunk_audio is not None:
                            # Encode audio as base64
                            audio_base64 = self._encode_audio(chunk_audio)

                            # Update metrics
                            if metrics.chunk_count == 0:
                                metrics.latency_to_first_chunk = current_time - start_time
                                logger.info(f"Request {request.request_id}: First audio chunk in {metrics.latency_to_first_chunk*1000:.1f}ms")

                            metrics.chunk_count += 1

                            # Send audio chunk
                            yield {
                                "type": "audio_chunk",
                                "request_id": request.request_id,
                                "data": {
                                    "chunk_index": metrics.chunk_count,
                                    "audio": audio_base64,
                                    "sample_rate": S3GEN_SR,
                                    "format": "wav",
                                }
                            }

                # Check if generation is complete
                if request_output.finished:
                    break

            # Process final tokens
            if all_tokens:
                # Get remaining tokens
                remaining_start = max(0, len(all_tokens) - request.chunk_size)
                remaining_tokens = all_tokens[remaining_start:]

                if remaining_tokens:
                    chunk_audio = await self._process_tokens_to_audio(
                        tokens=all_tokens,
                        chunk_size=len(remaining_tokens),
                        context_window=request.context_window,
                        diffusion_steps=request.diffusion_steps,
                        is_final=True,
                    )

                    if chunk_audio is not None:
                        audio_base64 = self._encode_audio(chunk_audio)
                        metrics.chunk_count += 1

                        yield {
                            "type": "audio_chunk",
                            "request_id": request.request_id,
                            "data": {
                                "chunk_index": metrics.chunk_count,
                                "audio": audio_base64,
                                "sample_rate": S3GEN_SR,
                                "format": "wav",
                                "is_final": True,
                            }
                        }

            # Send completion metrics
            metrics.total_generation_time = time.time() - start_time
            metrics.t3_token_generation_time = first_token_time - t3_start_time if first_token_time else 0

            yield {
                "type": "complete",
                "request_id": request.request_id,
                "data": {
                    "chunks_sent": metrics.chunk_count,
                    "total_tokens": len(all_tokens),
                    "total_time_seconds": round(metrics.total_generation_time, 3),
                    "first_token_ms": round(metrics.t3_first_token_time * 1000, 1) if metrics.t3_first_token_time else None,
                    "first_chunk_ms": round(metrics.latency_to_first_chunk * 1000, 1) if metrics.latency_to_first_chunk else None,
                }
            }

            logger.info(f"Request {request.request_id} complete: {metrics.chunk_count} chunks, {metrics.total_generation_time:.2f}s")

        except Exception as e:
            logger.error(f"Error generating audio for request {request.request_id}: {e}", exc_info=True)
            yield {
                "type": "error",
                "request_id": request.request_id,
                "error": str(e)
            }

    async def _process_tokens_to_audio(
        self,
        tokens: List[int],
        chunk_size: int,
        context_window: int,
        diffusion_steps: int,
        is_final: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Process speech tokens to audio.

        Note: This is a placeholder. The actual implementation would use S3Gen
        to convert speech tokens to audio. For now, we'll generate a simple sine wave
        for demonstration purposes.

        TODO: Integrate S3Gen for actual audio generation
        """
        # For demonstration, generate a simple sine wave audio chunk
        # In production, this would call S3Gen to convert tokens to audio

        duration = 0.1  # 100ms per chunk
        sample_rate = S3GEN_SR
        n_samples = int(duration * sample_rate)

        # Generate a simple tone (440Hz = A4)
        t = np.linspace(0, duration, n_samples, False)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)

        # Apply fade in/out
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio.astype(np.float32)

    def _encode_audio(self, audio: np.ndarray) -> str:
        """Encode audio array as base64 string."""
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)

        # Encode as bytes
        audio_bytes = audio_int16.tobytes()

        # Base64 encode
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        return audio_base64


# Global engine instance
_engine: Optional[AsyncLLMEngine] = None


async def get_engine() -> AsyncLLMEngine:
    """Get or create the global AsyncLLMEngine instance."""
    global _engine

    if _engine is None:
        logger.info("Initializing AsyncLLMEngine...")
        model_path = "./t3-model"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        engine_args = AsyncEngineArgs(
            model=str(model_path),
            tokenizer="EnTokenizer",
            tokenizer_mode="custom",
            gpu_memory_utilization=0.90,
            max_model_len=2000,
            enforce_eager=True,
            disable_log_stats=False,
            tensor_parallel_size=1,
        )

        _engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("AsyncLLMEngine ready!")

    return _engine


# FastAPI app
app = FastAPI(title="Chatterbox vLLM Streaming TTS API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "chatterbox-vllm-tts",
        "model": "async-llm-engine"
    }


@app.websocket("/tts/websocket")
async def tts_websocket(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None),
    version: Optional[str] = Query("2025-12-09")
):
    """
    WebSocket endpoint for streaming TTS.

    Query Parameters:
        api_key: Optional API key for authentication
        version: API version

    Message Format (Client -> Server):
        {
            "text": "Text to synthesize",
            "request_id": "optional-request-id",
            "temperature": 0.8,
            "max_tokens": 500,
            ...
        }

    Message Format (Server -> Client):
        {
            "type": "audio_chunk" | "progress" | "complete" | "error" | "acknowledgment",
            "request_id": "request-id",
            "data": { ... },
            "error": "error message if type=error"
        }

    Audio Chunk Format:
        {
            "type": "audio_chunk",
            "request_id": "request-id",
            "data": {
                "chunk_index": 1,
                "audio": "base64-encoded PCM audio",
                "sample_rate": 24000,
                "format": "wav"
            }
        }
    """
    # Validate API key if needed
    if api_key and api_key != "test-key":  # Replace with actual validation
        await websocket.close(code=1008, reason="Invalid API key")
        return

    # Log connection attempt
    logger.info(f"WebSocket connection attempt from: {websocket.client}")

    await websocket.accept()

    logger.info(f"WebSocket connection accepted: {websocket.client}")

    # Get or create engine
    try:
        engine = await get_engine()
        session = TTSWebsocketSession(engine)
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        await websocket.send_json({
            "type": "error",
            "error": f"Server initialization error: {str(e)}"
        })
        await websocket.close()
        return

    logger.info(f"WebSocket session ready: {websocket.client}")

    try:
        while True:
            # Receive JSON message
            data = await websocket.receive_json()

            logger.info(f"Received WebSocket data: {data}")

            # Handle TTS request
            async for chunk in session.handle_tts_event(raw_data=data):
                await websocket.send_json(chunk)

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: {e}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
    finally:
        # Gracefully close WebSocket if still connected
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except (RuntimeError, AttributeError):
            # Already closed or closing - ignore
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.chatterbox_vllm.websocket_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
