"""
Chatterbox vLLM WebSocket TTS Service

Provides WebSocket endpoint for streaming text-to-speech using token-level streaming
for optimal Time To First Audio (TTFA).
"""

import asyncio
import base64
import json
import logging
import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from chatterbox_vllm import ChatterboxTTSStreaming


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class GenerationRequest(BaseModel):
    """TTS generation request from client."""
    transcript: str
    context_id: str
    voice_id: Optional[str] = None
    language: str = "en"
    continue_: bool = True
    header_mode: Optional[str] = "none"  # "none", "first", or "all"
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationRequest":
        """Create from dictionary (for WebSocket JSON handling)."""
        return cls(
            transcript=data.get("transcript", ""),
            context_id=data.get("context_id", ""),
            voice_id=data.get("voice_id"),
            language=data.get("language", "en"),
            continue_=data.get("continue", True),
            header_mode=data.get("header_mode", "none"),
            seed=data.get("seed"),
        )


class CancelRequest(BaseModel):
    """Cancel request for ongoing generation."""
    context_id: str

    @classmethod
    def from_dict(cls, data: dict) -> "CancelRequest":
        return cls(context_id=data.get("context_id", ""))


# Global model instance
_tts_model: Optional[ChatterboxTTSStreaming] = None


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """Manage TTS model lifecycle."""
    global _tts_model

    logger.info("Initializing ChatterboxTTSStreaming model...")
    _tts_model = await ChatterboxTTSStreaming.from_pretrained(
        max_batch_size=16,
        max_model_len=1000,
    )
    logger.info("Model initialization complete")

    yield

    logger.info("Shutting down model...")
    await _tts_model.shutdown()
    _tts_model = None
    logger.info("Model shutdown complete")


def get_model() -> ChatterboxTTSStreaming:
    """Get the global TTS model instance."""
    if _tts_model is None:
        raise RuntimeError("Model not initialized")
    return _tts_model


class TTSWebsocketSession:
    """
    WebSocket session handler for TTS streaming.

    Handles incoming TTS requests and streams audio chunks back to the client
    using token-level streaming for optimal TTFA.
    """

    def __init__(self):
        self.active_contexts: Dict[str, bool] = {}

    async def handle_tts_event(
        self,
        raw_data: dict,
    ) -> AsyncGenerator[dict, None]:
        """
        Handle incoming TTS WebSocket event.

        Args:
            raw_data: Raw WebSocket message data

        Yields:
            Response dictionaries to send back to client
        """
        if "transcript" in raw_data:
            data = GenerationRequest.from_dict(raw_data)
            async for response in self._handle_generation_request(data):
                yield response

        elif "cancel" in raw_data:
            data = CancelRequest.from_dict(raw_data)
            async for response in self._handle_cancel_request(data):
                yield response

        else:
            logger.warning(f"Unknown message type: {raw_data.keys()}")
            yield self._error_response(
                error=f"Unknown message type: {list(raw_data.keys())}",
                context_id=raw_data.get("context_id", ""),
            )

    async def _handle_generation_request(
        self,
        data: GenerationRequest,
    ) -> AsyncGenerator[dict, None]:
        """Handle TTS generation request."""
        transcript = data.transcript.strip()

        logger.info(f"[{data.context_id}] TTS request: '{transcript[:50]}...'")

        if not data.continue_ or not transcript:
            # End of stream or empty transcript
            yield self._done_response(context_id=data.context_id)
            return

        # Mark context as active
        self.active_contexts[data.context_id] = True

        # Use provided seed or generate random one
        seed = data.seed or random.randint(1, 1000000)

        start_time = time.perf_counter_ns()
        total_chunks = 0
        first_chunk_time = None

        try:
            model = get_model()

            # Stream audio using token-level streaming
            async for chunk in model.stream_audio_tokens(
                prompt=transcript,
                language_id=data.language,
                temperature=0.8,
                exaggeration=0.5,
                max_tokens=1000,
            ):
                # Check if context was cancelled
                if not self.active_contexts.get(data.context_id, False):
                    logger.info(f"[{data.context_id}] Request cancelled")
                    break

                # Record first chunk time (TTFA)
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter_ns()
                    ttfa_ms = (first_chunk_time - start_time) / 1_000_000
                    logger.info(f"[{data.context_id}] First audio chunk in {ttfa_ms:.2f}ms")

                # Convert tensor to bytes
                chunk_bytes = self._tensor_to_wav_bytes(chunk, model.sr)

                total_chunks += 1

                # Send audio chunk
                yield self._audio_chunk_response(
                    audio_chunk=chunk_bytes,
                    context_id=data.context_id,
                    step_time=(time.perf_counter_ns() - start_time) / 1_000_000,
                )

                # Reset start time for next chunk
                start_time = time.perf_counter_ns()

            # Log completion stats
            if first_chunk_time:
                total_time = (time.perf_counter_ns() - first_chunk_time) / 1_000_000
                logger.info(
                    f"[{data.context_id}] Complete: {total_chunks} chunks, "
                    f"{total_time:.2f}ms total"
                )

        except Exception as e:
            logger.error(f"[{data.context_id}] Generation error: {e}", exc_info=True)
            yield self._error_response(
                error=str(e),
                context_id=data.context_id,
            )

        finally:
            # Mark context as inactive
            self.active_contexts.pop(data.context_id, None)

        # Send done response
        yield self._done_response(context_id=data.context_id)

    async def _handle_cancel_request(
        self,
        data: CancelRequest,
    ) -> AsyncGenerator[dict, None]:
        """Handle cancel request."""
        logger.info(f"[{data.context_id}] Cancel request received")

        # Mark context as inactive
        self.active_contexts[data.context_id] = False

        # Send done response
        yield self._done_response(context_id=data.context_id)

    def _audio_chunk_response(
        self,
        audio_chunk: bytes,
        context_id: str,
        step_time: float = 0,
    ) -> dict:
        """Generate audio chunk response."""
        return {
            "type": "audio_chunk",
            "data": base64.b64encode(audio_chunk).decode("utf-8"),
            "done": False,
            "status_code": 206,
            "step_time": round(step_time, 5),
            "context_id": context_id,
        }

    def _done_response(self, context_id: str) -> dict:
        """Generate completion response."""
        return {
            "type": "done",
            "done": True,
            "status_code": 200,
            "context_id": context_id,
        }

    def _error_response(self, error: str, context_id: str) -> dict:
        """Generate error response."""
        return {
            "type": "error",
            "error": error,
            "done": True,
            "status_code": 500,
            "context_id": context_id,
        }

    def _tensor_to_wav_bytes(
        self,
        tensor: "torch.Tensor",
        sample_rate: int,
    ) -> bytes:
        """Convert audio tensor to WAV bytes."""
        import io
        import torch
        import torchaudio as ta

        # Ensure tensor is on CPU and has correct shape
        audio = tensor.cpu()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Convert to bytes
        buffer = io.BytesIO()
        ta.save(buffer, audio, sample_rate, format="wav")
        buffer.seek(0)
        return buffer.read()
