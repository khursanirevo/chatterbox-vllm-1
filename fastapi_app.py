"""
FastAPI application with WebSocket TTS streaming endpoint.

Run with:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --ws-ping-interval 20
"""

import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware

from websocket_service import TTSWebsocketSession, lifespan_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create FastAPI app with lifespan
app = FastAPI(
    title="Chatterbox vLLM TTS Service",
    description="Streaming text-to-speech using Chatterbox with token-level streaming",
    version="1.0.0",
    lifespan=lifespan_manager,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "chatterbox-vllm-tts"}


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Chatterbox vLLM TTS Service",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/tts/websocket",
            "health": "/health",
        },
        "features": [
            "Token-level streaming for optimal TTFA",
            "Continuous batching for high throughput",
            "WebSocket support for real-time streaming",
        ],
    }


@app.websocket("/tts/websocket")
async def tts_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming TTS.

    Client should send JSON messages with the following structure:
    {
        "transcript": "Text to synthesize",
        "context_id": "unique_session_id",
        "language": "en",
        "continue": true,
        "voice_id": "optional_voice_id",
        "seed": 12345  # Optional seed for reproducibility
    }

    Server responds with:
    {
        "type": "audio_chunk",
        "data": "base64_encoded_wav",
        "done": false,
        "context_id": "session_id",
        "step_time": 123.45
    }

    When complete:
    {
        "type": "done",
        "done": true,
        "context_id": "session_id"
    }
    """
    # Accept WebSocket connection
    await websocket.accept()

    # Get query parameters
    params = websocket.query_params
    api_key = params.get("api_key")
    version = params.get("version", "2025-12-09")

    logger.info(f"WebSocket connection established (version: {version})")

    # Create session handler
    session = TTSWebsocketSession()

    try:
        while True:
            # Receive JSON message from client
            data = await websocket.receive_json()

            logger.debug(f"Received: {data}")

            # Handle the event and stream responses
            response_count = 0
            async for response in session.handle_tts_event(raw_data=data):
                try:
                    await websocket.send_text(json.dumps(response))
                    response_count += 1

                    # Log important events
                    if response.get("type") == "audio_chunk":
                        logger.debug(
                            f"[{response['context_id']}] Sent chunk "
                            f"(step_time: {response.get('step_time', 0):.2f}ms)"
                        )
                    elif response.get("type") == "done":
                        logger.info(f"[{response['context_id']}] Generation complete")
                    elif response.get("type") == "error":
                        logger.error(
                            f"[{response['context_id']}] Error: "
                            f"{response.get('error', 'Unknown')}"
                        )

                except Exception as e:
                    logger.error(f"Error sending response: {e}", exc_info=True)
                    break

            logger.debug(f"[{data.get('context_id', 'unknown')}] Sent {response_count} responses")

    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)

        # Try to send error message
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e),
                    "done": True,
                    "context_id": "unknown",
                }))
        except Exception:
            pass  # Already closed

    finally:
        # Gracefully close WebSocket if still connected
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except (RuntimeError, AttributeError):
            # Already closed or closing - ignore
            pass

        logger.info("WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn

    # Run the server
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        log_level="info",
    )
