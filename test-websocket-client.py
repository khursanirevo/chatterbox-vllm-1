#!/usr/bin/env python3
"""
Test client for the Chatterbox vLLM WebSocket TTS API.

This script demonstrates how to connect to the WebSocket endpoint and
receive streaming audio chunks.

Usage:
    # First, start the server:
    uv run python -m src.chatterbox_vllm.websocket_api

    # Then run this client:
    uv run python test-websocket-client.py
"""

import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

import websockets
import numpy as np
import scipy.io.wavfile as wavfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def tts_websocket_client(
    text: str,
    uri: str = "ws://localhost:8000/tts/websocket",
    api_key: Optional[str] = None,
    save_audio: bool = True,
    output_path: str = "websocket-output.wav",
):
    """
    Connect to WebSocket TTS API and receive streaming audio.

    Args:
        text: Text to synthesize
        uri: WebSocket URI
        api_key: Optional API key
        save_audio: Whether to save the audio to file
        output_path: Path to save audio file
    """
    # Collect audio chunks
    audio_chunks = []
    metrics = {}
    start_time = time.time()
    first_chunk_time = None

    # Build URI with API key if provided
    if api_key:
        uri = f"{uri}?api_key={api_key}"

    logger.info(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            # Send request
            request = {
                "text": text,
                "request_id": f"test-{int(time.time())}",
                "temperature": 0.8,
                "max_tokens": 500,
                "chunk_size": 25,
                "context_window": 50,
            }

            logger.info(f"Sending request: '{text}'")
            await websocket.send(json.dumps(request))

            # Receive responses
            async for message in websocket:
                response = json.loads(message)
                msg_type = response.get("type")
                request_id = response.get("request_id")

                logger.info(f"Received {msg_type} for request {request_id}")

                if msg_type == "acknowledgment":
                    logger.info(f"  Server acknowledged: {response.get('data', {}).get('message')}")

                elif msg_type == "progress":
                    data = response.get("data", {})
                    tokens = data.get("tokens_generated")
                    max_tokens = data.get("max_tokens")
                    logger.info(f"  Progress: {tokens}/{max_tokens} tokens generated")

                elif msg_type == "audio_chunk":
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        latency = first_chunk_time - start_time
                        logger.info(f"  ⚡ First audio chunk received in {latency*1000:.1f}ms")

                    data = response.get("data", {})
                    chunk_index = data.get("chunk_index")
                    audio_base64 = data.get("audio")
                    sample_rate = data.get("sample_rate", 24000)

                    # Decode audio
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32767.0

                    audio_chunks.append((sample_rate, audio_float32))
                    logger.info(f"  📻 Received chunk #{chunk_index}: {len(audio_float32)} samples @ {sample_rate}Hz")

                elif msg_type == "complete":
                    metrics = response.get("data", {})
                    total_time = time.time() - start_time
                    logger.info(f"  ✅ Complete: {metrics}")
                    logger.info(f"  Total time: {total_time:.2f}s")
                    break

                elif msg_type == "error":
                    error = response.get("error")
                    logger.error(f"  ❌ Error: {error}")
                    break

    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket error: {e}")
        return
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return

    # Save audio if requested
    if save_audio and audio_chunks:
        # Concatenate all chunks
        sample_rate = audio_chunks[0][0]
        all_audio = np.concatenate([chunk[1] for chunk in audio_chunks])

        # Convert to 16-bit PCM
        audio_int16 = (all_audio * 32767).astype(np.int16)

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, sample_rate, audio_int16)

        duration = len(all_audio) / sample_rate
        logger.info(f"  💾 Saved audio to {output_path}")
        logger.info(f"  Duration: {duration:.2f}s, Samples: {len(all_audio)}")

    return {
        "chunks": len(audio_chunks),
        "metrics": metrics,
        "first_chunk_latency_ms": (first_chunk_time - start_time) * 1000 if first_chunk_time else None,
    }


async def main():
    """Run WebSocket TTS client test."""
    # Test texts
    texts = [
        "Hello world, this is a test of the WebSocket streaming TTS API.",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer text to test the streaming capabilities of the text to speech system.",
    ]

    print("="*70)
    print("Chatterbox vLLM WebSocket TTS Client Test")
    print("="*70)

    for i, text in enumerate(texts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: '{text[:50]}...'")
        print(f"{'='*70}\n")

        result = await tts_websocket_client(
            text=text,
            save_audio=True,
            output_path=f"websocket-test-{i}.wav",
        )

        if result:
            print(f"\n✅ Test {i} complete:")
            print(f"  Chunks received: {result['chunks']}")
            print(f"  First chunk latency: {result['first_chunk_latency_ms']:.1f}ms" if result['first_chunk_latency_ms'] else "  First chunk latency: N/A")
            if result['metrics']:
                for key, value in result['metrics'].items():
                    print(f"  {key}: {value}")

        # Small delay between tests
        await asyncio.sleep(1)

    print(f"\n{'='*70}")
    print("All tests complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
