"""
Simple WebSocket client to test the Chatterbox vLLM TTS service.

Usage:
    python test_websocket_client.py
"""

import asyncio
import base64
import json
import wave
import os

import websockets
from websockets.asyncio.client import connect


async def test_tts_websocket():
    """Test the WebSocket TTS service."""

    uri = "ws://localhost:8000/tts/websocket"
    context_id = "test-session-001"

    # Test transcript
    transcript = "Hello! This is a test of the Chatterbox text to speech service with token-level streaming for better performance."

    print(f"Connecting to {uri}...")

    try:
        async with connect(uri) as websocket:
            print("Connected!")

            # Send TTS request
            request = {
                "transcript": transcript,
                "context_id": context_id,
                "language": "en",
                "continue": True,
                "seed": 42,  # Fixed seed for reproducibility
            }

            print(f"Sending request: {transcript[:50]}...")
            await websocket.send(json.dumps(request))

            # Receive responses
            chunk_count = 0
            audio_chunks = []
            start_time = None

            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "audio_chunk":
                    if start_time is None:
                        start_time = asyncio.get_event_loop().time()
                        print(f"🎵 First audio chunk received! TTFA: {data.get('step_time', 0):.2f}ms")

                    chunk_count += 1

                    # Decode base64 audio
                    audio_data = base64.b64decode(data["data"])
                    audio_chunks.append(audio_data)

                    print(f"  Received chunk {chunk_count} ({len(audio_data)} bytes)")

                elif data.get("type") == "done":
                    print(f"✅ Generation complete!")
                    break

                elif data.get("type") == "error":
                    print(f"❌ Error: {data.get('error')}")
                    break

            # Combine and save audio if we received chunks
            if audio_chunks:
                save_combined_audio(audio_chunks, f"test_output_{context_id}.wav")
                print(f"Saved {len(audio_chunks)} chunks to test_output_{context_id}.wav")

    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Is the server running?")
        print("Start the server with: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000")

    except Exception as e:
        print(f"❌ Error: {e}")


def save_combined_audio(chunks: list[bytes], filename: str):
    """Combine audio chunks and save as WAV file."""
    # Combine all chunks
    combined = b"".join(chunks)

    # Save to file
    with open(filename, "wb") as f:
        f.write(combined)

    print(f"Combined audio size: {len(combined)} bytes")


async def test_multiple_requests():
    """Test multiple concurrent requests."""
    uri = "ws://localhost:8000/tts/websocket"

    async def send_request(text: str, req_id: int):
        async with connect(uri) as websocket:
            context_id = f"test-{req_id:03d}"

            request = {
                "transcript": text,
                "context_id": context_id,
                "language": "en",
                "continue": True,
            }

            await websocket.send(json.dumps(request))

            chunk_count = 0
            while True:
                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "audio_chunk":
                    chunk_count += 1

                elif data.get("type") in ("done", "error"):
                    print(f"[{context_id}] Complete: {chunk_count} chunks")
                    break

    # Test texts
    texts = [
        "Short text.",
        "This is a medium length text that will take some time to process.",
        "This is a much longer text passage designed to test the continuous batching capabilities of the system. It should take significantly longer to process than the shorter texts, allowing us to see how well the system handles multiple concurrent requests with varying complexities.",
    ]

    print("Sending 3 concurrent requests...")

    await asyncio.gather(*[
        send_request(text, i) for i, text in enumerate(texts)
    ])

    print("All requests complete!")


if __name__ == "__main__":
    print("="*60)
    print("Chatterbox vLLM WebSocket TTS Test Client")
    print("="*60)
    print()

    # Run single request test
    print("Test 1: Single request")
    print("-"*60)
    asyncio.run(test_tts_websocket())

    print()
    print("="*60)
    print("Test 2: Multiple concurrent requests")
    print("-"*60)
    asyncio.run(test_multiple_requests())
