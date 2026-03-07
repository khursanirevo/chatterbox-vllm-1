#!/usr/bin/env python3
"""
Test client for WebSocket TTS API.
"""

import asyncio
import websockets
import json
import numpy as np
import torch
import torchaudio as ta
from pathlib import Path


async def test_websocket_tts(text: str, output_dir: str = "output/websocket_test"):
    """Test the WebSocket TTS endpoint."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save input text
    (output_path / "input.txt").write_text(text)

    uri = "ws://localhost:8000/ws/tts"

    print(f"Connecting to {uri}...")
    print(f"Text: {text}\n")

    audio_chunks = []
    start_time = asyncio.get_event_loop().time()
    first_chunk_time = None

    try:
        async with websockets.connect(uri) as websocket:
            # Send text
            await websocket.send(text)

            # Receive audio chunks and stats
            while True:
                try:
                    # Receive message (binary or text)
                    message = await websocket.recv()

                    if isinstance(message, bytes):
                        # Binary audio chunk
                        if first_chunk_time is None:
                            first_chunk_time = asyncio.get_event_loop().time()
                            latency_ms = (first_chunk_time - start_time) * 1000
                            print(f"⚡ First audio chunk: {latency_ms:.1f}ms")

                        # Convert bytes to numpy array
                        audio_np = np.frombuffer(message, dtype=np.float32)
                        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                        audio_chunks.append(audio_tensor)

                        chunk_duration = len(audio_np) / 24000
                        print(f"Received chunk {len(audio_chunks)}: {chunk_duration:.3f}s")

                    elif isinstance(message, str):
                        # JSON message (stats or error)
                        data = json.loads(message)

                        if data.get("type") == "complete":
                            print(f"\n✓ Complete: {data}")
                            break
                        elif data.get("type") == "error":
                            print(f"\n✗ Error: {data}")
                            break

                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break

    except Exception as e:
        print(f"Error: {e}")
        return

    # Save combined audio
    if audio_chunks:
        full_audio = torch.cat(audio_chunks, dim=-1)
        output_file = output_path / "full_audio.wav"
        ta.save(str(output_file), full_audio, 24000)

        print(f"\n{'='*50}")
        print(f"Saved to: {output_file}")
        print(f"Chunks: {len(audio_chunks)}")
        print(f"Duration: {full_audio.shape[-1] / 24000:.2f}s")
        print(f"{'='*50}")


if __name__ == "__main__":
    text = "Hello world, this is a test of the WebSocket streaming TTS system."
    asyncio.run(test_websocket_tts(text))
