# Chatterbox WebSocket TTS API

Real-time text-to-speech streaming via WebSocket.

## Features

- **<1s first chunk latency** using AsyncLLMEngine
- **Binary PCM streaming** for efficient audio transfer
- **Real-time statistics** (latency, RTF, duration)
- **Simple protocol** - client sends text, receives audio

## Quick Start

### Start Server

```bash
CUDA_VISIBLE_DEVICES=0 uv run python src/chatterbox_vllm/websocket_api.py
```

Server starts on `http://0.0.0.0:8000`
WebSocket endpoint: `ws://localhost:8000/ws/tts`

### Test Client

```bash
uv run python test_websocket_client.py
```

## Protocol

### Request

Client sends plain text:
```
Hello world, this is a test.
```

### Response

Server sends binary PCM audio chunks:
- Format: Float32 samples
- Sample rate: 24000 Hz
- Channels: Mono (1)

Server sends JSON statistics at end:
```json
{
  "type": "complete",
  "first_chunk_ms": 767,
  "duration_s": 4.5,
  "rtf": 0.65,
  "chunks": 20,
  "total_time_s": 2.9
}
```

## Example Clients

### Python

```python
import asyncio
import websockets
import json
import numpy as np
import torch
import torchaudio

async def text_to_speech(text):
    uri = "ws://localhost:8000/ws/tts"
    audio_chunks = []

    async with websockets.connect(uri) as ws:
        # Send text
        await ws.send(text)

        # Receive audio
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                audio = np.frombuffer(message, dtype=np.float32)
                audio_chunks.append(torch.from_numpy(audio))
            else:
                data = json.loads(message)
                if data.get("type") == "complete":
                    print(f"RTF: {data['rtf']}")
                    break

    # Combine and save
    full_audio = torch.cat(audio_chunks).unsqueeze(0)
    torchaudio.save("output.wav", full_audio, 24000)

asyncio.run(text_to_speech("Hello world"))
```

### JavaScript (Browser)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tts');
const audioChunks = [];

ws.onopen = () => {
  ws.send('Hello world from JavaScript');
};

ws.onmessage = async (event) => {
  if (event.data instanceof Blob) {
    // Binary audio chunk
    const arrayBuffer = await event.data.arrayBuffer();
    const audioData = new Float32Array(arrayBuffer);
    audioChunks.push(audioData);
  } else {
    // JSON stats
    const data = JSON.parse(event.data);
    console.log('Complete:', data);
    ws.close();
  }
};

// Play audio using Web Audio API (when ready)
function playAudio() {
  const audioContext = new AudioContext({sampleRate: 24000});
  const buffer = audioContext.createBuffer(1, audioChunks.length, 24000);
  buffer.getChannelData(0).set(audioChunks.flat());

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(audioContext.destination);
  source.start();
}
```

## API Endpoints

### WebSocket: `/ws/tts`

Streaming TTS endpoint.

**Input:** Plain text string
**Output:** Binary PCM chunks + JSON stats

### GET: `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "chatterbox-websocket-tts"
}
```

## Configuration

Server uses hardcoded generation parameters (matching `simple_stream.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `temperature` | 0.8 | Sampling temperature |
| `max_tokens` | 500 | Maximum tokens to generate |
| `chunk_size` | 25 | Tokens per audio chunk |
| `context_window` | 50 | Context tokens for continuity |
| `diffusion_steps` | 5 | S3Gen diffusion steps (faster) |

## Performance

- **First chunk latency**: ~767ms (with AsyncLLMEngine)
- **RTF**: 0.55-0.65 (generates faster than real-time)
- **Sample rate**: 24000 Hz
- **Audio format**: PCM Float32
