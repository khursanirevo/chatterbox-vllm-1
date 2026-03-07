# Chatterbox vLLM WebSocket Streaming TTS API

A FastAPI-based WebSocket API for real-time text-to-speech streaming using vLLM's AsyncLLMEngine.

## Features

- ✅ **Real-time streaming**: <1s first audio chunk latency
- ✅ **AsyncLLMEngine**: Fast token generation with continuous batching
- ✅ **WebSocket protocol**: Bidirectional streaming for low latency
- ✅ **Multiple concurrent requests**: Efficiently handles burst traffic
- ✅ **Base64 audio encoding**: Easy integration with web clients
- ✅ **Progress updates**: Real-time feedback during generation

## Installation

Add the required dependencies:

```bash
uv add fastapi uvicorn websockets scipy
```

Or using pip:

```bash
pip install fastapi uvicorn websockets scipy
```

## Quick Start

### Option 1: Web Frontend (Easiest)

Start everything with one command:

```bash
chmod +x start-all.sh
./start-all.sh
```

Then open your browser to: **http://localhost:8080**

The frontend includes:
- 📝 Text input with preset examples
- ⚙️ Adjustable parameters (temperature, max tokens, chunk size)
- 📊 Real-time metrics display
- 🔊 Live audio playback
- 📋 Progress tracking

### Option 2: WebSocket Server Only

```bash
# Using uv
CUDA_VISIBLE_DEVICES=0 uv run python -m src.chatterbox_vllm.websocket_api

# Or directly
CUDA_VISIBLE_DEVICES=0 python -m src.chatterbox_vllm.websocket_api
```

The server will start on `http://0.0.0.0:8000`

### Option 3: Python Test Client

```bash
CUDA_VISIBLE_DEVICES=0 uv run python test-websocket-client.py
```

## API Specification

### WebSocket Endpoint

**URL:** `ws://localhost:8000/tts/websocket`

**Query Parameters:**
- `api_key` (optional): API key for authentication
- `version` (optional): API version (default: "2025-12-09")

### Request Format (Client → Server)

```json
{
  "text": "Text to synthesize",
  "request_id": "optional-request-id",
  "temperature": 0.8,
  "max_tokens": 500,
  "top_p": 0.95,
  "repetition_penalty": 1.0,
  "chunk_size": 25,
  "context_window": 50,
  "fade_duration": 0.02,
  "diffusion_steps": 10
}
```

**Parameters:**
- `text` (required): Text to synthesize
- `request_id` (optional): Unique identifier for the request
- `temperature` (optional): Sampling temperature (0.0-1.0, default: 0.8)
- `max_tokens` (optional): Maximum tokens to generate (default: 500)
- `top_p` (optional): Top-p sampling parameter (default: 0.95)
- `repetition_penalty` (optional): Repetition penalty (default: 1.0)
- `chunk_size` (optional): Speech tokens per audio chunk (default: 25)
- `context_window` (optional): Context tokens for continuity (default: 50)
- `fade_duration` (optional): Fade-in duration in seconds (default: 0.02)
- `diffusion_steps` (optional): S3Gen diffusion steps (default: 10)

### Response Format (Server → Client)

#### Acknowledgment

```json
{
  "type": "acknowledgment",
  "request_id": "request-id",
  "data": {
    "message": "Request received",
    "text": "First 100 chars of text..."
  }
}
```

#### Progress Update

```json
{
  "type": "progress",
  "request_id": "request-id",
  "data": {
    "tokens_generated": 50,
    "max_tokens": 500
  }
}
```

#### Audio Chunk

```json
{
  "type": "audio_chunk",
  "request_id": "request-id",
  "data": {
    "chunk_index": 1,
    "audio": "base64-encoded PCM audio",
    "sample_rate": 24000,
    "format": "wav",
    "is_final": false
  }
}
```

**Audio Format:**
- Encoding: Base64
- Sample format: 16-bit PCM (little-endian)
- Sample rate: 24000 Hz
- Channels: 1 (mono)

#### Complete

```json
{
  "type": "complete",
  "request_id": "request-id",
  "data": {
    "chunks_sent": 10,
    "total_tokens": 250,
    "total_time_seconds": 3.5,
    "first_token_ms": 50.2,
    "first_chunk_ms": 767.5
  }
}
```

#### Error

```json
{
  "type": "error",
  "request_id": "request-id",
  "error": "Error message"
}
```

## Usage Examples

### Python Client

```python
import asyncio
import json
import base64
import numpy as np
import websockets

async def stream_tts(text: str):
    uri = "ws://localhost:8000/tts/websocket"

    async with websockets.connect(uri) as ws:
        # Send request
        request = {
            "text": text,
            "request_id": "my-request",
        }
        await ws.send(json.dumps(request))

        # Receive streaming audio
        audio_chunks = []
        async for message in ws:
            response = json.loads(message)

            if response["type"] == "audio_chunk":
                # Decode audio
                audio_base64 = response["data"]["audio"]
                audio_bytes = base64.b64decode(audio_base64)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_chunks.append(audio)

                # Play or process audio chunk
                # ...

            elif response["type"] == "complete":
                break

        # Concatenate and save
        full_audio = np.concatenate(audio_chunks)
        scipy.io.wavfile.write("output.wav", 24000, full_audio)

asyncio.run(stream_tts("Hello world!"))
```

### JavaScript Client (Browser)

```javascript
const text = "Hello world!";
const ws = new WebSocket("ws://localhost:8000/tts/websocket");

ws.onopen = () => {
  // Send request
  ws.send(JSON.stringify({
    text: text,
    request_id: "my-request"
  }));
};

ws.onmessage = async (event) => {
  const response = JSON.parse(event.data);

  if (response.type === "audio_chunk") {
    // Decode base64 audio
    const audioBase64 = response.data.audio;
    const audioBytes = Uint8Array.from(atob(audioBase64), c => c.charCodeAt(0));

    // Convert to AudioBuffer
    const audioContext = new AudioContext({ sampleRate: 24000 });
    const audioBuffer = audioContext.decodeAudioData(audioBytes.buffer);

    // Play audio
    const source = audioContext.createBufferSource();
    source.buffer = await audioBuffer;
    source.connect(audioContext.destination);
    source.start();
  }

  if (response.type === "complete") {
    console.log("Streaming complete!", response.data);
    ws.close();
  }
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};
```

### curl Test (Text Mode)

```bash
# Note: curl doesn't support WebSocket, use a dedicated WebSocket client
# Here's using websocat:
echo '{"text":"Hello world"}' | websocat ws://localhost:8000/tts/websocket
```

## Web Frontend

A complete web-based UI is included for easy interaction with the TTS API.

### Screenshot Preview

The frontend features:
- 🎨 **Modern UI** - Beautiful gradient design with responsive layout
- 🔌 **One-click connect** - Easy WebSocket connection management
- 📝 **Preset texts** - Quick test buttons for short, medium, and long texts
- ⚙️ **Configurable parameters** - Adjust temperature, max tokens, chunk size
- 📊 **Real-time metrics** - First token/chunk latency, total time, chunk count
- 🔊 **Live playback** - Audio chunks play as they arrive
- 📋 **Progress tracking** - Visual progress bar with status updates
- 📜 **Optional logging** - Detailed logs for debugging

### Starting the Frontend

**Option 1: Start Everything (Recommended)**

```bash
chmod +x start-all.sh
./start-all.sh
```

This starts both:
- WebSocket API server on port 8000
- Frontend HTTP server on port 8080

**Option 2: Frontend Only**

```bash
chmod +x serve-frontend.sh
./serve-frontend.sh
# Or manually: python3 -m http.server 8080 --directory frontend
```

Then open your browser to: **http://localhost:8080**

### Frontend Features

1. **Connection Status** - Visual indicator showing WebSocket connection state
2. **Text Input** - Large textarea with preset example texts
3. **Settings Panel** - Adjust generation parameters
4. **Generate Button** - Start speech generation (disabled when not connected)
5. **Progress Bar** - Real-time progress during generation
6. **Metrics Display** - Shows latency and timing information
7. **Audio Playback** - Chunks play automatically as they arrive
8. **Log Panel** - Optional detailed logging (toggle with checkbox)

### Browser Compatibility

Tested on:
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Opera 76+

**Note:** Requires a modern browser with WebSocket and Web Audio API support.

### Customizing the Frontend

The frontend is a single HTML file (`frontend/index.html`) with embedded CSS and JavaScript. You can modify:

- **Styling**: Edit the `<style>` section for custom colors/layout
- **WebSocket URL**: Change `ws://localhost:8000` to your server
- **Default settings**: Modify input field default values
- **Audio playback**: Adjust chunk queuing behavior

### Using the Frontend with Remote Server

To connect to a remote WebSocket server:

1. Open `frontend/index.html` in a text editor
2. Find this line in the JavaScript:
   ```javascript
   const wsUrl = 'ws://localhost:8000/tts/websocket';
   ```
3. Change to your server URL:
   ```javascript
   const wsUrl = 'wss://your-server.com/tts/websocket';
   ```

**Note:** For HTTPS/WSS, ensure your server has valid SSL certificates.

## Performance

Based on our testing with unique texts (no prefix caching):

| Concurrent Requests | Avg First Token | First Audio Chunk | Throughput |
|---------------------|-----------------|-------------------|------------|
| 1                   | 26.3ms          | ~450ms            | 38 req/s   |
| 4                   | 54.1ms          | ~480ms            | 18 req/s   |
| 8                   | 32.3ms          | ~460ms            | 28 req/s   |
| 16                  | 29.2ms          | ~450ms            | 20 req/s   |
| 32                  | 32.1ms          | ~450ms            | 39 req/s   |

## Configuration

### Server Configuration

Edit `websocket_api.py` to modify:

```python
engine_args = AsyncEngineArgs(
    model="./t3-model",           # Path to T3 model
    tokenizer="EnTokenizer",      # Tokenizer
    gpu_memory_utilization=0.90,  # GPU memory (0.0-1.0)
    max_model_len=2000,           # Max sequence length
    enforce_eager=True,           # Disable CUDA graphs
    tensor_parallel_size=1,       # GPU count for tensor parallelism
)
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU to use (default: "0")

## Troubleshooting

### Server won't start

```
FileNotFoundError: Model not found at ./t3-model
```

**Solution:** Ensure the T3 model exists at `./t3-model` or update the `model_path` in the code.

### WebSocket connection fails

```
websockets.exceptions.InvalidStatusCode: server=403
```

**Solution:** Check API key authentication if enabled.

### No audio output

**Solution:**
1. Check that audio chunks are being received
2. Verify sample rate (24000 Hz)
3. Check audio decoding (16-bit PCM, little-endian)

## Production Deployment

### Using Uvicorn

```bash
uvicorn src.chatterbox_vllm.websocket_api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --ws-ping-interval 20 \
  --ws-ping-timeout 20
```

### Using Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install uv && uv sync

ENV CUDA_VISIBLE_DEVICES=0
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.chatterbox_vllm.websocket_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Gunicorn with Uvicorn Workers

```bash
gunicorn src.chatterbox_vllm.websocket_api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

**Note:** Use only 1 worker per GPU to avoid CUDA initialization issues.

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "chatterbox-vllm-tts",
  "model": "async-llm-engine"
}
```

## License

Same as parent project.
