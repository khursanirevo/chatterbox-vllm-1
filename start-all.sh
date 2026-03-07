#!/bin/bash
# Start both WebSocket API server and frontend HTTP server

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    jobs -p | xargs -r kill
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "=================================="
echo "Chatterbox vLLM TTS - Full Stack"
echo "=================================="
echo ""

# Start WebSocket API server in background
echo "Starting WebSocket API server on port 8000..."
uv run python -m src.chatterbox_vllm.websocket_api &
WS_PID=$!
echo "  WebSocket API: http://localhost:8000/tts/websocket (PID: $WS_PID)"

# Wait a bit for WebSocket server to initialize
sleep 3

# Start frontend HTTP server in background
echo ""
echo "Starting frontend HTTP server on port 8080..."
python3 -m http.server 8080 --directory frontend &
HTTP_PID=$!
echo "  Frontend: http://localhost:8080 (PID: $HTTP_PID)"

echo ""
echo "=================================="
echo "✅ All servers started!"
echo "=================================="
echo ""
echo "📱 Open your browser to:"
echo "   http://localhost:8080"
echo ""
echo "📊 API documentation:"
echo "   http://localhost:8000/docs"
echo "   http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "=================================="
echo ""

# Wait for any background process to finish
wait
