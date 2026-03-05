#!/bin/bash
# Quick test script for CUDA MPS S3Gen implementation

set -e

echo "========================================"
echo "CUDA MPS S3Gen Quick Test"
echo "========================================"

# Set MPS pipe directory
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
echo "CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"

# Check if MPS is running
if pgrep -f "nvidia-cuda-mps-control" > /dev/null; then
    echo "✓ MPS daemon is running"
else
    echo "⚠ MPS daemon is not running"
    echo "Start it with: nvidia-cuda-mps-control -d"
    echo "Or run: ./start_mps.sh"
fi

# Check checkpoint directory
CKPT_DIR=${CHATTERBOX_CKPT:-./models/chatterbox}
if [ -d "$CKPT_DIR" ]; then
    echo "✓ Checkpoint directory: $CKPT_DIR"
else
    echo "⚠ Checkpoint directory not found: $CKPT_DIR"
    echo "Set CHATTERBOX_CKPT environment variable"
fi

echo ""
echo "Running implementation tests..."
echo "========================================"

uv run python test_mps_implementation.py

echo ""
echo "========================================"
echo "Test complete!"
echo "========================================"
