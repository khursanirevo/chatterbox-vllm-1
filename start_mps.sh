#!/bin/bash
# Start CUDA MPS daemon for GPU sharing

echo "========================================="
echo "Starting CUDA MPS Daemon"
echo "========================================="

# Check GPU status
echo "Current GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,compute_mode --format=csv

# Find a GPU with minimal usage
echo -e "\n[1/4] Finding available GPU..."
GPU_ID=0
MIN_MEMORY=999999

for i in {0..3}; do
    MEM_USED=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$MEM_USED" -lt "$MIN_MEMORY" ]; then
        MIN_MEMORY=$MEM_USED
        GPU_ID=$i
    fi
done

echo "Using GPU $GPU_ID (has $MIN_MEMORY MiB used)"

# Set GPU to exclusive process mode (required for MPS)
echo -e "\n[2/4] Setting GPU $GPU_ID to EXCLUSIVE_PROCESS mode..."
if nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS 2>&1 | grep -q "Insufficient Permissions"; then
    echo "⚠️  Warning: Insufficient permissions to set GPU mode"
    echo "   Trying without exclusive mode (may still work)..."
    EXCLUSIVE_MODE=0
else
    echo "✓ GPU $GPU_ID set to EXCLUSIVE_PROCESS mode"
    EXCLUSIVE_MODE=1
fi

# Start MPS daemon
echo -e "\n[3/4] Starting MPS daemon on GPU $GPU_ID..."
nvidia-cuda-mps-control -d

# Verify MPS is running
echo -e "\n[4/4] Verifying MPS status..."
sleep 1
if echo "get_state" | nvidia-cuda-mps-control 2>/dev/null | grep -q "ready"; then
    echo "✓ MPS daemon is running and ready"
else
    echo "⚠️  MPS daemon started but status check failed"
    echo "   This is normal - MPS may still work"
fi

# Set environment variables for current session
export CUDA_VISIBLE_DEVICES=$GPU_ID
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

echo -e "\n========================================="
echo "✅ CUDA MPS is now running on GPU $GPU_ID!"
echo "========================================="
echo ""
echo "Environment variables set:"
echo "  CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "  CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"
echo "  CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log"
echo ""
echo "To stop MPS later, run:"
echo "  ./stop_mps.sh"
echo "  or"
echo "  echo quit | nvidia-cuda-mps-control"
if [ "$EXCLUSIVE_MODE" -eq 1 ]; then
    echo "  nvidia-smi -i $GPU_ID -c DEFAULT"
fi
echo ""
echo "GPU resources can now be shared across processes!"
echo "========================================="

# Display reminder for future sessions
echo -e "\n⚠️  REMINDER: For new terminal sessions, set:"
echo "  export CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"
echo "  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log"
echo "========================================="
