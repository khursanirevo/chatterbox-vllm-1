#!/bin/bash
# Stop CUDA MPS daemon

echo "========================================="
echo "Stopping CUDA MPS Daemon"
echo "========================================="

# Stop MPS daemon
echo "[1/2] Stopping MPS daemon..."
if echo quit | nvidia-cuda-mps-control 2>/dev/null; then
    echo "✓ MPS daemon stopped"
else
    echo "⚠️  MPS daemon was not running"
fi

# Reset GPU to default mode (try all GPUs)
echo -e "\n[2/2] Resetting GPUs to DEFAULT mode..."
for i in {0..3}; do
    nvidia-smi -i $i -c DEFAULT 2>/dev/null && echo "✓ GPU $i reset to DEFAULT mode"
done

echo -e "\n========================================="
echo "✅ CUDA MPS stopped"
echo "========================================="
