#!/bin/bash
# scripts/train_lora.sh

set -e

echo "================================================="
echo "   Training CropGuard LoRA"
echo "================================================="

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "[INFO] Detected $NUM_GPUS GPUs. Launching multi-GPU training with torchrun..."
    torchrun --nproc_per_node=$NUM_GPUS training/train_lora.py
else
    echo "[INFO] Detected 1 GPU. Launching single-GPU training..."
    python3 training/train_lora.py
fi

echo "[OK] LoRA training pipeline complete."
