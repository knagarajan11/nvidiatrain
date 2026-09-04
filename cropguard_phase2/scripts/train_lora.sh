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

python3 training/train_lora.py

echo "[OK] LoRA training pipeline complete."
