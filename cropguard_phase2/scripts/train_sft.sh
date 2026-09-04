#!/bin/bash
# scripts/train_sft.sh

set -e

echo "================================================="
echo "   Training CropGuard Full SFT"
echo "================================================="

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

python3 training/train_sft.py

echo "[OK] SFT training script complete."
