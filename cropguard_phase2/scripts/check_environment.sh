#!/bin/bash
# scripts/check_environment.sh

echo "================================================="
echo "   CropGuard Phase 2 - Environment Check"
echo "================================================="

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "[PASS] nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[FAIL] nvidia-smi not found. This pipeline requires an NVIDIA DGX or GPU server."
    exit 1
fi

# Check Docker / Container Environment
if [ -f /.dockerenv ]; then
    echo "[INFO] Running inside a container."
else
    echo "[WARN] Not running inside a container. It is highly recommended to use the official NVIDIA NeMo container."
fi

# Check Python version
python3 --version

# Run GPU detection python script
python3 scripts/check_gpu.py

echo "Environment check completed."
