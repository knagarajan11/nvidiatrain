#!/bin/bash
# scripts/evaluate.sh

set -e

echo "================================================="
echo "   Evaluating CropGuard Models"
echo "================================================="

# Source environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

python3 evaluation/run_nemo_evaluator.py

python3 evaluation/error_analysis.py
python3 evaluation/generate_report.py

echo "[OK] Evaluation complete."
