#!/bin/bash
# scripts/download_all_datasets.sh

set -e

echo "================================================="
echo "   Downloading Datasets"
echo "================================================="

# Source environment variables if they exist
if [ -f ".env" ]; then
    export $(cat .env | grep -v '#' | awk '/=/ {print $1}')
fi

python3 datasets/download_plantvillage.py
python3 datasets/download_plantdoc.py
python3 datasets/download_rice1426.py

echo "[OK] All datasets downloaded successfully."
