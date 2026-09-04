#!/bin/bash
# scripts/prepare_all_datasets.sh

set -e

echo "================================================="
echo "   Preparing Datasets"
echo "================================================="

python3 datasets/prepare_plantvillage.py
python3 datasets/prepare_plantdoc.py
python3 datasets/prepare_rice1426.py

python3 datasets/normalize_labels.py
python3 datasets/validate_dataset.py
python3 datasets/split_dataset.py
python3 datasets/create_instruction_dataset.py

echo "[OK] All datasets prepared successfully."
