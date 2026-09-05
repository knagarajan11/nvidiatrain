#!/bin/bash
# scripts/run_pipeline.sh

set -e

# Default to running all stages
RUN_DOWNLOAD=true
RUN_DATA=true
RUN_TRAIN=true
RUN_EVALUATE=true
RUN_INFERENCE=true
FORCE_DOWNLOAD_ARG=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            RUN_DOWNLOAD=false
            RUN_DATA=false
            RUN_TRAIN=false
            RUN_EVALUATE=false
            RUN_INFERENCE=false
            case $STAGE in
                download) RUN_DOWNLOAD=true ;;
                data) RUN_DATA=true ;;
                train) RUN_TRAIN=true ;;
                evaluate) RUN_EVALUATE=true ;;
                inference) RUN_INFERENCE=true ;;
                *) echo "Unknown stage: $STAGE"; exit 1 ;;
            esac
            shift 2
            ;;
        --force|-f)
            FORCE_DOWNLOAD_ARG="--force"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting CropGuard Phase 2 Pipeline..."

# 1. Environment Verification
./scripts/check_environment.sh

# 2. Download
if [ "$RUN_DOWNLOAD" = true ]; then
    ./scripts/download_all_datasets.sh $FORCE_DOWNLOAD_ARG
fi

# 3. Data Prep
if [ "$RUN_DATA" = true ]; then
    ./scripts/prepare_all_datasets.sh
fi

# 4. Train
if [ "$RUN_TRAIN" = true ]; then
    ./scripts/train_lora.sh
    # Optionally: ./scripts/train_sft.sh if needed
fi

# 5. Evaluate
if [ "$RUN_EVALUATE" = true ]; then
    ./scripts/evaluate.sh
fi

# 6. Inference test
if [ "$RUN_INFERENCE" = true ]; then
    echo "================================================="
    echo "   Testing Inference"
    echo "================================================="
    # Create dummy image for inference test if it doesn't exist
    touch dummy_leaf.jpg
    python3 inference/predict.py --image dummy_leaf.jpg
    rm dummy_leaf.jpg
    echo "[OK] Inference test passed."
fi

echo "================================================="
echo "   CropGuard Phase 2 Pipeline Finished"
echo "================================================="
