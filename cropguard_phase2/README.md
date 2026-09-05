# CropGuard AI Phase 2

Phase 2 pipeline for fine-tuning the **NVIDIA Nemotron Nano 12B v2 VL** model on plant disease datasets (PlantVillage, PlantDoc, Rice1426) using NVIDIA NeMo (LoRA/PEFT) on a DGX GPU server.

## A. DGX Environment
This pipeline is designed for NVIDIA DGX servers (A100/H100).
Ensure GPUs are visible by running:
```bash
nvidia-smi
```

## B. Environment Setup
We highly recommend running this pipeline within the official **NVIDIA NeMo Container** to avoid dependency nightmares with CUDA, PyTorch, NCCL, and Megatron.

Using Apptainer (recommended for HPC/DGX nodes where `sudo` is unavailable). To avoid long load times if `squashfuse` is missing from your host, we build a permanent sandbox first:

**Step 1: Build the sandbox (Only needs to be done once)**
```bash
apptainer build --sandbox nemo_sandbox docker://nvcr.io/nvidia/nemo:24.09
```

**Step 2: Run the container from the sandbox (Instant loading)**
```bash
apptainer run --nv --bind $(pwd):/workspace --pwd /workspace nemo_sandbox bash
```

Inside the container, install the basic pip dependencies for data preparation:
```bash
pip install -r requirements.txt
```

## C. Authentication
You must provide access to HuggingFace and NGC to pull the Nemotron base models.
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
# Edit .env and add HF_TOKEN and NGC_API_KEY
```

## D. Download Datasets
Downloads PlantVillage, PlantDoc, and Rice1426.
```bash
./scripts/run_pipeline.sh --stage download
# or:
./scripts/download_all_datasets.sh

# To force re-download and refresh raw directory structure:
./scripts/run_pipeline.sh --stage download --force
# or:
./scripts/download_all_datasets.sh --force
```

## E. Prepare Datasets & Verification
Normalizes labels, validates schemas, splits data (train/val/test), and generates multimodal JSONL instructions for NeMo.

```bash
./scripts/run_pipeline.sh --stage data
# or:
./scripts/prepare_all_datasets.sh
```

### Verification & Quality Checks:
After preparation, verify dataset integrity and label normalization:

1. **Verify Label Normalization (Check for 0 UNKNOWNs)**:
   ```bash
   python datasets/debug_normalization.py
   ```
2. **Validate Image Files & Corrupt Images**:
   ```bash
   python datasets/validate_dataset.py
   ```
3. **Inspect Dataset Splits**:
   ```bash
   wc -l data/splits/*.jsonl
   ```
4. **Inspect Generated NeMo Instructions**:
   ```bash
   head -n 2 data/instructions/nemo_train.jsonl
   ```

## F. Train LoRA
Runs NeMo PEFT (LoRA) fine-tuning targeting the Nemotron language backbone.
```bash
./scripts/run_pipeline.sh --stage train
# or
./scripts/train_lora.sh
```
Configurations for LoRA (r, alpha, target modules) can be adjusted in `configs/lora.yaml`.

## G. Evaluate
Evaluates the base model and fine-tuned LoRA using NeMo Evaluator.
```bash
./scripts/run_pipeline.sh --stage evaluate
# or
./scripts/evaluate.sh
```

## H. Inference
Run inference on a single image:
```bash
python inference/predict.py --image leaf.jpg
python inference/predict.py --image leaf.jpg --json
```
Batch inference:
```bash
python inference/predict_batch.py --input data/test --output outputs/predictions/
```

## I. Locate Results
*   **`outputs/checkpoints/`**: NeMo checkpoints (.nemo) for LoRA and full SFT.
*   **`outputs/reports/`**: Error analysis CSVs, model comparison JSONs, dataset statistics.
*   **`outputs/predictions/`**: Batch inference outputs.
*   **`outputs/metrics/`**: Raw NeMo Evaluator outputs for base and finetuned models.

## Troubleshooting

*   **CUDA errors or GPU OOM**: Edit `configs/lora.yaml` and reduce `batch_size`. Verify `--nproc-per-node` matches your actual GPU count. Ensure your `nvcr.io` container version matches your DGX CUDA driver.
*   **NCCL errors**: If using multi-node DGX setups, verify infiniband configurations. In Apptainer, ensure your system limits are appropriately set for InfiniBand.
*   **Model download/authentication errors**: Verify `.env` tokens. Ensure you have accepted the model license terms on HuggingFace/NGC.
*   **NeMo version conflicts**: Ensure you are using the official NVIDIA NeMo container. Do not blindly `pip install -U everything` as it breaks tightly coupled Megatron-Core versions.
*   **Dataset errors**: Run `python datasets/validate_dataset.py` to identify missing or corrupt image paths.
*   **NeMo Evaluator errors**: Ensure `nel` (NeMo Evaluator Library) is installed correctly inside the container via pip or from source, depending on your NeMo version.
