# CropGuard AI Phase 2

Phase 2 pipeline for fine-tuning the **NVIDIA Nemotron Nano 12B v2 VL** model on plant disease datasets (PlantVillage, PlantDoc, Rice1426) using HuggingFace PEFT (LoRA) on a DGX GPU server.

## A. DGX Environment
This pipeline is designed for NVIDIA DGX servers (A100/H100).
Ensure GPUs are visible by running:
```bash
nvidia-smi
```

## B. Environment Setup
We highly recommend running this pipeline within the official **NVIDIA NeMo Container** to avoid dependency nightmares with CUDA, PyTorch, NCCL, and Megatron.

Using Apptainer (recommended for HPC/DGX nodes where `sudo` is unavailable). To avoid long load times if `squashfuse` is missing from your host, we build a permanent sandbox first:

> **Important:** Use **NeMo 25.02** or later. The `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` model requires `transformers >= 4.48` and `torch >= 2.5`, both of which ship in `nemo:25.02`. The older `nemo:24.09` container is **incompatible** and will produce `MultiModalData`, `VideoInput`, and `AutoModelForVision2Seq` import errors.

**Step 1: Build the sandbox (Only needs to be done once)**
```bash
apptainer build --sandbox nemo_sandbox_25 docker://nvcr.io/nvidia/nemo:25.02
```

**Step 2: Run the container from the sandbox (Instant loading)**
```bash
apptainer run --nv --bind $(pwd):/workspace --pwd /workspace nemo_sandbox_25 bash
```

Inside the container, install the required pip dependencies:
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

Also accept the model license on HuggingFace before running (required for gated models):
- https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16

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
Normalizes labels, validates schemas, splits data (train/val/test), and generates multimodal JSONL instructions for fine-tuning.

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
4. **Inspect Generated Instructions**:
   ```bash
   head -n 2 data/train/instruction_train.jsonl
   ```

## F. Train LoRA
Runs LoRA fine-tuning using the HuggingFace PEFT path targeting the Nemotron VL backbone.
```bash
./scripts/run_pipeline.sh --stage train
# or
./scripts/train_lora.sh
```
Configurations for LoRA (r, alpha, target modules, batch size, epochs) can be adjusted in `configs/lora.yaml`.

## G. Evaluate
Evaluates the base model and fine-tuned LoRA adapter.
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
*   **`outputs/checkpoints/cropguard_lora/`**: LoRA adapter weights (HuggingFace PEFT format).
*   **`outputs/reports/`**: Error analysis CSVs, model comparison JSONs, dataset statistics.
*   **`outputs/predictions/`**: Batch inference outputs.
*   **`outputs/metrics/`**: Evaluation outputs for base and fine-tuned models.

## Troubleshooting

*   **`MultiModalData` / `VideoInput` / `AutoModelForVision2Seq` import errors**: You are running inside `nemo:24.09`. Rebuild the sandbox with `nemo:25.02` — see Section B above.
*   **`transformers 5.x` disables PyTorch (torch 2.4 < 2.5 required)**: A previous pip upgrade installed transformers 5.x in `~/.local`. Run `pip uninstall transformers -y` inside the container, then rerun. Or rebuild sandbox with `nemo:25.02`.
*   **CUDA errors or GPU OOM**: Edit `configs/lora.yaml` and reduce `batch_size`. On a single GPU, the model auto-loads in 4-bit QLoRA to conserve VRAM.
*   **HuggingFace 401/404 errors**: Verify `HF_TOKEN` in `.env` and accept the model license at https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
*   **NCCL errors**: Verify infiniband configurations for multi-node DGX setups.
*   **NeMo version conflicts**: Use the official NVIDIA NeMo container. Do not blindly `pip install -U everything` as it breaks tightly coupled Megatron-Core versions.
*   **Dataset errors**: Run `python datasets/validate_dataset.py` to identify missing or corrupt image paths.
