# CropGuard AI Phase 2 Implementation Plan

This plan outlines the creation of an independent, reproducible training and evaluation pipeline for CropGuard AI on an NVIDIA DGX GPU server, fine-tuning an NVIDIA Nemotron VLM using NeMo and evaluating it with NeMo Evaluator.

## User Review Required
> [!IMPORTANT]
> **API Verification Complete:** Before proceeding, please review the results of the NVIDIA API verification:
> 1. **Model ID:** The target model will be `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL`. 
> 2. **NeMo Evaluator:** The current CLI tool for NeMo Evaluator is `nel` (NeMo Evaluator Library/Launcher). The `evaluation/run_nemo_evaluator.py` script will wrap the `nel eval run` commands rather than fabricating a non-existent Python API.
> 3. **Training Method:** We will utilize the official NeMo Framework's Parameter-Efficient Fine-Tuning (PEFT) capabilities (LoRA) specifically designed for VLMs, using `torchrun` and NeMo's VLM fine-tuning scripts.
> 4. **Environment:** Instead of a complex custom Apptainer definition or blind `pip install`, I highly recommend utilizing the official NVIDIA NeMo container (e.g., `nvcr.io/nvidia/nemo:latest` or `24.09`) which comes pre-packaged with PyTorch, CUDA, NeMo, Megatron, and PEFT optimized for DGX.

## Open Questions
> [!WARNING]
> 1. **Data Sources:** Retrieve from standard sources (Kaggle/HuggingFace).
> 2. **NeMo Container:** Yes, use `nvcr.io/nvidia/nemo`.
> 3. **HuggingFace Access:** Credentials will be provided in `.env`.

## Proposed Changes

### 1. Project Structure & Environment Setup
Establish the directory skeleton, configuration files, and environment validation scripts.

#### [NEW] `requirements.txt`
#### [NEW] `.env.example`
#### [NEW] `.gitignore`
#### [NEW] `configs/model.yaml`
#### [NEW] `configs/dataset.yaml`
#### [NEW] `configs/lora.yaml`
#### [NEW] `configs/sft.yaml`
#### [NEW] `configs/evaluator.yaml`
#### [NEW] `configs/label_mapping.yaml`
#### [NEW] `scripts/check_environment.sh`
#### [NEW] `scripts/check_gpu.py`

---
### 2. Dataset Preparation Pipeline
Scripts to independently download, clean, normalize, and format the datasets into multimodal instruction schemas compatible with NeMo.

#### [NEW] `scripts/download_all_datasets.sh`
#### [NEW] `scripts/prepare_all_datasets.sh`
#### [NEW] `datasets/download_plantvillage.py`
#### [NEW] `datasets/prepare_plantvillage.py`
#### [NEW] `datasets/download_plantdoc.py`
#### [NEW] `datasets/prepare_plantdoc.py`
#### [NEW] `datasets/download_rice1426.py`
#### [NEW] `datasets/prepare_rice1426.py`
#### [NEW] `datasets/normalize_labels.py`
#### [NEW] `datasets/create_instruction_dataset.py`
#### [NEW] `datasets/validate_dataset.py`
#### [NEW] `datasets/split_dataset.py`

---
### 3. Model Training (NeMo PEFT/LoRA)
Scripts to invoke NeMo's VLM fine-tuning routines with configurable LoRA adapters, supporting multi-GPU DGX setups.

#### [NEW] `scripts/train_lora.sh`
#### [NEW] `scripts/train_sft.sh`
#### [NEW] `training/train_lora.py`
#### [NEW] `training/train_sft.py`
#### [NEW] `training/merge_lora.py`
#### [NEW] `training/resume_training.py`

---
### 4. NeMo Evaluator & Analytics
Scripts to evaluate the base and fine-tuned models using NeMo Evaluator (`nel`), calculate metrics, and generate comparison reports.

#### [NEW] `scripts/evaluate.sh`
#### [NEW] `evaluation/run_nemo_evaluator.py`
#### [NEW] `evaluation/evaluate_model.py`
#### [NEW] `evaluation/evaluate_classification.py`
#### [NEW] `evaluation/evaluate_vlm.py`
#### [NEW] `evaluation/error_analysis.py`
#### [NEW] `evaluation/generate_report.py`

---
### 5. Inference Interface
Provide the required endpoints for Phase 3 integration and batch processing.

#### [NEW] `inference/predict.py`
#### [NEW] `inference/predict_batch.py`

---
### 6. Master Pipeline & Documentation
The overarching pipeline runner and the final README documenting architecture, setup, and troubleshooting.

#### [NEW] `scripts/run_pipeline.sh`
#### [NEW] `README.md`

## Verification Plan

### Automated Checks
- Run `scripts/check_environment.sh` and `check_gpu.py` to ensure accurate DGX hardware detection (A100/H100, memory, CUDA) and software compatibility.
- Run `datasets/validate_dataset.py` to check for data leakage between splits, duplicate images, and JSON schema validity.

### Manual Verification
- Review the generated `dataset_statistics.md` and `error_analysis.csv` to ensure sensible distributions.
- Dry-run the master script (`./scripts/run_pipeline.sh --stage data`) to confirm it executes the data preparation workflow end-to-end without failing.
- Check that NeMo model loading and Evaluator configs align with the verified `docs.nvidia.com` standards.
