---
name: nvidia
description: >-
  Use this skill when the user asks to perform NVIDIA-specific tasks, such as 
  triggering specialized NeMo training pipelines, interacting with Nemotron, or 
  handling GPU optimization for CropGuard.
---
# NVIDIA Best Practices Skill

This skill incorporates the official guidelines from the `NVIDIA/skills` repository (specifically `nemotron-customize` and `nemo-*` skills) tailored for Antigravity.

## Core Rules for NeMo and Nemotron Tasks:
1. **Safety First:** Never run environment dumps (`env`, `printenv`, broad `export`) or commands that expose secret values. Always ask the user to export API keys manually in their environment rather than inlining them in scripts.
2. **LoRA Fine-tuning Constraints:** Ensure that the exact base checkpoint/model and tokenizer used during adapter training are preserved and matched during any later merge or evaluation steps.
3. **Hardware-Aware Execution:** When running `train_lora.py` or similar scripts, confirm the GPU count. For small hardware setups (like 1-2 GPUs), always recommend LoRA/PEFT (which you currently do) instead of full parameter Megatron-Bridge fine-tuning.
4. **Validation:** Always validate dataset structures and JSONL formats before executing expensive training runs. Use the `datasets/debug_normalization.py` or `validate_dataset.py` scripts first.
5. **Clear Command Execution:** Provide complete, parameterized commands in a single response. Do not invent speculative flags for NeMo scripts that are not explicitly supported.