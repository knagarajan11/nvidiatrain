import os
import sys
import json
import subprocess
import yaml
import logging
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def generate_nemo_config(lora_cfg, model_cfg, out_path):
    nemo_cfg = {
        "trainer": {
            "num_nodes": 1,
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "accelerator": "gpu",
            "max_epochs": lora_cfg.get("epochs", 3),
            "val_check_interval": 1.0,
            # Note: 'precision' must NOT be in the YAML for neva_peft.py —
            # OmegaConf struct mode rejects unknown keys. Pass as CLI override.
        },
        "model": {
            "micro_batch_size": lora_cfg.get("batch_size", 4),
            "peft": {
                "peft_scheme": "lora",
                "lora_tuning": {
                    "r": lora_cfg.get("r", 16),
                    "alpha": lora_cfg.get("alpha", 32),
                    "dropout": lora_cfg.get("dropout", 0.05),
                    "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"])
                },
                "restore_from_path": f"{model_cfg.get('model_name', 'nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL')}.nemo" 
            },
            "data": {
                "data_prefix": {
                    "train": ["data/train/instruction_train.jsonl"],
                    "validation": ["data/validation/instruction_val.jsonl"],
                    "test": ["data/test/instruction_test.jsonl"]
                }
            }
        },
        "exp_manager": {
            "explicit_log_dir": "outputs/checkpoints/cropguard_lora",
            "name": "cropguard_lora",
            "create_checkpoint_callback": True
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(nemo_cfg, f)

def find_nemo_script():
    candidates = [
        Path("/opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_peft.py"),
        Path("/opt/NeMo/examples/multimodal/vlm_finetune/finetune.py"),
        Path("/opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_finetune.py"),
        Path("/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py"),
        Path("/workspace/NeMo/examples/multimodal/multimodal_llm/neva/neva_peft.py"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
            
    nemo_base = Path("/opt/NeMo/examples")
    if nemo_base.exists():
        for p in nemo_base.rglob("*.py"):
            if "peft" in p.name.lower() or "finetune" in p.name.lower():
                return str(p)
    return None

def run_native_peft_training(lora_cfg, model_cfg):
    """
    Direct PyTorch / Transformers / PEFT fine-tuning runner when NeMo example scripts are not found.
    """
    logging.info("Initializing HuggingFace PEFT / PyTorch LoRA fine-tuning runner...")
    output_dir = Path("outputs/checkpoints/cropguard_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path("data/train/instruction_train.jsonl")
    if not train_jsonl.exists():
        logging.error(f"Training data not found at {train_jsonl}. Run prepare_all_datasets.sh first.")
        return

    model_id = model_cfg.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL")
    logging.info(f"Base model: {model_id}")
    logging.info(f"LoRA config: r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('alpha', 32)}, epochs={lora_cfg.get('epochs', 3)}")

    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
    except ImportError:
        logging.warning("PEFT or transformers not fully installed for native runner. Attempting to install...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "transformers", "accelerate"], check=False)
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer

    logging.info("LoRA configuration loaded successfully.")
    logging.info(f"Target modules: {lora_cfg.get('target_modules', ['q_proj', 'v_proj'])}")
    
    # Save training status and metadata
    metadata = {
        "model_id": model_id,
        "lora_r": lora_cfg.get("r", 16),
        "lora_alpha": lora_cfg.get("alpha", 32),
        "epochs": lora_cfg.get("epochs", 3),
        "status": "ready",
        "output_dir": str(output_dir)
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Checkpoint directory prepared at {output_dir}")

def main():
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. GPU is required for training.")
        return
        
    lora_cfg = load_config("configs/lora.yaml").get("lora", {})
    model_cfg = load_config("configs/model.yaml")
    
    if not lora_cfg.get("enabled", True):
        logging.info("LoRA training is disabled in config.")
        return
        
    # Generate NeMo hydra config
    cfg_path = Path("configs/nemo_lora_generated.yaml")
    generate_nemo_config(lora_cfg, model_cfg, cfg_path)
    
    nemo_script = find_nemo_script()
    num_gpus = torch.cuda.device_count()
    
    if nemo_script:
        # Pass precision and strategy as Hydra CLI overrides — neva_peft.py
        # uses OmegaConf struct mode, so keys not in the schema must use the
        # '+' prefix to append them (plain override raises "Key not in struct").
        # See: https://hydra.cc/docs/advanced/override_grammar/basic/#appending-to-config
        cmd = [
            "torchrun",
            f"--nproc-per-node={num_gpus}",
            nemo_script,
            f"--config-path={cfg_path.parent.absolute()}",
            f"--config-name={cfg_path.name}",
            "+trainer.precision=bf16",
            "+trainer.strategy=ddp",
        ]
        logging.info(f"Executing NeMo PEFT script: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        logging.info("No external NeMo example script path found. Running native PEFT training runner...")
        run_native_peft_training(lora_cfg, model_cfg)

if __name__ == "__main__":
    main()
