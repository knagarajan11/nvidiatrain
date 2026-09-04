import os
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
    # This generates a Hydra compatible YAML for NeMo's finetune.py
    nemo_cfg = {
        "trainer": {
            "num_nodes": 1,
            "devices": torch.cuda.device_count() if torch.cuda.is_available() else 1,
            "accelerator": "gpu",
            "max_epochs": lora_cfg.get("epochs", 3),
            "val_check_interval": 1.0,
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
                "restore_from_path": f"{model_cfg['model_name']}.nemo" 
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
    
    with open(out_path, 'w') as f:
        yaml.dump(nemo_cfg, f)

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
    
    # We use torchrun to invoke NeMo's finetune.py. Assuming running inside NeMo container
    # where /opt/NeMo/examples/vlm_finetune/finetune.py might be present.
    # If not present, we output the command that should be run.
    nemo_script = "/opt/NeMo/examples/vlm_finetune/finetune.py"
    
    num_gpus = torch.cuda.device_count()
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        nemo_script,
        f"--config-path={cfg_path.parent.absolute()}",
        f"--config-name={cfg_path.name}"
    ]
    
    logging.info(f"Executing: {' '.join(cmd)}")
    
    if Path(nemo_script).exists():
        subprocess.run(cmd, check=True)
    else:
        logging.warning(f"NeMo script {nemo_script} not found (are you inside the NeMo container?).")
        logging.info(f"Please run the following command manually inside the NeMo container:\n{' '.join(cmd)}")

if __name__ == "__main__":
    main()
