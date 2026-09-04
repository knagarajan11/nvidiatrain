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

def main():
    sft_cfg = load_config("configs/sft.yaml").get("sft", {})
    
    if not sft_cfg.get("enabled", False):
        logging.info("Full SFT training is disabled in config. Skipping.")
        return
        
    logging.info("Full SFT is enabled. Preparing training.")
    # Similar to LoRA, but without PEFT config block
    
    nemo_script = "/opt/NeMo/examples/vlm_finetune/finetune.py"
    num_gpus = torch.cuda.device_count()
    
    cmd = [
        "torchrun",
        f"--nproc-per-node={num_gpus}",
        nemo_script,
        # args would point to sft hydra config
    ]
    
    logging.info(f"Please run the following command manually inside the NeMo container:\n{' '.join(cmd)}")

if __name__ == "__main__":
    main()
