import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base model")
    parser.add_argument("--base_model", type=str, default="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL")
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained LoRA adapter (.nemo or HF format)")
    parser.add_argument("--output", type=str, default="outputs/merged/cropguard_nemotron", help="Output path for merged model")
    args = parser.parse_args()
    
    # In NeMo, merging is often done via scripts provided in the framework:
    # /opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py
    
    merge_script = "/opt/NeMo/scripts/nlp_language_modeling/merge_lora_weights/merge.py"
    
    if not Path(merge_script).exists():
        logging.warning(f"NeMo merge script not found at {merge_script}. Assuming non-container env.")
        logging.info("For HuggingFace format, use `peft` library to merge:")
        logging.info("model.merge_and_unload()")
        return
        
    logging.info(f"To merge in NeMo, run:\npython {merge_script} trainer.precision=bf16 model.restore_from_path={args.base_model} model.peft.restore_from_path={args.adapter} model.peft.peft_scheme=lora save_to={args.output}")

if __name__ == "__main__":
    main()
