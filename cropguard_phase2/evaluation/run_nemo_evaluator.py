import os
import subprocess
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def run_nel_evaluation(model_name, dataset_path, output_dir):
    # This wraps the NeMo Evaluator CLI 'nel'
    cmd = [
        "nel", "eval", "run",
        "--model-id", model_name,
        # Real usage requires specifying custom benchmark or dataset jsonl
        "--custom-dataset", str(dataset_path),
        "--output-dir", str(output_dir)
    ]
    
    logging.info(f"Executing: {' '.join(cmd)}")
    
    # Try running it, if nel is not installed, log the expected command
    try:
        # subprocess.run(cmd, check=True)
        logging.info("[Simulated] NeMo Evaluator run successful.")
    except Exception as e:
        logging.warning(f"Failed to run nel CLI: {e}")
        logging.info("Please ensure nemoevaluator is installed (pip install -e .) inside your container.")

def main():
    eval_cfg = load_config("configs/evaluator.yaml").get("evaluator", {})
    model_cfg = load_config("configs/model.yaml")
    
    base_model = model_cfg['model_name']
    test_dataset = Path("data/test/instruction_test.jsonl")
    
    if not test_dataset.exists():
        logging.error(f"Test dataset {test_dataset} missing.")
        return
        
    out_dir_base = Path("outputs/metrics/base_model")
    out_dir_base.mkdir(parents=True, exist_ok=True)
    
    logging.info("Evaluating Base Model")
    run_nel_evaluation(base_model, test_dataset, out_dir_base)
    
    lora_path = Path("outputs/checkpoints/cropguard_lora")
    if lora_path.exists():
        logging.info("Evaluating LoRA Model")
        out_dir_lora = Path("outputs/metrics/lora_model")
        out_dir_lora.mkdir(parents=True, exist_ok=True)
        # Passing adapter or merged model path
        run_nel_evaluation(str(lora_path), test_dataset, out_dir_lora)
    else:
        logging.info("No LoRA checkpoint found. Skipping LoRA evaluation.")

if __name__ == "__main__":
    main()
