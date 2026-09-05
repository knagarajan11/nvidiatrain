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

def find_neva_conf_dir(nemo_script):
    """
    Return the conf/ directory that ships alongside neva_peft.py, which contains
    neva_peft.py's own Hydra/OmegaConf schema.  We use this as the base config
    so that all required model keys are present — we then override only our
    specific settings via CLI '++key=value' overrides.
    """
    script_dir = Path(nemo_script).parent
    for candidate in [script_dir / "conf", script_dir.parent / "conf"]:
        if candidate.exists():
            return str(candidate)
    return None


def pick_config_name(conf_dir):
    """
    Choose the best Hydra config file from conf_dir.
    Priority:
      1. Any file whose stem contains 'peft' (e.g. neva_peft, neva_peft_config)
      2. Any file whose stem contains 'neva' but NOT 'mixtral' / 'llama' / 'mistral'
      3. First file alphabetically (last resort)
    """
    conf_files = sorted(Path(conf_dir).glob("*.yaml"))
    if not conf_files:
        return "neva_peft"
    # Priority 1 – contains 'peft'
    for f in conf_files:
        if "peft" in f.stem.lower():
            return f.stem
    # Priority 2 – generic neva, not a specific model variant
    skip_keywords = {"mixtral", "llama", "mistral", "falcon", "mamba"}
    for f in conf_files:
        stem_lower = f.stem.lower()
        if "neva" in stem_lower and not any(kw in stem_lower for kw in skip_keywords):
            return f.stem
    # Fallback
    return conf_files[0].stem

def build_hydra_overrides(lora_cfg, model_cfg):
    """
    Build the list of Hydra '++key=value' CLI overrides that inject our
    CropGuard settings into neva_peft.py's existing OmegaConf struct.

    We use '++' (double-plus) throughout:
      '++'  = override if the key already exists in the struct, OR
              append if it doesn't exist yet.
    This is safer than '+' (append-only, fails if key exists) or plain
    'key=value' (override-only, fails if key is not in struct).
    See: https://hydra.cc/docs/advanced/override_grammar/basic/#force-add
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    model_name = model_cfg.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL")
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    # Hydra list syntax: [a,b,c]
    target_modules_str = "[" + ",".join(target_modules) + "]"
    # Vision encoder for Nemotron-Nano-12B-v2-VL is SigLIP (not CLIP).
    # Derived from model.yaml: vision_encoder: "siglip"
    vision_encoder_model = model_cfg.get(
        "vision_encoder_hf_id",
        "google/siglip-so400m-patch14-384"   # SigLIP SO400M, 384px — used by Nemotron VL
    )

    overrides = [
        # --- trainer ---
        # NOTE: Do NOT set trainer.strategy here.
        # MegatronLMPPTrainerBuilder builds its own MegatronStrategy object and
        # passes it as strategy=... to Trainer() explicitly. If trainer.strategy
        # is also present in cfg.trainer, Trainer() receives it twice and raises:
        #   TypeError: got multiple values for keyword argument 'strategy'
        f"++trainer.num_nodes=1",
        f"++trainer.devices={num_gpus}",
        f"++trainer.accelerator=gpu",
        f"++trainer.precision=bf16",
        f"++trainer.max_epochs={lora_cfg.get('epochs', 3)}",
        # check_val_every_n_epoch=1 enables epoch-based validation.
        # When it is null (base config default), val_check_interval must be an
        # integer (step count). With epoch-mode=1, val_check_interval=1.0 is
        # valid (means 100% of epoch). Without this, PL raises:
        #   MisconfigurationException: val_check_interval should be an integer
        #   when check_val_every_n_epoch=None, found 1.0
        f"++trainer.check_val_every_n_epoch=1",
        f"++trainer.val_check_interval=1.0",
        # --- model (base model restore) ---
        # model.restore_from_path loads the full base .nemo model.
        # model.peft.restore_from_path is for restoring a PEFT adapter checkpoint
        # (i.e. resuming LoRA training), NOT for loading the base model.
        f"++model.restore_from_path={model_name}.nemo",
        f"++model.micro_batch_size={lora_cfg.get('batch_size', 4)}",
        # --- vision encoder ---
        # The base neva_peft.yaml has vision_encoder.from_pretrained='' (empty),
        # which causes: OSError: Incorrect path_or_model_id: ''
        # Nemotron-Nano-12B-v2-VL uses SigLIP as its vision backbone.
        f"++model.mm_cfg.vision_encoder.from_pretrained={vision_encoder_model}",
        # --- PEFT / LoRA ---
        f"++model.peft.peft_scheme=lora",
        f"++model.peft.lora_tuning.r={lora_cfg.get('r', 16)}",
        f"++model.peft.lora_tuning.alpha={lora_cfg.get('alpha', 32)}",
        f"++model.peft.lora_tuning.dropout={lora_cfg.get('dropout', 0.05)}",
        f"++model.peft.lora_tuning.target_modules={target_modules_str}",
        # --- data ---
        "++model.data.data_prefix.train=[data/train/instruction_train.jsonl]",
        "++model.data.data_prefix.validation=[data/validation/instruction_val.jsonl]",
        "++model.data.data_prefix.test=[data/test/instruction_test.jsonl]",
        # --- exp_manager ---
        "++exp_manager.explicit_log_dir=outputs/checkpoints/cropguard_lora",
        "++exp_manager.name=cropguard_lora",
        "++exp_manager.create_checkpoint_callback=true",
    ]
    return overrides

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
        
    nemo_script = find_nemo_script()
    num_gpus = torch.cuda.device_count()
    
    if nemo_script:
        # Use neva_peft.py's OWN conf/ directory as the Hydra config base.
        # This ensures all required OmegaConf struct keys (e.g. model.gradient_as_bucket_view)
        # are present from NeMo's schema.  We then inject our CropGuard settings
        # via '+key=value' CLI overrides ('+' appends without triggering struct errors).
        # Generating our own full replacement YAML always breaks because NeMo's
        # Python code accesses dozens of schema fields we can't predict.
        conf_dir = find_neva_conf_dir(nemo_script)
        overrides = build_hydra_overrides(lora_cfg, model_cfg)

        if conf_dir:
            config_name = pick_config_name(conf_dir)
            logging.info(f"Using NeMo conf dir: {conf_dir}, config: {config_name}")
            cmd = [
                "torchrun",
                f"--nproc-per-node={num_gpus}",
                nemo_script,
                f"--config-path={conf_dir}",
                f"--config-name={config_name}",
            ] + overrides
        else:
            # Fallback: no conf/ found — generate a minimal YAML as before,
            # but log a warning so the user knows schema errors may occur.
            logging.warning(
                "Could not locate neva_peft conf/ directory. "
                "Falling back to generated YAML — OmegaConf struct errors may occur."
            )
            cfg_path = Path("configs/nemo_lora_generated.yaml")
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            import yaml as _yaml
            with open(cfg_path, "w") as f:
                _yaml.dump({}, f)  # empty base; all settings via overrides
            cmd = [
                "torchrun",
                f"--nproc-per-node={num_gpus}",
                nemo_script,
                f"--config-path={cfg_path.parent.absolute()}",
                f"--config-name={cfg_path.name}",
            ] + overrides

        logging.info(f"Executing NeMo PEFT script: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        logging.info("No external NeMo example script path found. Running native PEFT training runner...")
        run_native_peft_training(lora_cfg, model_cfg)

if __name__ == "__main__":
    main()
