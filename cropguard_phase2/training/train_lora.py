import os
import sys
import json
import subprocess
import yaml
import logging
from pathlib import Path
from typing import Optional
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_env_file(env_path: str = ".env") -> dict:
    """
    Parse a .env file and return key→value dict.
    Searches in priority order:
      1. The explicit env_path (relative to CWD)
      2. Same directory as this script file
      3. Parent directory of this script (project root)
    """
    search_paths = [
        Path(env_path),                              # CWD-relative (e.g. ./env)
        Path(__file__).parent / env_path,            # script dir
        Path(__file__).parent.parent / env_path,     # project root
    ]
    for p in search_paths:
        if p.exists():
            logging.info(f"Loading env from: {p.resolve()}")
            env = {}
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        env[k.strip()] = v.strip().strip('"').strip("'")
            return env
    return {}


def get_hf_token() -> Optional[str]:
    """
    Resolve HuggingFace token in priority order:
      1. HF_TOKEN env var (already set in shell)
      2. HUGGING_FACE_HUB_TOKEN env var
      3. .env file in the current working directory
      4. None (unauthenticated — will fail for gated models)
    """
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        env_vars = load_env_file(".env")
        token = env_vars.get("HF_TOKEN") or env_vars.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        # Also export to env so child processes (torchrun) inherit it
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        logging.info("HF_TOKEN loaded successfully.")
    else:
        logging.warning(
            "HF_TOKEN not found. Gated models (e.g. Nemotron-Nano-12B-v2-VL) "
            "require authentication. Set HF_TOKEN in your .env file or shell."
        )
    return token or None




def _fix_broken_local_transformers():
    """
    Detect and remove a user-local transformers 5.x that was partially installed
    by a previous pip attempt but is incompatible with the container's torch 2.4.

    Transformers 5.x requires torch >= 2.5.  When it finds torch 2.4 it prints:
      [transformers] Disabling PyTorch because PyTorch >= 2.5 is required
    and makes AutoModel* classes unavailable.

    Fix: remove ~/.local site-packages from sys.path, purge cached modules,
    and reimport from the container's system install (4.46.x).
    No pip or internet access required.
    """
    import sys
    import importlib

    try:
        import transformers as _tf
        major = int(_tf.__version__.split(".")[0])
        if major < 5:
            return  # system version is fine, nothing to do

        logging.warning(
            f"Found user-local transformers {_tf.__version__} (requires torch>=2.5, "
            "container has torch 2.4 — PyTorch will be disabled). "
            "Removing from sys.path and falling back to container's system install ..."
        )

        # Strip user-local paths from sys.path
        import site
        user_site = site.getusersitepackages()          # e.g. ~/.local/lib/python3.10/site-packages
        home_local = str(Path.home() / ".local" / "lib")
        sys.path = [
            p for p in sys.path
            if p != user_site and not p.startswith(home_local)
        ]
        logging.info(f"Stripped user-local site-packages: {user_site}")

        # Purge all cached transformers (and peft) modules so reimport is clean
        stale = [k for k in sys.modules if k.startswith(("transformers", "peft"))]
        for k in stale:
            del sys.modules[k]

        # Reimport — now resolves from the system/container install
        import transformers as _tf2
        logging.info(f"Reloaded transformers from system: {_tf2.__version__}")

    except Exception as exc:
        logging.warning(f"_fix_broken_local_transformers: {exc}")


def _patch_transformers_compat():
    """
    Inject compatibility stubs into transformers.processing_utils for symbols
    added in transformers 4.48+ that are absent in the NeMo container's 4.46.x.

    The Nemotron VL model's dynamic processing.py does:
      from transformers.processing_utils import (
          ImagesKwargs, MultiModalData, ProcessingKwargs,
          ProcessorMixin, Unpack, VideosKwargs
      )
    This patch adds stub TypedDicts so that import succeeds without any
    network access or pip upgrade.
    """
    import transformers.processing_utils as _pu
    import transformers

    ver = transformers.__version__
    logging.info(f"transformers version in container: {ver}")

    try:
        from typing import TypedDict
    except ImportError:
        from typing_extensions import TypedDict

    # Unpack: in typing since 3.11, else typing_extensions
    if not hasattr(_pu, "Unpack"):
        try:
            from typing import Unpack as _Unpack
        except ImportError:
            try:
                from typing_extensions import Unpack as _Unpack
            except ImportError:
                from typing import Any as _Unpack  # last resort stub
        _pu.Unpack = _Unpack
        logging.info("  patched: Unpack")

    # Stub TypedDicts for the remaining missing symbols
    _stubs = ["ImagesKwargs", "VideosKwargs", "ProcessingKwargs", "MultiModalData"]
    for _name in _stubs:
        if not hasattr(_pu, _name):
            _cls = type(_name, (dict,), {"__class_getitem__": classmethod(lambda cls, x: cls)})
            setattr(_pu, _name, _cls)
            logging.info(f"  patched: {_name}")


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# NeMo path helpers (only used when a local .nemo file exists)
# ─────────────────────────────────────────────────────────────────────────────

def find_neva_conf_dir(nemo_script):
    """
    Return the conf/ directory that ships alongside neva_peft.py so that NeMo's
    full OmegaConf struct schema is used as the Hydra base config.
    """
    script_dir = Path(nemo_script).parent
    for candidate in [script_dir / "conf", script_dir.parent / "conf"]:
        if candidate.exists():
            return str(candidate)
    return None


def pick_config_name(conf_dir):
    """
    Choose the best Hydra config file from conf_dir.
    Priority: files with 'peft' > generic 'neva' (not mixtral/llama) > first file.
    """
    conf_files = sorted(Path(conf_dir).glob("*.yaml"))
    if not conf_files:
        return "neva_peft"
    for f in conf_files:
        if "peft" in f.stem.lower():
            return f.stem
    skip_keywords = {"mixtral", "llama", "mistral", "falcon", "mamba"}
    for f in conf_files:
        stem_lower = f.stem.lower()
        if "neva" in stem_lower and not any(kw in stem_lower for kw in skip_keywords):
            return f.stem
    return conf_files[0].stem


def build_hydra_overrides(lora_cfg, model_cfg, nemo_path: str):
    """
    Build Hydra '++key=value' CLI overrides to inject CropGuard settings into
    neva_peft.py's OmegaConf struct.  Only called when a local .nemo file exists.

    '++' = override if key exists, append if it doesn't — safe for both cases.
    See: https://hydra.cc/docs/advanced/override_grammar/basic/#force-add

    NOTE: Do NOT include trainer.strategy — MegatronLMPPTrainerBuilder builds its
    own MegatronStrategy and passes it explicitly to Trainer(), so having it in
    cfg.trainer too raises: TypeError: multiple values for keyword argument 'strategy'
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    target_modules_str = "[" + ",".join(target_modules) + "]"
    vision_encoder_hf = model_cfg.get(
        "vision_encoder_hf_id",
        "google/siglip-so400m-patch14-384"  # SigLIP used by Nemotron-Nano VL
    )

    return [
        # --- trainer ---
        f"++trainer.num_nodes=1",
        f"++trainer.devices={num_gpus}",
        f"++trainer.accelerator=gpu",
        f"++trainer.precision=bf16",
        f"++trainer.max_epochs={lora_cfg.get('epochs', 3)}",
        # epoch-based validation so val_check_interval=1.0 (float) is valid
        f"++trainer.check_val_every_n_epoch=1",
        f"++trainer.val_check_interval=1.0",
        # --- model ---
        f"++model.restore_from_path={nemo_path}",
        f"++model.micro_batch_size={lora_cfg.get('batch_size', 4)}",
        f"++model.mm_cfg.vision_encoder.from_pretrained={vision_encoder_hf}",
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


# ─────────────────────────────────────────────────────────────────────────────
# Native HuggingFace PEFT training (used for HF-only models like Nemotron VL)
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalInstructionDataset(torch.utils.data.Dataset):
    """
    Reads a JSONL file where each line is a multimodal instruction sample:
      {"image": "<path>", "conversations": [{"from": "human", "value": "..."}, ...]}
    Tokenises text and returns pixel_values + input_ids for vision-language training.
    """

    def __init__(self, jsonl_path: str, processor, max_length: int = 512):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Build text from conversation turns
        convs = sample.get("conversations", [])
        text_parts = []
        for turn in convs:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                text_parts.append(f"User: {value}")
            elif role in ("gpt", "assistant"):
                text_parts.append(f"Assistant: {value}")
        text = "\n".join(text_parts)

        # Load image if present
        image_path = sample.get("image", None)
        image = None
        if image_path:
            try:
                from PIL import Image
                img_p = Path(image_path)
                if not img_p.is_absolute():
                    img_p = Path("data") / img_p
                if img_p.exists():
                    image = Image.open(img_p).convert("RGB")
            except Exception:
                pass  # fall back to text-only if image load fails

        try:
            if image is not None:
                enc = self.processor(
                    text=text, images=image,
                    return_tensors="pt", truncation=True, max_length=self.max_length,
                    padding="max_length"
                )
            else:
                enc = self.processor(
                    text=text,
                    return_tensors="pt", truncation=True, max_length=self.max_length,
                    padding="max_length"
                )
        except Exception:
            enc = self.processor(
                text=text,
                return_tensors="pt", truncation=True, max_length=self.max_length,
                padding="max_length"
            )

        # Squeeze batch dim added by processor
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # Labels = input_ids (causal LM); mask padding tokens with -100
        labels = item["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        item["labels"] = labels
        return item


def run_native_peft_training(lora_cfg: dict, model_cfg: dict):
    """
    HuggingFace PEFT / LoRA fine-tuning for Nemotron-Nano-12B-v2-VL.

    Used when the model is a HuggingFace model (no local .nemo file).
    Loads the model via AutoModel, wraps it with LoRA via PEFT, then trains
    with HuggingFace Trainer on the multimodal JSONL instruction dataset.
    """
    logging.info("═" * 60)
    logging.info("CropGuard: HuggingFace PEFT / LoRA training path")
    logging.info("═" * 60)

    output_dir = Path("outputs/checkpoints/cropguard_lora")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = Path("data/train/instruction_train.jsonl")
    val_jsonl   = Path("data/validation/instruction_val.jsonl")
    if not train_jsonl.exists():
        logging.error(f"Training data not found at {train_jsonl}. Run prepare_all_datasets.sh first.")
        return

    model_id    = model_cfg.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL")
    lora_r      = lora_cfg.get("r", 16)
    lora_alpha  = lora_cfg.get("alpha", 32)
    lora_drop   = lora_cfg.get("dropout", 0.05)
    target_mods = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    epochs      = lora_cfg.get("epochs", 3)
    batch_size  = lora_cfg.get("batch_size", 2)
    max_length  = model_cfg.get("max_seq_length", 512)

    logging.info(f"Model      : {model_id}")
    logging.info(f"LoRA       : r={lora_r}, alpha={lora_alpha}, dropout={lora_drop}")
    logging.info(f"Targets    : {target_mods}")
    logging.info(f"Epochs     : {epochs}, batch={batch_size}")

    # ── fix broken user-local transformers 5.x (if present) ─────────────────
    # A prior pip upgrade attempt may have installed transformers 5.x in
    # ~/.local, which requires torch>=2.5 (container has 2.4) and disables
    # AutoModel* entirely.  Strip it from sys.path, fall back to 4.46 system.
    _fix_broken_local_transformers()

    # ── compatibility patch for older transformers in NeMo container ─────────
    # The NeMo container ships with transformers 4.46.x, but
    # nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16's processing.py requires
    # symbols added in transformers 4.48+ (MultiModalData, etc.).
    # The DGX has no outbound PyPI access so we cannot upgrade via pip.
    # Instead we inject stub objects directly into transformers.processing_utils
    # so the model's dynamic processing.py module can import successfully.
    _patch_transformers_compat()

    # ── imports ──────────────────────────────────────────────────────────────
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoProcessor, AutoModelForVision2Seq,
            TrainingArguments, Trainer, BitsAndBytesConfig
        )
    except ImportError:
        logging.warning("peft/transformers not fully available; check container deps.")
        raise

    # ── authenticate with HuggingFace Hub ───────────────────────────────────
    # Nemotron-Nano-12B-v2-VL is a gated model — HF_TOKEN is required.
    # Token is loaded from .env or the environment.
    hf_token = get_hf_token()
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
        except Exception:
            pass  # login() may not be available in old hub versions; token passed directly

    # ── load processor ───────────────────────────────────────────────────────
    logging.info(f"Loading processor from {model_id} …")
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True,
        token=hf_token
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── load model in 4-bit or bf16 ─────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    logging.info(f"Loading model on {num_gpus} GPU(s) …")

    load_kwargs = dict(
        pretrained_model_name_or_path=model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    # Use 4-bit quant if only 1 GPU to save VRAM; multi-GPU uses bf16 directly
    if num_gpus == 1:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_cfg
            load_kwargs.pop("torch_dtype", None)
            logging.info("Using 4-bit QLoRA (single GPU — conserving VRAM)")
        except Exception:
            logging.info("BitsAndBytes 4-bit not available, using bf16")

    model = AutoModelForVision2Seq.from_pretrained(**load_kwargs)
    model.config.use_cache = False  # required for gradient checkpointing

    # ── apply LoRA ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_drop,
        target_modules=target_mods,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── datasets ─────────────────────────────────────────────────────────────
    logging.info("Building datasets …")
    train_dataset = MultimodalInstructionDataset(str(train_jsonl), processor, max_length)
    eval_dataset  = (
        MultimodalInstructionDataset(str(val_jsonl), processor, max_length)
        if val_jsonl.exists() else None
    )
    logging.info(f"Train samples : {len(train_dataset)}")
    if eval_dataset:
        logging.info(f"Val samples   : {len(eval_dataset)}")

    # ── training arguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="eval_loss" if eval_dataset else None,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
        remove_unused_columns=False,
    )

    # ── trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logging.info("Starting LoRA fine-tuning …")
    trainer.train()

    # ── save adapter ─────────────────────────────────────────────────────────
    logging.info(f"Saving LoRA adapter to {output_dir} …")
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logging.info("✓ Training complete. Adapter saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        logging.error("CUDA is not available. GPU is required for training.")
        return

    lora_cfg  = load_config("configs/lora.yaml").get("lora", {})
    model_cfg = load_config("configs/model.yaml")

    if not lora_cfg.get("enabled", True):
        logging.info("LoRA training is disabled in config.")
        return

    model_name = model_cfg.get("model_name", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL")

    # ── routing decision ─────────────────────────────────────────────────────
    # neva_peft.py requires a local .nemo file via model.restore_from_path.
    # Nemotron-Nano-12B-v2-VL is a HuggingFace model — no .nemo file exists.
    # Attempting to use neva_peft.py with a HF repo ID as the path fails with:
    #   OSError: Incorrect path_or_model_id: 'nvidia/...'
    # So we check for a local .nemo file first; if absent, go native HF PEFT.
    nemo_candidates = [
        Path(f"{model_name}.nemo"),
        Path(f"models/{Path(model_name).name}.nemo"),
        Path(f"/workspace/models/{Path(model_name).name}.nemo"),
    ]
    local_nemo = next((p for p in nemo_candidates if p.exists()), None)

    if local_nemo:
        logging.info(f"Found local .nemo file: {local_nemo} — using NeMo neva_peft path.")
        _run_nemo_peft(lora_cfg, model_cfg, str(local_nemo))
    else:
        logging.info(
            f"No local .nemo file found for '{model_name}'. "
            "Using HuggingFace PEFT path (downloads model from Hub if needed)."
        )
        run_native_peft_training(lora_cfg, model_cfg)


def _run_nemo_peft(lora_cfg: dict, model_cfg: dict, nemo_path: str):
    """Launch neva_peft.py via torchrun using NeMo's own conf/ schema."""
    nemo_script = find_nemo_script()
    num_gpus    = torch.cuda.device_count()

    if not nemo_script:
        logging.warning("No NeMo PEFT script found. Falling back to native HF PEFT.")
        run_native_peft_training(lora_cfg, model_cfg)
        return

    conf_dir  = find_neva_conf_dir(nemo_script)
    overrides = build_hydra_overrides(lora_cfg, model_cfg, nemo_path)

    if conf_dir:
        config_name = pick_config_name(conf_dir)
        logging.info(f"NeMo conf dir: {conf_dir}, config: {config_name}")
        cmd = [
            "torchrun", f"--nproc-per-node={num_gpus}",
            nemo_script,
            f"--config-path={conf_dir}",
            f"--config-name={config_name}",
        ] + overrides
    else:
        logging.warning("NeMo conf/ dir not found. OmegaConf struct errors may occur.")
        cfg_path = Path("configs/nemo_lora_generated.yaml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, "w") as f:
            yaml.dump({}, f)
        cmd = [
            "torchrun", f"--nproc-per-node={num_gpus}",
            nemo_script,
            f"--config-path={cfg_path.parent.absolute()}",
            f"--config-name={cfg_path.name}",
        ] + overrides

    logging.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
