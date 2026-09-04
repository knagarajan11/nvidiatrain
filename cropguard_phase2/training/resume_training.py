import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Resume training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt or .nemo to resume from")
    args = parser.parse_args()
    
    logging.info(f"Resuming training from checkpoint: {args.checkpoint}")
    logging.info("When using NeMo, ensure exp_manager.resume_if_exists=True and exp_manager.resume_from_checkpoint is set in the hydra config.")

if __name__ == "__main__":
    main()
