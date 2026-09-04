import json
import logging
from pathlib import Path
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_task_a(record):
    return {
        "image": record['image_path'],
        "instruction": "Identify the crop.",
        "response": record['crop']
    }

def create_task_b(record):
    return {
        "image": record['image_path'],
        "instruction": "Identify the crop and disease.",
        "response": f"Crop: {record['crop']}\nDisease: {record['disease']}"
    }

def create_task_d(record):
    # Structured JSON
    structured_res = {
        "crop": record['crop'],
        "disease": record['disease'],
        "visual_symptoms": []
    }
    return {
        "image": record['image_path'],
        "instruction": "Output the diagnosis as structured JSON.",
        "response": json.dumps(structured_res)
    }

def generate_multimodal_dataset(split_name, input_file, out_file):
    if not input_file.exists():
        logging.warning(f"File {input_file} does not exist.")
        return
        
    instructions = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            
            # For each image, randomly select a task or create multiple entries
            task_fns = [create_task_a, create_task_b, create_task_d]
            for task_fn in task_fns:
                # In Megatron-Energon / NeMo format, typically we format as standard conversational dict
                task_data = task_fn(record)
                nemo_format = {
                    "conversations": [
                        {"from": "user", "value": f"<image>\n{task_data['instruction']}"},
                        {"from": "assistant", "value": task_data['response']}
                    ],
                    "image": task_data['image']
                }
                instructions.append(nemo_format)
                
    # Shuffle instructions
    random.shuffle(instructions)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for r in instructions:
            f.write(json.dumps(r) + "\n")
            
    logging.info(f"Generated {len(instructions)} multimodal instructions for {split_name} -> {out_file}")

def main():
    splits = {
        "train": (Path("data/train/train.jsonl"), Path("data/train/instruction_train.jsonl")),
        "validation": (Path("data/validation/val.jsonl"), Path("data/validation/instruction_val.jsonl")),
        "test": (Path("data/test/test.jsonl"), Path("data/test/instruction_test.jsonl"))
    }
    
    for split_name, (in_path, out_path) in splits.items():
        generate_multimodal_dataset(split_name, in_path, out_path)

if __name__ == "__main__":
    main()
