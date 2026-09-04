import os
import json
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_mapping(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def normalize_label(crop, disease, mapping):
    crop_lower = crop.lower()
    
    if crop_lower in mapping:
        crop_mapping = mapping[crop_lower]
        disease_lower = disease.lower()
        
        for canonical_disease, info in crop_mapping.items():
            aliases = [a.lower() for a in info.get('aliases', [])]
            if disease_lower == canonical_disease.lower() or disease_lower in aliases:
                return crop_lower.capitalize(), canonical_disease
                
    return crop, "UNKNOWN"

def process_file(jsonl_path, mapping, out_path):
    normalized_data = []
    
    if not jsonl_path.exists():
        logging.warning(f"File {jsonl_path} does not exist.")
        return
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            crop, disease = record['crop'], record['disease']
            
            norm_crop, norm_disease = normalize_label(crop, disease, mapping)
            
            if norm_disease == "UNKNOWN":
                logging.warning(f"Ambiguous mapping for {crop} / {disease} -> {norm_disease}")
                
            record['crop'] = norm_crop
            record['disease'] = norm_disease
            normalized_data.append(record)
            
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in normalized_data:
            f.write(json.dumps(item) + "\n")
            
    logging.info(f"Normalized {len(normalized_data)} records to {out_path}")

def main():
    mapping_path = Path("configs/label_mapping.yaml")
    metadata_dir = Path("data/metadata")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = load_mapping(mapping_path)
    
    for dataset in ["plantvillage", "plantdoc", "rice1426"]:
        jsonl_path = metadata_dir / f"{dataset}.jsonl"
        out_path = out_dir / f"{dataset}_normalized.jsonl"
        logging.info(f"Normalizing labels for {dataset}")
        process_file(jsonl_path, mapping, out_path)

if __name__ == "__main__":
    main()
