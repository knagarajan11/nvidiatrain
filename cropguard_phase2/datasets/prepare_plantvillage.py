import os
import json
import csv
import logging
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_directory(raw_dir, source_name):
    metadata = []
    
    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        # Assuming PlantVillage format like "Tomato_Early_blight" or "Tomato__healthy"
        parts = class_name.replace("__", "_").split("_")
        crop = parts[0]
        disease = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"
        health_status = "Healthy" if "healthy" in class_name.lower() else "Diseased"
        
        for img_path in class_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            sample_id = str(uuid.uuid4())
            metadata.append({
                "sample_id": sample_id,
                "image_path": str(img_path.absolute()),
                "crop": crop,
                "disease": disease,
                "health_status": health_status,
                "source": source_name,
                "original_label": class_name
            })
            
    return metadata

def save_metadata(metadata, out_csv, out_jsonl):
    if not metadata:
        logging.warning("No metadata to save!")
        return
        
    # Save CSV
    keys = metadata[0].keys()
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metadata)
        
    # Save JSONL
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
            
    logging.info(f"Saved {len(metadata)} records to {out_csv} and {out_jsonl}")

def main():
    raw_dir = Path(os.getenv("PLANTVILLAGE_DIR", "data/raw/plantvillage"))
    out_dir = Path("data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Preparing PlantVillage from {raw_dir}")
    metadata = process_directory(raw_dir, "PlantVillage")
    
    save_metadata(metadata, out_dir / "plantvillage.csv", out_dir / "plantvillage.jsonl")

if __name__ == "__main__":
    main()
