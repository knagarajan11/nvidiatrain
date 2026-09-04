import argparse
import json
import logging
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def predict_batch(input_dir, output_dir, model_path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return
        
    images = list(input_dir.glob("*.*"))
    predictions = []
    
    for img in images:
        if img.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # Mocking inference
        pred = {
            "image_path": str(img.absolute()),
            "crop": "Tomato",
            "disease": "Early Blight",
            "model": "cropguard_lora_v001"
        }
        predictions.append(pred)
        
    # Generate JSONL
    with open(output_dir / "predictions.jsonl", 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
            
    # Generate CSV
    if predictions:
        keys = predictions[0].keys()
        with open(output_dir / "predictions.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(predictions)
            
    logging.info(f"Processed {len(predictions)} images. Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run batch inference")
    parser.add_argument("--input", type=str, required=True, help="Path to directory containing test images")
    parser.add_argument("--output", type=str, default="outputs/predictions/", help="Output directory for predictions")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/cropguard_lora", help="Path to model")
    args = parser.parse_args()
    
    predict_batch(args.input, args.output, args.model)

if __name__ == "__main__":
    main()
