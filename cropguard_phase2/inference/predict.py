import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def predict_single(image_path, model_path, output_json=False):
    # This is a stub for the actual model invocation
    # In a real environment, you'd load the NeMo VLM model or the HF base + LoRA peft
    # and run inference.
    
    if not Path(image_path).exists():
        logging.error(f"Image not found: {image_path}")
        return
        
    # Mocking prediction for demonstration
    prediction = {
        "crop": "Tomato",
        "disease": "Early Blight",
        "model": "cropguard_lora_v001"
    }
    
    if output_json:
        print(json.dumps(prediction))
    else:
        print(f"Crop: {prediction['crop']}")
        print(f"Disease: {prediction['disease']}")

def main():
    parser = argparse.ArgumentParser(description="Run single image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/cropguard_lora", help="Path to the trained model/adapter")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()
    
    predict_single(args.image, args.model, args.json)

if __name__ == "__main__":
    main()
