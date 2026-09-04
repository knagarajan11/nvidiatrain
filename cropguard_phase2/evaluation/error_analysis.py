import logging
import json
import csv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_errors(predictions, ground_truths, out_csv):
    errors = []
    
    for p, gt in zip(predictions, ground_truths):
        p_crop = p.get('crop', 'Unknown')
        p_disease = p.get('disease', 'Unknown')
        
        gt_crop = gt.get('crop', 'Unknown')
        gt_disease = gt.get('disease', 'Unknown')
        
        if p_crop != gt_crop or p_disease != gt_disease:
            errors.append({
                "image_path": gt.get("image_path", ""),
                "true_crop": gt_crop,
                "true_disease": gt_disease,
                "pred_crop": p_crop,
                "pred_disease": p_disease,
                "error_type": "wrong_crop" if p_crop != gt_crop else "wrong_disease"
            })
            
    if errors:
        keys = errors[0].keys()
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(errors)
            
    logging.info(f"Identified {len(errors)} errors. Saved to {out_csv}")

def main():
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "error_analysis.csv"
    
    # Mock data for demonstration since we don't have actual model preds here
    preds = [{"crop": "Tomato", "disease": "Early Blight"}]
    gts = [{"crop": "Tomato", "disease": "Late Blight", "image_path": "mock.jpg"}]
    
    analyze_errors(preds, gts, out_csv)

if __name__ == "__main__":
    main()
