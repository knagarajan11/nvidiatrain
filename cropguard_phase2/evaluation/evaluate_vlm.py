import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_vlm_generative(predictions, ground_truths):
    # Evaluates structured JSON validity and exact match
    valid_json = 0
    exact_match = 0
    
    for p, gt in zip(predictions, ground_truths):
        try:
            parsed = json.loads(p)
            valid_json += 1
            if parsed.get('crop') == gt.get('crop') and parsed.get('disease') == gt.get('disease'):
                exact_match += 1
        except:
            pass
            
    total = len(predictions)
    return {
        "json_validity": valid_json / total if total > 0 else 0,
        "exact_match_accuracy": exact_match / total if total > 0 else 0
    }

if __name__ == "__main__":
    logging.info("VLM evaluation functions loaded.")
