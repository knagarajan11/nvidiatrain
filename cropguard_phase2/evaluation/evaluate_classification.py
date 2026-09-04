import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_classification(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    mac_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    w_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc,
        "macro_f1": mac_f1,
        "weighted_f1": w_f1
    }

if __name__ == "__main__":
    logging.info("Classification evaluation functions loaded.")
