import os
import pandas as pd
import logging
import argparse
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global configuration ---
ALL_LABEL_MEANINGS = {
    "finance_SP": {
        0: "decrease by more than 1%",
        1: "remain neutral (i.e., between -1% and 1%)",
        2: "increase by more than 1%"
    },
    "LargeAQ": {
        0: "no heavy pollution: PM2.5 <75",
        1: "heavy pollution level: PM2.5 >=75"
    },
    # --- Power dataset label meanings ---
    "power": {
        0: "Avg. power will not be higher",
        1: "Avg. power will be higher"
    },
    # --- Traffic dataset label meanings ---
    "traffic": {
        0: "Occupancy decreases by >2",
        1: "Occupancy changes within [-2, 2]",
        2: "Occupancy increases by >2"
    }
}

def evaluate_predictions(args):
    """
    Read CSV file, handle invalid predictions, and compute performance metrics.

    Args:
        csv_path (str): Path to the prediction results CSV file.
    """
    dataset_name = args.dataset_name

    csv_path = os.path.join("rag_results", dataset_name, args.file_name)
    if not os.path.exists(csv_path):
        logging.error(f"File not found: {csv_path}")
        return

    logging.info(f"Loading prediction results from {csv_path}...")
    results_df = pd.read_csv(csv_path)

    # --- Preprocessing: change predictions of -1 to 1 (neutral) ---
    invalid_predictions = results_df[results_df['prediction'] == -1]
    if not invalid_predictions.empty:
        logging.warning(f"Found {len(invalid_predictions)} samples with prediction value -1. Changing them to 1 (neutral).")
        results_df['prediction'] = results_df['prediction'].replace(-1, 1)

    y_true = results_df['true_label']
    y_pred = results_df['prediction']
    classes = sorted(y_true.unique())
    num_classes = len(classes)

    logging.info("\n--- Performance Evaluation ---")
    logging.info(f"Total samples evaluated: {len(results_df)}")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    logging.info(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    logging.info(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    # --- AUROC calculation ---
    # Simulate probabilities from discrete predictions (one-hot encoding)
    if num_classes > 1:
        y_true_binarized = label_binarize(y_true, classes=classes)
        y_pred_binarized = label_binarize(y_pred, classes=classes)

        # Ensure y_pred_binarized has the correct number of columns even if some classes are not predicted
        if y_pred_binarized.shape[1] !=  y_pred_binarized.shape[1]:
            # Create an identity matrix as a lookup table
            eye_matrix = np.eye(num_classes)
            # Use integer indices to construct a complete one-hot encoding matrix
            y_pred_binarized = eye_matrix[y_pred.astype(int)]

        auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
        logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
        auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
        logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")

    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[ALL_LABEL_MEANINGS[dataset_name][i] for i in sorted(ALL_LABEL_MEANINGS[dataset_name].keys())], zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG prediction results from a CSV file.")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='traffic',
        choices=ALL_LABEL_MEANINGS.keys(),
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default='rag_with_images_and_labels_predictions.csv'
    )
    args = parser.parse_args()
    evaluate_predictions(args)