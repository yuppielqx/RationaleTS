import os
import json
import pickle
import logging
import re
import pandas as pd
from tqdm import tqdm
import argparse
import concurrent.futures
import numpy as np
import warnings

from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from agents import ImageOnlyPredictionAgent
from utils import encode_image_to_base64

warnings.filterwarnings("ignore")

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- Global Config ---
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
    "power": {
        0: "Avg. power will not be higher",
        1: "Avg. power will be higher"
    },
    "traffic": {
        0: "Occupancy decreases by >2",
        1: "Occupancy changes within [-2, 2]",
        2: "Occupancy increases by >2"
    }
}

def evaluate_results(results_df: pd.DataFrame, LABEL_MEANINGS: dict, dataset_name: str):
    """Calculates and prints performance metrics for the predictions."""
    invalid_predictions = results_df[results_df['prediction'] == -1]
    if not invalid_predictions.empty:
        logging.warning(
            f"Found {len(invalid_predictions)} samples with prediction value -1. Changing them to 1 (neutral).")
        results_df['prediction'] = results_df['prediction'].replace(-1, 1)

    y_true = results_df['true_label']
    y_pred = results_df['prediction']
    classes = sorted(y_true.unique())
    num_classes = len(classes)

    logging.info("\n--- Performance Evaluation ---")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    logging.info(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    logging.info(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")

    if num_classes > 1:
        y_true_binarized = label_binarize(y_true, classes=classes)
        y_pred_binarized = label_binarize(y_pred, classes=classes)

        # Ensure y_pred_binarized has the correct number of columns, especially for binary cases
        if y_pred_binarized.shape[1] != num_classes:
            # This can happen if not all classes are present in y_pred
            # We rebuild the one-hot encoding against the full class list
            y_pred_binarized = label_binarize(y_pred, classes=sorted(ALL_LABEL_MEANINGS[dataset_name].keys()))

        # Check again, if still not matching, it's a more complex issue.
        if y_true_binarized.shape[1] == y_pred_binarized.shape[1]:
            auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
            auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
            logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
            logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")
        else:
            logging.warning("Could not compute AUROC. Shape mismatch between true and predicted labels after binarization.")


    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[LABEL_MEANINGS[i] for i in sorted(LABEL_MEANINGS.keys())], zero_division=0))

def parse_json_from_string(text: str) -> dict:
    """Extracts and parses a JSON object from a string that might contain extra text."""
    match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
    if match:
        json_str = match.group(1) if match.group(1) else match.group(2)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error(f"Error: Found a JSON-like block, but failed to parse it:\n{json_str}")
            return None
    logging.error("Error: No valid JSON block found in the response.")
    return None

def process_sample(image_id, all_labels, direct_prediction_agent, IMAGE_DIR):
    """
    Processes a single test sample for direct VLM prediction.
    """
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    # Directly call the prediction agent, no retrieval
    logging.info(f"Generating direct prediction for image_id: {image_id}")
    response_str = direct_prediction_agent.execute(image_path=image_path)
    
    result = parse_json_from_string(response_str)
    if not result:
        result = {"prediction": -1, "reasoning": "Direct VLM prediction failed."}

    # Record results
    true_label = all_labels[image_id]
    result['id'] = image_id
    result['true_label'] = int(true_label)

    return result

def main(args):
    """
    Main function to run direct VLM predictions on the test set.
    """
    # --- 0. Set up config based on args ---
    DATASET_NAME = args.dataset_name
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    RESULTS_DIR = f"rag_results/{DATASET_NAME}"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    # --- 1. Create output directory ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. Load data and identify test set ---
    logging.info("Loading data and splitting into test set...")
    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    train_size = int(len(all_labels) * args.train_split)
    test_indices = range(train_size, len(all_labels), 1)
    logging.info(f"Total samples: {len(all_labels)}, Test samples: {len(test_indices)}")

    # --- 3. Initialize Agent ---
    direct_prediction_agent = ImageOnlyPredictionAgent(model_name=args.prediction_model, dataset_name=DATASET_NAME, image_url_format='dict')

    # --- 4. Process test data in parallel ---
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(
                process_sample,
                image_id, all_labels,
                direct_prediction_agent,
                IMAGE_DIR
            ): image_id for image_id in test_indices
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(test_indices), desc="Running Direct VLM Predictions"):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
            except Exception as exc:
                image_id = future_to_id[future]
                logging.error(f"Image ID {image_id} generated an exception: {exc}")

    # --- 5. Save and evaluate results ---
    if not final_results:
        logging.warning("No results were generated. Exiting.")
        return
        
    results_df = pd.DataFrame(final_results)
    output_path = os.path.join(RESULTS_DIR, args.prediction_model + "_" + args.output_file)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Direct VLM prediction results saved to {output_path}")

    evaluate_results(results_df, LABEL_MEANINGS, DATASET_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run direct VLM predictions on a time-series dataset.")
    parser.add_argument(
        '--dataset_name', type=str, default='power', choices=ALL_LABEL_MEANINGS.keys(),
        help='Name of the dataset to process.'
    )
    parser.add_argument(
        '--prediction_model', type=str, default='qwen-vl-max',
        help='Name of the VLM for direct prediction.'
    )
    parser.add_argument(
        '--train_split', type=float, default=0.8,
        help='Fraction of the data used for training (to determine the test set start).'
    )
    parser.add_argument(
        '--max_workers', type=int, default=8,
        help='Maximum number of threads for parallel processing.'
    )
    parser.add_argument(
        '--output_file', type=str, default='vlm_direct_predictions.csv',
        help='Name of the output CSV file for saving results.'
    )
    args = parser.parse_args()
    main(args)