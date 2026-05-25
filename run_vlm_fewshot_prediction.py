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
import random

from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from agents import ImageFewShotPredictionAgent
from utils import encode_image_to_base64

warnings.filterwarnings("ignore")

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# --- 全局配置 ---
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
    """计算并打印预测结果的性能指标。"""
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

        if y_pred_binarized.shape[1] != num_classes:
            y_pred_binarized = label_binarize(y_pred, classes=sorted(ALL_LABEL_MEANINGS[dataset_name].keys()))

        if y_true_binarized.shape[1] == y_pred_binarized.shape[1]:
            auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
            auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
            logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
            logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")
        else:
            logging.warning("Could not compute AUROC. Shape mismatch between true and predicted labels after binarization.")

    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[LABEL_MEANINGS[i] for i in sorted(LABEL_MEANINGS.keys())], zero_division=0))

def parse_json_from_string(text: str) -> dict:
    """从可能包含额外文本的字符串中提取并解析JSON对象。"""
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

def process_sample(image_id, all_labels, fewshot_agent, IMAGE_DIR, train_indices, num_shots, LABEL_MEANINGS):
    """
    使用 Few-shot ICL 方法处理单个测试样本。
    """
    test_image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(test_image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    # --- 1. 从训练集中随机选择 few-shot 样本 ---
    shot_indices = random.sample(train_indices, num_shots)
    few_shot_examples = []
    for shot_id in shot_indices:
        shot_image_path = os.path.join(IMAGE_DIR, f"{shot_id}.png")
        if os.path.exists(shot_image_path):
            shot_label = all_labels[shot_id]
            shot_label_meaning = LABEL_MEANINGS.get(shot_label, "Unknown")
            few_shot_examples.append({
                "id": shot_id,
                "image_path": shot_image_path,
                "label_meaning": shot_label_meaning
            })
        else:
            logging.warning(f"Few-shot example image {shot_id}.png not found, skipping this shot.")

    if not few_shot_examples:
        logging.error(f"Could not find any valid few-shot examples for test sample {image_id}. Skipping.")
        return None

    # --- 2. 执行 few-shot 预测代理 ---
    logging.info(f"Generating few-shot prediction for image_id: {image_id} using {len(few_shot_examples)} shots.")
    response_str = fewshot_agent.execute(
        test_image_path=test_image_path,
        examples=few_shot_examples
    )

    # --- 3. 解析并记录结果 ---
    result = parse_json_from_string(response_str)
    if not result:
        result = {"prediction": -1, "reasoning": "Few-shot prediction failed."}

    true_label = all_labels[image_id]
    result['id'] = image_id
    result['true_label'] = int(true_label)
    result['few_shot_example_ids'] = [ex['id'] for ex in few_shot_examples]

    return result

def main(args):
    """
    主函数，对测试集执行 Few-shot ICL 预测。
    """
    # --- 0. 根据参数设置配置 ---
    DATASET_NAME = args.dataset_name
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    RESULTS_DIR = f"rag_results/{DATASET_NAME}"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    # --- 1. 创建输出目录 ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. 加载数据并划分训练/测试集索引 ---
    logging.info("Loading data and splitting into train/test sets...")
    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    train_size = int(len(all_labels) * args.train_split)
    train_indices = list(range(0, train_size))
    test_indices = list(range(train_size, len(all_labels)))
    logging.info(f"Total samples: {len(all_labels)}, Train: {len(train_indices)}, Test: {len(test_indices)}")

    # --- 3. 初始化 Agent ---
    fewshot_agent = ImageFewShotPredictionAgent(
        model_name=args.prediction_model,
        dataset_name=DATASET_NAME,
        image_url_format='dict'
    )

    # --- 4. 并行处理测试数据 ---
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(
                process_sample,
                image_id, all_labels, fewshot_agent, IMAGE_DIR,
                train_indices, args.num_shots, LABEL_MEANINGS
            ): image_id for image_id in test_indices
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(test_indices), desc="Running Few-Shot Predictions"):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
            except Exception as exc:
                image_id = future_to_id[future]
                logging.error(f"Image ID {image_id} generated an exception: {exc}")

    # --- 5. 保存和评估结果 ---
    if not final_results:
        logging.warning("No results were generated. Exiting.")
        return

    results_df = pd.DataFrame(final_results)
    output_path = os.path.join(RESULTS_DIR, args.prediction_model + "_" + args.output_file)
    results_df.to_csv(output_path, index=False)
    logging.info(f"Few-shot prediction results saved to {output_path}")

    evaluate_results(results_df, LABEL_MEANINGS, DATASET_NAME)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Few-shot VLM predictions on a time-series dataset.")
    parser.add_argument(
        '--dataset_name', type=str, default='power', choices=ALL_LABEL_MEANINGS.keys(),
        help='Name of the dataset to process.'
    )
    parser.add_argument(
        '--prediction_model', type=str, default='gpt-4o-mini',
        help='Name of the VLM for few-shot prediction.'
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
        '--num_shots', type=int, default=5,
        help='Number of few-shot examples to use in the prompt.'
    )
    parser.add_argument(
        '--output_file', type=str, default='vlm_fewshot_predictions.csv',
        help='Name of the output CSV file for saving results.'
    )
    args = parser.parse_args()
    main(args)