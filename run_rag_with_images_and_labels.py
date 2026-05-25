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
import torch
import warnings
from tabpfn import TabPFNClassifier
import huggingface_hub
from tabpfn_extensions.embedding import TabPFNEmbedding
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from agents import AnalysisAgent, RAGWithImagesAndLabelsAgent
from utils import EmbeddingHandler
from sklearn.metrics.pairwise import cosine_similarity


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

def evaluate_results(results_df: pd.DataFrame, LABEL_MEANINGS: dict):
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
    logging.info(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    logging.info(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro'):.4f}")
    logging.info(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")

    if num_classes > 1:
        y_true_binarized = label_binarize(y_true, classes=classes)
        y_pred_binarized = label_binarize(y_pred, classes=classes)

        if y_pred_binarized.shape[1] != num_classes:
            y_pred_binarized = np.eye(num_classes)[y_pred.astype(int)]

        auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
        auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
        logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
        logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")

    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[LABEL_MEANINGS[i] for i in sorted(LABEL_MEANINGS.keys())], zero_division=0))

def parse_json_from_string(text: str) -> dict:
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

def min_max_scale(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

def process_sample(image_id, all_labels, embedding_handler, analysis_agent, rag_prediction_agent, IMAGE_DIR, top_k, retrieval_mode='text', hybrid_alpha=0.5, all_data_values=None, train_data_embeddings=None, query_embedding_tabpfn=None, LABEL_MEANINGS=None):
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    similar_examples = []
    query_text = ""
    if retrieval_mode in ['text', 'hybrid']:
        logging.info(f"Generating initial analysis for image_id: {image_id}")
        query_text = analysis_agent.execute(image_path=image_path)

    if retrieval_mode == 'hybrid':
        # a. 计算 TabPFN 相似度
        sims_tabpfn = cosine_similarity(query_embedding_tabpfn.reshape(1, -1), train_data_embeddings)[0]
        # b. 计算 文本语义 相似度
        query_embedding_text = embedding_handler.get_embeddings([query_text])
        sims_text = cosine_similarity(query_embedding_text, embedding_handler.coarse_embeddings)[0]
        # c. 加权融合
        hybrid_scores = hybrid_alpha * min_max_scale(sims_tabpfn) + (1 - hybrid_alpha) * min_max_scale(sims_text)
        # d. 排序取 Top-K
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        similar_examples = embedding_handler.rag_metadata.iloc[top_indices].to_dict('records')
    elif retrieval_mode == 'tabpfn':
        sims_tabpfn = cosine_similarity(query_embedding_tabpfn.reshape(1, -1), train_data_embeddings)[0]
        top_indices = np.argsort(sims_tabpfn)[-top_k:][::-1]
        similar_examples = embedding_handler.rag_metadata.iloc[top_indices].to_dict('records')
    else:  # 'text' mode
        similar_examples = embedding_handler.find_similar_reasoning_paths(query_text, top_k=top_k)

    logging.info("Executing RAG prediction with example images and labels...")
    rag_response_str = rag_prediction_agent.execute(
        image_path=image_path,
        examples=similar_examples,
        LABEL_MEANINGS=LABEL_MEANINGS
    )
    rag_result = parse_json_from_string(rag_response_str)
    if not rag_result:
        rag_result = {"prediction": -1, "reasoning": "RAG prediction failed."}

    true_label = all_labels[image_id]
    rag_result['id'] = image_id
    rag_result['true_label'] = int(true_label)
    rag_result['retrieved_examples'] = [int(ex['id']) for ex in similar_examples]
    rag_result['retrieved_examples_true_labels'] = [int(ex['true_label']) for ex in similar_examples]

    return rag_result

def main(args):
    DATASET_NAME = args.dataset_name
    DATA_PATH = f"dataset/{DATASET_NAME}/data.pkl"
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    DB_PATH = f"vector_db/{DATASET_NAME}/vector_db.pkl"
    RESULTS_DIR = f"rag_results/{DATASET_NAME}"
    TRAIN_EMB_PATH = f"dataset/{DATASET_NAME}/train_embeddings.pkl"
    TEST_EMB_PATH = f"dataset/{DATASET_NAME}/test_embeddings.pkl"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    train_size = int(len(all_labels) * args.train_split)
    test_indices = range(train_size, len(all_labels), 1)

    embedding_handler = EmbeddingHandler(model_name=args.embedding_model)
    embedding_handler.load_vector_db(DB_PATH)

    all_data_values, train_data_embeddings, test_data_embeddings = None, None, None
    if args.retrieval_mode in ['tabpfn', 'hybrid']:
        # Logic to load or generate TabPFN embeddings
        if os.path.exists(TRAIN_EMB_PATH) and os.path.exists(TEST_EMB_PATH):
            logging.info(f"Loading pre-computed TabPFN embeddings...")
            with open(TRAIN_EMB_PATH, 'rb') as f: train_data_embeddings = pickle.load(f)
            with open(TEST_EMB_PATH, 'rb') as f: test_data_embeddings = pickle.load(f)
        else:
            logging.error("Pre-computed TabPFN embeddings not found. Please run run_rag_prediction.py first to generate them.")
            return

    analysis_agent = AnalysisAgent(model_name=args.analysis_model, dataset_name=DATASET_NAME, image_url_format='dict')
    # Use the new agent for this baseline
    rag_prediction_agent = RAGWithImagesAndLabelsAgent(model_name=args.prediction_model, dataset_name=DATASET_NAME, image_url_format='dict')

    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(
                process_sample,
                image_id, all_labels,
                embedding_handler,
                analysis_agent,
                rag_prediction_agent,
                IMAGE_DIR,
                args.top_k,
                args.retrieval_mode,
                args.hybrid_alpha,
                all_data_values,
                train_data_embeddings,
                test_data_embeddings[image_id - train_size] if test_data_embeddings is not None else None,
                LABEL_MEANINGS
            ): image_id for image_id in test_indices
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(test_indices), desc="Running RAG w/ Images & Labels"):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
            except Exception as exc:
                image_id = future_to_id[future]
                logging.error(f"Image ID {image_id} generated an exception: {exc}")

    results_df = pd.DataFrame(final_results)
    output_path = os.path.join(RESULTS_DIR, args.output_file)
    results_df.to_csv(output_path, index=False)
    logging.info(f"RAG with images and labels prediction results saved to {output_path}")

    evaluate_results(results_df, LABEL_MEANINGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG predictions with example images and labels.")
    # Add arguments similar to run_rag_prediction.py
    parser.add_argument('--dataset_name', type=str, default='power', choices=ALL_LABEL_MEANINGS.keys())
    parser.add_argument('--analysis_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--prediction_model', type=str, default='gpt-4o-mini')
    parser.add_argument('--embedding_model', type=str, default='text-embedding-3-large')
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--retrieval_mode', type=str, default='hybrid', choices=['text', 'tabpfn', 'hybrid'])
    parser.add_argument('--hybrid_alpha', type=float, default=0.2)
    parser.add_argument('--output_file', type=str, default='rag_with_images_and_labels_predictions.csv')
    args = parser.parse_args()
    main(args)