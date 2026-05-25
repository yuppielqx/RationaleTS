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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize
from agents import AnalysisAgent, RAGPredictionAgent
from utils import EmbeddingHandler

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
    # --- 新增 power 数据集的标签含义 ---
    "power": {
        0: "Avg. power will not be higher",
        1: "Avg. power will be higher"
    },
    # --- 新增 traffic 数据集的标签含义 ---
    "traffic": {
        0: "Occupancy decreases by >2",
        1: "Occupancy changes within [-2, 2]",
        2: "Occupancy increases by >2"
    }
}

def evaluate_results(results_df: pd.DataFrame, LABEL_MEANINGS: dict):
    # --- 预处理：将预测值为-1的样本改为1 ---
    invalid_predictions = results_df[results_df['prediction'] == -1]
    if not invalid_predictions.empty:
        logging.warning(
            f"Found {len(invalid_predictions)} samples with prediction value -1. Changing them to 1 (neutral).")
        results_df['prediction'] = results_df['prediction'].replace(-1, 1)

    """计算并打印预测结果的性能指标。"""
    y_true = results_df['true_label']
    y_pred = results_df['prediction']
    classes = sorted(y_true.unique())
    num_classes = len(classes)

    logging.info("\n--- Performance Evaluation ---")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    logging.info(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro'):.4f}")
    logging.info(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # --- 新增 AUROC 计算逻辑 ---
    # 因为没有直接的概率输出，我们基于最终的离散预测来模拟概率（one-hot编码）
    # 这是一种计算 AUROC 的方式，尽管不如使用真实概率分数精确
    if num_classes > 1:
        y_true_binarized = label_binarize(y_true, classes=classes)
        y_pred_binarized = label_binarize(y_pred, classes=classes)

        # 确保即使某些类别没有被预测，y_pred_binarized 也有正确的列数
        if y_pred_binarized.shape[1] != y_pred_binarized.shape[1]:
            # 创建一个单位矩阵作为查找表
            eye_matrix = np.eye(num_classes)
            # 使用整数索引来构建完整的one-hot编码矩阵
            y_pred_binarized = eye_matrix[y_pred.astype(int)]

        auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
        auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
        logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
        logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")

    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[LABEL_MEANINGS[i] for i in sorted(LABEL_MEANINGS.keys())], zero_division=0))



def parse_json_from_string(text: str) -> dict:
    """
    从可能包含额外文本的字符串中提取并解析JSON对象。
    """
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
    """将分数归一化到 [0, 1] 区间，用于混合检索加权"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

def process_sample(image_id, all_labels, embedding_handler, analysis_agent, rag_prediction_agent, IMAGE_DIR, top_k, retrieval_mode='text', hybrid_alpha=0.5, all_data_values=None, train_data_embeddings=None, query_embedding_tabpfn=None):
    """
    处理单个测试样本的完整RAG流程。
    """
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    similar_examples = []

    # --- 1. 准备查询向量 (根据模式) ---
    
    # 文本查询 (Hybrid 或 Text 模式需要)
    query_text = ""
    if retrieval_mode in ['text', 'hybrid', 'worst']:
        logging.info(f"Generating initial analysis for image_id: {image_id}")
        query_text = analysis_agent.execute(image_path=image_path) # 忽略 usage 返回值

    # TabPFN 数值查询 (Hybrid 或 TabPFN 模式需要)
    # 注意：query_embedding_tabpfn 现在由 main 函数预计算并传入，
    # 确保它与 train_data_embeddings 处于同一个向量空间（经过了 Transformer 层）。
    pass

    # --- 2. 执行检索 ---
    
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
    elif retrieval_mode == 'worst':
        # a. 计算 TabPFN 相似度
        sims_tabpfn = cosine_similarity(query_embedding_tabpfn.reshape(1, -1), train_data_embeddings)[0]
        # b. 计算 文本语义 相似度
        query_embedding_text = embedding_handler.get_embeddings([query_text])
        sims_text = cosine_similarity(query_embedding_text, embedding_handler.coarse_embeddings)[0]
        # c. 加权融合
        hybrid_scores = hybrid_alpha * min_max_scale(sims_tabpfn) + (1 - hybrid_alpha) * min_max_scale(sims_text)
        # d. 排序取 Top-K
        top_indices = np.argsort(hybrid_scores)[:top_k][::-1]
        similar_examples = embedding_handler.rag_metadata.iloc[top_indices].to_dict('records')
    elif retrieval_mode == 'tabpfn':
        sims_tabpfn = cosine_similarity(query_embedding_tabpfn.reshape(1, -1), train_data_embeddings)[0]
        top_indices = np.argsort(sims_tabpfn)[-top_k:][::-1]
        similar_examples = embedding_handler.rag_metadata.iloc[top_indices].to_dict('records')
    else: # 'text' mode
        similar_examples = embedding_handler.find_similar_reasoning_paths(query_text, top_k=top_k)


    # c. 执行增强预测
    logging.info("Executing RAG-enhanced prediction...")
    rag_response_str = rag_prediction_agent.execute(
        image_path=image_path,
        examples=similar_examples
    )
    rag_result = parse_json_from_string(rag_response_str)
    if not rag_result:
        rag_result = {"prediction": 1, "reasoning": "RAG prediction failed."}

    # d. 记录结果
    true_label = all_labels[image_id]
    rag_result['id'] = image_id
    rag_result['true_label'] = int(true_label)
    rag_result['retrieved_examples'] = [int(ex['id']) for ex in similar_examples]
    rag_result['retrieved_examples_true_labels'] = [int(ex['true_label']) for ex in similar_examples]

    return rag_result


def main(args):
    """
    主函数，对测试集执行RAG增强的金融预测。
    """
    # --- 0. 根据参数动态设置配置 ---
    DATASET_NAME = args.dataset_name
    DATA_PATH = f"dataset/{DATASET_NAME}/data.pkl"
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    # 确定使用什么样式的向量数据库路径（根据 build_vector_db.py 的输出）
    DB_PATH = f"vector_db/{DATASET_NAME}/vector_db.pkl"
    RESULTS_DIR = f"rag_results/{DATASET_NAME}"
    TRAIN_EMB_PATH = f"dataset/{DATASET_NAME}/train_embeddings.pkl"
    TEST_EMB_PATH = f"dataset/{DATASET_NAME}/test_embeddings.pkl"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    # --- 1. 创建输出目录 ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 2. 加载数据和划分测试集 ---
    logging.info("Loading data and splitting into test set...")
    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    train_size = int(len(all_labels) * args.train_split)
    test_indices = range(train_size, len(all_labels), 1)
    logging.info(f"Total samples: {len(all_labels)}, Test samples: {len(test_indices)}")

    # --- 3. 加载向量数据库和初始化处理器 ---
    logging.info(f"Loading vector database from {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        logging.error("Vector database not found. Please run 'build_vector_db.py' first.")
        return

    embedding_handler = EmbeddingHandler(model_name=args.embedding_model)
    embedding_handler.load_vector_db(DB_PATH)

    # --- 3.5 准备 TabPFN 检索资源 (如果需要) ---
    tabpfn_classifier = None
    all_data_values = None
    train_data_embeddings = None
    test_data_embeddings = None

    if args.retrieval_mode in ['tabpfn', 'hybrid', 'worst']:
        # 检查是否已经存在预计算的 Embedding
        if os.path.exists(TRAIN_EMB_PATH) and os.path.exists(TEST_EMB_PATH):
            logging.info(f"Loading pre-computed TabPFN embeddings from {TRAIN_EMB_PATH} and {TEST_EMB_PATH}...")
            with open(TRAIN_EMB_PATH, 'rb') as f:
                train_data_embeddings = pickle.load(f)
            with open(TEST_EMB_PATH, 'rb') as f:
                test_data_embeddings = pickle.load(f)

            # train_data_embeddings = np.nan_to_num(train_data_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            # test_data_embeddings = np.nan_to_num(test_data_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            logging.info("Authenticating with HuggingFace...")
            huggingface_hub.login(token=os.getenv("HF_TOKEN"))

            logging.info("Initializing TabPFN for embedding generation...")
            # device='cpu' 确保兼容性，N_ensemble_configurations=4 提高速度
            tabpfn_classifier = TabPFNClassifier(device='cpu')
            
            # 加载原始数据 (用于获取测试样本的原始值)
            if os.path.exists(DATA_PATH):
                with open(DATA_PATH, 'rb') as f:
                    all_data_values = pickle.load(f)
                if isinstance(all_data_values, list):
                    all_data_values = np.array(all_data_values)
                
                # --- Generate Training Embeddings ---
                logging.info("Generating TabPFN embeddings for training data...")
                X_train = all_data_values[:train_size]
                y_train = np.array(all_labels[:train_size])
                
                if X_train.ndim == 3:
                    N, T, V = X_train.shape
                    X_train_flat = X_train.reshape(N, T*V)
                else:
                    X_train_flat = X_train
                    
                embedder = TabPFNEmbedding(tabpfn_classifier, n_fold=5)
                embedder.fit(X_train_flat, y_train)
                
                train_embs_layers = embedder.get_embeddings(X_train_flat, y_train, X_train_flat, data_source='train')
                train_data_embeddings = train_embs_layers[-1]
                # train_data_embeddings = np.nan_to_num(train_data_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                logging.info(f"Generated TabPFN embeddings for {len(train_data_embeddings)} training samples.")

                # --- Generate Test Embeddings ---
                logging.info("Generating TabPFN embeddings for test data...")
                X_test = all_data_values[train_size:]
                if X_test.ndim == 3:
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_test_flat = X_test
                
                # 使用训练集作为 Context 来生成测试集的 Embedding
                test_embs_layers = embedder.get_embeddings(X_train_flat, y_train, X_test_flat, data_source='test')
                test_data_embeddings = test_embs_layers[-1]
                # test_data_embeddings = np.nan_to_num(test_data_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                logging.info(f"Generated TabPFN embeddings for {len(test_data_embeddings)} test samples.")

                # --- Save Embeddings ---
                logging.info(f"Saving TabPFN embeddings to {TRAIN_EMB_PATH} and {TEST_EMB_PATH}...")
                with open(TRAIN_EMB_PATH, 'wb') as f:
                    pickle.dump(train_data_embeddings, f)
                with open(TEST_EMB_PATH, 'wb') as f:
                    pickle.dump(test_data_embeddings, f)
            else:
                logging.error(f"Data file {DATA_PATH} not found. Cannot perform TabPFN retrieval.")
                return

    # --- 4. 初始化Agents ---
    analysis_agent = AnalysisAgent(model_name=args.analysis_model, dataset_name=DATASET_NAME, image_url_format='dict')
    rag_prediction_agent = RAGPredictionAgent(model_name=args.prediction_model, dataset_name=DATASET_NAME, image_url_format='dict')

    # --- 5. 并行处理测试数据，执行RAG预测 ---
    final_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 为每个 image_id 提交一个任务
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
                test_data_embeddings[image_id - train_size] if test_data_embeddings is not None else None
            ): image_id for image_id in test_indices
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(test_indices), desc="Running RAG Predictions"):
            try:
                result = future.result()
                if result:
                    final_results.append(result)
                    print(json.dumps(result, indent=2, ensure_ascii=False))
            except Exception as exc:
                image_id = future_to_id[future]
                logging.error(f"Image ID {image_id} generated an exception: {exc}")

    # --- 6. 保存最终预测结果 ---
    results_df = pd.DataFrame(final_results)
    output_path = os.path.join(RESULTS_DIR, args.output_file)
    results_df.to_csv(output_path, index=False)
    logging.info(f"RAG prediction results saved to {output_path}")

    # --- 7. 计算并打印性能指标 ---
    results_df = pd.read_csv(output_path)

    evaluate_results(results_df, LABEL_MEANINGS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG-enhanced predictions on a time-series dataset.")
    parser.add_argument(
        '--dataset_name', type=str, default='power', choices=ALL_LABEL_MEANINGS.keys(),
        help='Name of the dataset to process.'
    )
    parser.add_argument(
        '--analysis_model', type=str, default='gemini-2.0-flash',
        help='Name of the VLM for the AnalysisAgent.'
    )
    parser.add_argument(
        '--prediction_model', type=str, default='gemini-2.0-flash',
        help='Name of the VLM for the RAGPredictionAgent.'
    )
    parser.add_argument(
        '--embedding_model', type=str, default='text-embedding-3-large',
        help='Name of the model for generating text embeddings.'
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
        '--top_k', type=int, default=5,
        help='Number of similar examples to retrieve for the RAG prompt.'
    )
    parser.add_argument(
        '--retrieval_mode', type=str, default='hybrid', choices=['text', 'tabpfn', 'hybrid', 'worst'],
        help='Retrieval mode: "text" (semantic), "tabpfn" (raw value embedding), or "hybrid" (weighted).'
    )
    parser.add_argument(
        '--hybrid_alpha', type=float, default=0.8,
        help='Weight for TabPFN similarity in hybrid mode (0.0 to 1.0). Text similarity weight is (1 - alpha).'
    )
    parser.add_argument(
        '--output_file', type=str, default='gemini-2.0-flash_rag_predictions.csv',
        help='Name of the output CSV file for saving results.'
    )
    args = parser.parse_args()
    main(args)