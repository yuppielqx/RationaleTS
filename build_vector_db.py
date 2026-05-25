import os
import json
import pickle
import logging
from tqdm import tqdm
import pandas as pd
import argparse
import concurrent.futures
import numpy as np
import warnings

from agents import EvaluatorAgent
from utils import EmbeddingHandler

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 所有数据集的标签含义 ---
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

def process_sample(doc_index, image_id, all_labels, evaluator_agent, LABEL_MEANINGS, IMAGE_DIR):
    """
    处理单个样本：为给定的图像ID生成黄金标准的推理路径。
    这是一个独立的工作函数，以便于并行处理。
    """
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    true_label = all_labels[image_id]
    true_label_meaning = LABEL_MEANINGS.get(true_label, "Unknown")

    # a. 生成黄金标准推理路径
    logging.info(f"Generating reasoning for image_id: {image_id} (True Label: {true_label_meaning})")
    reasoning_path = evaluator_agent.execute(
        true_label=true_label,
        true_label_meaning=true_label_meaning,
        image_path=image_path,
        llm_prediction=0,
        reasoning=""
    )

    # 返回包含所有必要信息的结果，以便后续排序和处理
    return {
        "doc_index": doc_index,
        "image_id": image_id,
        "true_label": true_label,
        "reasoning_path": reasoning_path
    }

def main(args):
    """
    主函数，用于生成、向量化并存储金融预测的黄金标准推理路径。
    """
    # --- 1. 根据参数动态设置配置 ---
    DATASET_NAME = args.dataset_name
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    DB_OUTPUT_DIR = f"vector_db/{DATASET_NAME}"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    # --- 1. 创建输出目录 ---
    os.makedirs(DB_OUTPUT_DIR, exist_ok=True)

    # --- 2. 加载数据和划分训练集 ---
    logging.info("Loading data and splitting into training set...")
    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    # 将前80%作为训练数据
    train_size = int(len(all_labels) * args.train_split)
    logging.info(f"Total samples: {len(all_labels)}, Training samples: {train_size}")

    # --- 3. 初始化Agent和Embedding处理器 ---
    evaluator_agent = EvaluatorAgent(model_name=args.evaluator_model, dataset_name=DATASET_NAME)
    embedding_handler = EmbeddingHandler(model_name=args.embedding_model)

    # --- 4. 并行生成推理路径 ---
    temp_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 创建一个future到文档索引的映射
        future_to_doc = {
            executor.submit(process_sample, i, i, all_labels, evaluator_agent, LABEL_MEANINGS, IMAGE_DIR): i
            for i in range(train_size)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_doc), total=train_size, desc="Generating Reasoning Paths"):
            try:
                result = future.result()
                if result:
                    temp_results.append(result)
            except Exception as exc:
                doc_index = future_to_doc[future]
                logging.error(f"Document index {doc_index} generated an exception: {exc}")

    # --- 5. 结果排序并准备向量化 ---
    # 确保结果与 train_indices 的原始顺序一致
    temp_results.sort(key=lambda x: x['doc_index'])

    reasoning_data_df = pd.DataFrame(temp_results)
    all_reasoning_texts = []
    all_chunk_texts = []
    chunk_to_doc_id_map = []

    for i, row in reasoning_data_df.iterrows():
        reasoning_path = row['reasoning_path']
        all_reasoning_texts.append(reasoning_path)
        chunks = [chunk.strip() for chunk in reasoning_path.strip().split('\n') if chunk.strip()]
        all_chunk_texts.extend(chunks)
        for _ in chunks:
            chunk_to_doc_id_map.append(i)

    # 移除临时的 doc_index 列
    metadata_df = reasoning_data_df.drop(columns=['doc_index'])
    # 将 image_id 重命名为 id 以匹配 RAG 代理的期望
    metadata_df = metadata_df.rename(columns={'image_id': 'id'})

    # --- 6. 批量进行向量化 ---
    logging.info("Vectorizing all reasoning paths (coarse-grained)...")
    coarse_embeddings = embedding_handler.get_embeddings(all_reasoning_texts)

    logging.info("Vectorizing all chunks (fine-grained)...")
    fine_embeddings = embedding_handler.get_embeddings(all_chunk_texts)

    # --- 7. 保存向量数据库和元数据 ---
    logging.info(f"Saving vector database to {DB_OUTPUT_DIR}...")
    db_data = {
        "metadata": metadata_df,
        "coarse_embeddings": coarse_embeddings,
        "fine_embeddings": fine_embeddings,
        "chunk_to_doc_id_map": chunk_to_doc_id_map,
        "all_chunk_texts": all_chunk_texts
    }

    with open(os.path.join(DB_OUTPUT_DIR, "gpt4o_vector_db.pkl"), "wb") as f:
        pickle.dump(db_data, f)

    logging.info("Vector database build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a vector database of reasoning paths for a given dataset.")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='power',
        choices=ALL_LABEL_MEANINGS.keys(),
        help='Name of the dataset to process.'
    )
    parser.add_argument(
        '--evaluator_model',
        type=str,
        default='gpt-4o',
        help='Name of the VLM to use for generating reasoning paths.'
    )
    parser.add_argument(
        '--embedding_model',
        type=str,
        default='text-embedding-3-large',
        help='Name of the model to use for generating text embeddings.'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.8,
        help='Fraction of the data to use for building the training vector database (e.g., 0.8 for 80%).'
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help='Maximum number of threads to use for parallel processing.'
    )
    args = parser.parse_args()
    main(args)