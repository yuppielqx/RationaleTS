import base64
import io
import os
import logging

from typing import List, Dict, Tuple
import json
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report, accuracy_score
from sklearn.preprocessing import label_binarize

openai.base_url = "https://open.xiaojingai.com/v1/"
# openai.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Initialize the OpenAI client for text embedding
client1 = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "")
    , base_url="https://open.xiaojingai.com/v1/"
)
client_qwen = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_image_to_base64(image_path):
    """Reads an image file and returns its base64 encoded string."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def evaluate_results(results_df: pd.DataFrame, LABEL_MEANINGS: dict):
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
        if y_pred_binarized.shape[1] != num_classes:
            y_pred_binarized = np.eye(num_classes)[y_pred.astype(int)]

        auroc_macro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='macro')
        auroc_micro = roc_auc_score(y_true_binarized, y_pred_binarized, multi_class='ovr', average='micro')
        logging.info(f"AUROC (Macro, OvR, from discrete predictions): {auroc_macro:.4f}")
        logging.info(f"AUROC (Micro, OvR, from discrete predictions): {auroc_micro:.4f}")

    logging.info("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=[LABEL_MEANINGS[i] for i in sorted(LABEL_MEANINGS.keys())], zero_division=0))


def call_qwen_api(messages, model_name, max_tokens=2048):
    """A generic function to call the Qwen API with a prepared messages payload."""
    try:

        # completion = client_qwen.chat.completions.create(
        #     model=model_name,
        #     messages=messages,
        # )
        # return completion.choices[0].message.content
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages
        )
        # print("输入的tokens数量为: ", response.usage.prompt_tokens)
        # print("输出的tokens数量为: ", response.usage.completion_tokens)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during API call: {e}"


class EmbeddingHandler:
    """
    使用 OpenAI 兼容的 API 处理文本编码和相似度计算。
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        logging.info(f"Initializing embedding handler with model '{model_name}'.")
        self.train_embeddings = None
        self.train_df = None
        # For RAG
        self.rag_metadata = None
        self.coarse_embeddings = None
        self.fine_embeddings = None
        self.chunk_to_doc_id_map = None
        self.all_chunk_texts = None

    def _get_embeddings_from_api(self, texts: List[str]) -> np.ndarray:
        """通过调用 API 为一批文本生成嵌入向量。"""
        try:
            response = client1.embeddings.create(
                input=texts,
                model=self.model_name,
                encoding_format="float"
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"Error calling embedding API: {e}")
            return np.array([])

    def get_embeddings(self, texts: List[str], batch_size: int = 512) -> np.ndarray:
        """
        为文本列表生成嵌入向量，支持自动批处理以避免API长度限制。

        Args:
            texts (List[str]): 需要向量化的文本列表。
            batch_size (int): 每个API调用的批次大小。

        Returns:
            np.ndarray: 包含所有文本嵌入向量的numpy数组。
        """
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._get_embeddings_from_api(batch)
            if batch_embeddings.size == 0:
                logging.error(f"Failed to get embeddings for batch starting at index {i}. Aborting.")
                return np.array([])
            all_embeddings.append(batch_embeddings)

        if not all_embeddings:
            return np.array([])
        
        return np.vstack(all_embeddings)

    def create_train_embeddings(self, train_df: pd.DataFrame, texts: Dict[int, str], round_1_results_path: str):
        """Creates and caches text embeddings for the training set."""
        logging.info("Creating text embeddings for the training set...")
        self.train_df = train_df
        train_texts = [texts[idx] for idx in self.train_df['index']]
        self.train_embeddings = self._get_embeddings(train_texts)

        # 预计算第一轮推理文本的嵌入
        round_1_df = pd.read_csv(round_1_results_path).set_index('id')
        train_reasonings = [round_1_df.loc[idx]['llm_reasoning'] if idx in round_1_df.index else "" for idx in self.train_df['index']]
        self.reasoning_embeddings = self._get_embeddings(train_reasonings)

        logging.info("Training set text embeddings created.")

    def load_vector_db(self, db_path: str):
        """加载预先构建的向量数据库。"""
        logging.info(f"Loading vector DB from {db_path}")
        with open(db_path, "rb") as f:
            db_data = pickle.load(f)
        self.rag_metadata = db_data["metadata"]
        self.coarse_embeddings = db_data["coarse_embeddings"]
        self.fine_embeddings = db_data["fine_embeddings"]
        self.chunk_to_doc_id_map = db_data["chunk_to_doc_id_map"]
        self.all_chunk_texts = db_data["all_chunk_texts"]
        logging.info("Vector DB loaded successfully.")

    def find_similar_reasoning_paths(self, query_text: str, top_k: int = 5, coarse_k: int = 20) -> List[Dict]:
        """
        执行两阶段检索来查找最相关的推理路径。
        
        Args:
            query_text (str): 用户的查询或初步分析文本。
            top_k (int): 最终返回的最相关示例数量。
            coarse_k (int): 粗筛阶段返回的候选数量。

        Returns:
            List[Dict]: 包含最相关示例的列表，每个示例是一个字典。
        """
        if self.coarse_embeddings is None or self.fine_embeddings is None:
            raise ValueError("Vector database must be loaded first using 'load_vector_db'.")

        query_embedding = self.get_embeddings([query_text])

        # --- 1. 粗筛阶段 ---
        coarse_similarities = cosine_similarity(query_embedding, self.coarse_embeddings)[0]
        # 获取前 coarse_k 个最相似的文档的索引
        coarse_candidate_indices = np.argsort(coarse_similarities)[-coarse_k:][::-1]

        # --- 2. 精排阶段 ---
        rerank_scores = []
        for doc_id in coarse_candidate_indices:
            # 找到属于这个文档的所有 fine-grained chunks
            chunk_indices = [i for i, map_id in enumerate(self.chunk_to_doc_id_map) if map_id == doc_id]
            if not chunk_indices:
                continue
            
            candidate_fine_embeddings = self.fine_embeddings[chunk_indices]
            
            # 计算查询与该文档所有chunks的相似度，并取最大值
            fine_similarities = cosine_similarity(query_embedding, candidate_fine_embeddings)[0]
            max_similarity = np.max(fine_similarities)
            rerank_scores.append((max_similarity, doc_id))

        # 根据精排分数（最大块相似度）对候选进行排序
        rerank_scores.sort(key=lambda x: x[0], reverse=True)
        
        # --- 3. 格式化并返回最终结果 ---
        top_doc_ids = [doc_id for _, doc_id in rerank_scores[:top_k]]
        
        return self.rag_metadata.iloc[top_doc_ids].to_dict('records')

    def find_diverse_reasoning_paths(self, query_text: str, top_n_per_class: int = 2) -> List[Dict]:
        """
        执行“先分组，后检索”的分层检索策略，为每个标签类别找到最相关的推理路径。

        Args:
            query_text (str): 用户的查询或初步分析文本。
            top_n_per_class (int): 每个类别要选择的顶部样本数量。

        Returns:
            List[Dict]: 包含多样化示例的列表。
        """
        if self.coarse_embeddings is None:
            raise ValueError("Vector database must be loaded first using 'load_vector_db'.")

        # 1. 获取查询向量
        query_embedding = self.get_embeddings([query_text])

        # 2. 按标签分组，在每个组内独立进行检索
        selected_examples = []
        # 获取数据集中存在的所有唯一标签
        unique_labels = sorted(self.rag_metadata['true_label'].unique())
        for label_id in unique_labels:
            
            # a. 找到属于当前标签的所有文档的索引
            group_indices = self.rag_metadata[self.rag_metadata['true_label'] == label_id].index.tolist()
            if not group_indices:
                continue
            
            # b. 提取该组的嵌入向量
            group_embeddings = self.coarse_embeddings[group_indices]
            
            # c. 在该组内计算相似度
            group_similarities = cosine_similarity(query_embedding, group_embeddings)[0]
            
            # d. 找到该组内最相似的 top_n_per_class 个样本的索引 (相对于group_indices)
            top_indices_in_group = np.argsort(group_similarities)[-top_n_per_class:][::-1]
            
            # e. 获取这些样本在原始metadata中的真实索引
            top_original_indices = [group_indices[i] for i in top_indices_in_group]
            
            # f. 获取并保存这些样本
            best_in_group = self.rag_metadata.iloc[top_original_indices].copy()
            best_in_group['similarity'] = group_similarities[top_indices_in_group]
            selected_examples.extend(best_in_group.to_dict('records'))

        # 3. 按总体相似度对所有选出的样本进行最终排序
        # 使用dict来去重，因为DataFrame的drop_duplicates可能不保留我们想要的第一个实例
        unique_examples_dict = {ex['id']: ex for ex in selected_examples}
        final_list = sorted(unique_examples_dict.values(), key=lambda x: x['similarity'], reverse=True)

        logging.info(f"Retrieved {len(final_list)} diverse examples for labels: {[ex['true_label'] for ex in final_list]}")
        
        return final_list
