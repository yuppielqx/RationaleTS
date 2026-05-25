import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import torch
import logging
from typing import List, Dict, Any

# --- 配置部分 ---

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API和模型配置
API_KEY = os.environ.get("OPENROUTER_API_KEY", "sk-or-v1-...")  # 强烈建议使用环境变量
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PREDICTION_MODEL = "openai/gpt-4o-mini"  # 用于预测的LLM模型
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 用于生成文本嵌入的BERT模型

# 文件和目录路径
DATA_DIR = "./dataset"
EXPL_DIR = "./expl_results/weather_ny/round_1/refinement"
ORIGINAL_TEXT_DIR = os.path.join(DATA_DIR, "weather_summary")
CITY = "weather_ny"
CITY_FULL_NAME = "New York City"


# --- 辅助函数和类 ---

def load_data(data_dir: str, city: str) -> (pd.DataFrame, Dict[int, str]):
    """
    加载数据集索引和标签，并读取所有原始天气摘要。
    """
    logging.info("开始加载数据...")
    try:
        # 假设有一个包含索引和标签的CSV文件，这比多个npy/pkl文件更易于管理
        # 如果文件不存在，则根据 notebook 中的逻辑创建它
        labels_file = os.path.join(data_dir, f"{city}_labels.csv")
        if not os.path.exists(labels_file):
            logging.warning(f"{labels_file} 不存在，将尝试从 .npy 文件创建。")
            # 从 notebook 加载数据
            with open(os.path.join(data_dir, 'indices.pkl'), 'rb') as f:
                indices = np.load(f, allow_pickle=True)
            gt_train = np.load(os.path.join(data_dir, 'gt_train.npy'))
            gt_val = np.load(os.path.join(data_dir, 'gt_val.npy'))

            # 注意：notebook中的测试集标签没有直接给出，这里我们合并训练集和验证集
            # 这是一个基于 notebook 内容的合理假设
            num_train = len(gt_train)
            all_indices = indices[:num_train + len(gt_val)]
            all_labels = np.concatenate([gt_train, gt_val])

            df = pd.DataFrame({'index': all_indices, 'label': all_labels})
            df.to_csv(labels_file, index=False)
            logging.info(f"已创建并保存标签文件到 {labels_file}")

        df = pd.read_csv(labels_file)

        # 加载天气文本
        texts = {}
        for idx in df['index']:
            file_path = os.path.join(ORIGINAL_TEXT_DIR, f"{city}_{idx}.txt")
            with open(file_path, 'r', encoding='utf-8') as f:
                texts[idx] = f.read()

        logging.info(f"成功加载 {len(df)} 个样本和 {len(texts)} 篇天气摘要。")
        return df, texts
    except FileNotFoundError as e:
        logging.error(f"加载数据失败: {e}。请确保所有必需的数据文件都在 '{data_dir}' 目录下。")
        raise


class EmbeddingHandler:
    """
    处理文本编码和相似度计算。
    """

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"初始化嵌入模型 '{model_name}'，使用设备: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.train_embeddings = None
        self.train_df = None

    def create_train_embeddings(self, train_df: pd.DataFrame, texts: Dict[int, str]):
        """为训练集创建并缓存文本嵌入。"""
        logging.info("正在为训练集创建文本嵌入...")
        self.train_df = train_df
        train_texts = [texts[idx] for idx in self.train_df['index']]
        self.train_embeddings = self.model.encode(train_texts, convert_to_tensor=True, show_progress_bar=True)
        logging.info("训练集文本嵌入创建完成。")

    def find_similar_samples(self, test_text: str, top_k: int = 3) -> Dict[str, List[Dict]]:
        """
        为给定的测试文本找到最相似的训练样本。
        """
        if self.train_embeddings is None or self.train_df is None:
            raise ValueError("必须先调用 create_train_embeddings。")

        test_embedding = self.model.encode(test_text, convert_to_tensor=True)

        # 计算余弦相似度
        similarities = cosine_similarity(
            test_embedding.cpu().numpy().reshape(1, -1),
            self.train_embeddings.cpu().numpy()
        )[0]

        self.train_df['similarity'] = similarities

        # 分别为每个类别选出 top_k
        rain_samples = self.train_df[self.train_df['label'] == 1].nlargest(top_k, 'similarity')
        not_rain_samples = self.train_df[self.train_df['label'] == 0].nlargest(top_k, 'similarity')

        def format_samples(df: pd.DataFrame) -> List[Dict]:
            results = []
            for _, row in df.iterrows():
                # 加载对应的精炼文本
                refined_text_path = os.path.join(EXPL_DIR, f"{row['index']}.txt")
                try:
                    with open(refined_text_path, 'r', encoding='utf-8') as f:
                        refined_text = f.read()
                except FileNotFoundError:
                    logging.warning(f"未找到精炼文本: {refined_text_path}，将留空。")
                    refined_text = "N/A"

                results.append({
                    "index": row['index'],
                    "label": row['label'],
                    "similarity": row['similarity'],
                    "refined_text": refined_text
                })
            return results

        return {
            "rain": format_samples(rain_samples),
            "not_rain": format_samples(not_rain_samples)
        }


class PredictionAgent:
    """
    使用LLM进行最终预测的代理。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        if not api_key or "sk-or-v1-..." in api_key:
            raise ValueError("请设置 OPENROUTER_API_KEY 环境变量或在代码中提供有效的 API Key。")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
        self.model = model

    def predict(self, test_sample_text: str, similar_samples: Dict[str, List[Dict]]) -> str:
        """
        构建prompt并调用LLM API进行预测。
        """
        system_prompt = (
            "You are a professional weather forecaster. Your task is to predict whether it will rain in the next 24 hours "
            f"in {CITY_FULL_NAME} based on a current weather summary and examples from the past."
        )

        user_prompt = f"Your task is to predict if it will rain in {CITY_FULL_NAME} in the next 24 hours.\n\n"
        user_prompt += "First, review these 6 historical examples to understand the patterns. "
        user_prompt += "Pay close attention to the outcome (rain or not rain) for each.\n\n"

        # 添加相似样本信息
        all_examples = similar_samples['rain'] + similar_samples['not_rain']
        for i, sample in enumerate(all_examples):
            outcome = "It rained." if sample['label'] == 1 else "It did not rain."
            user_prompt += f"--- Example {i + 1} ---\n"
            user_prompt += f"Historical Summary: {sample['refined_text']}\n"
            user_prompt += f"Outcome: {outcome}\n\n"

        user_prompt += "--- Current Situation ---\n"
        user_prompt += "Now, analyze the following current weather summary:\n"
        user_prompt += f"Summary: {test_sample_text}\n\n"

        user_prompt += "--- Your Prediction ---\n"
        user_prompt += "Based on all the information provided, will it rain or not rain? "
        user_prompt += "Respond with only 'rain' or 'not rain'."

        try:
            logging.info("正在向LLM发送预测请求...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=10,
            )
            prediction = response.choices[0].message.content.strip().lower()
            # 清理输出，确保只有 'rain' 或 'not rain'
            if 'not rain' in prediction:
                return 'not rain'
            elif 'rain' in prediction:
                return 'rain'
            else:
                logging.warning(f"LLM返回了意外的预测结果: '{prediction}'。将默认处理为 'not rain'。")
                return 'not rain'

        except Exception as e:
            logging.error(f"调用LLM API时出错: {e}")
            return "error"


# --- 主程序 ---

def main():
    """
    主执行函数。
    """
    # 1. 加载数据并划分为训练集和测试集
    all_data_df, texts = load_data(DATA_DIR, CITY)
    train_df, test_df = train_test_split(
        all_data_df, test_size=0.2, random_state=42, stratify=all_data_df['label']
    )
    logging.info(f"数据集划分完成: {len(train_df)} 个训练样本, {len(test_df)} 个测试样本。")

    # 2. 初始化嵌入和预测模块
    embedding_handler = EmbeddingHandler(EMBEDDING_MODEL)
    prediction_agent = PredictionAgent(API_KEY, OPENROUTER_BASE_URL, PREDICTION_MODEL)

    # 为训练集创建嵌入
    embedding_handler.create_train_embeddings(train_df, texts)

    # 3. 遍历测试集，执行预测流程
    predictions = []
    for i, test_sample in test_df.iterrows():
        test_idx = test_sample['index']
        test_label = test_sample['label']
        test_text = texts[test_idx]

        logging.info(f"--- 正在处理测试样本 #{test_idx} (真实标签: {test_label}) ---")

        # 3.1 为测试样本找到相似的训练样本
        similar_samples = embedding_handler.find_similar_samples(test_text, top_k=3)

        # 打印找到的相似样本信息（用于调试）
        logging.debug(f"为样本 {test_idx} 找到的相似'下雨'样本: {[s['index'] for s in similar_samples['rain']]}")
        logging.debug(f"为样本 {test_idx} 找到的相似'不下雨'样本: {[s['index'] for s in similar_samples['not_rain']]}")

        # 3.2 将所有信息输入预测Agent
        # 注意：这里没有使用图像，因为 notebook 中主要依赖文本和原型
        final_prediction = prediction_agent.predict(test_text, similar_samples)

        logging.info(f"样本 #{test_idx}: 真实标签 = {test_label}, 预测结果 = '{final_prediction}'")

        predictions.append({
            "index": test_idx,
            "true_label": test_label,
            "predicted_text": final_prediction
        })

    # 4. 评估和保存结果
    results_df = pd.DataFrame(predictions)
    class_mapping = {'not rain': 0, 'rain': 1, 'error': -1}
    results_df['predicted_label'] = results_df['predicted_text'].map(class_mapping)

    # 计算准确率
    correct_predictions = (results_df['true_label'] == results_df['predicted_label']).sum()
    total_predictions = len(results_df)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    logging.info(f"\n--- 预测完成 ---")
    logging.info(f"总测试样本数: {total_predictions}")
    logging.info(f"正确预测数: {correct_predictions}")
    logging.info(f"准确率: {accuracy:.2%}")

    # 保存结果到CSV
    results_df.to_csv("test_predictions.csv", index=False)
    logging.info("预测结果已保存到 'test_predictions.csv'")


if __name__ == "__main__":
    main()
