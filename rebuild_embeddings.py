import os
import pickle
import logging
from tqdm import tqdm
import pandas as pd

from utils import EmbeddingHandler

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 全局配置 ---
DATASET_NAME = "LargeAQ"
DB_OUTPUT_DIR = f"vector_db/{DATASET_NAME}"
DB_PATH = os.path.join(DB_OUTPUT_DIR, "vector_db.pkl")

def main():
    """
    主函数，用于从一个只包含元数据的向量数据库文件中恢复并生成嵌入向量。
    """
    # --- 1. 检查并加载不完整的向量数据库 ---
    if not os.path.exists(DB_PATH):
        logging.error(f"Vector database file not found at '{DB_PATH}'. Please run 'build_vector_db.py' first to generate the metadata.")
        return

    logging.info(f"Loading partial vector database from '{DB_PATH}'...")
    with open(DB_PATH, 'rb') as f:
        db_data = pickle.load(f)

    metadata_df = db_data.get("metadata")
    if metadata_df is None or not isinstance(metadata_df, pd.DataFrame):
        logging.error("The 'metadata' key is missing or is not a DataFrame in the vector DB file.")
        return

    logging.info(f"Successfully loaded metadata for {len(metadata_df)} documents.")

    # --- 2. 优先从文件中加载文本列表，如果不存在则从元数据重建 ---
    all_reasoning_texts = metadata_df['reasoning_path'].tolist()
    all_chunk_texts = db_data.get("all_chunk_texts")
    chunk_to_doc_id_map = db_data.get("chunk_to_doc_id_map")

    logging.info(f"Reconstructed {len(all_reasoning_texts)} coarse-grained texts and {len(all_chunk_texts)} fine-grained chunks.")

    # --- 3. 初始化Embedding处理器并生成向量 ---
    logging.info("Initializing EmbeddingHandler...")
    embedding_handler = EmbeddingHandler(model_name='text-embedding-3-large')

    logging.info("Vectorizing all chunks (fine-grained)...")
    fine_embeddings = embedding_handler.get_embeddings(all_chunk_texts)

    logging.info("Vectorizing all reasoning paths (coarse-grained)...")
    coarse_embeddings = embedding_handler.get_embeddings(all_reasoning_texts)

    if coarse_embeddings.size == 0 or fine_embeddings.size == 0:
        logging.error("Embedding generation failed. API call might have returned an error. Aborting.")
        return

    # --- 4. 重新组合并保存完整的向量数据库 ---
    logging.info("Re-assembling and saving the complete vector database...")
    complete_db_data = {
        "metadata": metadata_df,
        "coarse_embeddings": coarse_embeddings,
        "fine_embeddings": fine_embeddings,
        "chunk_to_doc_id_map": chunk_to_doc_id_map,
        "all_chunk_texts": all_chunk_texts
    }

    with open(DB_PATH, "wb") as f:
        pickle.dump(complete_db_data, f)

    logging.info(f"Successfully rebuilt and saved the complete vector database to '{DB_PATH}'.")

if __name__ == "__main__":
    main()