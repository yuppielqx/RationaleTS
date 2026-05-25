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
    """Compute and print performance metrics for prediction results."""
    y_true = results_df['true_label']
    y_pred = results_df['prediction']
    classes = sorted(y_true.unique())
    num_classes = len(classes)

    logging.info("\n--- Performance Evaluation ---")
    logging.info(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    logging.info(f"F1 Score (Macro): {f1_score(y_true, y_pred, average='macro'):.4f}")
    logging.info(f"F1 Score (Micro): {f1_score(y_true, y_pred, average='micro'):.4f}")
    logging.info(f"F1 Score (Weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # --- AUROC calculation ---
    # Since there are no direct probability outputs, we simulate probabilities from discrete predictions (one-hot encoding)
    # This is a way to compute AUROC, though less precise than using real probability scores
    if num_classes > 1:
        y_true_binarized = label_binarize(y_true, classes=classes)
        y_pred_binarized = label_binarize(y_pred, classes=classes)

        # Ensure y_pred_binarized has the correct number of columns even if some classes are not predicted
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
        # print("Input token count: ", response.usage.prompt_tokens)
        # print("Output token count: ", response.usage.completion_tokens)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during API call: {e}"


class EmbeddingHandler:
    """
    Handle text encoding and similarity computation using an OpenAI-compatible API.
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
        """Generate embeddings for a batch of texts by calling the API."""
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
        Generate embeddings for a list of texts, with automatic batching to avoid API length limits.

        Args:
            texts (List[str]): List of texts to vectorize.
            batch_size (int): Batch size per API call.

        Returns:
            np.ndarray: Numpy array containing all text embeddings.
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

        # Pre-compute embeddings for round-1 reasoning texts
        round_1_df = pd.read_csv(round_1_results_path).set_index('id')
        train_reasonings = [round_1_df.loc[idx]['llm_reasoning'] if idx in round_1_df.index else "" for idx in self.train_df['index']]
        self.reasoning_embeddings = self._get_embeddings(train_reasonings)

        logging.info("Training set text embeddings created.")

    def load_vector_db(self, db_path: str):
        """Load a pre-built vector database."""
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
        Perform two-stage retrieval to find the most relevant reasoning paths.

        Args:
            query_text (str): User query or preliminary analysis text.
            top_k (int): Number of most relevant examples to return.
            coarse_k (int): Number of candidates to return in the coarse stage.

        Returns:
            List[Dict]: List of the most relevant examples, each as a dictionary.
        """
        if self.coarse_embeddings is None or self.fine_embeddings is None:
            raise ValueError("Vector database must be loaded first using 'load_vector_db'.")

        query_embedding = self.get_embeddings([query_text])

        # --- 1. Coarse screening stage ---
        coarse_similarities = cosine_similarity(query_embedding, self.coarse_embeddings)[0]
        # Get indices of the top coarse_k most similar documents
        coarse_candidate_indices = np.argsort(coarse_similarities)[-coarse_k:][::-1]

        # --- 2. Fine-grained re-ranking stage ---
        rerank_scores = []
        for doc_id in coarse_candidate_indices:
            # Find all fine-grained chunks belonging to this document
            chunk_indices = [i for i, map_id in enumerate(self.chunk_to_doc_id_map) if map_id == doc_id]
            if not chunk_indices:
                continue
            
            candidate_fine_embeddings = self.fine_embeddings[chunk_indices]
            
            # Compute similarity between query and all chunks of this document, take the max
            fine_similarities = cosine_similarity(query_embedding, candidate_fine_embeddings)[0]
            max_similarity = np.max(fine_similarities)
            rerank_scores.append((max_similarity, doc_id))

        # Sort candidates by re-ranking score (max chunk similarity)
        rerank_scores.sort(key=lambda x: x[0], reverse=True)
        
        # --- 3. Format and return final results ---
        top_doc_ids = [doc_id for _, doc_id in rerank_scores[:top_k]]
        
        return self.rag_metadata.iloc[top_doc_ids].to_dict('records')

    def find_diverse_reasoning_paths(self, query_text: str, top_n_per_class: int = 2) -> List[Dict]:
        “””
        Perform a stratified “group-then-retrieve” strategy, finding the most relevant
        reasoning paths for each label class.

        Args:
            query_text (str): User query or preliminary analysis text.
            top_n_per_class (int): Number of top samples to select per class.

        Returns:
            List[Dict]: List of diverse examples.
        “””
        if self.coarse_embeddings is None:
            raise ValueError("Vector database must be loaded first using 'load_vector_db'.")

        # 1. Get query embedding
        query_embedding = self.get_embeddings([query_text])

        # 2. Group by label and retrieve independently within each group
        selected_examples = []
        # Get all unique labels present in the dataset
        unique_labels = sorted(self.rag_metadata['true_label'].unique())
        for label_id in unique_labels:
            
            # a. Find indices of all documents belonging to the current label
            group_indices = self.rag_metadata[self.rag_metadata['true_label'] == label_id].index.tolist()
            if not group_indices:
                continue
            
            # b. Extract embeddings for this group
            group_embeddings = self.coarse_embeddings[group_indices]
            
            # c. Compute similarity within this group
            group_similarities = cosine_similarity(query_embedding, group_embeddings)[0]
            
            # d. Find indices of the top_n_per_class most similar samples within this group (relative to group_indices)
            top_indices_in_group = np.argsort(group_similarities)[-top_n_per_class:][::-1]
            
            # e. Get the actual indices of these samples in the original metadata
            top_original_indices = [group_indices[i] for i in top_indices_in_group]
            
            # f. Retrieve and save these samples
            best_in_group = self.rag_metadata.iloc[top_original_indices].copy()
            best_in_group['similarity'] = group_similarities[top_indices_in_group]
            selected_examples.extend(best_in_group.to_dict('records'))

        # 3. Sort all selected samples by overall similarity
        # Use dict for deduplication, since DataFrame.drop_duplicates may not preserve the first instance
        unique_examples_dict = {ex['id']: ex for ex in selected_examples}
        final_list = sorted(unique_examples_dict.values(), key=lambda x: x['similarity'], reverse=True)

        logging.info(f"Retrieved {len(final_list)} diverse examples for labels: {[ex['true_label'] for ex in final_list]}")
        
        return final_list
