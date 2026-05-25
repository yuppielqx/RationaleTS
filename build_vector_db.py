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

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Label meanings for all datasets ---
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

def process_sample(doc_index, image_id, all_labels, evaluator_agent, LABEL_MEANINGS, IMAGE_DIR):
    """
    Process a single sample: generate a gold-standard reasoning path for the given image ID.
    This is a standalone worker function for parallel processing.
    """
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.png")
    if not os.path.exists(image_path):
        logging.warning(f"Image {image_id}.png not found, skipping.")
        return None

    true_label = all_labels[image_id]
    true_label_meaning = LABEL_MEANINGS.get(true_label, "Unknown")

    # a. Generate gold-standard reasoning path
    logging.info(f"Generating reasoning for image_id: {image_id} (True Label: {true_label_meaning})")
    reasoning_path = evaluator_agent.execute(
        true_label=true_label,
        true_label_meaning=true_label_meaning,
        image_path=image_path,
        llm_prediction=0,
        reasoning=""
    )

    # Return results with all necessary information for subsequent sorting and processing
    return {
        "doc_index": doc_index,
        "image_id": image_id,
        "true_label": true_label,
        "reasoning_path": reasoning_path
    }

def main(args):
    """
    Main function to generate, vectorize, and store gold-standard reasoning paths.
    """
    # --- 1. Configure settings dynamically from arguments ---
    DATASET_NAME = args.dataset_name
    IMAGE_DIR = f"dataset/{DATASET_NAME}/images"
    LABELS_PATH = f"dataset/{DATASET_NAME}/labels.pkl"
    DB_OUTPUT_DIR = f"vector_db/{DATASET_NAME}"
    LABEL_MEANINGS = ALL_LABEL_MEANINGS.get(DATASET_NAME)

    if LABEL_MEANINGS is None:
        logging.error(f"Dataset '{DATASET_NAME}' is not configured in ALL_LABEL_MEANINGS.")
        return

    # --- 1. Create output directory ---
    os.makedirs(DB_OUTPUT_DIR, exist_ok=True)

    # --- 2. Load data and split into training set ---
    logging.info("Loading data and splitting into training set...")
    with open(LABELS_PATH, 'rb') as f:
        all_labels = pickle.load(f)

    # Use the first train_split fraction as training data
    train_size = int(len(all_labels) * args.train_split)
    logging.info(f"Total samples: {len(all_labels)}, Training samples: {train_size}")

    # --- 3. Initialize agents and embedding handler ---
    evaluator_agent = EvaluatorAgent(model_name=args.evaluator_model, dataset_name=DATASET_NAME)
    embedding_handler = EmbeddingHandler(model_name=args.embedding_model)

    # --- 4. Generate reasoning paths in parallel ---
    temp_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a mapping from future to document index
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

    # --- 5. Sort results and prepare for vectorization ---
    # Ensure results follow the original order of train_indices
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

    # Remove temporary doc_index column
    metadata_df = reasoning_data_df.drop(columns=['doc_index'])
    # Rename image_id to id to match RAG agent expectations
    metadata_df = metadata_df.rename(columns={'image_id': 'id'})

    # --- 6. Batch vectorization ---
    logging.info("Vectorizing all reasoning paths (coarse-grained)...")
    coarse_embeddings = embedding_handler.get_embeddings(all_reasoning_texts)

    logging.info("Vectorizing all chunks (fine-grained)...")
    fine_embeddings = embedding_handler.get_embeddings(all_chunk_texts)

    # --- 7. Save vector database and metadata ---
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
        default='gpt-5',
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