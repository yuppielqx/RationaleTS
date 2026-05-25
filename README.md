# RationaleTS: RAG-based Multimodal Time Series Prediction

Retrieval-Augmented Generation (RAG) for multimodal time series prediction. A VLM generates reasoning paths from historical chart images to build a **rationale base** (vector DB). At inference, similar reasoning paths are retrieved as context to improve prediction accuracy.

> Datasets will be open-sourced soon.

## Pipeline

```
Training Data  →  Chart Images  →  VLM (EvaluatorAgent)  →  Reasoning Paths  →  Vector DB
                                                                                   ↓
Test Data      →  Chart Images  →  VLM (AnalysisAgent)   →  Query Embedding  →  RAG Retrieval
                                                                                   ↓
                                             Prediction  ←  VLM (RAGPredictionAgent)  ←  Top-K Examples
```

## Environment Setup

```bash
conda create -n rationalets python=3.9 pip
conda activate rationalets
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"      # OpenAI-compatible endpoint
export DASHSCOPE_API_KEY="your-key"   # Alibaba DashScope
export HF_TOKEN="your-token"          # HuggingFace (for TabPFN embeddings)
```

## Entry Points

### 1. `build_vector_db.py` — Build the Rationale Base

Uses a VLM (EvaluatorAgent) to generate gold-standard reasoning paths for each training sample, then vectorizes them into a vector database.

```bash
python build_vector_db.py \
  --dataset_name traffic \
  --evaluator_model gpt-5 \
  --embedding_model text-embedding-3-large \
  --train_split 0.8 \
  --max_workers 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `power` | Dataset: `traffic`, `finance_SP`, `LargeAQ`, `power` |
| `--evaluator_model` | `gpt-5` | VLM for generating reasoning paths |
| `--embedding_model` | `text-embedding-3-large` | Text embedding model |
| `--train_split` | `0.8` | Fraction of data for training |
| `--max_workers` | `8` | Parallel workers |

**Output:** `vector_db/{dataset}/gpt4o_vector_db.pkl`

### 2. `run_rag_prediction.py` — RAG-based Prediction

Retrieves similar reasoning paths from the vector DB and feeds them as context to a VLM for final prediction.

```bash
python run_rag_prediction.py \
  --dataset_name traffic \
  --retrieval_mode hybrid \
  --hybrid_alpha 0.8 \
  --top_k 5 \
  --output_file my_predictions.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `power` | Dataset to process |
| `--analysis_model` | `gpt-4o-mini` | VLM for analyzing test chart |
| `--prediction_model` | `gpt-4o-mini` | VLM for final prediction |
| `--embedding_model` | `text-embedding-3-large` | Text embedding model |
| `--retrieval_mode` | `hybrid` | `text`, `tabpfn`, `hybrid`, `worst` |
| `--hybrid_alpha` | `0.8` | TabPFN weight in hybrid (0–1) |
| `--top_k` | `5` | Number of examples to retrieve |
| `--output_file` | `gpt-4o-mini_rag_predictions.csv` | Output CSV name |

**Output:** `rag_results/{dataset}/{output_file}.csv`

### 3. `evaluate_rag_results.py` — Evaluate Predictions

Computes accuracy, F1, and AUROC metrics from prediction CSV files.

```bash
python evaluate_rag_results.py \
  --dataset_name traffic \
  --file_name my_predictions.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `traffic` | Dataset to evaluate |
| `--file_name` | (required) | CSV file in `rag_results/{dataset}/` |

## Repository Structure

```
├── build_vector_db.py            # Entry point: build rationale base
├── run_rag_prediction.py         # Entry point: RAG prediction
├── evaluate_rag_results.py       # Entry point: evaluate predictions
├── agents.py                     # Agent classes (Evaluator, Analysis, RAG)
├── utils.py                      # EmbeddingHandler, API calls, metrics
├── prompts.json                  # Per-dataset prompt templates
├── requirements.txt
├── .gitignore
└── README.md
```
