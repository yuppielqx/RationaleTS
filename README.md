# RationaleTS: RAG-based Multimodal Time Series Prediction

Retrieval-Augmented Generation (RAG) for multimodal time series prediction. A Vision Language Model (VLM) generates reasoning paths from historical chart images to build a "rationale base" (vector DB). At inference time, the system retrieves similar reasoning paths and feeds them as context to improve prediction accuracy.

## Pipeline

```
Training Data  →  Chart Images  →  VLM (EvaluatorAgent)  →  Reasoning Paths  →  Vector DB
                                                                                    ↓
Test Data      →  Chart Images  →  VLM (AnalysisAgent)   →  Query Embedding  →  RAG Retrieval
                                                                                    ↓
                                              Prediction  ←  VLM (RAGPredictionAgent)  ←  Top-K Examples
```

## Supported Datasets

| Dataset | Task | Classes |
|---------|------|---------|
| `traffic` | Traffic occupancy change (next hour) | 3: decrease >2, stable [-2,2], increase >2 |
| `finance_SP` | S&P 500 movement (next day) | 3: decrease >1%, neutral, increase >1% |
| `LargeAQ` | PM2.5 heavy pollution (next day) | 2: none (<75), heavy (>=75) |
| `power` | Wind turbine power increase (next 6h) | 2: no, yes |

## Environment Setup

```bash
conda create -n rationalets python=3.9 pip
conda activate rationalets
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY="your-key"      # OpenAI-compatible endpoint (xiaojing)
export DASHSCOPE_API_KEY="your-key"   # Alibaba DashScope
export HF_TOKEN="your-token"          # HuggingFace (for TabPFN embeddings)
```

## Main Entry Points

### 1. `build_vector_db.py` — Build the Rationale Base

Uses a VLM (EvaluatorAgent) to generate gold-standard reasoning paths for training samples, then embeddings them into a vector database.

```bash
python build_vector_db.py \
  --dataset_name traffic \
  --evaluator_model gpt-4o \
  --embedding_model text-embedding-3-large \
  --train_split 0.8 \
  --max_workers 8
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `power` | Dataset: `traffic`, `finance_SP`, `LargeAQ`, `power` |
| `--evaluator_model` | `gpt-4o` | VLM for generating reasoning paths |
| `--embedding_model` | `text-embedding-3-large` | Text embedding model |
| `--train_split` | `0.8` | Fraction of data for training |
| `--max_workers` | `8` | Parallel workers |

**Output:** `vector_db/{dataset}/gpt4o_vector_db.pkl`

### 2. `run_rag_prediction.py` — RAG-based Prediction

Retrieves similar reasoning paths from the vector DB and uses them as context for prediction.

```bash
python run_rag_prediction.py \
  --dataset_name traffic \
  --analysis_model gemini-2.0-flash \
  --prediction_model gemini-2.0-flash \
  --retrieval_mode hybrid \
  --hybrid_alpha 0.8 \
  --top_k 5 \
  --output_file my_predictions.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_name` | `power` | Dataset to process |
| `--analysis_model` | `gemini-2.0-flash` | VLM for analyzing test chart |
| `--prediction_model` | `gemini-2.0-flash` | VLM for final prediction |
| `--retrieval_mode` | `hybrid` | `text`, `tabpfn`, `hybrid`, `worst` |
| `--hybrid_alpha` | `0.8` | TabPFN weight in hybrid (0–1) |
| `--top_k` | `5` | Number of examples to retrieve |
| `--output_file` | `gemini-2.0-flash_rag_predictions.csv` | Output CSV name |

**Output:** `rag_results/{dataset}/{output_file}.csv`

## Traffic Dataset

The traffic dataset uses hourly sensor data from a Madrid station (Jan–Jul 2019). 722 samples with 12-hour sliding windows and 7 features: NO2, windSpeed, Temp, Humidity, SolarRad, intensity, Occupancy.

### Quick Start with Pre-built Results

The repo includes intermediate results for the traffic dataset so you can run predictions immediately:

```bash
# Run RAG prediction with pre-built gemini-2.5 vector DB
python run_rag_prediction.py \
  --dataset_name traffic \
  --analysis_model gemini-2.0-flash \
  --prediction_model gemini-2.0-flash \
  --retrieval_mode hybrid \
  --hybrid_alpha 0.8 \
  --top_k 5
```

The pre-built `gemini2.5_vector_db.pkl` (61MB) is included in `vector_db/traffic/`. To use other evaluator models, rebuild with `build_vector_db.py`.

### Included Intermediate Results

| Path | Description |
|------|-------------|
| `dataset/traffic/images/` | 722 chart images (one per sample) |
| `dataset/traffic/data.pkl` | Preprocessed time series (722, 12, 7) |
| `dataset/traffic/labels.pkl` | Ground truth labels |
| `dataset/traffic/train_embeddings.pkl` | TabPFN embeddings (training set) |
| `dataset/traffic/test_embeddings.pkl` | TabPFN embeddings (test set) |
| `dataset/traffic/vector_db.csv` | Reasoning path metadata |
| `vector_db/traffic/gemini2.5_vector_db.pkl` | Pre-built vector DB (Gemini 2.5) |
| `rag_results/traffic/*.csv` | All prediction results (55 experiments) |

## Other Entry Points

| Script | Description |
|--------|-------------|
| `run_llm_direct_prediction.py` | Text-only direct prediction |
| `run_llm_cot_prediction.py` | Text-only Chain-of-Thought prediction |
| `run_llm_fewshot_prediction.py` | Text-only few-shot prediction |
| `run_vlm_direct_prediction.py` | Image-only direct prediction |
| `run_vlm_fewshot_prediction.py` | Image-only few-shot prediction |
| `run_rag_random_prediction.py` | RAG with random retrieval (baseline) |
| `evaluate_rag_results.py` | Compute metrics from prediction CSVs |

## Repository Structure

```
├── build_vector_db.py              # Main entry: build rationale base
├── run_rag_prediction.py           # Main entry: RAG prediction
├── agents.py                       # Agent classes (Evaluator, Analysis, RAG, etc.)
├── utils.py                        # EmbeddingHandler, API calls, evaluation
├── prompts.json                    # Per-dataset prompt templates
├── workflow.py                     # Workflow orchestration
├── dataset/traffic/                # Traffic dataset (images, embeddings, labels)
├── vector_db/traffic/              # Pre-built vector DB (gemini-2.5)
├── rag_results/traffic/            # Prediction results (55 experiments)
├── run_*.sh                        # Shell scripts for experiments
└── layers/                         # Transformer layers for PatchTST
```
