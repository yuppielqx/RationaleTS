# Explainable multi-modal time series prediction with LLM-in-the-Loop

## Required Environment Dependencies
```
torch==2.0.0
numpy==1.24.4
scikit-learn==1.2.2
pandas==1.5.3
transformers==4.37.2
openai==1.58.1
```
## Environment Installation
1. Create a new environment 
```
conda create -n timexl python=3.9 pip -c conda-forge -c pytorch
```
2. Activate the environment 
```
conda activate timexl
```
3. Install the requirements
```
pip install -r requirements.txt
```
## File Structure
```
├── dataset
│   ├── weather_summary
│   ├── timexl_agent.ipynb
│   ├── time_series_ny.pkl
│   └── ...
├── expl_results
├── get_text_embedding.py
├── model_weather_mm.py
└── train_encoder.py
```
## Usage
1. Use the selected language model to generate the text embedding for encoder training.
   ```
   python3.9 get_text_embedding.py
   ```
2. Train the multi-modal prototype-based encoder to get the initial prediction and explanations.
   ```
   python3.9 train_encoder.py
   ```
3. Navigate to ```./dataset/timexl_agent.ipynb``` for LLM prediction, reflection and refinement. 
