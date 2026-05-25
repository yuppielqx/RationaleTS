import os
import argparse
import random
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from agents import PredictionAgent, EvaluationAgent, RefinementAgent
from utils import parse_llm_response

class WorkflowManager:
    """Orchestrates the entire Prediction-Evaluation-Refinement loop."""
    def __init__(self, args):
        self.args = args
        self._setup_paths()
        # Pass the dataset name to the agents
        self.prediction_agent = PredictionAgent(self.args.model_name, self.args.dataset_name)
        self.evaluation_agent = EvaluationAgent(self.args.model_name, self.args.dataset_name)
        self.refinement_agent = RefinementAgent(self.args.refinement_model_name, self.args.dataset_name)

    def _setup_paths(self):
        if self.args.image_dir is None: self.args.image_dir = f'./dataset/{self.args.dataset_name}/images/'
        if self.args.text_dir is None: self.args.text_dir = f'./dataset/{self.args.dataset_name}/txt/'
        if self.args.rain_pkl_path is None: self.args.rain_pkl_path = f'./dataset/{self.args.dataset_name}/rain.pkl'
        if self.args.base_output_dir is None: self.args.base_output_dir = f'./expl_results/{self.args.dataset_name}/'

    def run(self):
        # --- Setup for Resume and Early Stopping ---
        start_round = 1
        current_text_dir = self.args.text_dir
        for i in range(self.args.num_loops, 0, -1):
            potential_refinement_dir = os.path.join(self.args.base_output_dir, f'round_{i}', 'refinement')
            if os.path.exists(potential_refinement_dir):
                print(f"Found completed round {i}. Resuming from round {i+1}.")
                start_round = i + 1
                current_text_dir = potential_refinement_dir
                break
        
        best_f1_score = -1
        early_stopping_counter = 0

        # --- Get Train/Test Split ---
        all_files = sorted([int(f.split('.')[0]) for f in os.listdir(self.args.image_dir) if f.lower().endswith('.png')])
        split_index = int(len(all_files) * 0.8)
        train_files = all_files[:split_index]
        print(f"Found {len(train_files)} training files to process from dataset: {self.args.dataset_name}")

        # --- Master Loop ---
        for n in range(start_round, self.args.num_loops + 1):
            # 1. Prediction Round
            round_output_dir = self._run_prediction_round(n, train_files, current_text_dir)

            # round_output_dir = os.path.join(self.args.base_output_dir, f"round_1")
            
            # 2. Consolidation and Evaluation Round
            consolidated_csv, f1_score_current = self._consolidate_data_for_round(round_output_dir)

            # 3. Early Stopping Check
            if f1_score_current > best_f1_score:
                best_f1_score = f1_score_current
                early_stopping_counter = 0
                print(f"F1 score improved to {best_f1_score:.4f}. Resetting early stopping counter.")
            else:
                early_stopping_counter += 1
                print(f"F1 score did not improve. Counter: {early_stopping_counter}/{self.args.patience}")
            
            if early_stopping_counter >= self.args.patience:
                print(f"F1 score has not improved for {self.args.patience} rounds. Stopping early.")
                break

            # 4. Evaluation and Refinement Round
            refined_text_dir = self._run_evaluation_and_refinement_round(n, consolidated_csv)
            
            # 5. Set up the input for the next iteration
            current_text_dir = refined_text_dir

        print(f"\n\n{'='*30} Full Pipeline Complete {'='*30}")
        print(f"Completed processing up to round {n} for dataset '{self.args.dataset_name}'.")

    def _run_prediction_round(self, round_num, train_files, current_text_dir):
        print(f"\n{'='*20} Starting Prediction Round {round_num} {'='*20}")
        round_output_dir = os.path.join(self.args.base_output_dir, f"round_{round_num}")
        reasoning_dir = os.path.join(round_output_dir, "llm_reasoning")
        os.makedirs(reasoning_dir, exist_ok=True)
        
        predictions_data = []
        for sample_id in train_files:
            image_filename = f"{sample_id}.png"
            image_path = os.path.join(self.args.image_dir, image_filename)
            text_path = os.path.join(current_text_dir, f"{sample_id}.txt")

            if not os.path.exists(text_path):
                print(f"Warning: Text file not found for {image_filename} in {current_text_dir}. Skipping.")
                continue

            with open(text_path, 'r') as f:
                text_content = f.read().strip()

            print(f"\nProcessing for Round {round_num}: {image_filename}")
            response = self.prediction_agent.execute(image_path=image_path, text_content=text_content)
            prediction, reasoning = parse_llm_response(response)
            
            if prediction != -1:
                predictions_data.append({"id": sample_id, "llm_prediction": prediction})
                reasoning_path = os.path.join(reasoning_dir, f"{sample_id}_reasoning.txt")
                with open(reasoning_path, 'w') as f:
                    f.write(reasoning)
                print(f"  - Parsed Prediction: {prediction}, Saved reasoning to {reasoning_path}")
            else:
                print(f"  - Warning: Could not parse prediction for {image_filename}")

        predictions_csv_path = os.path.join(round_output_dir, "llm_predictions.csv")
        pd.DataFrame(predictions_data).to_csv(predictions_csv_path, index=False)
        print(f"\nRound {round_num} predictions saved to {predictions_csv_path}")
        return round_output_dir

    def _consolidate_data_for_round(self, round_output_dir):
        print(f"\n--- Consolidating and Evaluating data for Round output at {round_output_dir} ---")
        predictions_csv_path = os.path.join(round_output_dir, "llm_predictions.csv")
        reasoning_dir = os.path.join(round_output_dir, "llm_reasoning")
        pred_df = pd.read_csv(predictions_csv_path)

        with open(self.args.rain_pkl_path, 'rb') as f:
            rain_data = np.array(pkl.load(f)).astype(np.int64)

        with open(os.path.join('./dataset', self.args.dataset_name, 'indices.pkl'), 'rb') as f:
            indices_list = list(pkl.load(f))
        
        id_to_pos_index = {val: i for i, val in enumerate(indices_list)}
        
        def get_true_label_from_rain_list(sample_id):
            pos_index = id_to_pos_index.get(sample_id)
            if pos_index is not None and pos_index + 1 < len(rain_data):
                return rain_data[pos_index + 1]
            return -1

        merged_df = pred_df.copy()
        merged_df['true_label'] = merged_df['id'].apply(get_true_label_from_rain_list)
        merged_df = merged_df[merged_df['true_label'] != -1]

        merged_df['original_summary'] = merged_df['id'].apply(lambda x: open(os.path.join(self.args.text_dir, f"{x}.txt")).read().strip())
        merged_df['llm_reasoning'] = merged_df['id'].apply(lambda x: open(os.path.join(reasoning_dir, f"{x}_reasoning.txt")).read().strip())
        
        consolidated_csv_path = os.path.join(round_output_dir, "consolidated_data.csv")
        merged_df.to_csv(consolidated_csv_path, index=False)
        print(f"Consolidated data for this round saved to {consolidated_csv_path}")

        f1 = f1_score(merged_df['true_label'], merged_df['llm_prediction'])
        auc = roc_auc_score(merged_df['true_label'], merged_df['llm_prediction'])
        print(f"--- Round Metrics --- F1: {f1:.4f}, AUC: {auc:.4f} ---")
        
        return consolidated_csv_path, f1

    def _run_evaluation_and_refinement_round(self, round_num, consolidated_csv_path):
        print(f"\n{'='*20} Starting Evaluation & Refinement for Round {round_num} {'='*20}")
        round_output_dir = os.path.join(self.args.base_output_dir, f"round_{round_num}")
        df = pd.read_csv(consolidated_csv_path)
        
        evaluation_dir = os.path.join(round_output_dir, "evaluation")
        refinement_dir = os.path.join(round_output_dir, "refinement")
        os.makedirs(evaluation_dir, exist_ok=True)
        os.makedirs(refinement_dir, exist_ok=True)

        evaluation_results = []
        refinement_results = []

        for index, row in df.iterrows():
            sample_id = row['id']
            print(f"\nProcessing for Eval/Refine Round {round_num}: {sample_id}")
            image_path = os.path.join(self.args.image_dir, f"{sample_id}.png")

            print("  - Calling Evaluation Agent...")
            evaluation_response = self.evaluation_agent.execute(
                true_label=row['true_label'],
                llm_prediction=row['llm_prediction'],
                reasoning=row['llm_reasoning'],
                image_path=image_path
            )
            eval_path = os.path.join(evaluation_dir, f"{sample_id}.txt")
            with open(eval_path, 'w') as f:
                f.write(evaluation_response)
            print(f"    - Saved evaluation to {eval_path}")
            evaluation_results.append({"id": sample_id, "evaluation_text": evaluation_response})

            print("  - Calling Refinement Agent...")
            refinement_response = self.refinement_agent.execute(
                summary=row['original_summary'],
                evaluation_analysis=evaluation_response
            )
            refine_path = os.path.join(refinement_dir, f"{sample_id}.txt")
            with open(refine_path, 'w') as f:
                f.write(refinement_response)
            print(f"    - Saved refined summary to {refine_path}")
            refinement_results.append({"id": sample_id, "refined_summary": refinement_response})

        eval_csv_path = os.path.join(round_output_dir, "evaluations.csv")
        pd.DataFrame(evaluation_results).to_csv(eval_csv_path, index=False)
        print(f"\nConsolidated evaluations for Round {round_num} saved to {eval_csv_path}")

        refine_csv_path = os.path.join(round_output_dir, "refinements.csv")
        pd.DataFrame(refinement_results).to_csv(refine_csv_path, index=False)
        print(f"Consolidated refinements for Round {round_num} saved to {refine_csv_path}")

        print(f"\nRound {round_num} evaluation and refinement complete.")
        return refinement_dir
