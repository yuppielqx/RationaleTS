import argparse
from workflow import WorkflowManager

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-round Prediction-Evaluation-Refinement loop.')
    parser.add_argument('-N', '--num_loops', type=int, default=10, help='Number of times to run the refinement loop.')
    parser.add_argument('--patience', type=int, default=2, help='Number of rounds to wait for F1 score improvement before stopping.')
    parser.add_argument('--dataset_name', type=str, default='weather_ny', help='Name of the dataset (e.g., weather_ny).')
    parser.add_argument('--model_name', type=str, default='qwen3-vl-plus', help='The primary VLM model for prediction and evaluation.')
    parser.add_argument('--refinement_model_name', type=str, default='qwen3-max', help='The text-only LLM for the refinement agent.')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing the input images.')
    parser.add_argument('--text_dir', type=str, default=None, help='Directory containing the original text summaries.')
    parser.add_argument('--rain_pkl_path', type=str, default=None, help='Path to the rain labels pkl file.')
    parser.add_argument('--base_output_dir', type=str, default=None, help='Base directory to save all round outputs.')
    args = parser.parse_args()

    manager = WorkflowManager(args)
    manager.run()
