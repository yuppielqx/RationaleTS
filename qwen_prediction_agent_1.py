import base64
import io
from PIL import Image
from openai import OpenAI
import os
import random
import pandas as pd
import pickle as pkl
import argparse
from sklearn.metrics import f1_score, roc_auc_score

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
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


def call_qwen_api(messages, model_name, max_tokens=2048):
    """A generic function to call the Qwen API with a prepared messages payload."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during API call: {e}"


def parse_llm_response(response_text):
    """Extracts the prediction tag and the reasoning text from the LLM's response."""
    prediction = -1
    reasoning = response_text
    if "<Rain>" in response_text:
        prediction = 1
        reasoning = response_text.split("<Rain>", 1)[1].strip()
    elif "<Not Rain>" in response_text:
        prediction = 0
        reasoning = response_text.split("<Not Rain>", 1)[1].strip()
    return prediction, reasoning


def run_prediction_round(round_num, train_files, image_dir, current_text_dir, base_output_dir, model_name):
    """Runs a round of inference and saves predictions and reasoning."""
    print(f"\n{'=' * 20} Starting Prediction Round {round_num} {'=' * 20}")
    round_output_dir = os.path.join(base_output_dir, f"round_{round_num}")
    reasoning_dir = os.path.join(round_output_dir, "llm_reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)

    predictions_data = []
    system_prompt = (
        "You are a meteorological reasoning expert. "
        "Your goal is to predict whether it will rain the next day based on the last 24 hours of meteorological data."
        "You must reason step-by-step, identifying relevant patterns such as humidity trends, pressure drops, temperature changes, or wind direction shifts."
        "Give your final answer and explain your reasoning clearly."
        "Never make random guesses — your reasoning must be meteorologically plausible."
        "Output both your final prediction and your reasoning."
    )

    for sample_id in train_files:
        image_filename = f"{sample_id}.png"
        image_path = os.path.join(image_dir, image_filename)
        text_path = os.path.join(current_text_dir, f"{sample_id}.txt")

        if not os.path.exists(text_path):
            print(f"Warning: Text file not found for {image_filename} in {current_text_dir}. Skipping.")
            continue

        with open(text_path, 'r') as f:
            text_content = f.read().strip()

        print(f"\nProcessing for Round {round_num}: {image_filename}")

        prompt_part_1 = (
            f"Analyze the provided time-series chart and the following weather summary. Based on all this information, will it rain in the next 24 hours?\n\n"
            f"Weather Summary: {text_content}"
        )
        prompt_part_2 = "Format your prediction as either <Rain> or <Not Rain>, followed by your reasoning."

        content_parts = []
        content_parts.append({"type": "text", "text": prompt_part_1})
        image_b64 = encode_image_to_base64(image_path)
        if image_b64:
            content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"})
        content_parts.append({"type": "text", "text": prompt_part_2})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts}
        ]

        response = call_qwen_api(messages, model_name=model_name)

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


def consolidate_data_for_round(round_output_dir, original_text_dir, rain_pkl_path):
    """Consolidates data for a given round, calculates metrics, and returns the F1 score."""
    print(f"\n--- Consolidating and Evaluating data for Round output at {round_output_dir} ---")
    predictions_csv_path = os.path.join(round_output_dir, "llm_predictions.csv")
    reasoning_dir = os.path.join(round_output_dir, "llm_reasoning")
    pred_df = pd.read_csv(predictions_csv_path)

    with open(rain_pkl_path, 'rb') as f:
        rain_data = np.array(pkl.load(f)).astype(np.int64)

    with open('./dataset/weather_ny/indices.pkl', 'rb') as f:
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

    merged_df['original_summary'] = merged_df['id'].apply(
        lambda x: open(os.path.join(original_text_dir, f"{x}.txt")).read().strip())
    merged_df['llm_reasoning'] = merged_df['id'].apply(
        lambda x: open(os.path.join(reasoning_dir, f"{x}_reasoning.txt")).read().strip())

    consolidated_csv_path = os.path.join(round_output_dir, "consolidated_data.csv")
    merged_df.to_csv(consolidated_csv_path, index=False)
    print(f"Consolidated data for this round saved to {consolidated_csv_path}")

    f1 = f1_score(merged_df['true_label'], merged_df['llm_prediction'])
    auc = roc_auc_score(merged_df['true_label'], merged_df['llm_prediction'])
    print(f"--- Round Metrics --- F1: {f1:.4f}, AUC: {auc:.4f} ---")

    return consolidated_csv_path, f1


def evaluation_agent(true_label, llm_prediction, reasoning, image_path, model_name):
    """Builds a prompt and calls the API to evaluate the correctness of a prediction."""
    system_prompt = (
        "You are an evaluation agent specialized in analyzing reasoning errors of meteorological models."
        "Your task is to:"
        "1. Compare the prediction with the ground truth."
        "2. Identify errors or weaknesses in the reasoning process."
        "3. Suggest improvements to the prediction prompt so that future predictions are more accurate and physically consistent."
        "Be specific in diagnosing which aspects of reasoning (e.g., pressure interpretation, trend analysis, humidity threshold) were incorrect or incomplete."
        "Do not re-predict the weather; just analyze the reasoning quality and suggest prompt refinements."
    )
    prompt_part_1 = (
        f"Please evaluate the following prediction based on the evidence provided.\n\n"
        f"- Ground Truth: {'Rain' if true_label == 1 else 'Not Rain'}\n"
        f"- LLM's Prediction: {'Rain' if llm_prediction == 1 else 'Not Rain'}\n"
        f"- LLM's Reasoning for its prediction: {reasoning}"
    )
    prompt_part_2 = (
        f"Now, looking at the provided time-series chart, please perform your analysis as instructed in your system role."
    )

    content_parts = [{"type": "text", "text": prompt_part_1}]
    image_b64 = encode_image_to_base64(image_path)
    if image_b64:
        content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"})
    content_parts.append({"type": "text", "text": prompt_part_2})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts}
    ]

    return call_qwen_api(messages, model_name=model_name)


def refinement_agent(summary, evaluation_analysis, model_name):
    """Builds a prompt and calls the API to refine a summary based on an evaluation."""
    system_prompt = (
        "You are a text summary optimization expert. "
        "Your task is to rewrite and improve a weather summary based on a critical evaluation of a previous attempt to improve its meteorological reasoning accuracy."
    )
    user_prompt = (
        f"An initial weather summary was written, but a follow-up evaluation found potential flaws. Use the evaluation to write a new, improved weather summary.\n\n"
        f"- Original Weather Summary: {summary}\n"
        f"- Evaluation of the original summary's reasoning: {evaluation_analysis}\n\n"
        f"Please write a new, refined, and more accurate weather summary based on the feedback in the evaluation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return call_qwen_api(messages, model_name=model_name)


def run_evaluation_and_refinement_round(round_num, consolidated_csv_path, image_dir, base_output_dir, model_name,
                                        refinement_model_name):
    """Runs the evaluation and refinement stages for a given round."""
    print(f"\n{'=' * 20} Starting Evaluation & Refinement for Round {round_num} {'=' * 20}")
    round_output_dir = os.path.join(base_output_dir, f"round_{round_num}")
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
        image_path = os.path.join(image_dir, f"{sample_id}.png")

        # 1. Call Evaluation Agent (uses the main VLM)
        print("  - Calling Evaluation Agent...")
        evaluation_response = evaluation_agent(
            true_label=row['true_label'],
            llm_prediction=row['llm_prediction'],
            reasoning=row['llm_reasoning'],
            image_path=image_path,
            model_name=model_name
        )
        eval_path = os.path.join(evaluation_dir, f"{sample_id}.txt")
        with open(eval_path, 'w') as f:
            f.write(evaluation_response)
        print(f"    - Saved evaluation to {eval_path}")
        evaluation_results.append({"id": sample_id, "evaluation_text": evaluation_response})

        # 2. Call Refinement Agent (uses the specified text-only LLM)
        print("  - Calling Refinement Agent...")
        refinement_response = refinement_agent(
            summary=row['original_summary'],
            evaluation_analysis=evaluation_response,
            model_name=refinement_model_name
        )
        refine_path = os.path.join(refinement_dir, f"{sample_id}.txt")
        with open(refine_path, 'w') as f:
            f.write(refinement_response)
        print(f"    - Saved refined summary to {refine_path}")
        refinement_results.append({"id": sample_id, "refined_summary": refinement_response})

    # Save consolidated results for the round
    eval_csv_path = os.path.join(round_output_dir, "evaluations.csv")
    pd.DataFrame(evaluation_results).to_csv(eval_csv_path, index=False)
    print(f"\nConsolidated evaluations for Round {round_num} saved to {eval_csv_path}")

    refine_csv_path = os.path.join(round_output_dir, "refinements.csv")
    pd.DataFrame(refinement_results).to_csv(refine_csv_path, index=False)
    print(f"Consolidated refinements for Round {round_num} saved to {refine_csv_path}")

    print(f"\nRound {round_num} evaluation and refinement complete.")
    return refinement_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a multi-round Prediction-Evaluation-Refinement loop.')
    parser.add_argument('-N', '--num_loops', type=int, default=2, help='Number of times to run the refinement loop.')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of rounds to wait for F1 score improvement before stopping.')
    parser.add_argument('--dataset_name', type=str, default='weather_ny',
                        help='Name of the dataset (e.g., weather_ny).')
    parser.add_argument('--model_name', type=str, default='qwen3-vl-plus',
                        help='The primary VLM model for prediction and evaluation.')
    parser.add_argument('--refinement_model_name', type=str, default='qwen3-max',
                        help='The text-only LLM for the refinement agent.')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory containing the input images.')
    parser.add_argument('--text_dir', type=str, default=None, help='Directory containing the original text summaries.')
    parser.add_argument('--rain_pkl_path', type=str, default=None, help='Path to the rain labels pkl file.')
    parser.add_argument('--base_output_dir', type=str, default=None, help='Base directory to save all round outputs.')

    args = parser.parse_args()

    # --- Dynamically construct paths ---
    if args.image_dir is None: args.image_dir = f'./dataset/{args.dataset_name}/images/'
    if args.text_dir is None: args.text_dir = f'./dataset/{args.dataset_name}/txt/'
    if args.rain_pkl_path is None: args.rain_pkl_path = f'./dataset/{args.dataset_name}/rain.pkl'
    if args.base_output_dir is None: args.base_output_dir = f'./expl_results/{args.dataset_name}/'

    # --- Setup for Resume and Early Stopping ---
    start_round = 1
    current_text_dir = args.text_dir
    for i in range(args.num_loops, 0, -1):
        potential_refinement_dir = os.path.join(args.base_output_dir, f'round_{i}', 'refinement')
        if os.path.exists(potential_refinement_dir):
            print(f"Found completed round {i}. Resuming from round {i + 1}.")
            start_round = i + 1
            current_text_dir = potential_refinement_dir
            break

    best_f1_score = -1
    early_stopping_counter = 0

    # --- Master Loop ---
    all_files = sorted([int(f.split('.')[0]) for f in os.listdir(args.image_dir) if f.lower().endswith('.png')])
    split_index = int(len(all_files) * 0.8)
    train_files = all_files[:split_index]
    print(f"Found {len(train_files)} training files to process from dataset: {args.dataset_name}")

    for n in range(start_round, args.num_loops + 1):
        # 1. Prediction Round
        round_output_dir = run_prediction_round(n, train_files, args.image_dir, current_text_dir, args.base_output_dir,
                                                args.model_name)

        # 2. Consolidation and Evaluation Round
        consolidated_csv, f1_score_current = consolidate_data_for_round(round_output_dir, args.text_dir,
                                                                        args.rain_pkl_path)

        # 3. Early Stopping Check
        if f1_score_current > best_f1_score:
            best_f1_score = f1_score_current
            early_stopping_counter = 0
            print(f"F1 score improved to {best_f1_score:.4f}. Resetting early stopping counter.")
        else:
            early_stopping_counter += 1
            print(f"F1 score did not improve. Counter: {early_stopping_counter}/{args.patience}")

        if early_stopping_counter >= args.patience:
            print("F1 score has not improved for {args.patience} rounds. Stopping early.")
            break

        # 4. Evaluation and Refinement Round
        refined_text_dir = run_evaluation_and_refinement_round(n, consolidated_csv, args.image_dir,
                                                               args.base_output_dir, args.model_name,
                                                               args.refinement_model_name)

        # 5. Set up the input for the next iteration
        current_text_dir = refined_text_dir

    print(f"\n\n{'=' * 30} Full Pipeline Complete {'=' * 30}")
    print(f"Completed processing up to round {n} for dataset '{args.dataset_name}'.")
