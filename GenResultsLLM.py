import numpy as np
import pickle as pkl
import os
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from PatchTST import Model as PatchTST
from patchtst_trainer import Trainer
import argparse
import base64
import io
from PIL import Image
from openai import OpenAI


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


def construct_prompt(test_text_content, test_image_b64, top_k_class_0, top_k_class_1, k, image_base_path,
                     text_base_path):
    """
    Constructs the detailed multi-modal prompt for the vision-language model.
    """
    content_parts = []

    # Add the main instruction
    content_parts.append({
        "type": "text",
        "text": "You are an expert weather forecaster. Your task is to predict whether it will rain based on a primary time-series chart and its text summary. To help you, I will provide several historical examples that a separate analysis model has identified as being highly influential. Analyze all the provided information and make a final prediction with a brief explanation."
    })

    # Add the primary test text summary
    content_parts.append({"type": "text", "text": f"\\n--- Primary Text Summary to Analyze ---\\n{test_text_content}"})

    # Add the primary test image to analyze
    if test_image_b64:
        content_parts.append({"type": "text", "text": "\\n--- Primary Chart to Analyze ---"})
        content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{test_image_b64}"})

    # Add the influential "Not Rain" examples
    content_parts.append(
        {"type": "text", "text": f"\\n--- Top {k} Influential Historical Examples (True Label: Not Rain) ---"})
    for i, (kl, label, orig_idx, prob_0) in enumerate(top_k_class_0):
        img_path = os.path.join(image_base_path, f's_{orig_idx}.png')
        txt_path = os.path.join(text_base_path, f's_{orig_idx}.txt')
        try:
            with open(txt_path, 'r') as f:
                txt_content = f.read().strip()
        except FileNotFoundError:
            txt_content = "Text summary not found."
        content_parts.append({"type": "text", "text": f"\\nExample {i + 1} (Not Rain): {txt_content}"})
        img_b64 = encode_image_to_base64(img_path)
        if img_b64:
            content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"})

    # Add the influential "Rain" examples
    content_parts.append(
        {"type": "text", "text": f"\\n--- Top {k} Influential Historical Examples (True Label: Rain) ---"})
    for i, (kl, label, orig_idx, prob_0) in enumerate(top_k_class_1):
        img_path = os.path.join(image_base_path, f's_{orig_idx}.png')
        txt_path = os.path.join(text_base_path, f's_{orig_idx}.txt')
        try:
            with open(txt_path, 'r') as f:
                txt_content = f.read().strip()
        except FileNotFoundError:
            txt_content = "Text summary not found."
        content_parts.append({"type": "text", "text": f"\\nExample {i + 1} (Rain): {txt_content}"})
        img_b64 = encode_image_to_base64(img_path)
        if img_b64:
            content_parts.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"})

    # Add the final, FORMATTED instruction
    content_parts.append({
        "type": "text",
        "text": "\\nBased on the primary chart, its summary, and all the influential historical examples, will it rain? Please format your response as follows: First, provide your final prediction by writing either `<Rain>` or `<Not Rain>`. Second, provide a brief justification for your decision."
    })

    return content_parts


def main(args):
    """
    Main function to run the explainable inference workflow.
    """
    # --- 1. Load Data and Models ---

    # Load test data using the function from train_patchtst
    from train_patchtst import load_and_process_data
    _, _, test_loader = load_and_process_data(args)
    print("Test data loaded successfully.")

    # Load the pre-trained PatchTST model
    patchtst_model = PatchTST(args)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'patchtst_checkpoint.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        patchtst_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Please run training first.")
        return
    patchtst_model.to(device)
    patchtst_model.eval()
    print(f"PatchTST model loaded from {checkpoint_path}")

    # Load the pre-computed logits and probabilities for the train/val sets
    try:
        train_logits = np.load(os.path.join(args.checkpoint_dir, 'train_logits.npy'))
        val_logits = np.load(os.path.join(args.checkpoint_dir, 'val_logits.npy'))
        all_logits = np.concatenate((train_logits, val_logits), axis=0)

        train_probs = np.load(os.path.join(args.checkpoint_dir, 'train_probs.npy'))
        val_probs = np.load(os.path.join(args.checkpoint_dir, 'val_probs.npy'))
        all_probs = np.concatenate((train_probs, val_probs), axis=0)
    except FileNotFoundError as e:
        print(f"Error: Could not find pre-computed logit/prob files in {args.checkpoint_dir}. {e}")
        print("Please ensure the model has been trained and these files were generated.")
        return
    print(f"Loaded and combined train/val logits and probabilities.")

    # Recreate the data context to get labels and original indices
    data_path = './dataset/weather_ny/'
    with open(os.path.join(data_path, 'indices.pkl'), 'rb') as f:
        original_indices = pkl.load(f)
    with open(os.path.join(data_path, 'rain.pkl'), 'rb') as f:
        rain_labels = pkl.load(f)


    data_size = len(original_indices)
    num_train = int(data_size * 0.6)
    num_test = int(data_size * 0.2)
    num_vali = data_size - num_train - num_test

    seq_len_day = 1

    train_idx = np.arange(num_train - seq_len_day)
    val_idx = np.arange(num_train - seq_len_day, num_train + num_vali - seq_len_day)
    test_idx = np.arange(num_train + num_vali - seq_len_day, num_train + num_vali + num_test - seq_len_day)

    all_indices_map = np.concatenate((train_idx, val_idx), axis=0)
    all_original_indices = [original_indices[i] for i in all_indices_map]
    all_true_labels = [int(rain_labels[i + args.pred_len]) for i in all_indices_map]

    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # --- 2. Setup for Final Evaluation ---
    llm_predictions = []
    ground_truth_labels = []
    summary_save_dir = os.path.join(args.checkpoint_dir, 'test_summary')
    os.makedirs(summary_save_dir, exist_ok=True)
    print(f"LLM reasoning will be saved to: {summary_save_dir}")

    # --- 3. Main Loop: Iterate through each test sample ---
    for test_sample_index, (batch_x, batch_y) in enumerate(test_loader):  # Corrected loop

        # Get the correct batch index from the recreated test_idx array
        batch_idx = test_idx[test_sample_index]

        print(f"\\n{'=' * 80}")
        print(f"Processing Test Sample #{test_sample_index} | Original Index Mapping: {batch_idx}")
        print(f"{'=' * 80}")

        # --- 3a. Get PatchTST prediction for the current test sample ---
        with torch.no_grad():
            batch_x = batch_x.float().to(device)
            test_logits = patchtst_model(batch_x, None, None, None)
            test_probs = torch.softmax(test_logits, dim=1)
            _, test_prediction = torch.max(test_probs, 1)

        print(f"PatchTST Prediction: {test_prediction.item()} (True Label: {batch_y.item()})")
        print(f"PatchTST Probabilities: {test_probs.cpu().numpy()}")

        # --- 3b. Calculate KL Divergence against all train/val samples ---
        T = 2.0
        test_logits_tensor = test_logits.cpu()
        kl_divergences = []
        for i in range(all_logits.shape[0]):
            train_val_logits_tensor = torch.from_numpy(all_logits[i])
            p_test_soft = F.softmax(test_logits_tensor / T, dim=-1)
            log_q_train_val_soft = F.log_softmax(train_val_logits_tensor / T, dim=-1)
            kl_div = F.kl_div(log_q_train_val_soft.unsqueeze(0), p_test_soft, reduction='batchmean')
            kl_divergences.append(kl_div.item())

        # --- 3c. Find Top-K similar samples ---
        bound_data = list(zip(kl_divergences, all_true_labels, all_original_indices, all_probs[:, 0]))
        class_0_samples = sorted([item for item in bound_data if item[1] == 0], key=lambda x: x[0])
        class_1_samples = sorted([item for item in bound_data if item[1] == 1], key=lambda x: x[0])

        top_k_class_0 = class_0_samples[:args.k]
        top_k_class_1 = class_1_samples[:args.k]

        print(f"\\nFound Top {args.k} most similar samples for each class.")

        # --- 3d. Construct the multi-modal prompt ---
        image_base_path = './dataset/weather_ny/images/'
        text_base_path = './dataset/weather_ny/txt/'

        test_sample_orig_idx_from_batch = original_indices[batch_idx]
        test_image_path = os.path.join(image_base_path, f's_{test_sample_orig_idx_from_batch}.png')
        test_text_path = os.path.join(text_base_path, f's_{test_sample_orig_idx_from_batch}.txt')

        try:
            with open(test_text_path, 'r') as f:
                test_text_content = f.read().strip()
        except FileNotFoundError:
            test_text_content = "Primary text summary not found."

        test_image_b64 = encode_image_to_base64(test_image_path)

        content_parts = construct_prompt(test_text_content, test_image_b64, top_k_class_0, top_k_class_1, args.k,
                                         image_base_path, text_base_path)

        messages = [{"role": "user", "content": content_parts}]

        # --- 3e. Get final explanation and parse the result ---
        try:
            response = client.chat.completions.create(
                model='qwen-vl-max-0809',
                messages=messages,
                max_tokens=2048
            )
            llm_explanation = response.choices[0].message.content
            print("\\n--- Final Explanation from Vision-Language Model ---")
            print(llm_explanation)

            # Parse the prediction from the response
            if "<Rain>" in llm_explanation:
                parsed_prediction = 1
            elif "<Not Rain>" in llm_explanation:
                parsed_prediction = 0
            else:
                parsed_prediction = -1  # Sentinel value for parsing failure
                print("Warning: Could not parse <Rain> or <Not Rain> from LLM output.")

            # Accumulate results if parsing was successful
            if parsed_prediction != -1:
                llm_predictions.append(parsed_prediction)
                ground_truth_labels.append(batch_y.item())

            # Save the full reasoning text
            reason_filename = f"reason_{test_sample_orig_idx_from_batch}.txt"
            reason_save_path = os.path.join(summary_save_dir, reason_filename)
            with open(reason_save_path, 'w') as f:
                f.write(llm_explanation)
            print(f"Saved LLM reasoning to {reason_save_path}")

        except Exception as e:
            print(f"\\n--- Error during API call for sample {test_sample_index} ---")
            print(e)

    # --- 4. Final Evaluation ---
    if ground_truth_labels:  # Ensure we have results to evaluate
        f1 = f1_score(ground_truth_labels, llm_predictions)
        # Note: AUC on hard 0/1 labels is less informative but calculated as requested.
        auc = roc_auc_score(ground_truth_labels, llm_predictions)

        print("\\n" + "=" * 80)
        print("--- Final LLM Performance Metrics (based on parsed predictions) ---")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC Score: {auc:.4f}")
        print(f"Evaluated on {len(ground_truth_labels)} samples.")
        print("=" * 80)
    else:
        print("\\nNo valid LLM predictions were parsed to calculate final metrics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run explainable inference on the PatchTST model.')

    # Core arguments for inference
    parser.add_argument('--k', type=int, default=3, help='Number of top similar samples to retrieve for each class')
    parser.add_argument('--checkpoint_dir', type=str, default='expl_results/weather_ny',
                        help='Path to load checkpoints and save results')
    parser.add_argument('--seed', type=int, default=10, help='Random seed for data loading')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference (should always be 1)')

    # Model architecture arguments (required to reconstruct the model)
    parser.add_argument('--seq_len', type=int, default=24, help='Input sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='Number of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--enc_in', type=int, default=5, help='Encoder input size')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes')
    parser.add_argument('--factor', type=int, default=1, help='Attention factor')

    # Dummy argument needed by load_and_process_data
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length (dummy for this script)')

    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: For this inference script, it is highly recommended to set batch_size to 1.")

    main(args)