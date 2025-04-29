# inference.py (or add to notebook)

import torch
import torch.nn as nn # Needed for generate_square_subsequent_mask
from text_utils import load_tokenizer # Assuming tokenizer is loaded here or passed
from seg_text_unet_model import JointSegTextUNet # To load model structure
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score # MOD: Import METEOR
from rouge_score import rouge_scorer # MOD: Import ROUGE scorer
import time
import warnings # To suppress potential warnings


@torch.no_grad() # Ensure no gradients are computed during inference
def generate_finding(model, tokenizer, image_tensor, device, max_length=50):
    """
    Generates a text finding for a single input image tensor using autoregressive decoding.

    Args:
        model: The trained JointSegTextUNet model with loaded weights.
        tokenizer: The loaded tokenizer.
        image_tensor: A single preprocessed image tensor (C, H, W) already on the correct device.
        device: The torch device ('cuda' or 'cpu').
        max_length: Maximum number of tokens to generate.

    Returns:
        A string containing the generated finding.
    """
    model.eval() # Set model to evaluation mode

    # --- 1. Add Batch Dimension to Image ---
    image_batch = image_tensor.unsqueeze(0).to(device) # Shape: [1, C, H, W]

    # --- 2. Encode Image to get Memory ---
    # Manually run the visual encoding part of the model's forward pass
    f0 = model.conv_input(image_batch)
    f1 = model.visual_encoder.stage1(f0)
    f2 = model.visual_encoder.stage2(f1)
    f3 = model.visual_encoder.stage3(f2)
    f4 = model.visual_encoder.stage4(f3)
    memory = f4.flatten(2).permute(0, 2, 1) # (B=1, H'*W', C)
    memory = model.visual_feature_proj(memory) # (B=1, 49, embed_dim)

    # --- 3. Initialize Sequence with SOS token ---
    # Shape: [1, 1] (Batch Size = 1, Sequence Length = 1)
    current_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

    # --- 4. Autoregressive Loop ---
    for _ in range(max_length - 1): # Max length includes SOS, so loop max_length-1 times
        # --- Prepare decoder inputs ---
        tgt_embed = model.text_embedding(current_ids) # (1, current_seq_len, embed_dim)

        # Apply positional encoding (handling batch_first convention)
        tgt_embed_pe = tgt_embed.permute(1, 0, 2) # (current_seq_len, 1, embed_dim)
        tgt_embed_pe = model.positional_encoding(tgt_embed_pe)
        tgt_embed_pe = tgt_embed_pe.permute(1, 0, 2) # (1, current_seq_len, embed_dim)

        # Create causal mask for the current sequence length
        current_seq_len = current_ids.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len, device=device) # (current_seq_len, current_seq_len)

        # Pass through the Transformer Decoder
        # NOTE: No padding mask needed as we generate one sequence without padding
        decoder_output = model.transformer_decoder(
            tgt=tgt_embed_pe,
            memory=memory,
            tgt_mask=tgt_mask
            # tgt_key_padding_mask=None # No padding during generation
        ) # Output shape: (1, current_seq_len, embed_dim)

        # Get logits for the *last* token prediction
        last_token_logits = model.text_output_layer(decoder_output[:, -1, :]) # Shape: (1, vocab_size)

        # Find the token ID with the highest probability (greedy search)
        predicted_id = torch.argmax(last_token_logits, dim=-1) # Shape: (1)

        # Append the predicted ID to the sequence
        # Shape becomes: [1, current_seq_len + 1]
        current_ids = torch.cat([current_ids, predicted_id.unsqueeze(0)], dim=1)

        # Stop if EOS token is predicted
        if predicted_id.item() == tokenizer.eos_token_id:
            break

    # --- 5. Decode IDs to Text ---
    # Exclude SOS token (index 0) and potentially EOS token at the end
    generated_ids = current_ids[0, 1:].tolist() # Get IDs as a list, remove SOS
    if generated_ids[-1] == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1] # Remove EOS if present

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False) # Set skip_special_tokens as desired

    return generated_text

# --- MOD: New Evaluation Function ---
def evaluate_model(model_path, model_config, test_loader, tokenizer, device, display_limit=5):
    """
    Loads a trained model, evaluates it on the test set, calculates BLEU, METEOR, ROUGE scores,
    and displays results for a limited number of samples.

    Args:
        model_path (str): Path to the saved model state_dict (.pth file).
        model_config (dict): Dictionary containing model hyperparameters
                               (vocab_size, embed_dim, nhead, etc., including pad_token_id).
        test_loader (DataLoader): DataLoader for the test set.
        tokenizer: The loaded tokenizer instance.
        device: The torch device ('cuda' or 'cpu').
        display_limit (int): How many samples to display image/text for.

    Returns:
        dict: A dictionary containing the calculated metrics.
              Returns None if evaluation fails.
    """

    print("--- Starting Evaluation ---")
    # Suppress warnings during evaluation (e.g., from rouge_score)
    warnings.filterwarnings("ignore")

    pad_id = model_config['pad_token_id']
    max_text_len = model_config['max_text_seq_len'] # Get max length from config

    # --- Load Model ---
    print("Initializing model for inference...")
    try:
        model_inf = JointSegTextUNet(**model_config).to(device) # Unpack config dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_inf.load_state_dict(state_dict)
        model_inf.to(device)
        print(f"Model weights loaded successfully from {model_path}")
        model_inf.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        warnings.filterwarnings("default") # Restore warnings
        return None

    # --- Download NLTK data if needed ---
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)

    # --- Evaluation Loop ---
    # MOD: Lists to store texts for different metrics
    predictions_tokenized = []
    references_tokenized = [] # List of lists [[ref1_toks], [ref2_toks], ...]
    predictions_raw = []
    references_raw = []

    samples_processed = 0
    samples_displayed = 0
    eval_start_time = time.time()

    print(f"Evaluating on {len(test_loader.dataset)} test samples...")
    try:
        for batch_idx, (images_batch, _, _, target_ids_batch, _) in enumerate(test_loader):
            for i in range(images_batch.size(0)):
                single_image_tensor = images_batch[i].to(device)
                ground_truth_ids = target_ids_batch[i].tolist()

                # Process Ground Truth
                ground_truth_text = "[Error decoding]"
                reference_tokens = []
                try:
                    if pad_id in ground_truth_ids:
                        first_pad_index = ground_truth_ids.index(pad_id)
                        ground_truth_ids = ground_truth_ids[:first_pad_index]
                    ground_truth_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                    reference_tokens = nltk.word_tokenize(ground_truth_text.lower())
                    references_raw.append(ground_truth_text) # Store raw for ROUGE
                    references_tokenized.append([reference_tokens]) # Store tokenized list (in list) for BLEU/METEOR
                except Exception as decode_e:
                    print(f"Warning: Error decoding GT for sample {samples_processed + 1}: {decode_e}")
                    samples_processed += 1
                    continue # Skip sample

                # Generate Prediction
                generated_finding = "[Error generating]"
                predicted_tokens = []
                try:
                    # Use the existing generate_finding function
                    generated_finding = generate_finding(
                        model=model_inf, tokenizer=tokenizer, image_tensor=single_image_tensor,
                        device=device, max_length=max_text_len
                    )
                    predicted_tokens = nltk.word_tokenize(generated_finding.lower())
                    predictions_raw.append(generated_finding) # Store raw for ROUGE
                    predictions_tokenized.append(predicted_tokens) # Store tokenized for BLEU/METEOR
                except Exception as gen_e:
                    print(f"Warning: Error generating for sample {samples_processed + 1}: {gen_e}")
                    # Remove corresponding reference if prediction failed
                    references_raw.pop(); references_tokenized.pop()
                    samples_processed += 1
                    continue # Skip sample

                # Display if limit not reached
                if samples_displayed < display_limit:
                    print(f"\n--- Displaying Sample {samples_processed + 1} ---")
                    try:
                        img_display = single_image_tensor.cpu().permute(1, 2, 0).numpy()
                        img_display = np.clip(img_display, 0, 1)
                        plt.figure(figsize=(5, 5))
                        if img_display.shape[2] == 1: plt.imshow(img_display.squeeze(), cmap='gray')
                        else: plt.imshow(img_display)
                        plt.title(f"Original Image (Sample {samples_processed + 1})")
                        plt.axis('off')
                        plt.show()
                    except Exception as plot_e:
                        print(f"Error displaying image: {plot_e}")
                    print(f"Ground Truth: {ground_truth_text}")
                    print(f"Predicted:    {generated_finding}")
                    samples_displayed += 1

                samples_processed += 1

            if batch_idx % 10 == 0 and batch_idx > 0:
                 print(f"  Processed {samples_processed}/{len(test_loader.dataset)} samples...")

    except Exception as loop_e:
         print(f"\nAn error occurred during the evaluation loop: {loop_e}")
         import traceback
         traceback.print_exc()

    eval_end_time = time.time()
    print(f"\nFinished evaluation loop ({samples_processed} samples processed) in {eval_end_time - eval_start_time:.2f} seconds.")

    # --- Calculate Metrics ---
    results = {}
    num_valid_samples = len(predictions_tokenized)
    if num_valid_samples > 0 and num_valid_samples == len(references_tokenized) == len(predictions_raw) == len(references_raw):
        print(f"\nCalculating Metrics (Based on {num_valid_samples} Valid Samples)...")

        # BLEU
        chencherry = SmoothingFunction()
        bleu_scores = {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-3': 0.0, 'BLEU-4': 0.0}
        for i in range(num_valid_samples):
             bleu_scores['BLEU-1'] += sentence_bleu(references_tokenized[i], predictions_tokenized[i], weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
             bleu_scores['BLEU-2'] += sentence_bleu(references_tokenized[i], predictions_tokenized[i], weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
             bleu_scores['BLEU-3'] += sentence_bleu(references_tokenized[i], predictions_tokenized[i], weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
             bleu_scores['BLEU-4'] += sentence_bleu(references_tokenized[i], predictions_tokenized[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        for k in bleu_scores: results[k] = bleu_scores[k] / num_valid_samples

        # METEOR
        total_meteor = 0.0
        for i in range(num_valid_samples):
            # meteor_score expects list of reference lists, and a hypothesis list
            total_meteor += meteor_score(references_tokenized[i], predictions_tokenized[i])
        results['METEOR'] = total_meteor / num_valid_samples

        # ROUGE-L
        rouge_l_f1 = 0.0
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for i in range(num_valid_samples):
            # ROUGE scorer takes raw strings
            scores = scorer.score(references_raw[i], predictions_raw[i])
            rouge_l_f1 += scores['rougeL'].fmeasure # F1-score for ROUGE-L
        results['ROUGE-L'] = rouge_l_f1 / num_valid_samples

        print("Metrics Calculation Complete.")
        # Print nicely formatted results
        for metric, score in results.items(): print(f"  {metric}: {score:.4f}")

    else:
        print("Could not calculate metrics (not enough valid prediction/reference pairs found).")

    warnings.filterwarnings("default") # Restore warnings
    print("--- Evaluation Finished ---")
    return results
