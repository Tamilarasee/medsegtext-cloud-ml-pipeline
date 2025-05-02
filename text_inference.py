# inference.py (or add to notebook)

import torch
import torch.nn as nn # Needed for generate_square_subsequent_mask
import torch.nn.functional as F # Needed for interpolate/sigmoid
from text_utils import load_tokenizer # Assuming tokenizer is loaded here or passed
from med_seg_text_model import JointSegTextUNet # To load model structure
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score # MOD: Import METEOR
from rouge_score import rouge_scorer # MOD: Import ROUGE scorer
import time
import warnings # To suppress potential warnings
from train import dice_score, iou_score # Import metrics calculation


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

# --- MOD: Updated Evaluation Function for Joint Output ---
def evaluate_model(model_path, model_config, test_loader, tokenizer, device, display_limit=5):
    """
    Loads a JOINTLY trained model, evaluates it on the test set, calculates
    segmentation (Dice/IoU) and text (BLEU/METEOR/ROUGE) metrics,
    and displays results for a limited number of samples.
    """
    print("--- Starting Evaluation (Joint Model) ---")
    warnings.filterwarnings("ignore")
    pad_id = model_config['pad_token_id']
    max_text_len = model_config['max_text_seq_len']

    # --- Load Model ---
    print("Initializing model for inference...")
    try:
        model_inf = JointSegTextUNet(**model_config).to(device)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model_inf.load_state_dict(state_dict)
        model_inf.to(device)
        print(f"Model weights loaded successfully from {model_path}")
        model_inf.eval()
    except Exception as e:
        print(f"Error loading model: {e}"); warnings.filterwarnings("default"); return None

    # --- NLTK Data Check --- (Keep existing block)
    try: nltk.data.find('tokenizers/punkt')
    except: print("Downloading punkt..."); nltk.download('punkt', quiet=True)
    try: nltk.data.find('corpora/wordnet')
    except: print("Downloading wordnet..."); nltk.download('wordnet', quiet=True)
    # try: nltk.data.find('????/omw-1.4') # Wordnet might require omw-1.4 too
    # except: nltk.download('omw-1.4', quiet=True) # Uncomment if meteor throws omw error


    # --- Evaluation Loop ---
    predictions_tokenized = []
    references_tokenized = []
    predictions_raw = []
    references_raw = []
    # MOD: Accumulators for segmentation metrics
    total_dice_score = 0.0
    total_iou_score = 0.0

    samples_processed = 0
    samples_displayed = 0
    eval_start_time = time.time()

    print(f"Evaluating on {len(test_loader.dataset)} test samples...")
    try:
        # --- Loop through ENTIRE test loader ---
        for batch_idx, (images_batch, masks_batch, _, target_ids_batch, _) in enumerate(test_loader): # Get masks now

            # --- Iterate through items in the current batch ---
            for i in range(images_batch.size(0)):
                single_image_tensor = images_batch[i].to(device)
                single_mask_tensor = masks_batch[i].to(device) # Ground truth mask
                ground_truth_ids = target_ids_batch[i].tolist()

                # --- Generate Segmentation Prediction ---
                seg_output_logits = None
                pred_mask_display = None # For visualization
                try:
                     # Call model in segmentation mode
                     seg_output_logits = model_inf(image=single_image_tensor.unsqueeze(0), mode='segmentation') # Add batch dim for model call
                     # Calculate Dice/IoU for ALL samples (using logits)
                     batch_dice = dice_score(seg_output_logits, single_mask_tensor.unsqueeze(0)).item()
                     batch_iou = iou_score(seg_output_logits, single_mask_tensor.unsqueeze(0)).item()
                     total_dice_score += batch_dice
                     total_iou_score += batch_iou
                     # Prepare mask for display (only needed for first few)
                     if samples_displayed < display_limit:
                          pred_mask_display = torch.sigmoid(seg_output_logits).squeeze().detach().cpu().numpy() > 0.5
                except Exception as seg_e:
                     print(f"Warning: Error during segmentation for sample {samples_processed + 1}: {seg_e}")
                     # Decide if you want to skip text generation too if seg fails
                     # samples_processed += 1; continue # Option: Skip sample entirely

                # --- Process Ground Truth Text ---
                ground_truth_text = "[Error decoding]"
                reference_tokens = []
                try:
                    if pad_id in ground_truth_ids:
                        first_pad_index = ground_truth_ids.index(pad_id)
                        ground_truth_ids = ground_truth_ids[:first_pad_index]
                    ground_truth_text = tokenizer.decode(ground_truth_ids, skip_special_tokens=True)
                    reference_tokens = nltk.word_tokenize(ground_truth_text.lower())
                    references_raw.append(ground_truth_text)
                    references_tokenized.append([reference_tokens])
                except Exception as decode_e:
                    print(f"Warning: Error decoding GT for sample {samples_processed + 1}: {decode_e}")
                    samples_processed += 1; continue # Skip text metrics for this sample

                # --- Generate Predicted Text ---
                generated_finding = "[Error generating]"
                predicted_tokens = []
                try:
                    generated_finding = generate_finding( # Calls internal encoder again
                        model=model_inf, tokenizer=tokenizer, image_tensor=single_image_tensor,
                        device=device, max_length=max_text_len
                    )
                    predicted_tokens = nltk.word_tokenize(generated_finding.lower())
                    predictions_raw.append(generated_finding)
                    predictions_tokenized.append(predicted_tokens)
                except Exception as gen_e:
                    print(f"Warning: Error generating finding for sample {samples_processed + 1}: {gen_e}")
                    references_raw.pop(); references_tokenized.pop() # Remove corresponding ref
                    samples_processed += 1; continue # Skip text metrics for this sample

                # --- Display if limit not reached ---
                if samples_displayed < display_limit:
                    print(f"\n--- Displaying Sample {samples_processed + 1} ---")
                    # --- Display Image, GT Mask, Pred Mask ---
                    try:
                        img_display = single_image_tensor.cpu().permute(1, 2, 0).numpy()
                        gt_mask_display = single_mask_tensor.squeeze().cpu().numpy()
                        img_display = np.clip(img_display, 0, 1)

                        # Create figure with 3 subplots
                        fig, axes = plt.subplots(1, 3, figsize=(15, 5)) # Adjust figsize

                        # Original Image
                        if img_display.shape[2] == 1: axes[0].imshow(img_display.squeeze(), cmap='gray')
                        else: axes[0].imshow(img_display)
                        axes[0].set_title(f"Original Image (Sample {samples_processed + 1})")
                        axes[0].axis('off')

                        # Ground Truth Mask
                        axes[1].imshow(gt_mask_display, cmap='gray')
                        axes[1].set_title("Ground Truth Mask")
                        axes[1].axis('off')

                        # Predicted Mask
                        if pred_mask_display is not None:
                             axes[2].imshow(pred_mask_display, cmap='gray')
                             axes[2].set_title(f"Predicted Mask\nDice: {batch_dice:.4f}, IoU: {batch_iou:.4f}")
                             axes[2].axis('off')
                        else:
                             axes[2].set_title("Pred Mask Error")
                             axes[2].axis('off')

                        plt.tight_layout()
                        plt.show()
                    except Exception as plot_e:
                        print(f"Error displaying images/masks: {plot_e}")

                    # --- Print Text ---
                    print(f"Ground Truth Text: {ground_truth_text}")
                    print(f"Predicted Text:    {generated_finding}")
                    samples_displayed += 1

                samples_processed += 1

            if batch_idx % 10 == 0 and batch_idx > 0: print(f"  Processed {samples_processed}/{len(test_loader.dataset)} samples...")

    except Exception as loop_e: print(f"\nAn error occurred: {loop_e}"); import traceback; traceback.print_exc()

    eval_end_time = time.time(); print(f"\nFinished loop ({samples_processed} samples) in {eval_end_time - eval_start_time:.2f}s.")

    # --- Calculate Metrics ---
    results = {}
    num_valid_text_samples = len(predictions_tokenized)
    num_seg_samples = samples_processed # Or count samples where seg succeeded if errors happened

    # Calculate Text Metrics
    if num_valid_text_samples > 0 and num_valid_text_samples == len(references_tokenized):
        print(f"\nCalculating Text Metrics (Based on {num_valid_text_samples} Valid Samples)...")
        # BLEU
        chencherry = SmoothingFunction(); bleu_scores = {f'BLEU-{i}': 0.0 for i in range(1, 5)}
        weights = [(1,0,0,0), (0.5,0.5,0,0), (0.33,0.33,0.33,0), (0.25,0.25,0.25,0.25)]
        for i in range(num_valid_text_samples):
            for j, w in enumerate(weights):
                bleu_scores[f'BLEU-{j+1}'] += sentence_bleu(references_tokenized[i], predictions_tokenized[i], weights=w, smoothing_function=chencherry.method1)
        for k in bleu_scores: results[k] = bleu_scores[k] / num_valid_text_samples
        # METEOR
        total_meteor = sum(meteor_score(references_tokenized[i], predictions_tokenized[i]) for i in range(num_valid_text_samples))
        results['METEOR'] = total_meteor / num_valid_text_samples
        # ROUGE-L
        rouge_l_f1 = 0.0; scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        for i in range(num_valid_text_samples): rouge_l_f1 += scorer.score(references_raw[i], predictions_raw[i])['rougeL'].fmeasure
        results['ROUGE-L'] = rouge_l_f1 / num_valid_text_samples
    else:
        print("Could not calculate text metrics.")

    # MOD: Calculate Segmentation Metrics
    if num_seg_samples > 0:
         print(f"\nCalculating Segmentation Metrics (Based on {num_seg_samples} Samples)...")
         results['Avg Dice'] = total_dice_score / num_seg_samples
         results['Avg IoU'] = total_iou_score / num_seg_samples
    else:
         print("Could not calculate segmentation metrics.")


    # --- Print Final Results ---
    print("\n--- Overall Evaluation Metrics ---")
    if results:
        for metric, score in results.items(): print(f"  {metric}: {score:.4f}")
    else:
        print("  No valid metrics calculated.")

    warnings.filterwarnings("default")
    print("--- Evaluation Finished ---")
    return results
