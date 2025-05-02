import torch
import torch.optim as optim
from loss import TextCrossEntropyLoss, CombinedLoss, JointLoss
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum() + 1e-8)

def iou_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-8)

# def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
#     model.to(device)
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0
#         train_dice, train_iou = 0, 0 
#         for batch_idx, (images, masks, texts) in enumerate(train_loader):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad()
#             preds = model(images, texts)
#             loss = criterion(preds, masks)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#             # Calculate metrics on training batch
#             batch_dice = dice_score(preds, masks).item()
#             batch_iou = iou_score(preds, masks).item()
#             train_dice += batch_dice
#             train_iou += batch_iou

#             if batch_idx % 500 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}, Batch Dice: {batch_dice:.4f}, Batch IoU: {batch_iou:.4f}")

#         # Validation phase
#         val_loss = 0
#         val_dice, val_iou = 0, 0 
#         model.eval()
#         with torch.no_grad():
#             for images, masks, texts in val_loader:
#                 images, masks = images.to(device), masks.to(device)
#                 preds = model(images, texts)
#                 loss = criterion(preds, masks)
#                 val_loss += loss.item()

#                 # Calculate and accumulate metrics on validation batch
#                 val_dice += dice_score(preds, masks).item()
#                 val_iou += iou_score(preds, masks).item()

#         # Print epoch results with correct averages
#         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Train Dice: {train_dice/len(train_loader):.4f}, Train IoU: {train_iou/len(train_loader):.4f}, \n"
#               f"Val Loss: {val_loss/len(val_loader):.4f}, Val Dice: {val_dice/len(val_loader):.4f}, Val IoU: {val_iou/len(val_loader):.4f}")

#     torch.save(model.state_dict(), "mmi_unet_model.pth")
#     print("Model saved successfully!")


# MOD: Adapt the training function for Text Generation (Phase 1)
def train_text_phase(model, train_loader, val_loader, pad_token_id, epochs=20, learning_rate=1e-4):
    """Trains the text generation part of the model."""
    model.to(device)
    # MOD: Use Adam optimizer and specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # MOD: Instantiate the text loss function
    criterion = TextCrossEntropyLoss(pad_token_id=pad_token_id).to(device)

    print("Starting Text Generation Training Phase...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # train_dice, train_iou = 0, 0 # Comment out segmentation metrics

        # MOD: Update loop variable names based on collate_fn output
        for batch_idx, (images, _, input_ids, target_ids, target_padding_mask) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            input_ids = input_ids.to(device)      # Decoder input
            target_ids = target_ids.to(device)    # Loss target
            # target_padding_mask = target_padding_mask.to(device) # Needed if model uses it explicitly, but loss handles padding

            # --- MOD: Add debug prints for input_ids range ---
            if batch_idx == 0 and epoch == 0: # Print only for the very first batch
                 print(f"DEBUG: input_ids shape: {input_ids.shape}")
                 print(f"DEBUG: input_ids min value: {torch.min(input_ids)}")
                 print(f"DEBUG: input_ids max value: {torch.max(input_ids)}")
                 print(f"DEBUG: Expected vocab size (num_embeddings): {model.text_embedding.num_embeddings}") # Access vocab size from embedding layer
                 # Also check target_ids just in case, though error points to input_ids via tgt_text_indices
                 print(f"DEBUG: target_ids min value: {torch.min(target_ids)}")
                 print(f"DEBUG: target_ids max value: {torch.max(target_ids)}")
            # --- End debug prints ---

            optimizer.zero_grad()

            # MOD: Call model in 'text' mode and pass correct arguments
            # Assumes model forward signature is: forward(self, image, tgt_text_indices=None, ...)
            # and tgt_text_indices corresponds to input_ids
            # The model internally handles masks based on padding/causality
            text_logits = model(image=images, tgt_text_indices=input_ids, mode='text')

            # MOD: Calculate text loss
            loss = criterion(text_logits, target_ids)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # batch_dice = dice_score(preds, masks).item() # Comment out segmentation metrics
            # batch_iou = iou_score(preds, masks).item() # Comment out segmentation metrics
            # train_dice += batch_dice
            # train_iou += batch_iou

            if batch_idx > 0 and batch_idx % 100 == 0: # Log every 100 batches
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}] - Train Loss: {loss.item():.4f}")

        # Validation phase
        val_loss = 0
        # val_dice, val_iou = 0, 0 # Comment out segmentation metrics
        model.eval()
        print(f"--- Running Validation for Epoch {epoch+1} ---")
        with torch.no_grad():
            # MOD: Update loop variable names
            for images, _, input_ids, target_ids, target_padding_mask in val_loader:
                # Move data to device
                images = images.to(device)
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                # target_padding_mask = target_padding_mask.to(device)

                # MOD: Call model in 'text' mode
                text_logits = model(image=images, tgt_text_indices=input_ids, mode='text')

                # MOD: Calculate text loss
                loss = criterion(text_logits, target_ids)
                val_loss += loss.item()

                # val_dice += dice_score(preds, masks).item() # Comment out segmentation metrics
                # val_iou += iou_score(preds, masks).item() # Comment out segmentation metrics

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} Completed:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Val Loss:   {avg_val_loss:.4f}")
        # print(f"  Train Dice: {train_dice/len(train_loader):.4f}, Train IoU: {train_iou/len(train_loader):.4f}") # Comment out
        # print(f"  Val Dice: {val_dice/len(val_loader):.4f}, Val IoU: {val_iou/len(val_loader):.4f}") # Comment out
        print("-" * 30)

    # MOD: Save model with a different name maybe
    save_path = "medsegtext_text_phase.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}!")


# MOD: Add Joint Training Function
def train_joint_phase(model, train_loader, val_loader, pad_token_id,
                      epochs=20, learning_rate=1e-4,
                      seg_loss_weight=1.0, text_loss_weight=1.0, model_save_path="medsegtext_joint.pth"):
    """Trains both segmentation and text generation parts of the model jointly."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Use the new JointLoss criterion
    criterion = JointLoss(
        pad_token_id=pad_token_id,
        seg_loss_weight=seg_loss_weight,
        text_loss_weight=text_loss_weight
    ).to(device)

    print(f"Starting Joint Training Phase (Seg Weight: {seg_loss_weight}, Text Weight: {text_loss_weight})...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_train_seg_loss = 0
        total_train_text_loss = 0

        for batch_idx, (images, masks, input_ids, target_ids, _) in enumerate(train_loader):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)          # Ground truth masks
            input_ids = input_ids.to(device)  # Input text IDs (shifted right)
            target_ids = target_ids.to(device)# Target text IDs

            optimizer.zero_grad()

            # Call model in 'joint' mode
            seg_output_logits, text_logits = model(
                image=images,
                tgt_text_indices=input_ids,
                mode='joint'
            )

            # Calculate joint loss
            # Ensure outputs are not None if training logic requires both
            if seg_output_logits is None or text_logits is None:
                 print(f"Warning: Skipping batch {batch_idx} due to None output in joint mode during training.")
                 continue

            total_loss, seg_loss, text_loss = criterion(
                seg_preds_logits=seg_output_logits,
                text_logits=text_logits,
                seg_targets=masks,
                text_targets=target_ids
            )

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_train_seg_loss += seg_loss.item()
            total_train_text_loss += text_loss.item()

            if batch_idx > 0 and batch_idx % 100 == 0: # Log progress
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}]")
                print(f"    Avg Batch Loss (Total): {total_loss.item():.4f} (Seg: {seg_loss.item():.4f}, Text: {text_loss.item():.4f})")

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_seg_loss = 0
        total_val_text_loss = 0
        total_val_dice = 0
        total_val_iou = 0
        print(f"--- Running Validation for Epoch {epoch+1} ---")
        with torch.no_grad():
            for images, masks, input_ids, target_ids, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                # Call model in 'joint' mode
                seg_output_logits, text_logits = model(
                    image=images,
                    tgt_text_indices=input_ids,
                    mode='joint'
                )

                # Calculate joint loss
                # Handle None outputs if necessary for validation logic
                if seg_output_logits is None or text_logits is None:
                     print(f"Warning: Skipping validation batch due to None output in joint mode.")
                     continue # Or decide how to handle validation if only one task runs

                total_loss, seg_loss, text_loss = criterion(
                    seg_preds_logits=seg_output_logits,
                    text_logits=text_logits,
                    seg_targets=masks,
                    text_targets=target_ids
                )

                total_val_loss += total_loss.item()
                total_val_seg_loss += seg_loss.item()
                total_val_text_loss += text_loss.item()

                # Calculate segmentation metrics (using logits)
                total_val_dice += dice_score(seg_output_logits, masks).item()
                total_val_iou += iou_score(seg_output_logits, masks).item()

        # Calculate Averages
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_seg_loss = total_train_seg_loss / len(train_loader)
        avg_train_text_loss = total_train_text_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_seg_loss = total_val_seg_loss / len(val_loader)
        avg_val_text_loss = total_val_text_loss / len(val_loader)
        avg_val_dice = total_val_dice / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1}/{epochs} Completed in {epoch_end_time - epoch_start_time:.2f}s:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Seg: {avg_train_seg_loss:.4f}, Text: {avg_train_text_loss:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f} (Seg: {avg_val_seg_loss:.4f}, Text: {avg_val_text_loss:.4f})")
        print(f"  Val Dice:   {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")
        print("-" * 50)

    # Save the final model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved successfully to {model_save_path}!")


