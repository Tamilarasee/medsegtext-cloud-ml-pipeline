import torch
import torch.nn as nn
import torch.nn.functional as F # Import F

# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Convert logits to probabilities
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice  # Since we want to minimize the loss

# Segmentation - Define combined loss function (Dice + CrossEntropy) 
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_preds_logits, seg_targets):
        dice = self.dice_loss(seg_preds_logits, seg_targets)
        bce = self.bce_loss(seg_preds_logits, seg_targets.float())
        return dice + bce

# --- MOD: Add Text Generation Loss ---
class TextCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss for text generation, ignoring padding tokens.
    """
    def __init__(self, pad_token_id):
        super().__init__()
        # Use ignore_index to skip padding tokens during loss calculation
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits (Batch, SeqLen, VocabSize)
            targets: Target token IDs (Batch, SeqLen)
        """
        # CrossEntropyLoss expects logits as (Batch, NumClasses, ...) or (N, C)
        # and targets as (Batch, ...) or (N)
        # Reshape logits: (B, T, V) -> (B*T, V)
        # Reshape targets: (B, T) -> (B*T)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)

        loss = self.loss_fn(logits_flat, targets_flat)
        return loss

# --- MOD: Add Joint Loss --- Seg + Text
class JointLoss(nn.Module):
    """
    Combines segmentation loss (e.g., CombinedLoss) and text loss
    (TextCrossEntropyLoss) with weighting.
    """
    def __init__(self, pad_token_id, seg_loss_weight=1.0, text_loss_weight=1.0):
        super().__init__()
        self.seg_loss_fn = CombinedLoss() # Uses Dice + BCEWithLogits
        self.text_loss_fn = TextCrossEntropyLoss(pad_token_id)
        self.seg_weight = seg_loss_weight
        self.text_weight = text_loss_weight
        print(f"Initialized JointLoss with Seg Weight: {self.seg_weight}, Text Weight: {self.text_weight}")

    def forward(self, seg_preds_logits, text_logits, seg_targets, text_targets):
        """
        Args:
            seg_preds_logits: Raw logits from segmentation decoder head (B, 1, H, W).
            text_logits: Raw logits from text decoder head (B, T, V).
            seg_targets: Ground truth segmentation mask (B, 1, H, W).
            text_targets: Ground truth text token IDs (B, T).
        """
        # Handle cases where one output might be None (e.g., during validation on only one task)
        seg_loss = torch.tensor(0.0, device=seg_targets.device)
        text_loss = torch.tensor(0.0, device=text_targets.device)
        
        if seg_preds_logits is not None:
             seg_loss = self.seg_loss_fn(seg_preds_logits, seg_targets.float())
        
        if text_logits is not None:
             text_loss = self.text_loss_fn(text_logits, text_targets)

        combined_loss = (self.seg_weight * seg_loss) + (self.text_weight * text_loss)

        # Return individual losses for logging purposes
        return combined_loss, seg_loss, text_loss





