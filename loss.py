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

# Define combined loss function (Dice + CrossEntropy)
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.dice_loss(preds, targets) + self.ce_loss(preds, targets.float())

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





