import torch
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

# --- Tokenizer Initialization ---

SPECIAL_TOKENS = {
    "bos_token": "[SOS]", # Start Of Sequence
    "eos_token": "[EOS]", # End Of Sequence
    "pad_token": "[PAD]", # Pad Token
    "unk_token": "[UNK]", # Unknown Token (often already present)
}
# This is no longer needed, we will pass the local downloaded model path directly
#TOKENIZER_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"

def load_tokenizer(tokenizer_path):
    """Loads the tokenizer and adds special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Check if special tokens need to be added
    newly_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(f"Added {newly_added} special tokens.")
    print(f"Tokenizer Vocabulary Size: {tokenizer.vocab_size}")
    print(f"SOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")
    print(f"EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
    print(f"PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")

    # Store pad_token_id for easy access later, e.g., in collate_fn
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    return tokenizer

# --- Text Processing Function ---

def tokenize_and_prepare_text(text, tokenizer, max_length=50):
    """
    Tokenizes text, adds special tokens (SOS/EOS), and truncates.
    Returns a list of token IDs.
    """
    # Tokenize the input text
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False, # We will add SOS/EOS manually
        max_length=max_length - 2, # Account for adding SOS and EOS
        truncation=True
    )

    # Add SOS and EOS tokens
    processed_token_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.eos_token_id]

    return processed_token_ids

# --- Collate Function for DataLoader ---

def collate_fn_text(batch, tokenizer):
    """
    Processes a batch of (image, mask, token_ids) tuples for text generation training.
    Handles padding and creates input/target sequences for the decoder.

    Args:
        batch: A list of tuples, where each tuple is (image_tensor, mask_tensor, token_id_list).
        tokenizer: The loaded tokenizer instance (needed for PAD ID).

    Returns:
        A tuple containing:
        - Batched images (Tensor)
        - Batched masks (Tensor)
        - Input token IDs (Tensor, padded, shifted right with SOS) (B, T)
        - Target token IDs (Tensor, padded, with EOS) (B, T)
        - Target padding mask (Tensor, boolean) (B, T) - True where padded
    """
    images, masks, token_ids_list = zip(*batch)

    # Stack images and masks
    images_batch = torch.stack(images, dim=0)
    masks_batch = torch.stack(masks, dim=0) # Assuming masks are already tensors

    # Convert token ID lists to tensors
    token_tensors = [torch.tensor(ids, dtype=torch.long) for ids in token_ids_list]

    # Pad sequences
    # pad_sequence expects (SeqLen, Batch), so transpose later if needed by model
    padded_sequences = pad_sequence(
        token_tensors,
        batch_first=True, # Make output (Batch, SeqLen)
        padding_value=tokenizer.pad_token_id
    )

    # Create input sequences (shifted right: SOS token + sequence[:-1])
    # Example: [SOS, w1, w2, EOS, PAD] -> [SOS, w1, w2, EOS] for input
    #          [SOS, w1, w2, EOS, PAD] -> [w1, w2, EOS, PAD] for target
    input_ids = padded_sequences[:, :-1]

    # Create target sequences (sequence[1:])
    # Targets should not predict based on the final PAD token from the input
    target_ids = padded_sequences[:, 1:]

    # Create padding mask for the targets (True where padded)
    # Needed by the TransformerDecoder's tgt_key_padding_mask
    target_padding_mask = (target_ids == tokenizer.pad_token_id)

    return images_batch, masks_batch, input_ids, target_ids, target_padding_mask
