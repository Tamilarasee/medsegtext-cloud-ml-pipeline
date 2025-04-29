import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math
import inspect
class ConvNeXtEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ConvNeXt-Tiny from Hugging Face
        url = "facebook/convnext-tiny-224"
        self.convnext = AutoModel.from_pretrained(url, trust_remote_code=True)

        # ConvNeXt stages to extract feature maps
        self.stage1 = self.convnext.encoder.stages[0]  # Produces F1 (H/4 x W/4 x 96)
        self.stage2 = self.convnext.encoder.stages[1]  # Produces F2 (H/8 x W/8 x 192)
        self.stage3 = self.convnext.encoder.stages[2]  # Produces F3 (H/16 x W/16 x 384)
        self.stage4 = self.convnext.encoder.stages[3]  # Produces F4 (H/32 x W/32 x 768)


# class CXR_BERT_Encoder(nn.Module):
#     def __init__(self, max_length=512):
#         super().__init__()
        
#         # Load pretrained CXR-BERT from Hugging Face
#         url = "microsoft/BiomedVLP-CXR-BERT-specialized"
#         self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
#         self.cxrbert = AutoModel.from_pretrained(url, trust_remote_code=True)
#         self.max_length = max_length

#         # Freeze the BERT layers
#         for param in self.cxrbert.parameters():
#             param.requires_grad = False

#     def forward(self, text_inputs):
#         tokens = self.tokenizer(
#             text_inputs, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True,
#             max_length=self.max_length
#         )
#         tokens = {k: v.to(self.cxrbert.device) for k, v in tokens.items()}
#         outputs = self.cxrbert(**tokens)  
#         text_features = outputs.last_hidden_state
#         return text_features


# class ITM(nn.Module):
#     def __init__(self,visual_channels, text_channels,num_heads=8):
#         super().__init__()
#         self.conv_image = nn.Conv2d(visual_channels,visual_channels, kernel_size=2, stride=2)
#         self.linear_text = nn.Linear(text_channels, visual_channels)
#         self.mhsa = nn.MultiheadAttention(embed_dim=visual_channels, num_heads=num_heads, batch_first=True)
#         self.layer_norm = nn.LayerNorm(visual_channels)
#         self.mhca = nn.MultiheadAttention(embed_dim=visual_channels, num_heads=num_heads, batch_first=True)        
#         self.conv_transpose = nn.ConvTranspose2d(visual_channels, visual_channels, kernel_size=2, stride=2)
#         self.ffn = nn.Sequential(
#             nn.Linear(visual_channels, visual_channels * 4),
#             nn.GELU(),
#             nn.Linear(visual_channels * 4, visual_channels),
#             nn.Dropout(0.1)
#         )
#         self.linear_text_2 = nn.Linear(visual_channels, text_channels)

#     def forward(self, visual_features, text_features):
#         # MHSA Image
#         visual_reduction_fc = self.conv_image(visual_features)
        
#         # Reshape for LayerNorm (BCHW → BNC)
#         batch, channels, height, width = visual_reduction_fc.shape
#         visual_reduction_fc = visual_reduction_fc.reshape(batch, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
#         visual_reduction_fc = self.layer_norm(visual_reduction_fc)
        
#         # Apply self-attention (already in correct shape [B, H*W, C])
#         visual_fc_mhsa, _ = self.mhsa(visual_reduction_fc, visual_reduction_fc, visual_reduction_fc)
#         visual_f_sa = self.layer_norm(visual_fc_mhsa + visual_reduction_fc)

#         # MHSA Text (no reshaping needed - already in BNC format)
#         text_reduction_t = self.linear_text(text_features)  
#         text_reduction_t = self.layer_norm(text_reduction_t)
#         text_mhsa, _ = self.mhsa(text_reduction_t, text_reduction_t, text_reduction_t)
#         text_t_sa = self.layer_norm(text_mhsa + text_reduction_t)

#         # MHCA Image and Upsampling
#         visual_f_mhca, _ = self.mhca(visual_f_sa, text_t_sa, text_t_sa)
#         visual_f_ca = self.layer_norm(visual_f_mhca + visual_f_sa)
#         visual_ffn = self.ffn(visual_f_ca)
#         visual_ffn_norm = self.layer_norm(visual_ffn + visual_f_ca)
        
#         # Reshape back to BCHW for conv_transpose
#         visual_ffn_norm = visual_ffn_norm.permute(0, 2, 1).reshape(batch, channels, height, width)
#         visual_up_f_itm = self.conv_transpose(visual_ffn_norm)

#         # MHCA Text
#         text_mhca, _ = self.mhca(text_t_sa, visual_f_sa, visual_f_sa)
#         text_t_ca = self.layer_norm(text_mhca + text_t_sa)
#         text_ffn = self.ffn(text_t_ca)
#         text_ffn_norm = self.layer_norm(text_ffn + text_t_ca)
#         text_up_f_itm = self.linear_text_2(text_ffn_norm)

#         return visual_up_f_itm, text_up_f_itm


class ECA(nn.Module):
    """Constructs a ECA module with adaptive kernel size.
    
    Args:
        channel: Number of channels of the input feature map
        gamma: Parameter controlling kernel size calculation (default=2)
        beta: Parameter controlling kernel size calculation (default=1)
    """
    def __init__(self, channel, gamma=2, beta=1):
        super(ECA, self).__init__()
        # Calculate adaptive kernel size based on channel dimension
        t = int(abs(math.log(channel, 2) + beta) / gamma)
        k_size = t if t % 2 else t + 1  # ensure odd kernel size
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50): # MOD: Reduced max_len default
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it moves with the model

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class JointSegTextUNet(nn.Module):
    def __init__(self, seg_out_channels=1,
                 vocab_size=120, # MOD: Placeholder vocab size (116 unique + SOS, EOS, PAD, UNK?)
                 embed_dim=768,  # MOD: Embedding dim matching ConvNeXt F4 channels
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=3072, # MOD: Typically 4*embed_dim
                 max_text_seq_len=50, # MOD: Max length for text sequences
                 dropout=0.1,
                 pad_token_id=0): # MOD: Add pad_token_id argument (default=0 just in case)
        super().__init__()

        # MOD: Store pad_token_id
        self.pad_token_id = pad_token_id



        self.visual_encoder = ConvNeXtEncoder()

        # --- MOD: Standard Input Convolution (Keep) ---
        # Using the same features as original MMI-UNet's ConvNeXt stages
        self.conv_input = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.visual_features_channels = [96, 192, 384, 768] # Keep track of encoder channels

        # --- MOD: Text Decoder Components (New) ---
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id) # MOD: Also useful to set padding_idx in Embedding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_text_seq_len)

        # Input projection from visual features to Transformer's expected dim (if needed)
        # RATCHET uses DenseNet-121 which has 1024 features out. ConvNeXt-Tiny has 768.
        # If embed_dim is different from final visual features (768), add projection.
        # Assuming embed_dim == 768 for now.
        self.visual_feature_proj = nn.Identity() # Or nn.Linear if projection needed
        if self.visual_features_channels[-1] != embed_dim:
             self.visual_feature_proj = nn.Linear(self.visual_features_channels[-1], embed_dim)


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # MOD: Use batch_first=True for easier tensor manipulation
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.text_output_layer = nn.Linear(embed_dim, vocab_size) # Map to vocab

    def forward(self, image, tgt_text_indices=None, mode='joint'):
        """
        Args:
            image: Input image tensor (B, C, H, W)
            tgt_text_indices: Target text indices (shifted right) for training (B, T_tgt)
                                Required when mode includes 'text'.
            mode: 'text', 'segmentation', or 'joint'. Controls which parts run.
                  For now, we focus on 'text'.
        """

        # --- Visual Encoding ---
        # MOD: Directly compute visual features without ITM
        f0 = self.conv_input(image) # (B, 96, H, W) - Assuming input is 224x224
        f1 = self.visual_encoder.stage1(f0) # (B, 96, H/4, W/4) - 56x56
        f2 = self.visual_encoder.stage2(f1) # (B, 192, H/8, W/8) - 28x28
        f3 = self.visual_encoder.stage3(f2) # (B, 384, H/16, W/16) - 14x14
        f4 = self.visual_encoder.stage4(f3) # (B, 768, H/32, W/32) - 7x7

        # --- Text Decoding Path ---
        text_logits = None
        if mode == 'text' or mode == 'joint':
            if tgt_text_indices is None:
                raise ValueError("tgt_text_indices must be provided for text generation mode")

            # Prepare visual features for Transformer Decoder ('memory')
            # Input: f4 (B, C=768, H'=7, W'=7)
            # Expected by TransformerDecoderLayer (batch_first=True): (B, S_mem, E)
            # S_mem = H'*W', E = embed_dim
            memory = f4.flatten(2).permute(0, 2, 1) # (B, H'*W', C=768)
            # MOD: Project visual features if embed_dim doesn't match C
            memory = self.visual_feature_proj(memory) # (B, 49, embed_dim=768)

            # Prepare target text
            # tgt_text_indices: (B, T_tgt) - Assumed to be padded, shifted right with SOS
            tgt_embed = self.text_embedding(tgt_text_indices) # (B, T_tgt, embed_dim)
            # MOD: PositionalEncoding expects (T_tgt, B, E), but our TransformerDecoderLayer is batch_first.
            # Let's adapt PositionalEncoding or adjust tensor shapes here.
            # Adapting PE forward call to match batch_first convention:
            # Need to transpose for PE and back.
            tgt_embed = tgt_embed.permute(1, 0, 2) # (T_tgt, B, embed_dim)
            tgt_embed = self.positional_encoding(tgt_embed) # Apply PE
            tgt_embed = tgt_embed.permute(1, 0, 2) # (B, T_tgt, embed_dim)

            # Generate causal mask for target sequence
            tgt_seq_len = tgt_text_indices.size(1)
            # MOD: Need device placement for mask
            device = tgt_text_indices.device
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device) # (T_tgt, T_tgt)

            # Generate padding mask for target sequence
            # MOD: Use self.pad_token_id passed during initialization
            tgt_padding_mask = (tgt_text_indices == self.pad_token_id) # (B, T_tgt), True where padded

            # Pass through Transformer Decoder
            # Input: tgt=(B, T_tgt, E), memory=(B, S_mem, E), tgt_mask=(T_tgt, T_tgt), tgt_key_padding_mask=(B, T_tgt)
            decoder_output = self.transformer_decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask
            ) # Output: (B, T_tgt, E)

            # Final linear layer to get logits
            text_logits = self.text_output_layer(decoder_output) # (B, T_tgt, vocab_size)

        # --- Segmentation Decoding Path (Placeholder) ---
        seg_output = None
        if mode == 'segmentation' or mode == 'joint':
            # MOD: This part needs to be implemented later using f1, f2, f3, f4
            # skip_connections = [f3, f2, f1] # Example: Use direct encoder outputs
            # x = f4 # Start decoding from the bottleneck
            # for i, dec_layer in enumerate(self.seg_decoder):
            #     x = dec_layer(x, skip_connections[i])
            # x = F.interpolate(x, size=image.shape[2:], mode='bilinear', align_corners=False)
            # seg_output = self.seg_final_conv(x)
            # seg_output = torch.sigmoid(seg_output) # Assuming binary segmentation
            print("WARN: Segmentation path not implemented yet.")
            pass # Placeholder

        # --- Return requested outputs ---
        if mode == 'text':
            return text_logits
        elif mode == 'segmentation':
            return seg_output # Currently None
        elif mode == 'joint':
            return seg_output, text_logits # Currently (None, text_logits)
        else:
            raise ValueError(f"Invalid mode: {mode}")

