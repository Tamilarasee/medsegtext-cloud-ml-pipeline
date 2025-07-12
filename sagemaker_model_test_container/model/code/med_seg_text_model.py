import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math
import inspect
# --- MOD: Import the new Segmentation Decoder ---
from seg_decoder import SegmentationDecoder # DoubleConv is not needed directly here

class ConvNeXtEncoder(nn.Module):
    #def __init__(self):
    def __init__(self, convnext_model_path):
        super().__init__()

        # Update - Removed the Hugging Face URL and replaced it with the local path to reduce dependency on internet connection and download time for deployment
        # Load pretrained ConvNeXt-Tiny from Hugging Face
        # url = "facebook/convnext-tiny-224"
        # self.convnext = AutoModel.from_pretrained(url, trust_remote_code=True)

        # Load pretrained ConvNeXt-Tiny from the provided local path
        self.convnext = AutoModel.from_pretrained(convnext_model_path, trust_remote_code=True)

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
                 vocab_size=30524, # Use actual vocab size
                 embed_dim=768,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=3072, # MOD: Typically 4*embed_dim
                 max_text_seq_len=50, # MOD: Max length for text sequences
                 dropout=0.1,
                 pad_token_id=0,
                 convnext_model_path=None): # Adding this arguement to pass local path to ConvNeXt model
        super().__init__()

        if convnext_model_path is None:
            raise ValueError("A path to the ConvNeXt model must be provided.")

        self.pad_token_id = pad_token_id
        #self.visual_encoder = ConvNeXtEncoder()
        # Pass the local path to the visual encoder
        self.visual_encoder = ConvNeXtEncoder(convnext_model_path=convnext_model_path)

        # --- Input Convolution ---
        self.conv_input = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.visual_features_channels = [96, 192, 384, 768] # Keep track of encoder channels

        # --- Text Decoder Components ---
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_token_id) # MOD: Also useful to set padding_idx in Embedding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout, max_text_seq_len)

        # Input projection from visual features to Transformer's expected dim (if needed)
        # RATCHET uses DenseNet-121 which has 1024 features out. ConvNeXt-Tiny has 768.
        # If embed_dim is different from final visual features (768), add projection.
       
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
        self.text_output_layer = nn.Linear(embed_dim, vocab_size)

        # --- MOD: Segmentation Decoder Component ---
        self.seg_decoder = SegmentationDecoder(
            encoder_channels=self.visual_features_channels, # Pass the channel list
            out_channels=seg_out_channels
        )

    def forward(self, image, tgt_text_indices=None, mode='joint'):
        """
        Args:
            image: Input image tensor (B, C, H, W)
            tgt_text_indices: Target text indices (shifted right) for training (B, T_tgt).
                                Required when mode includes 'text' or 'joint'.
            mode: 'text', 'segmentation', or 'joint'. Controls which parts run.
        """
        # --- Visual Encoding ---

        f0 = self.conv_input(image) # (B, 96, H, W) - Assuming input is 224x224
        f1 = self.visual_encoder.stage1(f0) # (B, 96, H/4, W/4) - 56x56
        f2 = self.visual_encoder.stage2(f1) # (B, 192, H/8, W/8) - 28x28
        f3 = self.visual_encoder.stage3(f2) # (B, 384, H/16, W/16) - 14x14
        f4 = self.visual_encoder.stage4(f3) # (B, 768, H/32, W/32) - 7x7

        # Store features in the order expected by seg_decoder [f1, f2, f3, f4]
        encoder_features = [f1, f2, f3, f4]

        # --- Text Decoding Path ---
        text_logits = None
        # MOD: Allow joint mode even if tgt_text_indices is None during inference maybe?
        # Let's keep the check strict for now during training.
        if mode == 'text' or mode == 'joint':
            if tgt_text_indices is None and (mode == 'text' or self.training): # Require text for training text/joint
                 raise ValueError("tgt_text_indices must be provided for text/joint training mode")
            if tgt_text_indices is not None:
                memory = f4.flatten(2).permute(0, 2, 1)
                memory = self.visual_feature_proj(memory)
                tgt_embed = self.text_embedding(tgt_text_indices)
                tgt_embed = tgt_embed.permute(1, 0, 2)
                tgt_embed = self.positional_encoding(tgt_embed)
                tgt_embed = tgt_embed.permute(1, 0, 2)
                tgt_seq_len = tgt_text_indices.size(1)
                device = tgt_text_indices.device
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len, device=device)
                tgt_padding_mask = (tgt_text_indices == self.pad_token_id)
                decoder_output = self.transformer_decoder(
                    tgt=tgt_embed, memory=memory,
                    tgt_mask=tgt_mask,  # This mask helps it learn autoregressive behavior
                    tgt_key_padding_mask=tgt_padding_mask
                )
                text_logits = self.text_output_layer(decoder_output)

        # --- MOD: Segmentation Decoding Path ---
        seg_output_raw = None
        if mode == 'segmentation' or mode == 'joint':
            # Remove the old placeholder comment/pass
            seg_output_raw = self.seg_decoder(encoder_features) # Output is H/4 x W/4

            # Upsample output to original image size
            seg_output_raw = F.interpolate(
                seg_output_raw,
                size=image.shape[2:], # Original H, W
                mode='bilinear',
                align_corners=False
            )
            # We return raw logits, activation (sigmoid) is handled by loss/inference

        # --- Return requested outputs ---
        if mode == 'text':
            if text_logits is None: raise RuntimeError("Text mode selected but text_logits not computed.")
            return text_logits
        elif mode == 'segmentation':
             if seg_output_raw is None: raise RuntimeError("Seg mode selected but seg_output_raw not computed.")
             return seg_output_raw
        elif mode == 'joint':
             # Ensure both are computed or handle None if inference logic changes later
             if seg_output_raw is None or text_logits is None:
                 # This might happen during inference if text isn't generated but seg is needed
                 # Or if training joint but text input is missing - needs careful handling
                 print(f"Warning: Joint mode returning potentially None values (Seg: {seg_output_raw is not None}, Text: {text_logits is not None})")
             return seg_output_raw, text_logits
        else:
            raise ValueError(f"Invalid mode: {mode}")

