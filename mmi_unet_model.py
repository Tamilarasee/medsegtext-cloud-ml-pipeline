import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import math

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


class CXR_BERT_Encoder(nn.Module):
    def __init__(self, max_length=512):
        super().__init__()
        
        # Load pretrained CXR-BERT from Hugging Face
        url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)
        self.cxrbert = AutoModel.from_pretrained(url, trust_remote_code=True)
        self.max_length = max_length

        # Freeze the BERT layers
        for param in self.cxrbert.parameters():
            param.requires_grad = False

    def forward(self, text_inputs):
        tokens = self.tokenizer(
            text_inputs, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_length
        )
        outputs = self.cxrbert(**tokens)  
        text_features = outputs.last_hidden_state
        return text_features


class ITM(nn.Module):
    def __init__(self, visual_channels, text_channels, num_heads=8):
        super().__init__()
        self.conv_image = nn.Conv2d(visual_channels, visual_channels, kernel_size=2, stride=2)
        self.linear_text = nn.Linear(text_channels, visual_channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=visual_channels, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm([visual_channels])  # Specify correct normalization shape
        self.mhca = nn.MultiheadAttention(embed_dim=visual_channels, num_heads=num_heads, batch_first=True)        
        self.conv_transpose = nn.ConvTranspose2d(visual_channels, visual_channels, kernel_size=2, stride=2)
        self.ffn = nn.Sequential(
            nn.Linear(visual_channels, visual_channels * 4),
            nn.GELU(),
            nn.Linear(visual_channels * 4, visual_channels),
            nn.Dropout(0.1)
        )
        self.linear_text_2 = nn.Linear(visual_channels, text_channels)

    def forward(self, visual_features, text_features):
        # MHSA Image
        visual_reduction_fc = self.conv_image(visual_features)
        
        # Reshape for LayerNorm (BCHW → BNC)
        batch, channels, height, width = visual_reduction_fc.shape
        visual_reduction_fc = visual_reduction_fc.permute(0, 2, 3, 1)  # [B, H, W, C]
        visual_reduction_fc = visual_reduction_fc.reshape(batch * height * width, channels)  # [B*H*W, C]
        visual_reduction_fc = self.layer_norm(visual_reduction_fc)  # Apply LayerNorm
        visual_reduction_fc = visual_reduction_fc.reshape(batch, height * width, channels)  # [B, H*W, C]

        # Apply self-attention
        visual_fc_mhsa, _ = self.mhsa(visual_reduction_fc, visual_reduction_fc, visual_reduction_fc)
        
        # Apply LayerNorm to residual connection
        visual_f_sa = visual_fc_mhsa + visual_reduction_fc
        visual_f_sa = visual_f_sa.reshape(batch * height * width, channels)
        visual_f_sa = self.layer_norm(visual_f_sa)
        visual_f_sa = visual_f_sa.reshape(batch, height * width, channels)

        # MHSA Text (no reshaping needed - already in BNC format)
        text_reduction_t = self.linear_text(text_features)  
        text_reduction_t = self.layer_norm(text_reduction_t)
        text_mhsa, _ = self.mhsa(text_reduction_t, text_reduction_t, text_reduction_t)
        text_t_sa = self.layer_norm(text_mhsa + text_reduction_t)

        # MHCA Image and Upsampling
        visual_f_mhca, _ = self.mhca(visual_f_sa, text_t_sa, text_t_sa)
        visual_f_ca = self.layer_norm(visual_f_mhca + visual_f_sa)
        visual_ffn = self.ffn(visual_f_ca)
        visual_ffn_norm = self.layer_norm(visual_ffn + visual_f_ca)
        
        # Reshape back to BCHW for conv_transpose
        visual_ffn_norm = visual_ffn_norm.permute(0, 2, 1).reshape(batch, channels, height, width)
        visual_up_f_itm = self.conv_transpose(visual_ffn_norm)

        # MHCA Text
        text_mhca, _ = self.mhca(text_t_sa, visual_f_sa, visual_f_sa)
        text_t_ca = self.layer_norm(text_mhca + text_t_sa)
        text_ffn = self.ffn(text_t_ca)
        text_ffn_norm = self.layer_norm(text_ffn + text_t_ca)
        text_up_f_itm = self.linear_text_2(text_ffn_norm)

        return visual_up_f_itm, text_up_f_itm


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
        
# Decoder to be written later
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.eca = ECA(in_channels)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
       

    def forward(self, x, skip):
        skip_eca = self.eca(skip) # ITM output as skip connection
        up_x = self.conv_transpose(x) #previous encoder output
        concat_x = torch.cat((skip_eca, up_x), dim=1)
        decoder_output = self.conv(concat_x)
        return decoder_output

    

class MMI_UNet(nn.Module):
    def __init__(self, out_channels=1, features=[96, 192, 384, 768]):
        super().__init__()

        self.visual_encoder = ConvNeXtEncoder()
        self.text_encoder = CXR_BERT_Encoder()
        
           
        self.itm_1 = ITM(features[0], 768, num_heads=4)    
        self.itm_2 = ITM(features[1], 768, num_heads=8)    
        self.itm_3 = ITM(features[2], 768, num_heads=8)    
        self.itm_4 = ITM(features[3], 768, num_heads=12)    
       
        self.conv_input = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)

       
        # Decoder (Upsampling Path)
        self.decoder = nn.ModuleList() 
                
        for feature in reversed(features):
            self.decoder.append(Decoder(feature, feature//2))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    
    def forward(self, x, text_inputs):

        text_bert_features = self.text_encoder(text_inputs)
        self.skip_connections = []
        
        text_features_1 = text_bert_features
        x = self.conv_input(x)
        visual_encoder_output_f1 = self.visual_encoder.stage1(x)        
        visual_itm_1, text_itm_1 = self.itm_1(visual_encoder_output_f1, text_features_1)
        self.skip_connections.append(visual_itm_1)

        text_features_2 = text_bert_features + text_itm_1
        visual_encoder_output_f2 = self.visual_encoder.stage2(visual_encoder_output_f1 + visual_itm_1)
        visual_itm_2, text_itm_2 = self.itm_2(visual_encoder_output_f2, text_features_2)
        self.skip_connections.append(visual_itm_2)

        text_features_3 = text_bert_features + text_itm_2
        visual_encoder_output_f3 = self.visual_encoder.stage3(visual_encoder_output_f2 + visual_itm_2)
        visual_itm_3, text_itm_3 = self.itm_3(visual_encoder_output_f3, text_features_3)
        self.skip_connections.append(visual_itm_3)

        text_features_4 = text_bert_features + text_itm_3
        visual_encoder_output_f4 = self.visual_encoder.stage4(visual_encoder_output_f3 + visual_itm_3)
        visual_itm_4, text_itm_4 = self.itm_4(visual_encoder_output_f4, text_features_4)

        skip_connections = self.skip_connections[::-1]
        decoder_output = self.decoder[0](visual_itm_4, skip_connections[0])

        for i in range(1, len(self.decoder)): 
            decoder_output = self.decoder[i](decoder_output, skip_connections[i])

        # Segmentation Head
         # Upsample to the input size (224x224)
        decoder_output_up = F.interpolate(decoder_output, size=(224, 224), mode='bilinear', align_corners=False)

        # Final output layer (Segmentation Head) with sigmoid
        seg_head_ouput = self.final_conv(decoder_output_up)
        seg_head_ouput = torch.sigmoid(seg_head_ouput)

        return seg_head_ouput

