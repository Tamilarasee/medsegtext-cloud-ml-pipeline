import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SegmentationDecoder(nn.Module):
    """
    Standard U-Net style decoder. Uses ConvTranspose2d for upsampling.
    Takes encoder features (skip connections) and bottleneck features.
    """
    def __init__(self, encoder_channels, out_channels=1):
        """
        Args:
            encoder_channels (list): List of channel counts from encoder stages
                                     in order [f1, f2, f3, f4].
                                     Example: [96, 192, 384, 768]
            out_channels (int): Number of output segmentation channels.
        """
        super().__init__()

        if not isinstance(encoder_channels, list) or len(encoder_channels) != 4:
            raise ValueError("encoder_channels must be a list of 4 channel counts.")

        self.encoder_channels = encoder_channels
        self.num_stages = len(encoder_channels) - 1 # 3 upsampling stages needed

        # ModuleLists for up-convolution and double-convolution blocks
        self.up_convs = nn.ModuleList()
        self.convs = nn.ModuleList()

        # Decoder path: Upsample f4 -> Combine with f3 -> Upsample -> Combine with f2 -> Upsample -> Combine with f1
        in_ch = self.encoder_channels[-1] # Start with bottleneck channels (e.g., 768 from f4)
        for i in range(self.num_stages): # i = 0, 1, 2
            # Target channel count for this stage (output of DoubleConv)
            # Corresponds to channels of f3, f2, f1
            out_ch = self.encoder_channels[-2-i] # e.g., 384, 192, 96

            # Up-convolution: halves channels from in_ch to out_ch
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # Double convolution: input is out_ch (from skip) + out_ch (from up_conv)
            self.convs.append(DoubleConv(out_ch * 2, out_ch))

            in_ch = out_ch # Input channels for next up_conv stage

        # Final 1x1 convolution layer
        # Input channels = output channels of the last decoder stage (e.g., 96)
        self.final_conv = nn.Conv2d(self.encoder_channels[0], out_channels, kernel_size=1)

    def forward(self, encoder_features):
        """
        Args:
            encoder_features (list): List of feature maps [f1, f2, f3, f4].
                                     f1=[B,C1,H/4,W/4], f2=[B,C2,H/8,W/8], ... f4=[B,C4,H/32,W/32]
        Returns:
            Tensor: Segmentation map before final upsampling/activation. (Size H/4 x W/4)
        """
        if not isinstance(encoder_features, list) or len(encoder_features) != 4:
             raise ValueError("encoder_features must be a list of 4 tensors [f1, f2, f3, f4].")

        f1, f2, f3, f4 = encoder_features
        skip_connections = [f3, f2, f1] # Order needed for decoder loop (match f3, f2, f1)

        x = f4 # Start with bottleneck features

        for i in range(self.num_stages): # i = 0, 1, 2
            x = self.up_convs[i](x)    # Upsample (e.g., 768->384 channels, H/32->H/16)
            skip = skip_connections[i] # Get corresponding skip connection (f3, f2, f1)

            # Handle potential size mismatch after ConvTranspose2d
            if x.shape != skip.shape:
                 x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([skip, x], dim=1) # Concatenate (e.g., [384+384])
            x = self.convs[i](x)           # Apply DoubleConv (e.g., 768 -> 384)

        # Final output convolution
        x = self.final_conv(x) # Map to output channels (e.g., 96 -> 1)

        return x # Output resolution is H/4 x W/4
