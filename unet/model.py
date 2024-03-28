"""
A PyTorch implementation of UNet (https://arxiv.org/pdf/1505.04597.pdf)

Based on implementation by (https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201)

"""

import numpy as np
import torch
import torch.nn as nn


class conv_block(nn.Module):
    """UNet 3x3 convolution block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class encoder_block(nn.Module):
    """UNet encoder block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    """UNet decoder block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(2*out_c, out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x
        

class UNet(nn.Module):
    """UNet class based on https://arxiv.org/pdf/1505.04597.pdf."""
    def __init__(self, num_out_channels: int = 3):
        super().__init__()
        
        self.num_out_channels = num_out_channels

        # Encoder
        self.down1 = encoder_block(3, 64)
        self.down2 = encoder_block(64, 128)
        self.down3 = encoder_block(128, 256)
        self.down4 = encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = conv_block(512, 1024)
        
        # Decoder
        self.up1 = decoder_block(1024, 512)
        self.up2 = decoder_block(512, 256)
        self.up3 = decoder_block(256, 128)
        self.up4 = decoder_block(128, 64)
        
        self.new_head(num_out_channels=self.num_out_channels)

    def new_head(self, num_out_channels: int = 3):
        self.num_out_channels = num_out_channels
        # 1x1 convolution classifier
        self.classifier = nn.Conv2d(64, num_out_channels, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        s1, d1 = self.down1(inputs)
        s2, d2 = self.down2(d1)
        s3, d3 = self.down3(d2)
        s4, d4 = self.down4(d3)
        
        b = self.bottleneck(d4)
        
        u1 = self.up1(b, s4)
        u2 = self.up2(u1, s3)
        u3 = self.up3(u2, s2)
        u4 = self.up4(u3, s1)
        
        out = self.classifier(u4)
        
        return out
        
    