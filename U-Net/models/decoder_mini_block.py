import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from double_conv2d import DoubleConv


class DecoderMiniBlock(LightningModule):
    def __init__(self, in_channels, out_channels, ce_method="upsample", dropout=0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.double_conv2d = DoubleConv(in_channels, out_channels)
        
        if ce_method == "upsample":
            self.ce = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif ce_method == "transposed_convolution":
            self.ce = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            raise Exception("Invalid Channel Expansion Method!")
        
    def forward(self, x, skip):
        x = self.ce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv2d(x)
        x = self.dropout(x)
        return x
