import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from models.double_conv2d import DoubleConv


class DecoderMiniBlock(LightningModule):
    def __init__(self, in_channels, out_channels, ce_method="upsample", dropout=0):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.double_conv2d = DoubleConv(in_channels + out_channels, out_channels)
        
        if ce_method == "upsample":
            self.ce = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        elif ce_method == "transposed_convolution":
            self.ce = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            raise Exception("Invalid Channel Expansion Method!")
        
    def forward(self, x, skip):
        x = self.ce(x)
        # Adjust the size of x or skip here if they don't match
        delta_height = x.shape[2] - skip.shape[2]
        delta_width = x.shape[3] - skip.shape[3]

        if delta_height != 0 or delta_width != 0:
            if delta_height > 0 or delta_width > 0:
                # Crop x if it's larger
                x = x[:, :, delta_height//2 : x.shape[2] - delta_height//2, delta_width//2 : x.shape[3] - delta_width//2]
            else:
                # Crop skip if it's larger
                skip = skip[:, :, -delta_height//2 : skip.shape[2] + delta_height//2, -delta_width//2 : skip.shape[3] + delta_width//2]

        x = torch.cat([x, skip], dim=1)
        x = self.double_conv2d(x)
        x = self.dropout(x)
        return x

