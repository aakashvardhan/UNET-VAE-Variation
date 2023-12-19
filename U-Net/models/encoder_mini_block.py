import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from models.double_conv2d import DoubleConv




class EncoderMiniBlock(LightningModule):
    def __init__(self, in_channels, out_channels, stride=2, cr_method="max_pooling", dropout=0):
        super().__init__()
        self.double_conv2d = DoubleConv(in_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        if stride > 1:
            if cr_method == "max_pooling":
                self.cr = nn.MaxPool2d(kernel_size=2, stride=stride)
            elif cr_method == "strided_convolution":
                self.cr = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=stride)
            else:
                raise Exception("Invalid Channel Reduction Method!")
            
    def forward(self, x):
        x = self.double_conv2d(x)
        x = self.dropout(x)
        skip = x
        if hasattr(self, "cr"):
            x = self.cr(x)
        return x, skip

