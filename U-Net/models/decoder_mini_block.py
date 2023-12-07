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
        
        
