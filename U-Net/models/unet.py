import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from encoder_mini_block import EncoderMiniBlock
from decoder_mini_block import DecoderMiniBlock

class UNet(LightningModule):
    
    def __init__(self, in_channels, out_channels, n_filters, dropout=0, cr_method="max_pooling",ce_method="upsample"):
        super().__init__()
        
        # Contraction / Encoding Block
        self.enc1 = EncoderMiniBlock(in_channels,
                                     n_filters // 8,
                                     dropout=dropout,
                                    cr_method=cr_method)
        
        self.enc2 = EncoderMiniBlock(n_filters // 8,
                                     n_filters // 4,
                                     dropout=dropout,
                                    cr_method=cr_method)
        
        self.enc3 = EncoderMiniBlock(n_filters // 4,
                                     n_filters // 2,
                                     dropout=dropout,
                                    cr_method=cr_method)
        
        self.enc4 = EncoderMiniBlock(n_filters // 2,
                                     n_filters,
                                     dropout=dropout,
                                    cr_method=cr_method)
        
        # Expansion / Decoding Block
        self.dec1 = DecoderMiniBlock(n_filters,
                                     n_filters // 2,
                                     dropout=dropout,
                                    ce_method=ce_method)
        
        self.dec2 = DecoderMiniBlock(n_filters // 2,
                                     n_filters // 4,
                                     dropout=dropout,
                                    ce_method=ce_method)
        
        self.dec3 = DecoderMiniBlock(n_filters // 4,
                                     n_filters // 8,
                                     dropout=dropout,
                                    ce_method=ce_method)
        
        # Final Layer
        self.final_layer = nn.Conv2d(n_filters // 8, out_channels, kernel_size=1)
        
        # Assert in_channels is 3 and in_channels == out_channels
        assert in_channels == 3, "in_channels must be 3"
        assert in_channels == out_channels, "in_channels must be equal to out_channels"
        
    def forward(self, x):
        # Contraction / Encoding Block
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, _ = self.enc4(x)
        
        # Expansion / Decoding Block
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        x = self.final_layer(x)
        
        return x
        
        