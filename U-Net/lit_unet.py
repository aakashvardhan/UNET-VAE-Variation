import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule




class LitUNet(LightningModule):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config