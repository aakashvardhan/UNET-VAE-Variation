import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule




class DiceLoss(LightningModule):
    # Multi class Dice Loss for Image Segmentation using softmax
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, y_pred, y_true):
        # y_pred: (N, C, H, W)
        # y_true: (N, H, W)
        prob = y_pred
        if self.config['softmax_dim'] is not None:
            prob = F.softmax(y_pred, dim=self.config['softmax_dim'])(y_pred)
        # y_pred = F.softmax(y_pred, dim=self.config['softmax_dim'])
        y_true = F.one_hot(y_true, num_classes=self.config['num_classes']).permute(0, 3, 1, 2).float()
        # y_true: (N, C, H, W)
        
        # dice loss
        numerator = 2 * torch.sum(prob * y_true, dim=(2, 3))
        denominator = torch.sum(prob ** 2 + y_true ** 2, dim=(2, 3))
        dice_loss = 1 - (numerator + 1) / (denominator + 1)
        
        return dice_loss.mean()
    
    
    