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
        # Ensure y_true is of type torch.long for F.one_hot
        if y_true.dtype != torch.long:
            y_true = y_true.long()

        # Apply softmax to y_pred if required
        prob = y_pred
        if self.config['softmax_dim'] is not None:
            prob = nn.Softmax(dim=self.config['softmax_dim'])(y_pred)

        # Convert y_true to one-hot encoding
        # Shape of y_true after one_hot will be (N, H, W, num_classes)
        y_true = F.one_hot(y_true, num_classes=self.config['num_classes'])

        # Rearrange the tensor dimensions to (N, num_classes, H, W)
        y_true = y_true.permute(0, 3, 1, 2).float()

        # Calculate dice loss
        numerator = 2 * torch.sum(prob * y_true, dim=(2, 3))
        denominator = torch.sum(prob.pow(2) + y_true.pow(2), dim=(2, 3))
        dice_loss = 1 - (numerator + 1) / (denominator + 1)

        return dice_loss.mean()


# y_pred = torch.tensor([[[[0.7, 0.3], [0.4, 0.6]], [[0.3, 0.7], [0.6, 0.4]]]])
# y_true = torch.tensor([[[0, 1], [1, 0]]])
# config = {'softmax_dim': 1, 'num_classes': 2}
# dice_loss = DiceLoss(config)
# assert dice_loss(y_pred, y_true) < 0.2