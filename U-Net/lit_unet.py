import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from models.unet import UNet
from dice_loss import DiceLoss


class LitUNet(LightningModule):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = UNet(config,
                          in_channels=3,
                          out_channels=3,
                          n_filters=64,
                          dropout=0.05)
        
        if self.config['loss_method'] == 'dice_loss':
            self.loss_fn = DiceLoss(config)
        elif self.config['loss_method'] == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        
        def forward(self, x):
            return self.model(x)
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
            scheduler = {
                'scheduler':torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=self.config['max_lr'],
                                                            steps_per_epoch=int(len(self.train_dataloader())),
                                                            epochs=self.config['epochs']),
                'interval':'step',
                'frequency':1}
            return [optimizer], [scheduler]
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            self.log('train_loss', loss)
            return loss