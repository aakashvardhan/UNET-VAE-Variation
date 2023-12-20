import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from models.unet import UNet
from dice_loss import DiceLoss
from config import get_config
import torchmetrics
class LitUNet(LightningModule):
    def __init__(self,
                 config,
                 lr=1e-3,
                 max_lr=1e-3):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = UNet(config,
                          in_channels=3,
                          out_channels=3,
                          n_filters=64,
                          dropout=0.05)
        self.lr = lr
        self.max_lr = max_lr
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=config['num_classes'])
        
        if self.config['loss_method'] == 'dice_loss':
            self.loss_fn = DiceLoss(config)
        elif self.config['loss_method'] == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler':torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=self.max_lr,
                                                        steps_per_epoch=int(len(self.trainer.datamodule.train_dataloader())),
                                                        epochs=self.config['epochs']),
            'interval':'step',
            'frequency':1}
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        #accuracy metrics
        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        
        #log loss
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['mask']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        #accuracy metrics
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        #log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
            
            
            