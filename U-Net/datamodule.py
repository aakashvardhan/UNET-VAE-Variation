from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from dataset import oxford_IIIT as ox
import pandas as pd

class DataModule(LightningDataModule):
    def __init__(self,
                 config,
                sep=',',
                pin_memory: bool = True):
        super().__init__()
        self.sep = sep
        self.config = config
        self.pin_memory = pin_memory
        
        self.train_img_lst = []
        self.train_mask_lst = []
        self.val_img_lst = []
        self.val_mask_lst = []
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_pd = pd.read_csv(self.config['train_f'], sep=self.sep, header=None)[0].to_list()
            val_pd = pd.read_csv(self.config['val_f'], sep=self.sep, header=None)[0].to_list()

            self.train_img_lst = [os.path.join(self.config['img_dir'], i + ".jpg") for i in train_pd]
            self.train_mask_lst = [os.path.join(self.config['mask_dir'], i + ".png") for i in train_pd]
            self.val_img_lst = [os.path.join(self.config['img_dir'], i + ".jpg") for i in val_pd]
            self.val_mask_lst = [os.path.join(self.config['mask_dir'], i + ".png") for i in val_pd]

            self.train_dataset = ox.OxfordIIIT(self.train_img_lst, self.train_mask_lst)
            self.val_dataset = ox.OxfordIIIT(self.val_img_lst, self.val_mask_lst)
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'],
                          pin_memory=self.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'],
                          pin_memory=self.pin_memory,
                          shuffle=False)