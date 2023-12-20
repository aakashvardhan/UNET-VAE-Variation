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
        train_pd = pd.read_csv(self.config['train_f'], sep=self.sep, header=None)[0].tolist()
        val_pd = pd.read_csv(self.config['val_f'], sep=self.sep, header=None)[0].tolist()
        
        [self.train_img_lst.append(os.path.join(self.config['img_dir'], i + ".jpg")) for i in train_pd]
        [self.train_mask_lst.append(os.path.join(self.config['mask_dir'], i + ".png")) for i in train_pd]
        [self.val_img_lst.append(os.path.join(self.config['img_dir'], i + ".jpg")) for i in val_pd]
        [self.val_mask_lst.append(os.path.join(self.config['mask_dir'], i + ".png")) for i in val_pd]
        print(f"Number of training images: {len(self.train_img_lst)}")
        print(f"Number of training masks: {len(self.train_mask_lst)}")
        print(f"Number of validation images: {len(self.val_img_lst)}")
        print(f"Number of validation masks: {len(self.val_mask_lst)}")
        print("Data preparation complete!")
            
    def train_dataloader(self) -> DataLoader:
        self.train_dataset = ox.OxfordIIIT(self.train_img_lst, self.train_mask_lst)
        return DataLoader(self.train_dataset,
                          batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'],
                          pin_memory=self.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self) -> DataLoader:
        self.val_dataset = ox.OxfordIIIT(self.val_img_lst, self.val_mask_lst)
        return DataLoader(self.val_dataset,
                          batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'],
                          pin_memory=self.pin_memory,
                          shuffle=False)