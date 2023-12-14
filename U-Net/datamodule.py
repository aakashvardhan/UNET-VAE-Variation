from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from dataset import oxford_IIIT as ox
import pandas as pd

class DataModule(LightningDataModule):
    def __init__(self,
                 train_f: str,
                val_f: str,
                img_dir: str,
                mask_dir: str,
                batch_size: int,
                num_workers: int,
                sep=',',
                pin_memory: bool = True):
        super().__init__()
        self.sep = sep
        self.train_f = train_f
        self.val_f = val_f
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_img_lst = []
        self.train_mask_lst = []
        self.val_img_lst = []
        self.val_mask_lst = []
        
    def prepare_data(self):
        train_pd = pd.read_csv(self.train_f, sep=self.sep, header=None)[0].tolist()
        val_pd = pd.read_csv(self.val_f, sep=self.sep, header=None)[0].tolist()
        
        [self.train_img_lst.append(os.path.join(self.img_dir, i + ".jpg")) for i in train_pd]
        [self.train_mask_lst.append(os.path.join(self.mask_dir, i + ".png")) for i in train_pd]
        [self.val_img_lst.append(os.path.join(self.img_dir, i + ".jpg")) for i in val_pd]
        [self.val_mask_lst.append(os.path.join(self.mask_dir, i + ".png")) for i in val_pd]
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = ox.OxfordIIIT(self.train_img_lst, self.train_mask_lst)
            self.val_dataset = ox.OxfordIIIT(self.val_img_lst, self.val_mask_lst)
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)