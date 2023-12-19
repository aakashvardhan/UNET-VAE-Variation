import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import numpy as np
import os


class OxfordIIIT(Dataset):
    
    def __init__(self,
                 imgs_file: list[str],
                 masks_file: list[str],
                 transform_img: transforms.Compose = None,
                 transform_mask: transforms.Compose = None,
                 img_size: tuple[int, int] = (224, 224)):
        super().__init__()
        self.imgs_file = imgs_file
        self.masks_file = masks_file
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.img_size = img_size
        
    def __len__(self):
        return len(self.imgs_file)
    
    def __getitem__(self, idx: int):
        imgs = self.imgs_file[idx]
        masks = self.masks_file[idx]
        if self.transform_img is None:
            self.transform_img = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5]),
            ])
            transformed_img = self.transform_img(imgs)
            self.transform_mask = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
            transformed_mask = self.transform_mask(masks) * 255 - 1.0
        return transformed_img, transformed_mask
    

