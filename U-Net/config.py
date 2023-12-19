import os
import torch
def get_config():
    return {
        'batch_size': 32,
        'train_f': './data/trainval.txt',
        'val_f': './data/test.txt',
        'img_dir': './kaggle/input/oxford-iiit-pet-dataset/images',
        'mask_dir': './kaggle/input/oxford-iiit-pet-dataset/annotations/trimaps',
        'num_workers': os.cpu_count(),
        'lr': 1e-3,
        'expansion_method': 'upsample',
        'compression_method': 'max_pooling',
        'num_classes': 3,
        'softmax_dim': 1,
        'loss_method': 'dice_loss',
        'epochs': 24,
        'max_lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }