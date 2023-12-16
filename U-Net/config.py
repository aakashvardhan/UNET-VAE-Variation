import os

def get_config():
    return {
        'batch_size': 32,
        'train_f': './data/trainval.txt',
        'val_f': './data/test.txt',
        'img_dir': './kaggle/input/oxford-iiit-pet-dataset/images',
        'mask_dir': './kaggle/input/oxford-iiit-pet-dataset/annotations/trimaps',
        'num_workers': os.cpu_count(),
        'lr': 1e-3,
        'expansion_method': 'max_pooling',
        'compression_method': 'upsample',
        'num_classes': 3,
        'softmax_dim': 1,
    }