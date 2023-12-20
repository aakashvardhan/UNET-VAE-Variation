from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class OxfordIIIT(Dataset):
    def __init__(self, imgs_file, masks_file, transform_img=None, transform_mask=None, img_size=(224, 224)):
        super().__init__()
        self.imgs_file = imgs_file
        self.masks_file = masks_file
        self.img_size = img_size

        # Set default transformations if none provided
        if transform_img is None:
            self.transform_img = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform_img = transform_img

        if transform_mask is None:
            self.transform_mask = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor()
            ])
        else:
            self.transform_mask = transform_mask

    def __len__(self):
        return len(self.imgs_file)

    def __getitem__(self, idx):
        img_path = self.imgs_file[idx]
        mask_path = self.masks_file[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        transformed_img = self.transform_img(img)
        transformed_mask = self.transform_mask(mask) * 255.0 - 1.0

        return {'image': transformed_img, 'mask': transformed_mask}

    def show_img_mask(self, idx):
        img = Image.open(self.imgs_file[idx])
        mask = Image.open(self.masks_file[idx])

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.show()

