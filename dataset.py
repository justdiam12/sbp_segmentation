import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.io as io

class SBPDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [file for file in os.listdir(image_dir) if file.endswith('.mat')]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        color1 = [0, 0, 255]    # Blue
        color2 = [255, 0, 0]    # Red
        color3 = [255, 255, 0]  # Yellow
        color4 = [0, 255, 0]    # Green
        color5 = [0, 255, 255]  # Cyan
        color6 = [0, 0, 0]      # Black   
        # print(self.images)

        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mat = io.loadmat(image_path)
        image = np.array(mat['AMP'], dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        # image = np.expand_dims(image, axis=0)
        # Switch dimension 3 with dim 1 for 
        mask = np.array(mat['label'], dtype=np.float32)
        # print(mask.shape)
        mask = np.transpose(mask, (2, 0, 1))

        # mask1 = np.all(mask == np.array(color1), axis=-1) # Blue
        # mask2 = np.all(mask == np.array(color2), axis=-1)
        # mask3 = np.all(mask == np.array(color3), axis=-1)
        # mask4 = np.all(mask == np.array(color4), axis=-1)
        # mask5 = np.all(mask == np.array(color5), axis=-1)
        # mask6 = np.all(mask == np.array(color6), axis=-1)

        # mask = np.stack([mask1, mask2, mask3, mask4, mask5, mask6], axis=0).astype(np.uint8)

        # if self.transform is not None:
        #     augmentations = self.transform(image=image, mask=mask)
        #     image = augmentations["image"]
        #     mask = augmentations["mask"]

        return image, mask