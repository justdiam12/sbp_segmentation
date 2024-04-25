import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from utils import load_checkpoint
import torch.nn as nn
import torch.optim as optim
from UNet.unet import UNet
import numpy as np
from PIL import Image
import scipy.io as io

def convert_to_RGB(input_tensor, output_directory):
    colors = np.array([
        np.array([0, 0, 0]),       # Black
        np.array([0, 0, 255]),     # Blue
        np.array([255, 0, 0]),     # Red
        np.array([255, 255, 0]),   # Yellow
        np.array([0, 255, 0]),     # Green
        np.array([0, 255, 255]),   # Cyan
    ])
    
    image_array = input_tensor
    print(input_tensor.shape)
    rgb_image = np.zeros((256, 256, 3), dtype=np.float32)

    for i in range(6):
        rows, columns = np.where(image_array[:,:,i] == 1)       
        
        for c in range(3):
            for row, col in zip(rows, columns):
                rgb_image[row, col, c] = colors[i,c]
   
    rgb_image_pil = Image.fromarray(rgb_image.astype(np.uint8))
    rgb_image_pil.save(output_directory)


train_dir = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Train/"
val_dir = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Validation/"
train_images = [file for file in os.listdir(train_dir) if file.endswith('.mat')]
val_images = [file for file in os.listdir(val_dir) if file.endswith('.mat')]
for i in range(len(train_images)):
    image_directory = train_dir + train_images[i]
    mat = io.loadmat(image_directory)
    output = np.array(mat["label"], dtype=np.float32)

    output_name = train_images[i]
    output_name = output_name[:-4] + ".png"
    output_directory =  "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Label/" +  output_name
    convert_to_RGB(output, output_directory)

for i in range(len(val_images)):
    image_directory = val_dir + val_images[i]
    mat = io.loadmat(image_directory)
    output = np.array(mat["label"], dtype=np.float32)
    output_name = val_images[i]
    output_name = output_name[:-4] + ".png"
    output_directory =  "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Label/" +  output_name
    convert_to_RGB(output, output_directory)