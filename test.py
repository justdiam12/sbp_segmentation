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
    print(colors.shape)
    
    image_array = input_tensor.numpy()
    rgb_image = np.zeros((256, 256, 3), dtype=np.float64)

    for i in range(6):
        rows, columns = np.where(image_array[i,:,:] == 1)       
        
        for c in range(3):
            for row, col in zip(rows, columns):
                rgb_image[row, col, c] = colors[i,c]
   
    rgb_image_pil = Image.fromarray(rgb_image.astype(np.uint8))
    rgb_image_pil.save(output_directory)


model_path = '/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/UNet_Epoch50.pth.tar'
epochs = 50
model = UNet(in_channels=1, out_channels=6)
load_checkpoint(torch.load(model_path), model)
model.eval()

with torch.no_grad():
    train_dir = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Train/"
    val_dir = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Validation/"
    test_dir = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Test/"
    train_images = [file for file in os.listdir(train_dir) if file.endswith('.mat')]
    val_images = [file for file in os.listdir(val_dir) if file.endswith('.mat')]
    test_images = [file for file in os.listdir(test_dir) if file.endswith('.mat')]

    # for i in range(len(train_images)):
    #     image_directory = train_dir + train_images[i]
    #     mat = io.loadmat(image_directory)
    #     image = np.array(mat["AMP"], dtype=np.float32)
    #     # image = np.array(Image.open(image_directory).convert("L"), dtype=np.float32)
    #     image = np.expand_dims(image, axis=0)
    #     image = np.expand_dims(image, axis=0)
    #     image = torch.from_numpy(image).float()
    #     preds = torch.sigmoid(model(image))
    #     preds = (preds > 0.5).float()
    #     output = preds.squeeze(0)
    #     output_name = train_images[i]
    #     output_name = output_name[:-4] + ".png"
    #     output_directory =  "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Output_" + str(epochs) + "_Epochs/" +  output_name
    #     convert_to_RGB(output, output_directory)

    # for i in range(len(val_images)):
    #     image_directory = val_dir + val_images[i]
    #     mat = io.loadmat(image_directory)
    #     image = np.array(mat["AMP"], dtype=np.float32)
    #     # image = np.array(Image.open(image_directory).convert("L"), dtype=np.float32)
    #     image = np.expand_dims(image, axis=0)
    #     image = np.expand_dims(image, axis=0)
    #     image = torch.from_numpy(image).float()
    #     preds = torch.sigmoid(model(image))
    #     preds = (preds > 0.5).float()
    #     output = preds.squeeze(0)
    #     output_name = val_images[i]
    #     output_name = output_name[:-4] + ".png"
    #     output_directory =  "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Output_" + str(epochs) + "_Epochs/" +  output_name
    #     convert_to_RGB(output, output_directory)
    
    for i in range(len(test_images)):
        image_directory = test_dir + test_images[i]
        mat = io.loadmat(image_directory)
        image = np.array(mat["AMP"], dtype=np.float32)
        # image = np.array(Image.open(image_directory).convert("L"), dtype=np.float32)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
        preds = torch.sigmoid(model(image))
        preds = (preds > 0.5).float()
        output = preds.squeeze(0)
        output_name = test_images[i]
        output_name = output_name[:-4] + ".png"
        output_directory =  "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Test_Output_" + str(epochs) + "_Epochs/" +  output_name
        convert_to_RGB(output, output_directory)