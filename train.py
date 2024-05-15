import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from UNet.unet import UNet

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    # save_predictions_as_imgs
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 6
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v6/Train'
TRAIN_MASK_DIR = 'E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v6/Train'
VAL_IMG_DIR = 'E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v6/Validation'
VAL_MASK_DIR = 'E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v6/Validation'

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            # print(data.shape)
            # print(targets.shape)
            predictions = model(data)
            # print(predictions.shape)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNet(in_channels=2, out_channels=6).float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform
    )

    scaler = torch.cuda.amp.GradScaler()

    epoch_saves = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1} of {NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        if epoch+1 in epoch_saves:
            save_checkpoint(checkpoint, f"UNet_Epoch{epoch+1}.pth.tar")

        check_accuracy(val_loader, model, device=DEVICE)

        ### Save prediction as images ###
        # save_predictions_as_imgs(
        #     val_loader, model, folder="/Users/justindiamond/Documents/Documents/UW-APL/NN_Models/Saved_Images/", device=DEVICE
        # )

if __name__ == '__main__':
    main()