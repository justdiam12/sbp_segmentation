import torch
import torchvision
from dataset import SBPDataset
from torch.utils.data import DataLoader 

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = SBPDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = True,
    )

    val_ds = SBPDataset(
        image_dir = val_dir,
        mask_dir = val_maskdir,
        transform = val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size = batch_size,
        num_workers = num_workers,
        pin_memory = pin_memory,
        shuffle = False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cpu'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    model.train()

def save_predictions_as_imgs(
        loader, model, folder="/Users/justindiamond/Documents/Documents/UW-APL/NN_Models/Saved_Images", device='mps'
):
    model.eval()
    for idx, (x,y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/y_{idx}.png")

    model.train()