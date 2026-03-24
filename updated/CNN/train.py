from pathlib import Path
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets_split import create_splits


# CONFIG
@dataclass
class TrainConfig:
    seed: int = 42
    image_size: int = 28
    batch_size: int = 128
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15

config = TrainConfig()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAMES = ["octmnist", "pathmnist", "tissuemnist"]
DATA_DIR = Path("datasets")
NPZ_DIR = DATA_DIR 
OUT_DIR = Path("CNN\\outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# SIMPLE CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels:int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(), nn.MaxPool2d(2),
        )
        feature_size = config.image_size // 8
        flattened_dim = 128 * feature_size * feature_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# DATALOADERS
def make_dataloaders(npz_file):

    data = np.load(npz_file)

    num_classes = int(np.max(data["train_labels"]) + 1)

    datasets = create_splits(npz_file)
  

    train_loader = DataLoader(datasets['train'], batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers)

    val_loader = DataLoader(datasets['val'], batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
    
    test_loader = DataLoader(datasets['test'], batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    calib_loader = DataLoader(datasets['calib'], batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    eval_loader = DataLoader(datasets['eval'], batch_size=config.batch_size,
                         shuffle=False, num_workers=config.num_workers)


    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "calib": calib_loader,
        "eval": eval_loader
    }

    return loaders, num_classes


# TRAIN / VALIDATION
def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total = 0
    correct = 0
    loss_sum = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, 1)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        total += batch_size
        correct += (preds == labels).sum().item()

    return loss_sum / total, correct / total

# FIT MODEL
def fit_dataset_model(dataset_name):
    npz_file = NPZ_DIR / f"{dataset_name}.npz"
    loaders, num_classes = make_dataloaders(npz_file)
    in_channels = 1
    if dataset_name in ["pathmnist"]:
        in_channels = 3

    model = SimpleCNN(num_classes, in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_acc = 0.0
    best_model_path = OUT_DIR / f"{dataset_name}_best_model.pth"

    for epoch in range(config.epochs):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion)

        print(f"{dataset_name} | Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model ({best_val_acc:.4f}) → {best_model_path}")
    
    test_loss, test_acc = run_epoch(model, loaders["test"], criterion)
    print(f"{dataset_name} | Test Acc: {test_acc:.4f}")

    return model


def main():
    for dataset_name in DATASET_NAMES:

        print(f"Training {dataset_name}...")

        fit_dataset_model(dataset_name)



if __name__ == "__main__":
    main()
