import os
import csv
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import argparse 
from datasets_split import create_splits

# === Output Directory Setup ===
output_dir = "training_results"
os.makedirs(output_dir, exist_ok=True)

# === Device setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

# === Dataset loader ===
def make_dataloaders(npz_file):

    data = np.load(npz_file)
    num_classes = int(np.max(data["train_labels"]) + 1)

    datasets = create_splits(npz_file)
    batch_size = 256
    num_workers = 4

    train_loader = DataLoader(datasets['train'], batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(datasets['val'], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    test_loader = DataLoader(datasets['test'], batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    calib_loader = DataLoader(datasets['calib'], batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    eval_loader = DataLoader(datasets['eval'], batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "calib": calib_loader,
        "eval": eval_loader
    }

    return loaders, num_classes


# === ResNet18 Model Setup ===
def training(train_loader, val_loader, num_class, output_dir, model_name, num_epochs=15):
    model = models.resnet18(pretrained=True)

    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False
    )

    # Freeze most layers
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, num_class)
    )

    model = model.to(device)

    # === Loss and Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )

    # === Training Loop ===
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, model_name)

    for epoch in range(num_epochs):

        # === Training ===
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=1)
            correct_train += (preds == label).sum().item()
            total_train += label.size(0)

            running_train_loss += loss.item()

            # print every 20 batches to monitor progress (will take a while)
            if (i + 1) % 20 == 0:
                current_acc = correct_train / total_train
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, "
                      f"Train Loss: {running_train_loss/(i+1):.4f}, Train Acc: {current_acc:.4f}")

        train_acc = correct_train / total_train
        running_train_loss /= len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)

                output = model(data)
                loss = criterion(output, label)

                preds = output.argmax(dim=1)
                correct_val += (preds == label).sum().item()
                total_val += label.size(0)

                val_loss += loss.item()

        val_acc = correct_val / total_val
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: "
              f"Train Acc: {train_acc:.4f}, Train Loss: {running_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(running_train_loss)
        val_losses.append(val_loss)

        # === Save Best Model ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1}")

    # === Plot Loss ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"Training plot saved to {loss_plot_path}")


def main(num_classes, dataset_name, saved_path,
         model_name=f"resnet18model.pth", num_epochs=15):
    model_name = f"{dataset_name}_{model_name}"
    npz_file = f"datasets/{dataset_name}.npz"

    loaders, _ = make_dataloaders(npz_file)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    training(train_loader, val_loader,
             num_classes, saved_path,
             model_name, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 on medical images')

    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Dataset name (e.g., octmnist)')
    parser.add_argument('--saved_path', type=str, default='training_results',
                        help='Path to save results')
    parser.add_argument('--model_name', type=str, default='resnet18model.pth',
                        help='Model filename')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of epochs')

    args = parser.parse_args()

    main(args.num_classes,
         args.dataset_name,
         args.saved_path,
         args.model_name,
         args.num_epochs)
