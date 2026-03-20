import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datasets_split import create_splits, make_dataloaders

# === Output Directory Setup ===
output_dir = "training_results"
os.makedirs(output_dir, exist_ok=True)

# === Device setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)


# === Model Definition ===
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18Small(nn.Module):
    """
    ResNet18 for 28x28 input. No aggressive stem downsampling.
    Returns softmax probabilities by default (for CP), or logits for training.
    """
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 28x28 -> 28x28 -> 14x14 -> 7x7 -> 4x4
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return F.softmax(logits, dim=1)





# === Training ===
def training(train_loader, val_loader, num_class, in_channels, output_dir, model_name, num_epochs=15):
    model = ResNet18Small(num_classes=num_class, in_channels=in_channels).to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

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
            loss = criterion(torch.log(output), label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = output.argmax(dim=1)
            correct_train += (preds == label).sum().item()
            total_train += label.size(0)
            running_train_loss += loss.item()

            if (i + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, "
                      f"Train Loss: {running_train_loss/(i+1):.4f}, "
                      f"Train Acc: {correct_train/total_train:.4f}")

        train_acc = correct_train / total_train
        running_train_loss /= len(train_loader)
        scheduler.step()

        # === Validation ===
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(torch.log(output), label)
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
    plt.savefig(os.path.join(output_dir, "training_plot.png"))
    plt.close()


def main(num_classes, dataset_name, saved_path, model_name="resnet18model.pth", num_epochs=15):
    model_name = f"{dataset_name}_{model_name}"
    npz_file = f"datasets/{dataset_name}.npz"

    loaders, num_classes, in_channels = make_dataloaders(npz_file, dataset_name)

    training(loaders["train"], loaders["val"],
             num_classes, in_channels,
             saved_path, model_name, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18Small on medical images')
    parser.add_argument('--num_classes',   type=int, required=True)
    parser.add_argument('--dataset_name',  type=str, required=True)
    parser.add_argument('--saved_path',    type=str, default='training_results')
    parser.add_argument('--model_name',    type=str, default='resnet18model.pth')
    parser.add_argument('--num_epochs',    type=int, default=15)
    args = parser.parse_args()

    main(args.num_classes, args.dataset_name, args.saved_path,
         args.model_name, args.num_epochs)