import os
import csv
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
import argparse 
#from sklearn.metrics import roc_curve, auc

# === Output Directory Setup ===
output_dir = "training_results" #Where results are saved
os.makedirs(output_dir, exist_ok=True) #Make folder 

# === Device setup ===
device = "cuda" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(1234) 
if device == 'cuda':
    torch.cuda.manual_seed_all(1234)

# === Dataset loader ===
def load_dataset_from_csv(folder, modality_type, set_type):
    dataset_dir = os.path.join("./", folder, "images", modality_type, set_type) #Get path for image pathway
    labels_csv = modality_type + "_" + set_type + "_labels.csv"
    labels_csv = os.path.join(folder,  labels_csv) #Labels path
    #labels_csv  = "datasets/octmnist_train_labels.csv"
    image_paths, labels = [], []
    #Loop through all labels and get the label
    with open(labels_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile) 
        for row in reader:
            image = row["image_index"] + ".png"
            image_paths.append(os.path.join(dataset_dir, image))
            labels.append(int(row["label_index"]))
    return image_paths, labels

# === Custom Dataset ===
class PiiDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #Opens image and returns it
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label, self.image_paths[idx]

# === Data Transformations ===
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load and split dataset ===
def load_split_dataset(modality_type):
    training_image_paths, training_labels = load_dataset_from_csv("datasets", modality_type, "train") 
    validation_image_paths, validation_labels = load_dataset_from_csv("datasets", modality_type, "val") 
    train_dataset = PiiDataset(training_image_paths, training_labels, transform=transform)
    val_dataset = PiiDataset(validation_image_paths, validation_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader, training_labels

# === ResNet18 Model Setup ===
def training(train_loader, val_loader, training_labels, num_class, output_dir, model_name, num_epochs=15):
    model = models.resnet18(pretrained=True)
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

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
    class_counts = []
    for i in range(num_class):
        class_counts.append(training_labels.count(i)) 
    weights = [1.0 / c for c in class_counts]
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

    # === Training Loop ===
    epochs = num_epochs
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, model_name)

    for epoch in range(epochs):
        # === Training ===
        model.train()
        running_train_loss, train_acc = 0.0, 0.0
        for data, label, _ in train_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)

            probs = torch.softmax(output, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            train_acc += acc.item() / len(train_loader)
            running_train_loss += loss.item() / len(train_loader)

        # === Validation ===
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for data, label, paths in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)

                acc = (output.argmax(dim=1) == label).float().mean()
                val_acc += acc.item() / len(val_loader)
                val_loss += loss.item() / len(val_loader)

                probs = torch.softmax(output, dim=1)
        
                preds = output.argmax(dim=1)
                
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Train Loss: {running_train_loss:.4f} | Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        train_losses.append(running_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1}")


    # === Training/Validation Loss Plot ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training plot saved to {loss_plot_path}")

def main(num_classes,  dataset_name, saved_path, model_name = "resnet18model.pth", num_epochs=15):
    train_loader, val_loader, training_labels = load_split_dataset(dataset_name)
    training(train_loader, val_loader, training_labels, num_classes, saved_path, model_name, num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ResNet18 on medical images')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., octmnist)')
    parser.add_argument('--saved_path', type=str, default='training_results', help='Path to save results')
    parser.add_argument('--model_name', type=str, default='resnet18model.pth', help='Model filename')
    parser.add_argument('--num_epochs', type=int, default=15, help='Number of epochs')
    
    args = parser.parse_args()
    
    main(args.num_classes, args.dataset_name, args.saved_path, 
         args.model_name, args.num_epochs)
'''
TODO: 
break code into functions
Take in dataset name, number of classes and where we want the model path to be saved 
Build predict.py
'''

