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
    labels_csv = os.path.join(folder, labels_csv) #Labels path
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

training_image_paths, training_labels = load_dataset_from_csv("datasets") 
validation_image_paths, validation_labels = load_dataset_from_csv("datasets") 
train_dataset = PiiDataset(training_image_paths, training_labels, transform=transform)
val_dataset = PiiDataset(validation_image_paths, validation_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# === ResNet18 Model Setup ===

model = models.resnet18(pretrained=True)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.2), #Should it be lower? --> could cause underfitting? 
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 4)
)
model = model.to(device)



# === Loss and Optimizer ===
class_counts = [training_labels.count(0), training_labels.count(1), training_labels.count(2), training_labels.count(3)]
weights = [1.0 / c for c in class_counts]
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)

# === Training Loop ===
epochs = 15
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_model_path = os.path.join(output_dir, "retnet_diverse_trainingset_model.pth")

# ROC tracking



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

'''
TODO: 
break code into functions
Take in dataset name, number of classes and where we want the model path to be saved 
Build predict.py
'''

