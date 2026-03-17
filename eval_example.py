import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
from eval import evaluate, ALPHA, SCORING_FUNCTIONS
from CPWrapper import CPWrapper
from scoring_functions import oneminussoftmax, APS  # add more as needed
import torchvision.transforms as transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)   # wrapper contract: model outputs softmax vector
        )

    def forward(self, x):
        return self.net(x)
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model  = SmallCNN().to(device)


# ── 3. Train ──────────────────────────────────────────────────────────────────

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# NLLLoss because the model already outputs probabilities (post-softmax)
criterion = nn.NLLLoss()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Reserve 10 000 training samples for calibration — never seen by the model
train_set, calib_set = random_split(full_train, [40_000, 10_000])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)

print("Training...")
model.train()
for epoch in range(5):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(torch.log(model(x) + 1e-9), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/5  loss={total_loss/len(train_loader):.4f}")


evaluate(model, calib_set, test_loader, 10, device="cpu")