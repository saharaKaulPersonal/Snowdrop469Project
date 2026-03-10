"""
Quick smoke test for CPWrapper using a small CNN trained on MNIST.

Split strategy (per the paper):
  - train set  → model training
  - val set    → calibration (unseen during training, as required)
  - test set   → evaluate empirical coverage of the prediction sets

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from CPWrapper import CPWrapper
from scoring_functions import oneminussoftmax, APS


# ── 1. Data ──────────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set   = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Reserve 10 000 training samples for calibration — never seen by the model
train_set, calib_set = random_split(full_train, [50_000, 10_000])

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False)


# ── 2. Small CNN ──────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
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

print("Training...")
model.train()
for epoch in range(3):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(torch.log(model(x) + 1e-9), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3  loss={total_loss/len(train_loader):.4f}")


alpha   = 0.1   # target error rate → we want ≥ 90 % coverage
wrapper = CPWrapper(model, alpha=alpha, scoring_fn=oneminussoftmax, device=device)

print("\nCalibrating on held-out calibration set (unseen during training)...")
wrapper.fit(calib_set)   # Dataset passed directly — no manual DataLoader needed
print(f"  qhat (threshold) = {wrapper.threshold:.4f}")


# ── 6. Evaluate coverage on the test set ─────────────────────────────────────

correct_in_set = 0
total          = 0
set_sizes      = []

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        pred_sets = wrapper.predict(x)
        for pred_set, label in zip(pred_sets, y.numpy()):
            correct_in_set += int(label in pred_set)
            set_sizes.append(len(pred_set))
            total += 1

empirical_coverage = correct_in_set / total
print(f"\nResults over {total} test samples:")
print(f"  Empirical coverage : {empirical_coverage:.4f}  (target ≥ {1-alpha:.2f})")
print(f"  Average set size   : {np.mean(set_sizes):.3f}")
print(f"  Coverage guarantee : {'PASS' if empirical_coverage >= 1 - alpha else 'FAIL'}")