from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ====================
# Transform configs
# ====================

train_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ====================
# Dataset class
# ====================
class MedMNISTNpzDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        if img.ndim == 2:
            img = Image.fromarray(img.astype("uint8"), mode="L")
        else:
            img = Image.fromarray(img.astype("uint8"), mode="RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ====================
# Create splits function
# ====================
def create_splits(npz_file, test_fraction=0.25, calib_fraction=0.6, seed=42):
    """
    Splits dataset into train/val/test, with calibration data drawn only from test set.
    - test_fraction: fraction of total data used for test+calibration
    - calib_fraction: fraction of the test set used as calibration
    """
    data = np.load(npz_file)
    # combine all data into one pool
    images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]], axis=0)
    labels = np.concatenate([data["train_labels"].flatten(),
                             data["val_labels"].flatten(),
                             data["test_labels"].flatten()])

    rng = np.random.default_rng(seed)
    total_idx = np.arange(len(images))
    rng.shuffle(total_idx)

    # --- Split off test pool ---
    n_test_total = int(test_fraction * len(total_idx))
    test_pool_idx = total_idx[:n_test_total]
    remaining_idx = total_idx[n_test_total:]

    # --- Split test pool into calibration + actual test ---
    n_calib = int(calib_fraction * len(test_pool_idx))
    calib_idx = test_pool_idx[:n_calib]
    test_idx = test_pool_idx[n_calib:]

    # --- Split remaining data into train/val ---
    n_val = int(0.15 * len(remaining_idx))  # 15% of remaining for val
    val_idx = remaining_idx[:n_val]
    train_idx = remaining_idx[n_val:]

    # --- Build datasets ---
    datasets = {
        "train": MedMNISTNpzDataset(images[train_idx], labels[train_idx], transform=train_transform),
        "val": MedMNISTNpzDataset(images[val_idx], labels[val_idx], transform=eval_transform),
        "test": MedMNISTNpzDataset(images[test_idx], labels[test_idx], transform=eval_transform),
        "calib": MedMNISTNpzDataset(images[calib_idx], labels[calib_idx], transform=eval_transform),
        "eval": MedMNISTNpzDataset(images[test_idx], labels[test_idx], transform=eval_transform)
    }

    return datasets
