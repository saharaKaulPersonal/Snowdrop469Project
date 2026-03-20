from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, DataLoader
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

# === Dataset loader ===
def make_dataloaders(npz_file, dataset_name):
    data = np.load(npz_file)
    num_classes = int(np.max(data["train_labels"]) + 1)

    datasets = create_splits(npz_file)
    batch_size = 256
    num_workers = 4

    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size,
                          shuffle=(split == "train"), num_workers=num_workers)
        for split in ["train", "val", "test", "calib", "eval"]
    }

    # pathmnist is RGB, everything else is grayscale
    in_channels = 3 if dataset_name == "pathmnist" else 1

    return loaders, num_classes, in_channels

# ====================
# Create splits function
# ====================
def create_splits(npz_file, test_fraction=0.25, calib_fraction=0.6, seed=42):
    """
    Splits dataset into train/val/test/calib/eval.
    Split indices are saved alongside the .npz file on first run and
    reloaded on every subsequent run â€” so the split is always identical.

    Cache file: <npz_stem>_split_indices.npz  (next to the dataset file)
    """
    npz_path = Path(npz_file)
    cache_path = npz_path.with_name(npz_path.stem + "_split_indices.npz")

    data = np.load(npz_file)
    images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]], axis=0)
    labels = np.concatenate([
        data["train_labels"].flatten(),
        data["val_labels"].flatten(),
        data["test_labels"].flatten()
    ])

    
    if cache_path.exists():
        print(f"[create_splits] Loading cached split indices from {cache_path}")
        cached = np.load(cache_path)
        train_idx = cached["train_idx"]
        val_idx   = cached["val_idx"]
        test_idx  = cached["test_idx"]
        calib_idx = cached["calib_idx"]

    
    else:
        print(f"[create_splits] Computing splits and saving indices to {cache_path}")
        rng = np.random.default_rng(seed)
        total_idx = np.arange(len(images))
        rng.shuffle(total_idx)

        # Split off test pool
        n_test_total = int(test_fraction * len(total_idx))
        test_pool_idx = total_idx[:n_test_total]
        remaining_idx = total_idx[n_test_total:]

        # Split test pool â†’ calibration + actual test
        n_calib   = int(calib_fraction * len(test_pool_idx))
        calib_idx = test_pool_idx[:n_calib]
        test_idx  = test_pool_idx[n_calib:]

        # Split remaining â†’ train / val
        n_val      = int(0.15 * len(remaining_idx))
        val_idx    = remaining_idx[:n_val]
        train_idx  = remaining_idx[n_val:]

        np.savez(
            cache_path,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            calib_idx=calib_idx,
        )

    
    datasets = {
        "train": MedMNISTNpzDataset(images[train_idx], labels[train_idx], transform=train_transform),
        "val":   MedMNISTNpzDataset(images[val_idx],   labels[val_idx],   transform=eval_transform),
        "test":  MedMNISTNpzDataset(images[test_idx],  labels[test_idx],  transform=eval_transform),
        "calib": MedMNISTNpzDataset(images[calib_idx], labels[calib_idx], transform=eval_transform),
        "eval":  MedMNISTNpzDataset(images[test_idx],  labels[test_idx],  transform=eval_transform),
    }

    return datasets