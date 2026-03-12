from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "octmnist"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "CNN" / "checkpoints" / DATASET_NAME / "best.pt"
IMAGE_SIZE = 64


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        feature_size = IMAGE_SIZE // 8
        flattened_dim = 128 * feature_size * feature_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    head_weight_key = "classifier.4.weight"
    if head_weight_key not in state_dict:
        raise KeyError(f"Expected key '{head_weight_key}' in checkpoint state_dict")

    num_classes = int(state_dict[head_weight_key].shape[0])
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_softmax_vector(model: nn.Module, image_path: Path, device: torch.device) -> list[float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    return [float(p) for p in probs]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run octmnist model inference and print full softmax vector.")
    parser.add_argument("--image", required=True, help="Path to a single image file")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint (.pt)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    model = load_model(Path(args.checkpoint), device)
    softmax_vector = infer_softmax_vector(model, Path(args.image), device)

    print(json.dumps(softmax_vector))


if __name__ == "__main__":
    main()
