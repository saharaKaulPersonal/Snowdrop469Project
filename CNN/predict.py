from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.model import MultiLabelCNN
from src.transforms import build_eval_transform
from src.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for multi-label CNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    class_names = ckpt["class_names"]
    config = ckpt["config"]
    model_cfg = config["model"]

    model = MultiLabelCNN(
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = resolve_device(args.device)
    model = model.to(device)

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = build_eval_transform(config["image_size"])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probs_list = probs.tolist()
    pred_index = int(torch.argmax(probs).item())

    print(f"Image: {image_path}")
    print(f"Softmax vector: {probs_list}")
    if 0 <= pred_index < len(class_names):
        print(f"Predicted class: {class_names[pred_index]} (p={probs_list[pred_index]:.4f})")


if __name__ == "__main__":
    main()
