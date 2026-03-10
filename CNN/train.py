from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MultiLabelImageDataset
from src.model import MultiLabelCNN
from src.transforms import build_eval_transform, build_train_transform
from src.utils import load_config, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-label CNN classifier")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Comma-separated class names in CSV column order",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training from",
    )
    return parser.parse_args()


def build_loaders(config: dict, class_names: List[str]):
    train_dataset = MultiLabelImageDataset(
        csv_path=config["train_csv"],
        image_root=config["image_root"],
        class_names=class_names,
        transform=build_train_transform(config["image_size"]),
    )
    val_dataset = MultiLabelImageDataset(
        csv_path=config["val_csv"],
        image_root=config["image_root"],
        class_names=class_names,
        transform=build_eval_transform(config["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True)
        and config["num_workers"] > 0,
        prefetch_factor=config.get("prefetch_factor", 2)
        if config["num_workers"] > 0
        else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config.get("pin_memory", True),
        persistent_workers=config.get("persistent_workers", True)
        and config["num_workers"] > 0,
        prefetch_factor=config.get("prefetch_factor", 2)
        if config["num_workers"] > 0
        else None,
    )
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    threshold: float,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    all_targets = []
    all_preds = []

    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(images)
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_targets.append(targets.detach().cpu())
            all_preds.append(preds.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
    }


def load_training_state(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
) -> tuple[int, float]:
    checkpoint = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_micro_f1 = float(checkpoint.get("best_val_micro_f1", -1.0))
    return start_epoch, best_val_micro_f1


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    class_names: List[str],
    config: dict,
    best_val_micro_f1: float,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "config": config,
        "best_val_micro_f1": best_val_micro_f1,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    if not class_names:
        raise ValueError("At least one class name is required")

    config = load_config(args.config)
    config["model"]["num_classes"] = len(class_names)

    set_seed(config["seed"])
    device = resolve_device(config["device"])
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))

    train_loader, val_loader = build_loaders(config, class_names)

    model = MultiLabelCNN(
        num_classes=config["model"]["num_classes"],
        base_channels=config["model"]["base_channels"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    use_amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / config["best_model_name"]
    last_model_path = save_dir / config.get("last_model_name", "last_model.pt")

    history = []
    best_val_micro_f1 = -1.0
    epochs_without_improvement = 0
    start_epoch = 1

    resume_candidate = args.resume or config.get("resume_checkpoint")
    if resume_candidate:
        resume_path = Path(resume_candidate)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_val_micro_f1 = load_training_state(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )
        print(
            f"Resumed from {resume_path} at epoch {start_epoch} "
            f"with best val micro-F1={best_val_micro_f1:.4f}"
        )

    for epoch in range(start_epoch, config["epochs"] + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            threshold=config["threshold"],
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            threshold=config["threshold"],
            scaler=None,
            use_amp=use_amp,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train micro-F1 {train_metrics['micro_f1']:.4f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val micro-F1 {val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                path=model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                class_names=class_names,
                config=config,
                best_val_micro_f1=best_val_micro_f1,
            )
        else:
            epochs_without_improvement += 1

        if config.get("save_last_checkpoint", True):
            save_checkpoint(
                path=last_model_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                class_names=class_names,
                config=config,
                best_val_micro_f1=best_val_micro_f1,
            )

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    save_json({"history": history, "best_val_micro_f1": best_val_micro_f1}, save_dir / "history.json")
    print(f"Best checkpoint saved at: {model_path}")


if __name__ == "__main__":
    main()
