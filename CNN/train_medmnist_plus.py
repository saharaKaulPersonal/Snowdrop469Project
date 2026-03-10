from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import medmnist
import numpy as np
import torch
from medmnist import INFO
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import MultiLabelCNN
from src.transforms import build_eval_transform, build_train_transform
from src.utils import load_config, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3 classifiers on MedMNIST+ datasets"
    )
    parser.add_argument("--config", type=str, default="medmnist_plus_experiments.yaml")
    return parser.parse_args()


def build_medmnist_datasets(dataset_flag: str, image_size: int, download: bool):
    if dataset_flag not in INFO:
        raise ValueError(f"Unknown MedMNIST dataset flag: {dataset_flag}")

    dataset_info = INFO[dataset_flag]
    data_class = getattr(medmnist, dataset_info["python_class"])

    train_ds = data_class(
        split="train",
        transform=build_train_transform(image_size),
        download=download,
        size=image_size,
        as_rgb=True,
    )
    val_ds = data_class(
        split="val",
        transform=build_eval_transform(image_size),
        download=download,
        size=image_size,
        as_rgb=True,
    )
    return train_ds, val_ds, dataset_info


def make_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if dataset.split == "train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def labels_to_targets(labels: torch.Tensor, task: str, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = labels.to(device)
    if task == "multi-label, binary-class":
        return labels.float()

    labels = labels.view(-1).long()
    if task == "binary-class":
        return labels
    if task == "multi-class":
        return labels
    raise ValueError(f"Unsupported task type: {task}")


def predict_from_logits(logits: torch.Tensor, task: str, threshold: float) -> torch.Tensor:
    if task == "multi-label, binary-class":
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()

    pred_idx = torch.argmax(logits, dim=1)
    return pred_idx


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, task: str):
    if task == "multi-label, binary-class":
        micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return float(micro), float(macro)

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(micro), float(macro)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
    num_classes: int,
    threshold: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    y_true_batches = []
    y_pred_batches = []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = labels_to_targets(labels, task, num_classes, device)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                logits = model(images)
                if task == "multi-label, binary-class":
                    loss = criterion(logits, targets)
                else:
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

            preds = predict_from_logits(logits.detach(), task, threshold)
            if task == "multi-label, binary-class":
                y_true_batches.append(targets.detach().cpu())
                y_pred_batches.append(preds.detach().cpu())
            else:
                y_true_batches.append(targets.detach().cpu())
                y_pred_batches.append(preds.detach().cpu())

    y_true = torch.cat(y_true_batches).numpy()
    y_pred = torch.cat(y_pred_batches).numpy()
    micro_f1, macro_f1 = compute_f1(y_true, y_pred, task)

    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def train_one_experiment(global_cfg: dict, experiment_cfg: dict, device: torch.device):
    model_name = experiment_cfg["model_name"]
    dataset_flag = experiment_cfg["dataset_flag"]
    image_size = int(experiment_cfg.get("image_size", global_cfg["image_size"]))
    batch_size = int(experiment_cfg.get("batch_size", global_cfg["batch_size"]))
    epochs = int(experiment_cfg.get("epochs", global_cfg["epochs"]))

    train_ds, val_ds, info = build_medmnist_datasets(
        dataset_flag,
        image_size,
        download=bool(global_cfg.get("medmnist_download", True)),
    )
    task = info["task"]
    class_names = list(info["label"].values())
    num_classes = len(class_names)

    print(
        f"\n=== Experiment: {model_name} | dataset={dataset_flag} | "
        f"task={task} | classes={num_classes} | "
        f"train={len(train_ds)} | val={len(val_ds)} ==="
    )

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        num_workers=global_cfg["num_workers"],
        pin_memory=global_cfg.get("pin_memory", True),
        persistent_workers=global_cfg.get("persistent_workers", True),
        prefetch_factor=global_cfg.get("prefetch_factor", 2),
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        num_workers=global_cfg["num_workers"],
        pin_memory=global_cfg.get("pin_memory", True),
        persistent_workers=global_cfg.get("persistent_workers", True),
        prefetch_factor=global_cfg.get("prefetch_factor", 2),
    )

    model = MultiLabelCNN(num_classes=num_classes, base_channels=global_cfg["model"]["base_channels"]).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=global_cfg["learning_rate"],
        weight_decay=global_cfg["weight_decay"],
    )

    use_amp = bool(global_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(global_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"{model_name}.pt"
    history_path = save_dir / f"{model_name}_history.json"

    best_val_micro_f1 = -1.0
    patience = int(global_cfg["early_stopping_patience"])
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            task=task,
            num_classes=num_classes,
            threshold=float(global_cfg["threshold"]),
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task=task,
            num_classes=num_classes,
            threshold=float(global_cfg["threshold"]),
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        print(
            f"[{model_name}] Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} train_microF1={train_metrics['micro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_microF1={val_metrics['micro_f1']:.4f}"
        )

        if val_metrics["micro_f1"] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics["micro_f1"]
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "dataset_flag": dataset_flag,
                    "task": task,
                    "config": global_cfg,
                    "experiment": experiment_cfg,
                    "best_val_micro_f1": best_val_micro_f1,
                },
                best_model_path,
            )
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[{model_name}] Early stopping triggered.")
            break

    save_json(
        {
            "model_name": model_name,
            "dataset_flag": dataset_flag,
            "task": task,
            "num_classes": num_classes,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "best_val_micro_f1": best_val_micro_f1,
            "history": history,
        },
        history_path,
    )
    print(f"[{model_name}] Best checkpoint: {best_model_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    experiments = cfg.get("experiments", [])
    if len(experiments) != 3:
        raise ValueError("Config must define exactly 3 experiments.")

    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))

    for experiment in experiments:
        train_one_experiment(cfg, experiment, device)


if __name__ == "__main__":
    main()
