# CNN Multi-Label Image Classifier

Minimal PyTorch project for multi-label image classification using a custom CNN and `BCEWithLogitsLoss`.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset format

Place images under:

- `data/images/...`

Create CSV files:

- `data/train.csv`
- `data/val.csv`

CSV schema (header must match exactly):

```csv
image_path,class_a,class_b,class_c
cat/img001.jpg,1,0,1
dog/img002.jpg,0,1,0
```

- `image_path`: path relative to `image_root` (default `data/images`)
- Each class column: binary label `0` or `1`

## 3) Configure

Edit `config.yaml` to set:

- `image_size`, `batch_size`, `epochs`
- optimizer params (`learning_rate`, `weight_decay`)
- checkpoint output (`save_dir`, `best_model_name`)
- loader/runtime params (`num_workers`, `prefetch_factor`, `amp`, `resume_checkpoint`)

For ~30k images on a single GPU node, good starting values are:

- `batch_size: 64` (or highest that fits GPU memory)
- `num_workers: 8` (match `--cpus-per-task` on Slurm)
- `amp: true`
- `persistent_workers: true`
- `prefetch_factor: 4`
- `epochs: 20-40`

## 4) Train

```bash
python train.py --config config.yaml --classes class_a,class_b,class_c
```

This will:

- train with BCE-with-logits for multi-label targets
- report train/validation loss and micro-F1
- save best model to `checkpoints/best_model.pt`
- save training history to `checkpoints/history.json`

## 5) Predict

```bash
python predict.py --checkpoint checkpoints/best_model.pt --image data/images/cat/img001.jpg
```

Optional flags:

- `--threshold 0.4`
- `--device cpu` or `--device cuda`

## 6) Train on Slurm (HPC)

A template job script is included at `scripts/train_slurm.sh`.

This script trains **3 classifiers** on **MedMNIST+** datasets in one job using `train_medmnist_plus.py`.

1. Edit `medmnist_plus_experiments.yaml` to set your 3 experiments (`model_name`, `dataset_flag`, optional `batch_size`/`epochs`).
2. Adjust Slurm resources in `scripts/train_slurm.sh` (`--gres`, `--cpus-per-task`, `--mem`, `--time`) for your cluster.
3. Submit:

```bash
mkdir -p logs
sbatch scripts/train_slurm.sh
```

Default experiment config includes:

- `octmnist_classifier` (`octmnist`, 4 classes)
- `tissuemnist_classifier` (`tissuemnist`, 8 classes)
- `pathmnist_classifier` (`pathmnist`, 9 classes)

Each model is saved with its own name in `checkpoints/`, for example:

- `checkpoints/bloodmnist_classifier.pt`
- `checkpoints/tissuemnist_classifier.pt`
- `checkpoints/pathmnist_classifier.pt`

Each experiment also writes its own history JSON:

- `checkpoints/<model_name>_history.json`

MedMNIST data access behavior follows the official API style:

- Set `medmnist_download: true` to auto-download (`download=True`).
- Set `medmnist_download: false` to use already-downloaded files (`download=False`).

## 7) Resume manually

```bash
python train.py --config config.yaml --classes class_a,class_b,class_c --resume checkpoints/last_model.pt
```

## 8) Train 3 MedMNIST+ models locally (no Slurm)

```bash
python train_medmnist_plus.py --config medmnist_plus_experiments.yaml
```

## Notes

- The model is intentionally compact for a clean starting point.
- If your labels are imbalanced, a next step is adding per-class positive weights to `BCEWithLogitsLoss`.
