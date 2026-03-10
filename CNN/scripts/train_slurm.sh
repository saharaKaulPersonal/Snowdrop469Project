#!/bin/bash
#SBATCH --job-name=medmnist-plus-3x
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

mkdir -p logs checkpoints

module purge
# module load python/3.10 cuda/12.1

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python train_medmnist_plus.py --config medmnist_plus_experiments.yaml
