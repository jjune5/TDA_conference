#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TLCGNN_CORES=${SLURM_CPUS_PER_TASK:-32}

echo "[INFO] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} args=$*"
nvidia-smi -L || true
python -u pipelines.py "$@"
echo "[INFO] done"
