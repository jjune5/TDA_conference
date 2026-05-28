#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=s1-cornell
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/s1-cornell-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/s1-cornell-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TLCGNN_CORES=${SLURM_CPUS_PER_TASK:-16}

echo "[INFO] host=$(hostname) start=$(date)"

python -u noise_robust_exp.py \
    --datasets Cornell \
    --ps 0 5 10 20 \
    --variants PI no-PI GDC-PI \
    --graph_seeds 3 \
    --init_per_graph 10 \
    --cores ${SLURM_CPUS_PER_TASK:-16} \
    --results_dir results/noise_robust_solid/Cornell \
    --epochs 2000

echo "[INFO] done $(date)"
