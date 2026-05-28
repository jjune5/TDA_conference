#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=s1-chameleon
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/s1-chameleon-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/s1-chameleon-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TLCGNN_CORES=${SLURM_CPUS_PER_TASK:-32}

echo "[INFO] host=$(hostname) start=$(date) args=$*"

python -u noise_robust_exp.py \
    --datasets Chameleon \
    --ps 0 5 10 20 \
    --variants PI no-PI GDC-PI \
    --graph_seeds 3 \
    --init_per_graph 10 \
    --cores ${SLURM_CPUS_PER_TASK:-32} \
    --results_dir results/noise_robust_solid/Chameleon \
    --epochs 2000

echo "[INFO] done $(date)"
