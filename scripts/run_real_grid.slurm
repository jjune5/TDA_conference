#!/bin/bash
# Real-data LP grid cell: one (dataset, topo_mode, backend) over 3 seeds, on 1 GPU.
# Usage: sbatch scripts/run_real_grid.slurm <dataset> <topo_mode> <backend|none> <tag> [cap]
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/hetero_pdg_lp
export PYTHONPATH=/mnt/data/users/junyoungpark/code/TLC-GNN:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4

DS=$1; MODE=$2; BACKEND=$3; TAG=$4; CAP=${5:-1000}
EXTRA=""; [ "$BACKEND" != "none" ] && EXTRA="--topo-backend $BACKEND"
echo "[INFO] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} DS=$DS MODE=$MODE BACKEND=$BACKEND CAP=$CAP start=$(date)"
for s in 0 1 2; do
  echo "=== $DS / $TAG / seed $s ==="
  python -u -m hetero_pdg.train_hetero_lp --dataset "$DS" --topo-mode "$MODE" $EXTRA \
      --max-target-edges "$CAP" --epochs 100 --seed "$s" --device cuda \
      --output-dir "runs/real/${DS}_${TAG}_s${s}" 2>&1 | grep -E '"feature_kind"|"n_test"|"test_auc"|"test_ap"|Error|Traceback' || true
done
echo "[INFO] done $(date)"
