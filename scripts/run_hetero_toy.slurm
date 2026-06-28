#!/bin/bash
# Lightweight toy run: all topology modes with the FALLBACK backend (CPU, self-contained).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=hpdg-toy
#SBATCH --output=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/hetero_pdg_lp
export PYTHONUNBUFFERED=1

echo "[INFO] host=$(hostname) start=$(date)"
for M in no_topology metapath_topology_concat metapath_topology_attention unified_filter_topology; do
  echo "=== $M (fallback) ==="
  python -u -m hetero_pdg.train_hetero_lp --dataset toy --target-rel paper,cites,paper \
      --topo-mode "$M" --topo-backend fallback --epochs 30 --device cpu \
      --output-dir "runs/toy_${M}"
done
echo "[INFO] done $(date)"
