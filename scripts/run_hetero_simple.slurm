#!/bin/bash
# "simple" run: planted positive control across modes (fallback), plus an OPTIONAL
# real backend smoke. Real backends need the TLC-GNN engine on PYTHONPATH.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=hpdg-simple
#SBATCH --output=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/hetero_pdg_lp
export PYTHONUNBUFFERED=1

echo "[INFO] host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} start=$(date)"
# fallback positive control (topology should help vs no_topology when planted)
for M in no_topology metapath_topology_concat metapath_topology_attention unified_filter_topology; do
  echo "=== $M (fallback, planted) ==="
  python -u -m hetero_pdg.train_hetero_lp --topo-mode "$M" --topo-backend fallback \
      --planted --epochs 60 --device cpu --output-dir "runs/simple_${M}"
done
# OPTIONAL real backend smoke (requires TLC-GNN repo on PYTHONPATH)
export PYTHONPATH=/mnt/data/users/junyoungpark/code/TLC-GNN:${PYTHONPATH:-}
echo "=== metapath_topology_concat (real_tlc, planted) ==="
python -u -m hetero_pdg.train_hetero_lp --topo-mode metapath_topology_concat \
    --topo-backend real_tlc --planted --epochs 30 --device cpu \
    --output-dir runs/simple_real_tlc || echo "[WARN] real_tlc smoke failed (engine?)"
echo "[INFO] done $(date)"
