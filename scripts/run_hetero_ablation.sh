#!/bin/bash
# Phase-3 ablation on the toy graph. Toy metrics are SANITY CHECKS ONLY -- not
# performance claims. Pass "--planted" as $1 to use the positive control.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=hpdg-ablation
#SBATCH --output=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/hetero_pdg_lp/slurm_logs/%x-%j.err

set -euo pipefail
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/hetero_pdg_lp
export PYTHONUNBUFFERED=1

PLANTED="${1:-}"                  # pass --planted for the positive control
C=(--epochs 60 --seed 0 --device cpu)
[ -n "$PLANTED" ] && C+=("$PLANTED")

run(){ python -u -m hetero_pdg.train_hetero_lp "$@" "${C[@]}"; }

run --topo-mode no_topology                                                 --output-dir runs/abl_no_topology
run --topo-mode metapath_topology_attention --topo-backend fallback         --output-dir runs/abl_metapath_attention
run --topo-mode collapsed_topology --filtration-mode pair_conditioned        --output-dir runs/abl_pair_conditioned
run --topo-mode metapath_topology_concat --edge-filtration-mode relation_delay --output-dir runs/abl_relation_delay
run --topo-mode typed_cycle_topology                                         --output-dir runs/abl_typed_cycle
run --topo-mode multi_slice_topology --num-slices 3                          --output-dir runs/abl_multi_slice
echo "[done] ablation metrics under runs/abl_*/metrics.json"
