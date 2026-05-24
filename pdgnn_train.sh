#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.train_pdgnn_lp \
  --data ./data/PDGNN/PubMed_LP_hop2_n10000_train.pkl \
  --out ./data/PDGNN/checkpoints/pdgnn_lp.pt \
  --hidden 32 --layers 3 --epochs 50 --lr 1e-3
