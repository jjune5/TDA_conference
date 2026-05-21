#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/data/users/junyoungpark/code/TLC-GNN/slurm_logs/%x-%j.err
source /mnt/data/users/junyoungpark/miniforge3/etc/profile.d/conda.sh
conda activate tlcgnn
cd /mnt/data/users/junyoungpark/code/TLC-GNN
python -m Knowledge_Distillation.prepare_data_LP_modern --name PubMed --max_edges 10000
