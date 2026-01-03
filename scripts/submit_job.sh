#!/bin/bash
# SLURM job submission script

#SBATCH --job-name=mcts_routing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Activate environment
source venv/bin/activate

# Run training
python experiments/train.py --config configs/training/h100_cluster.yaml

