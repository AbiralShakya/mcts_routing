#!/bin/bash
#SBATCH --job-name=train_critic_only
#SBATCH --output=logs/train_critic_only_%j.out
#SBATCH --error=logs/train_critic_only_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

# SLURM script for training only the critic model
# Usage: sbatch scripts/slurm/train_critic_only.sh

set -e

echo "Starting critic-only training job on della"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules (if needed)
module load anaconda3

# Activate conda environment
source activate mcts_routing

# Set environment variables
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"
cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing

# Check if we have critic training data
if [ ! -d "data/critic_data" ] || [ -z "$(ls -A data/critic_data 2>/dev/null)" ]; then
    echo "ERROR: No critic training data found in data/critic_data/"
    echo "Please run: python scripts/generate_critic_data.py --model_path checkpoints/checkpoint_epoch_200.pt --output_dir data/critic_data --num_samples 1000 --use_synthetic"
    exit 1
fi

# Check if we have a diffusion checkpoint for shared encoders
if [ ! -f "checkpoints/checkpoint_epoch_200.pt" ]; then
    echo "WARNING: No diffusion checkpoint found at checkpoints/checkpoint_epoch_200.pt"
    echo "The critic will train with random shared encoders (not recommended)"
    echo "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

echo "Training data directory: data/routing_states"
echo "Critic data directory: data/critic_data"
echo "Checkpoint directory: checkpoints"
echo "Device: cuda"

# Run critic-only training
python experiments/train_joint.py \
      --config configs/training/della_critic.yaml \
      --data_dir data/routing_states \
      --checkpoint_dir checkpoints \
      --phase critic \
      --device cuda

echo "Critic training completed successfully!"
echo "End time: $(date)"
