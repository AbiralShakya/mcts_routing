#!/bin/bash
#SBATCH --job-name=generate_critic_data
#SBATCH --output=logs/generate_critic_data_%j.out
#SBATCH --error=logs/generate_critic_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=as0714@princeton.edu

# SLURM script for generating critic training data using trained diffusion model
# Usage: sbatch scripts/slurm/generate_critic_data.sh

set -e

echo "Starting critic data generation job on della"
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

# Create directories
mkdir -p logs
mkdir -p data/critic_data

# Check if diffusion model exists
if [ ! -f "checkpoints/checkpoint_epoch_200.pt" ]; then
    echo "ERROR: No diffusion checkpoint found at checkpoints/checkpoint_epoch_200.pt"
    echo "Please train the diffusion model first."
    exit 1
fi

echo "Generating critic training data..."
echo "Diffusion model: checkpoints/checkpoint_epoch_200.pt"
echo "Output directory: data/critic_data"
echo "Number of samples: 1000"
echo "Using synthetic routing (no nextpnr)"

# Generate critic training data
python scripts/generate_critic_data.py \
    --model_path checkpoints/checkpoint_epoch_200.pt \
    --output_dir data/critic_data \
    --num_samples 1000 \
    --use_synthetic \
    --device cuda

echo "Critic data generation completed successfully!"
echo "Generated training data in: data/critic_data"
echo "Number of files: $(ls -1 data/critic_data/*.pkl 2>/dev/null | wc -l)"
echo "End time: $(date)"
echo ""
echo "Next step: Train critic model with:"
echo "  sbatch scripts/slurm/train_critic_only.sh"


