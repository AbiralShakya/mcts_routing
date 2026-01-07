#!/bin/bash
#SBATCH --job-name=run_mcts_gpu
#SBATCH --output=logs/run_mcts_gpu_%j.out
#SBATCH --error=logs/run_mcts_gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=as0714@princeton.edu

# SBATCH script for running MCTS inference with GPU
# Usage: sbatch scripts/slurm/run_mcts_gpu.sh

set -e

echo "Starting MCTS inference job on della"
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

# Create logs directory
mkdir -p logs

echo "Running MCTS inference with GPU..."
echo "Config: configs/mcts/base.yaml"
echo "Data dir: data/routing_states"
echo "Device: cuda"

# Run MCTS inference
python experiments/run_mcts.py \
    --config configs/mcts/base.yaml \
    --data_dir data/routing_states \
    --num_samples 10 \
    --num_iterations 500 \
    --device cuda

echo "MCTS inference completed successfully!"
echo "End time: $(date)"
echo ""
echo "Check output files for results:"
echo "  - logs/run_mcts_gpu_${SLURM_JOB_ID}.out"
echo "  - Any generated result files"


