#!/bin/bash
#SBATCH --job-name=route_diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_diffusion_%j.out
#SBATCH --error=logs/train_diffusion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu

# Setup environment
source venv/bin/activate

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Use $SCRATCH for data and checkpoints if available
DATA_DIR=${SCRATCH:-.}/data
CHECKPOINT_DIR=${SCRATCH:-.}/checkpoints

# Create directories if they don't exist
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

# Run training with srun for proper multi-GPU setup
srun python experiments/train_routing.py \
    --config configs/training/della_diffusion.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --distributed \
    --seed 42

echo "Training job completed"

