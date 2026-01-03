#!/bin/bash
#SBATCH --job-name=route_critic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_critic_%j.out
#SBATCH --error=logs/train_critic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu

# Setup environment
source venv/bin/activate

# Set up distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Use $SCRATCH for data and checkpoints if available
DATA_DIR=${SCRATCH:-.}/data
CHECKPOINT_DIR=${SCRATCH:-.}/checkpoints

# Create directories if they don't exist
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

# Run critic training
srun python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --distributed \
    --seed 42

echo "Critic training job completed"

