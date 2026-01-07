#!/bin/bash
#SBATCH --job-name=route_mcts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/mcts_%j.out
#SBATCH --error=logs/mcts_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=as0714@princeton.edu

# Project root
PROJECT_ROOT="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"
cd $PROJECT_ROOT

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Setup environment
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated venv"
elif command -v conda &> /dev/null && conda info --envs | grep -q mcts_routing; then
    conda activate mcts_routing
    echo "Activated conda environment"
else
    echo "Warning: No virtual environment found, using system Python"
fi

# Paths
DATA_DIR="$PROJECT_ROOT/data/routing_states"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
JOINT_CHECKPOINT="$CHECKPOINT_DIR/joint_model.pt"

mkdir -p logs

echo "========================================"
echo "MCTS Routing Inference"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Joint checkpoint: $JOINT_CHECKPOINT"

# Check checkpoint exists
if [ ! -f "$JOINT_CHECKPOINT" ]; then
    echo "WARNING: Joint checkpoint not found at $JOINT_CHECKPOINT"
    echo "Looking for separate checkpoints..."

    DIFFUSION_CKPT=$(ls -t $CHECKPOINT_DIR/checkpoint_epoch_*.pt 2>/dev/null | head -1)
    CRITIC_CKPT=$(ls -t $CHECKPOINT_DIR/critic/critic_epoch_*.pt 2>/dev/null | head -1)

    if [ -n "$DIFFUSION_CKPT" ] && [ -n "$CRITIC_CKPT" ]; then
        echo "Found: $DIFFUSION_CKPT"
        echo "Found: $CRITIC_CKPT"

        python experiments/run_mcts.py \
            --diffusion_checkpoint $DIFFUSION_CKPT \
            --critic_checkpoint $CRITIC_CKPT \
            --config configs/training/della_diffusion.yaml \
            --data_dir $DATA_DIR \
            --num_samples 10 \
            --num_iterations 500 \
            --ucb_c 1.41 \
            --critic_threshold 0.3 \
            --seed 42
    else
        echo "ERROR: No checkpoints found. Train models first."
        echo "Run: sbatch scripts/slurm/train_joint.sh"
        exit 1
    fi
else
    echo "Using joint checkpoint"
    python experiments/run_mcts.py \
        --joint_checkpoint $JOINT_CHECKPOINT \
        --data_dir $DATA_DIR \
        --num_samples 10 \
        --num_iterations 500 \
        --ucb_c 1.41 \
        --critic_threshold 0.3 \
        --seed 42
fi

echo ""
echo "========================================"
echo "MCTS inference completed!"
echo "========================================"
