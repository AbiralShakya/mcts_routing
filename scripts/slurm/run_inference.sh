#!/bin/bash
#SBATCH --job-name=route_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# Project root
PROJECT_ROOT="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"
cd $PROJECT_ROOT

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Setup environment
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "Activated venv"
elif [ -f "$HOME/.local/bin/activate" ]; then
    source "$HOME/.local/bin/activate"
    echo "Activated home venv"
elif command -v conda &> /dev/null && conda info --envs | grep -q mcts_routing; then
    conda activate mcts_routing
    echo "Activated conda environment"
else
    echo "Warning: No virtual environment found, using system Python"
fi

# Paths
CHECKPOINT="$PROJECT_ROOT/checkpoints/checkpoint_epoch_160.pt"
DATA_DIR="$PROJECT_ROOT/data/routing_states"

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Available checkpoints:"
    ls -la $PROJECT_ROOT/checkpoints/
    exit 1
fi

mkdir -p logs

echo "Running inference..."
echo "Checkpoint: $CHECKPOINT"
echo "Data dir: $DATA_DIR"

python scripts/run_inference.py \
    --checkpoint $CHECKPOINT \
    --data_dir $DATA_DIR \
    --num_samples 5 \
    --num_steps 100 \
    --seed 42

echo "Inference job completed"
