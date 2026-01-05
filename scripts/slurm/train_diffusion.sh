#!/bin/bash
#SBATCH --job-name=route_diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_diffusion_%j.out
#SBATCH --error=logs/train_diffusion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=as0714@princeton.edu

# Project root
PROJECT_ROOT="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"
cd $PROJECT_ROOT

# Set PYTHONPATH so 'src' module can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Setup environment - try multiple options
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

# Data directories
DATA_DIR="$PROJECT_ROOT/data/routing_states"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

# Verify data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    exit 1
fi

PKL_COUNT=$(ls -1 "$DATA_DIR"/*.pkl 2>/dev/null | wc -l)
if [ "$PKL_COUNT" -eq 0 ]; then
    echo "ERROR: No .pkl files found in $DATA_DIR"
    exit 1
fi
echo "Found $PKL_COUNT routing samples in $DATA_DIR"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# Run single-GPU training (no distributed)
python experiments/train_routing.py \
    --config configs/training/della_diffusion.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --seed 42

echo "Training job completed"
