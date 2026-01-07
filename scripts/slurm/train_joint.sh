#!/bin/bash
#SBATCH --job-name=route_joint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_joint_%j.out
#SBATCH --error=logs/train_joint_%j.err
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
CONFIG="$PROJECT_ROOT/configs/training/della_diffusion.yaml"

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p $CHECKPOINT_DIR/critic
mkdir -p logs

echo "========================================"
echo "Joint Training Pipeline"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Config: $CONFIG"
echo "PYTHONPATH: $PYTHONPATH"

# Check data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found at $DATA_DIR"
    echo "Generating synthetic data first..."
    python scripts/generate_synthetic_diffusion_data.py \
        --output_dir $DATA_DIR \
        --num_samples 2000 \
        --grid_size 20 \
        --seed 42
fi

PKL_COUNT=$(ls -1 "$DATA_DIR"/*.pkl 2>/dev/null | wc -l)
echo "Found $PKL_COUNT routing samples"

# Run joint training (diffusion first, then critic)
echo ""
echo "Starting joint training..."
python experiments/train_joint.py \
    --config $CONFIG \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --phase all \
    --critic_samples 10000 \
    --seed 42 \
    --device cuda

echo ""
echo "========================================"
echo "Joint training completed!"
echo "========================================"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "Joint model: $CHECKPOINT_DIR/joint_model.pt"
