#!/bin/bash
#SBATCH --job-name=route_critic_adroit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/train_critic_%j.out
#SBATCH --error=logs/train_critic_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=as0714@princeton.edu

# Project root (Adroit uses /scratch/network/$USER)
PROJECT_ROOT="/scratch/network/as0714/mcts_routing"
cd $PROJECT_ROOT

# Set PYTHONPATH so 'src' module can be found
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"
echo "Activated virtual environment: $(which python)"

# Data directories
DATA_DIR="$PROJECT_ROOT/data"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/critic"

# Verify critic data exists
if [ -f "$DATA_DIR/adversarial_critic_data.pkl" ]; then
    echo "Found adversarial_critic_data.pkl"
elif [ -f "$DATA_DIR/critic_data.pkl" ]; then
    echo "Found critic_data.pkl"
else
    echo "ERROR: No critic data found in $DATA_DIR"
    exit 1
fi

# Create directories
mkdir -p $CHECKPOINT_DIR
mkdir -p logs

echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# Run single-GPU training (no distributed)
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --seed 42

echo "Critic training job completed"
