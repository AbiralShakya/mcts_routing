#!/bin/bash
# Setup for Princeton Della cluster

set -e

echo "Setting up Della cluster environment..."

# Load modules
module load anaconda3/2023.9
module load cuda/11.8
module load gcc/11.2.0

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .

# Setup data directories in $SCRATCH (faster than home)
if [ -n "$SCRATCH" ]; then
    mkdir -p $SCRATCH/checkpoints
    mkdir -p $SCRATCH/data
    mkdir -p $SCRATCH/logs
    echo "Created directories in $SCRATCH:"
    echo "  - $SCRATCH/checkpoints"
    echo "  - $SCRATCH/data"
    echo "  - $SCRATCH/logs"
else
    echo "Warning: $SCRATCH not set, using local directories"
    mkdir -p checkpoints
    mkdir -p data
    mkdir -p logs
fi

# Create logs directory locally too
mkdir -p logs

echo "Setup complete!"
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "To submit training job:"
echo "  sbatch scripts/slurm/train_diffusion.sh"

