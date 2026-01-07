#!/bin/bash
# Setup for Princeton Adroit cluster - Simplified version

set -e

echo "Setting up Adroit cluster environment..."

# Try to load basic modules (if available)
echo "Loading modules..."

# Try to load CUDA if available (optional) - temporarily disable exit on error
set +e
echo "Trying to load CUDA module..."
module load cuda 2>/dev/null
CUDA_LOADED=$?
set -e

if [ $CUDA_LOADED -eq 0 ]; then
    echo "Loaded CUDA module"
else
    echo "CUDA module not available - will install CPU PyTorch"
fi

# Try to load GCC if available (optional)
set +e
echo "Trying to load GCC module..."
module load gcc 2>/dev/null
GCC_LOADED=$?
set -e

if [ $GCC_LOADED -eq 0 ]; then
    echo "Loaded GCC module"
else
    echo "GCC module not available - using system compiler"
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please load a Python module manually or install Python."
    echo "Try: module load python"
    exit 1
fi

echo "Using Python: $(which python3)"
python3 --version

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Activated virtual environment: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch - use CUDA version for Adroit (GPUs available via SLURM)
echo "Installing PyTorch..."

# Adroit has GPUs available via SLURM, so install CUDA version
echo "Installing CUDA PyTorch (GPUs available via SLURM jobs)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Note: CUDA will only be available when running on GPU nodes via SLURM

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install project in editable mode
pip install -e .

# Setup data directories in user's scratch space
SCRATCH_DIR="/scratch/network/as0714"
if [ -d "$SCRATCH_DIR" ]; then
    mkdir -p $SCRATCH_DIR/mcts_routing/checkpoints
    mkdir -p $SCRATCH_DIR/mcts_routing/data
    mkdir -p $SCRATCH_DIR/mcts_routing/logs
    echo "Created directories in $SCRATCH_DIR/mcts_routing:"
    echo "  - $SCRATCH_DIR/mcts_routing/checkpoints"
    echo "  - $SCRATCH_DIR/mcts_routing/data"
    echo "  - $SCRATCH_DIR/mcts_routing/logs"
else
    echo "Warning: Scratch directory not found, using local directories"
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
echo "  sbatch scripts/slurm/train_diffusion_adroit.sh"
echo ""
echo "To run training directly:"
echo "  source venv/bin/activate && python experiments/train_routing.py --config configs/training/della_diffusion.yaml --data_dir data/routing_states --checkpoint_dir checkpoints --seed 42"

