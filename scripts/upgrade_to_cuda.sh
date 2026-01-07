#!/bin/bash
# Upgrade existing CPU PyTorch installation to CUDA version
# Run this after setup_adroit.sh if you want GPU support

set -e

echo "Upgrading PyTorch to CUDA version..."

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Please activate your virtual environment first:"
    echo "  source venv/bin/activate"
    exit 1
fi

# Uninstall CPU versions
echo "Uninstalling CPU PyTorch..."
pip uninstall torch torchvision torchaudio -y

# Install CUDA versions
echo "Installing CUDA PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo "Upgrade complete!"
echo ""
echo "Note: CUDA will only be available when running on GPU nodes via SLURM jobs."
echo "The login node doesn't have GPUs, but SLURM will allocate them for your jobs."
