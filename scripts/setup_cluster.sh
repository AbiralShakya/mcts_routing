#!/bin/bash
# Cluster setup script for H100s at Princeton

set -e

echo "Setting up cluster environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "Setup complete!"

