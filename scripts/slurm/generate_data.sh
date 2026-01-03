#!/bin/bash
#SBATCH --job-name=gen_routing_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/gen_data_%j.out
#SBATCH --error=logs/gen_data_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu

# Setup environment
source venv/bin/activate

# Use $SCRATCH for data if available
DATA_DIR=${SCRATCH:-.}/data
INPUT_DIR=${INPUT_DIR:-/path/to/nextpnr/designs}

# Create directories if they don't exist
mkdir -p $DATA_DIR
mkdir -p logs

# Generate routing data (CPU-bound, uses nextpnr)
python scripts/generate_routing_data.py \
    --input_dir $INPUT_DIR \
    --output_dir $DATA_DIR/routing_states \
    --num_samples 10000 \
    --nextpnr_path nextpnr-xilinx

echo "Data generation completed"

