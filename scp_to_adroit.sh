#!/bin/bash
# Copy project files to Adroit cluster
# Usage: ./scp_to_adroit.sh

echo "Copying project to Adroit cluster..."

# Copy the entire project (excluding checkpoints, data, logs)
rsync -avz --exclude='checkpoints/' --exclude='data/' --exclude='logs/' --exclude='.git/' \
    /Users/abiralshakya/Documents/mcts_routing/ \
    as0714@adroit.princeton.edu:/scratch/network/as0714/mcts_routing/

echo "Project copied to Adroit!"
echo "Now SSH to adroit and run: bash scripts/setup_adroit.sh"
