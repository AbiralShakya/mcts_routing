#!/bin/bash

# Single SCP command to copy all necessary files to della
# Usage: ./scp_all_to_della.sh

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "Copying all files to della in one command..."

# Use rsync for efficient transfer (or tar+scp)
# Option 1: Using rsync (more efficient, preserves structure)
rsync -avz --progress \
    src/shared/ \
    src/mcts/search.py \
    src/diffusion/model.py \
    src/critic/gnn.py \
    src/critic/training.py \
    experiments/train_routing.py \
    experiments/train_critic.py \
    experiments/run_mcts_inference.py \
    experiments/full_pipeline.py \
    scripts/generate_routing_data.py \
    scripts/generate_critic_data.py \
    scripts/generate_synthetic_diffusion_data.py \
    scripts/create_test_netlist.py \
    scripts/run_inference.sh \
    scripts/slurm/train_diffusion.sh \
    scripts/slurm/train_critic.sh \
    configs/training/della_diffusion.yaml \
    configs/training/della_critic.yaml \
    ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/

echo ""
echo "Files copied successfully!"
echo ""
echo "Note: Files are copied to the project root. The directory structure is preserved."
echo "Next steps on della:"
echo "  1. Generate test netlist: python scripts/create_test_netlist.py --output data/test_netlist.json"
echo "  2. Generate critic data: python scripts/generate_critic_data.py --model_path checkpoints/checkpoint_epoch_200.pt --output_dir data/critic_data --num_samples 1000 --use_synthetic"
echo "  3. Train critic: sbatch scripts/slurm/train_critic.sh"
echo "  4. Run inference: python experiments/run_mcts_inference.py --config configs/mcts/base.yaml --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt --netlist_file data/test_netlist.json --output_dir results"

