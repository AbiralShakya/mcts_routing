#!/bin/bash

# Single SCP command using tar for maximum efficiency
# This creates a tarball and transfers it in one go

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "Creating tarball and copying to della..."

# Create tarball with all necessary files
tar -czf /tmp/mcts_routing_update.tar.gz \
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
    configs/training/della_critic.yaml

# Copy tarball to della
scp /tmp/mcts_routing_update.tar.gz ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/

# Extract on della
ssh ${DELLA_USER}@${DELLA_HOST} "cd ${DELLA_BASE} && tar -xzf mcts_routing_update.tar.gz && rm mcts_routing_update.tar.gz"

# Clean up local tarball
rm /tmp/mcts_routing_update.tar.gz

echo ""
echo "All files copied and extracted on della!"
echo ""
echo "Files updated:"
echo "  - src/shared/ (shared encoders)"
echo "  - src/mcts/search.py (None critic fix + 7 features)"
echo "  - src/diffusion/model.py (shared encoder support)"
echo "  - src/critic/ (shared encoder support)"
echo "  - experiments/ (training scripts)"
echo "  - scripts/ (data generation + inference)"
echo "  - configs/ (training configs)"

