#!/bin/bash
# SINGLE COMMAND - Copy everything in one go using tar piped through SSH
# This is the most efficient single command approach

tar -czf - \
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
    2>/dev/null | ssh as0714@della.princeton.edu "cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing && tar -xzf -" && echo "✅ All files copied successfully!"

