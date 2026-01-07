#!/bin/bash
# Command to run MCTS inference on della

cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing

# Run MCTS with basic parameters
python experiments/run_mcts.py \
    --config configs/mcts/base.yaml \
    --data_dir data/routing_states \
    --num_samples 5 \
    --num_iterations 100 \
    --device cuda


