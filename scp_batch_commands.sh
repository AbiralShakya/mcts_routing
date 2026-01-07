#!/bin/bash
# Batch SCP commands for all the files

echo "Copying files to della..."

# Individual commands (copy-paste these):
scp src/diffusion/model.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/src/diffusion/

scp src/critic/gnn.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/src/critic/

scp experiments/train_joint.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/experiments/

scp experiments/run_mcts.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/experiments/

# Optional visualization scripts
scp scripts/visualize_single_sample.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/scripts/
scp scripts/visualize_routing_data.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/scripts/

echo "All files copied!"



