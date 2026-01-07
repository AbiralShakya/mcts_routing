#!/bin/bash
# QUICK SCP - Copy all changes to della (minimal output)

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "Copying all changes to della..."

# Core fixes
scp src/critic/gnn.py src/mcts/search.py src/mcts/node.py src/diffusion/model.py src/critic/training.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/

# Experiments
scp experiments/run_mcts.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/

# Configs
scp configs/diffusion/ddim.yaml configs/diffusion/ddpm.yaml configs/mcts/base.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/

# SLURM
scp scripts/slurm/train_critic_only.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/

echo "✅ All files copied successfully!"
echo ""
echo "Ready to test:"
echo "  python experiments/run_mcts.py --config configs/mcts/base.yaml"
echo "  sbatch scripts/slurm/train_critic_only.sh"


