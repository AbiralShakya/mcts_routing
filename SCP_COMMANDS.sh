#!/bin/bash
# SCP commands to copy all modified and new files to della
# Run these commands one by one or copy-paste the entire block

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "Copying files to della..."

# 1. Modified core files
echo "1. Copying modified training file (max_pips fix)..."
scp src/training/routing_trainer.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/training/

echo "2. Copying modified MCTS search (None critic fix + 7 features)..."
scp src/mcts/search.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/

# 2. New experiment scripts
echo "3. Copying new experiment scripts..."
scp experiments/train_joint.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/
scp experiments/run_mcts.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/

# 3. New SLURM scripts
echo "4. Copying new SLURM scripts..."
scp scripts/slurm/train_joint.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/
scp scripts/slurm/run_mcts.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/

# 4. Helper scripts (if they exist)
if [ -f "scripts/create_test_netlist.py" ]; then
    echo "5. Copying helper scripts..."
    scp scripts/create_test_netlist.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/
    scp scripts/run_inference.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/
fi

echo ""
echo "✅ All files copied successfully!"
echo ""
echo "Next steps on della:"
echo "  1. Submit joint training: sbatch scripts/slurm/train_joint.sh"
echo "  2. Or run MCTS inference: sbatch scripts/slurm/run_mcts.sh"



