#!/bin/bash
# COMPREHENSIVE SCP - Copy ALL necessary files to della after git pull
# This includes all modified/new files from the latest changes

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "🔄 Starting comprehensive SCP to della..."
echo "Files include: critical MCTS fixes (time-aware critic + K-branch expansion)"

# 1. Modified core models (critical fixes)
echo "1. Copying modified diffusion model (DDIM support + shared encoders)..."
scp src/diffusion/model.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/diffusion/

echo "2. Copying modified critic (TIME-AWARE conditioning)..."
scp src/critic/gnn.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/

echo "3. Copying modified critic training..."
scp src/critic/training.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/

echo "4. Copying modified MCTS search (K-branch expansion)..."
scp src/mcts/search.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/

echo "5. Copying modified MCTS node (time-aware Q)..."
scp src/mcts/node.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/

# 6. Modified experiments
echo "6. Copying modified MCTS runner (DDIM support)..."
scp experiments/run_mcts.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/

# 7. New configs
echo "7. Copying new diffusion configs..."
scp configs/diffusion/ddim.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/diffusion/
scp configs/diffusion/ddpm.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/diffusion/

echo "8. Copying modified MCTS config..."
scp configs/mcts/base.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/mcts/

# 8. SLURM scripts (if any new ones)
echo "9. Copying new SLURM scripts..."
scp scripts/slurm/train_critic_only.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/

echo ""
echo "✅ COMPREHENSIVE SCP COMPLETE!"
echo ""
echo "📋 Files copied:"
echo "   🔧 Core fixes:"
echo "     - Time-aware critic (src/critic/gnn.py)"
echo "     - K-branch MCTS expansion (src/mcts/search.py)"
echo "     - DDIM support in diffusion (src/diffusion/model.py)"
echo "     - Time-aware Q-values (src/mcts/node.py)"
echo ""
echo "   📊 New configs:"
echo "     - DDIM config (configs/diffusion/ddim.yaml)"
echo "     - DDPM config (configs/diffusion/ddpm.yaml)"
echo "     - Updated MCTS config (configs/mcts/base.yaml)"
echo ""
echo "🚀 Next steps on della:"
echo "   1. Test MCTS: python experiments/run_mcts.py --config configs/mcts/base.yaml"
echo "   2. Train critic: sbatch scripts/slurm/train_critic_only.sh"
echo "   3. Full pipeline: python experiments/full_pipeline.py"
echo ""
echo "🎯 Key improvements:"
echo "   - Critic now understands timestep context (no more aggressive pruning)"
echo "   - MCTS expands 4 branches per node (true exploration vs rejection sampling)"
echo "   - DDIM support for 20x faster inference"
echo "   - Time-aware Q-values for better UCB balancing"


