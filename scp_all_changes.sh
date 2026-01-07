#!/bin/bash
# SCP ALL CHANGES - Comprehensive script to copy all modified/new files to della after git pull
# This includes the critical MCTS architecture fixes (time-aware critic + K-branch expansion)

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "🚀 Starting comprehensive SCP of ALL changes to della..."
echo "This includes critical MCTS fixes: time-aware critic + K-branch expansion"
echo ""

# Function to check if scp succeeded
check_scp() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ FAILED: $1"
        exit 1
    fi
}

# 1. Core architecture fixes (CRITICAL)
echo "🔧 Copying CORE ARCHITECTURE FIXES:"
echo "   - Time-aware critic (prevents aggressive pruning)"
scp src/critic/gnn.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/
check_scp "src/critic/gnn.py"

echo "   - K-branch MCTS expansion (true exploration vs rejection sampling)"
scp src/mcts/search.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/
check_scp "src/mcts/search.py"

echo "   - Time-aware Q-values for UCB balancing"
scp src/mcts/node.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/
check_scp "src/mcts/node.py"

echo "   - DDIM support + stochastic denoising"
scp src/diffusion/model.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/diffusion/
check_scp "src/diffusion/model.py"

# 2. Training updates
echo ""
echo "🏃 Copying TRAINING UPDATES:"
echo "   - Time-aware critic training"
scp src/critic/training.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/
check_scp "src/critic/training.py"

# 3. Experiment updates
echo ""
echo "🧪 Copying EXPERIMENT UPDATES:"
echo "   - DDIM integration + time-aware critic calls"
scp experiments/run_mcts.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/
check_scp "experiments/run_mcts.py"

# 4. New configs
echo ""
echo "⚙️  Copying NEW/MODIFIED CONFIGS:"
echo "   - DDIM config (20x faster inference)"
scp configs/diffusion/ddim.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/diffusion/
check_scp "configs/diffusion/ddim.yaml"

echo "   - DDPM config (standard diffusion)"
scp configs/diffusion/ddpm.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/diffusion/
check_scp "configs/diffusion/ddpm.yaml"

echo "   - Updated MCTS config (K=4 branches)"
scp configs/mcts/base.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/mcts/
check_scp "configs/mcts/base.yaml"

# 5. New SLURM scripts
echo ""
echo "📊 Copying NEW SLURM SCRIPTS:"
echo "   - Critic-only training script"
scp scripts/slurm/train_critic_only.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/
check_scp "scripts/slurm/train_critic_only.sh"

echo ""
echo "🎉 ALL FILES SUCCESSFULLY COPIED TO DELLA!"
echo ""
echo "📋 SUMMARY OF CHANGES:"
echo "   🔧 CRITICAL FIXES:"
echo "     ✓ Time-aware critic (no more aggressive pruning)"
echo "     ✓ K-branch MCTS expansion (true exploration)"
echo "     ✓ Time-aware Q-values (better UCB balancing)"
echo "     ✓ DDIM support (20x faster inference)"
echo ""
echo "   📊 NEW FEATURES:"
echo "     ✓ DDIM/DDPM configs for different inference speeds"
echo "     ✓ Updated MCTS config with K=4 branches"
echo "     ✓ Critic-only training script"
echo ""
echo "🚀 READY TO TEST ON DELLA:"
echo "   1. Test MCTS: python experiments/run_mcts.py --config configs/mcts/base.yaml"
echo "   2. Train critic: sbatch scripts/slurm/train_critic_only.sh"
echo "   3. Full pipeline: python experiments/full_pipeline.py"
echo ""
echo "🎯 IMPACT:"
echo "   - MCTS now explores multiple branches per node"
echo "   - Critic understands timestep context"
echo "   - No more infinite recursion from pruning everything"
echo "   - Practical inference speeds with DDIM"


