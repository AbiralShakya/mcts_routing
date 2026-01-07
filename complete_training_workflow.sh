#!/bin/bash
# COMPLETE TRAINING WORKFLOW - All steps to get MCTS working

echo "🔄 MCTS ROUTING TRAINING WORKFLOW"
echo "=================================="
echo ""

# Step 1: Generate critic training data
echo "📊 STEP 1: Generate Critic Training Data"
echo "   Using trained diffusion model to create (partial_state, score) pairs"
echo "   Command: sbatch scripts/slurm/generate_critic_data.sh"
echo ""

# Step 2: Train critic model
echo "🏃 STEP 2: Train Critic Model"
echo "   Train time-aware critic using generated data"
echo "   Command: sbatch scripts/slurm/train_critic_only.sh"
echo ""

# Step 3: Run MCTS inference
echo "🎯 STEP 3: Run MCTS Inference"
echo "   Test the complete system with diffusion + critic"
echo "   Command: sbatch scripts/slurm/run_mcts_gpu.sh"
echo ""

echo "📋 PREREQUISITES:"
echo "   ✅ Diffusion model trained (checkpoints/checkpoint_epoch_200.pt)"
echo "   ⏳ Critic training data (generated in Step 1)"
echo "   ⏳ Critic model trained (trained in Step 2)"
echo "   ⏳ MCTS working (tested in Step 3)"
echo ""

echo "🚀 QUICK START (run these in order):"
echo "   1. sbatch scripts/slurm/generate_critic_data.sh"
echo "   2. sbatch scripts/slurm/train_critic_only.sh"
echo "   3. sbatch scripts/slurm/run_mcts_gpu.sh"
echo ""

echo "⚡ FAST TEST (run directly):"
echo "   python experiments/run_mcts.py --config configs/mcts/base.yaml --data_dir data/routing_states --num_samples 3 --num_iterations 50 --device cuda"


