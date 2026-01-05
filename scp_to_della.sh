#!/bin/bash

# Script to SCP all necessary files to della cluster
# Usage: ./scp_to_della.sh

DELLA_USER="as0714"
DELLA_HOST="della.princeton.edu"
DELLA_BASE="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing"

echo "Copying files to della..."

# 1. Source code - shared encoders (NEW)
echo "Copying shared encoder modules..."
scp -r src/shared ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/

# 2. Modified source files
echo "Copying modified diffusion model..."
scp src/diffusion/model.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/diffusion/

echo "Copying modified critic model..."
scp src/critic/gnn.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/

echo "Copying modified critic training..."
scp src/critic/training.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/critic/

echo "Copying modified MCTS search..."
scp src/mcts/search.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/src/mcts/

# 3. Training scripts
echo "Copying training scripts..."
scp experiments/train_routing.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/
scp experiments/train_critic.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/
scp experiments/run_mcts_inference.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/
scp experiments/full_pipeline.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/experiments/

# 4. Data generation scripts
echo "Copying data generation scripts..."
scp scripts/generate_routing_data.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/
scp scripts/generate_critic_data.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/
scp scripts/generate_synthetic_diffusion_data.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/

# 5. Config files
echo "Copying config files..."
scp configs/training/della_diffusion.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/training/
scp configs/training/della_critic.yaml ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/configs/training/

# 6. SLURM scripts
echo "Copying SLURM scripts..."
scp scripts/slurm/train_diffusion.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/
scp scripts/slurm/train_critic.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_BASE}/scripts/slurm/

echo ""
echo "All files copied successfully!"
echo ""
echo "Next steps on della:"
echo "1. Generate training data: python scripts/generate_synthetic_diffusion_data.py --output_dir data/routing_states --num_samples 2000"
echo "2. Submit diffusion training: sbatch scripts/slurm/train_diffusion.sh"
echo "3. After training, generate critic data: python scripts/generate_critic_data.py --model_path checkpoints/checkpoint_epoch_200.pt --output_dir data/critic_data --num_samples 1000 --use_synthetic"
echo "4. Submit critic training: sbatch scripts/slurm/train_critic.sh"
