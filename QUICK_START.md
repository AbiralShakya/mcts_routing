# Quick Start Guide - Della

## Your Training Results

**Excellent progress!** Your model achieved:
- **Final Loss: 0.856** (down from ~0.99)
- **Improvement: ~14%** - This is great learning!

## 1. Run Diffusion Inference (Standalone)

```bash
ssh as0714@della.princeton.edu
cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"

# Simple inference (diffusion only, no critic)
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

**Or use the helper script:**
```bash
./scripts/run_inference.sh --no-critic --netlist data/test_netlist.json
```

## 2. Train Critic Model

### Step 1: Generate Critic Data
```bash
python scripts/generate_critic_data.py \
    --model_path checkpoints/checkpoint_epoch_200.pt \
    --output_dir data/critic_data \
    --num_samples 1000 \
    --use_synthetic
```

### Step 2: Train Critic
```bash
# Using SLURM
sbatch scripts/slurm/train_critic.sh

# Or directly
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir data/critic_data \
    --checkpoint_dir checkpoints/critic \
    --seed 42
```

### Step 3: Monitor
```bash
tail -f logs/train_critic_*.out
```

## 3. Run Full MCTS (Diffusion + Critic)

After critic training completes:
```bash
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --critic_checkpoint checkpoints/critic/critic_epoch_100.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

**Or use helper script:**
```bash
./scripts/run_inference.sh \
    --diffusion checkpoints/checkpoint_epoch_200.pt \
    --critic checkpoints/critic/critic_epoch_100.pt \
    --netlist data/test_netlist.json
```

## 4. Run with Shared Encoders

### Copy Updated Files First:
```bash
# From your local machine
scp experiments/train_routing.py experiments/train_critic.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/experiments/
```

### Train Diffusion with Shared Encoders:
```bash
# Config already has shared_encoders.enabled: true
sbatch scripts/slurm/train_diffusion.sh
```

### Train Critic with Shared Encoders:
Edit `configs/training/della_critic.yaml`:
```yaml
shared_encoders:
  use_shared_encoders: true
  diffusion_checkpoint: checkpoints/checkpoint_epoch_200.pt
```

Then train:
```bash
sbatch scripts/slurm/train_critic.sh
```

The critic will automatically load shared encoders from the diffusion checkpoint.

## File Checklist

Make sure these are on della:
- [x] `checkpoints/checkpoint_epoch_200.pt` (your trained model)
- [ ] `src/shared/` directory (for shared encoders)
- [ ] Updated `experiments/train_routing.py` (for shared encoder support)
- [ ] Updated `experiments/train_critic.py` (for shared encoder support)
- [ ] `scripts/generate_critic_data.py` (for generating critic data)
- [ ] `experiments/run_mcts_inference.py` (for inference)

## Quick Commands Reference

```bash
# Check checkpoints
ls -lh checkpoints/checkpoint_epoch_*.pt
ls -lh checkpoints/critic/critic_epoch_*.pt

# View logs
tail -f logs/train_diffusion_*.out
tail -f logs/train_critic_*.out

# Check if shared encoders are used
grep -i "shared" logs/train_*.out
```

