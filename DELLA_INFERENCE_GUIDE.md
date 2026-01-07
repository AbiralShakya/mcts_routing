# Della Inference and Training Guide

## 1. Run Diffusion Inference (Standalone)

After training completes, you can run inference with just the diffusion model:

```bash
# SSH into della
ssh as0714@della.princeton.edu
cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing

# Set environment
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"
source venv/bin/activate  # or conda activate mcts_routing

# Run inference (you'll need a test netlist file)
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

**Note:** You'll need a netlist file. If you don't have one, you can create a simple test netlist or use synthetic data.

## 2. Train Critic Model

### Step 1: Generate Critic Training Data

```bash
# Generate critic training data from diffusion trajectories
python scripts/generate_critic_data.py \
    --model_path checkpoints/checkpoint_epoch_200.pt \
    --output_dir data/critic_data \
    --num_samples 1000 \
    --use_synthetic

# This will create: data/critic_data/critic_data.pkl
```

### Step 2: Train Critic Model

**Option A: Using SLURM (recommended)**
```bash
sbatch scripts/slurm/train_critic.sh
```

**Option B: Direct run**
```bash
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir data/critic_data \
    --checkpoint_dir checkpoints/critic \
    --seed 42
```

**Monitor training:**
```bash
tail -f logs/train_critic_*.out
```

**Expected output:** `checkpoints/critic/critic_epoch_*.pt` files

## 3. Run Full MCTS Inference (Diffusion + Critic)

After both models are trained:

```bash
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --critic_checkpoint checkpoints/critic/critic_epoch_100.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

## 4. Run with Shared Encoders

### Step 1: Copy Updated Files to Della

From your local machine:
```bash
# Copy updated training script with shared encoder support
scp experiments/train_routing.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/experiments/

# Copy updated critic training script
scp experiments/train_critic.py as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/experiments/

# Copy updated configs (if not already done)
scp configs/training/della_diffusion.yaml configs/training/della_critic.yaml as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/configs/training/
```

### Step 2: Train Diffusion with Shared Encoders

The config already has `shared_encoders.enabled: true`, so just run:

```bash
sbatch scripts/slurm/train_diffusion.sh
```

Or directly:
```bash
python experiments/train_routing.py \
    --config configs/training/della_diffusion.yaml \
    --data_dir data/routing_states \
    --checkpoint_dir checkpoints \
    --seed 42
```

The updated script will automatically create shared encoders.

### Step 3: Train Critic with Shared Encoders

**Update the critic config** to load shared encoders from diffusion checkpoint:

Edit `configs/training/della_critic.yaml`:
```yaml
shared_encoders:
  use_shared_encoders: true
  diffusion_checkpoint: checkpoints/checkpoint_epoch_200.pt  # Path to diffusion checkpoint
```

Then train:
```bash
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir data/critic_data \
    --checkpoint_dir checkpoints/critic \
    --seed 42
```

The critic will load the shared encoders from the diffusion checkpoint and use them.

### Step 4: Run MCTS with Shared Encoders

The MCTS inference script automatically uses shared encoders if they're in the models:

```bash
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --critic_checkpoint checkpoints/critic/critic_epoch_100.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

## Quick Reference Commands

### Check Training Status
```bash
# Check diffusion checkpoints
ls -lh checkpoints/checkpoint_epoch_*.pt

# Check critic checkpoints
ls -lh checkpoints/critic/critic_epoch_*.pt

# View latest training logs
tail -f logs/train_diffusion_*.out
tail -f logs/train_critic_*.out
```

### Verify Shared Encoders
```bash
# Check if shared encoders are being used (look for log message)
grep "shared encoder" logs/train_diffusion_*.out
grep "shared encoder" logs/train_critic_*.out
```

## Troubleshooting

### Issue: "No module named 'src.shared'"
**Solution:** Make sure you copied the `src/shared/` directory:
```bash
scp -r src/shared as0714@della.princeton.edu:/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing/src/
```

### Issue: Critic training fails to load shared encoders
**Solution:** Make sure:
1. Diffusion checkpoint exists: `checkpoints/checkpoint_epoch_200.pt`
2. Config has correct path: `diffusion_checkpoint: checkpoints/checkpoint_epoch_200.pt`
3. Updated `train_critic.py` is on della

### Issue: MCTS inference needs netlist file
**Solution:** Create a simple test netlist or use synthetic:
```bash
# Generate a test netlist (if you have a generator)
# Or use one of your training netlists
```

## Training Loss Analysis

Your final loss of **0.856** is actually **excellent**! 

- Started at: ~0.99 (with bad data) or ~0.98 (with good data)
- Final: 0.856
- **Improvement: ~0.12-0.14** (14% relative improvement)

This is much better than the first run that got stuck at 0.997. The model is learning meaningful patterns in the routing data.

The learning rate schedule (cosine annealing) worked well - it started at 1e-4 and decayed to near zero by epoch 200, which is expected behavior.

