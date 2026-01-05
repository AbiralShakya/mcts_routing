# Della Cluster Run Guide

## Files Already Copied

The following files have been copied to della:
- `src/shared/` - Shared encoder modules
- `src/diffusion/model.py` - Updated diffusion model
- `src/critic/gnn.py` - Updated critic model  
- `src/critic/training.py` - Updated critic training
- `src/mcts/search.py` - Updated MCTS search
- `experiments/train_routing.py` - Diffusion training script
- `experiments/train_critic.py` - Critic training script
- `experiments/run_mcts_inference.py` - MCTS inference script
- `scripts/generate_*.py` - Data generation scripts
- `configs/training/della_*.yaml` - Training configs
- `scripts/slurm/train_*.sh` - SLURM submission scripts

## Step-by-Step Workflow on Della

### 1. SSH into Della

```bash
ssh as0714@della.princeton.edu
cd /scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing
```

### 2. Set Up Environment (if not already done)

```bash
# Activate your Python environment
# Option 1: Virtual environment
source venv/bin/activate

# Option 2: Conda
conda activate mcts_routing

# Set PYTHONPATH
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"
```

### 3. Generate Diffusion Training Data

```bash
# Create data directory
mkdir -p data/routing_states

# Generate synthetic training data with meaningful latents
python scripts/generate_synthetic_diffusion_data.py \
    --output_dir data/routing_states \
    --num_samples 2000 \
    --grid_size 20 \
    --min_nets 5 \
    --max_nets 20 \
    --seed 42
```

**Expected output:** `data/routing_states/sample_*.pkl` files

### 4. Train Diffusion Model

```bash
# Submit SLURM job
sbatch scripts/slurm/train_diffusion.sh

# Or run directly (if on GPU node)
python experiments/train_routing.py \
    --config configs/training/della_diffusion.yaml \
    --data_dir data/routing_states \
    --checkpoint_dir checkpoints \
    --seed 42
```

**Monitor job:**
```bash
# Check job status
squeue -u as0714

# View logs
tail -f logs/train_diffusion_*.out
tail -f logs/train_diffusion_*.err
```

**Expected output:** `checkpoints/checkpoint_epoch_*.pt` files (saved every 10 epochs)

### 5. Generate Critic Training Data

After diffusion training completes (or use an existing checkpoint):

```bash
# Create critic data directory
mkdir -p data/critic_data

# Generate critic training data from diffusion trajectories
python scripts/generate_critic_data.py \
    --model_path checkpoints/checkpoint_epoch_200.pt \
    --output_dir data/critic_data \
    --num_samples 1000 \
    --use_synthetic

# Or use real netlists (if available)
# python scripts/generate_critic_data.py \
#     --model_path checkpoints/checkpoint_epoch_200.pt \
#     --netlist_dir data/netlists \
#     --output_dir data/critic_data \
#     --num_samples 1000
```

**Expected output:** `data/critic_data/critic_data.pkl`

### 6. Train Critic Model

```bash
# Submit SLURM job
sbatch scripts/slurm/train_critic.sh

# Or run directly
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir data/critic_data \
    --checkpoint_dir checkpoints/critic \
    --seed 42
```

**Monitor job:**
```bash
tail -f logs/train_critic_*.out
tail -f logs/train_critic_*.err
```

**Expected output:** `checkpoints/critic/critic_epoch_*.pt` files

### 7. Run MCTS Inference (Optional)

After both models are trained:

```bash
# Create results directory
mkdir -p results

# Run MCTS inference
python experiments/run_mcts_inference.py \
    --config configs/mcts/base.yaml \
    --diffusion_checkpoint checkpoints/checkpoint_epoch_200.pt \
    --critic_checkpoint checkpoints/critic/critic_epoch_100.pt \
    --netlist_file data/test_netlist.json \
    --output_dir results \
    --seed 42
```

## Quick Reference Commands

### Check Data
```bash
# Count training samples
ls -1 data/routing_states/*.pkl | wc -l

# Check critic data
ls -lh data/critic_data/critic_data.pkl
```

### Monitor Training
```bash
# Watch diffusion training logs
tail -f logs/train_diffusion_*.out

# Watch critic training logs  
tail -f logs/train_critic_*.out

# Check GPU usage
nvidia-smi
```

### Check Checkpoints
```bash
# List diffusion checkpoints
ls -lh checkpoints/checkpoint_epoch_*.pt

# List critic checkpoints
ls -lh checkpoints/critic/critic_epoch_*.pt
```

### Cancel Jobs
```bash
# List your jobs
squeue -u as0714

# Cancel a job
scancel <job_id>
```

## Troubleshooting

### Issue: "No module named 'src.shared'"
**Solution:** Make sure PYTHONPATH is set:
```bash
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"
```

### Issue: "No .pkl files found"
**Solution:** Generate data first:
```bash
python scripts/generate_synthetic_diffusion_data.py --output_dir data/routing_states --num_samples 2000
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size in config files:
- `configs/training/della_diffusion.yaml`: `batch_size: 16` (instead of 32)
- `configs/training/della_critic.yaml`: `batch_size: 8` (instead of 16)

### Issue: Loss stuck at ~1.0
**Solution:** This means data has meaningless latents. Regenerate data:
```bash
python scripts/generate_synthetic_diffusion_data.py --output_dir data/routing_states --num_samples 2000
```

## Expected Training Times

- **Diffusion training:** ~2-4 hours for 200 epochs (single GPU)
- **Critic data generation:** ~30 minutes for 1000 samples
- **Critic training:** ~1-2 hours for 100 epochs (single GPU)
- **MCTS inference:** ~5-10 minutes per netlist (depends on iterations)

## Next Steps After Training

1. **Evaluate models:** Check validation loss in logs
2. **Visualize results:** Plot training curves
3. **Run inference:** Test on real netlists
4. **Compare baselines:** Run comparison experiments

