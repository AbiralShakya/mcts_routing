# Della Cluster Training Setup

This document describes how to train the routing diffusion model and critic on Princeton's Della cluster.

## Quick Start

1. **Setup environment on Della:**
   ```bash
   ssh della.princeton.edu
   cd /path/to/mcts_routing
   bash scripts/setup_della.sh
   ```

2. **Generate training data:**
   ```bash
   sbatch scripts/slurm/generate_data.sh
   ```

3. **Train diffusion model:**
   ```bash
   bash scripts/submit_training.sh diffusion
   ```

4. **Train critic:**
   ```bash
   bash scripts/submit_training.sh critic
   ```

## Directory Structure

```
scripts/
├── setup_della.sh              # Della cluster setup
├── submit_training.sh          # Job submission helper
├── monitor_training.sh         # Monitor training progress
├── generate_routing_data.py    # Generate routing training data
├── generate_critic_data.py     # Generate critic training data
└── slurm/
    ├── train_diffusion.sh      # Diffusion training job
    ├── train_critic.sh         # Critic training job
    └── generate_data.sh        # Data generation job

experiments/
├── train_routing.py            # Routing diffusion training entry
└── train_critic.py             # Critic training entry

configs/training/
├── routing_diffusion.yaml      # Base routing training config
├── della_diffusion.yaml        # Della-specific config
└── della_critic.yaml           # Critic training config
```

## Setup

### 1. Initial Setup

Run the setup script to:
- Load required modules (anaconda3, cuda, gcc)
- Create virtual environment
- Install PyTorch with CUDA 11.8
- Install project dependencies
- Setup $SCRATCH directories

```bash
bash scripts/setup_della.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

## Data Generation

### Generate Routing Training Data

This script loads nextpnr designs, runs the router, and converts to RoutingState format:

```bash
python scripts/generate_routing_data.py \
    --input_dir /path/to/nextpnr/designs \
    --output_dir $SCRATCH/data/routing_states \
    --num_samples 10000
```

Or submit as SLURM job:

```bash
sbatch scripts/slurm/generate_data.sh
```

### Generate Critic Training Data

This script samples partial routing states from diffusion trajectories and records final scores:

```bash
python scripts/generate_critic_data.py \
    --model_path checkpoints/diffusion_model.pt \
    --netlist_dir /path/to/netlists \
    --output_dir $SCRATCH/data/critic_data \
    --num_samples 1000
```

## Training

### Train Routing Diffusion Model

**Local (single GPU):**
```bash
python experiments/train_routing.py \
    --config configs/training/routing_diffusion.yaml \
    --data_dir $SCRATCH/data/routing_states \
    --checkpoint_dir $SCRATCH/checkpoints
```

**Cluster (multi-GPU with DDP):**
```bash
bash scripts/submit_training.sh diffusion
```

Or directly:
```bash
sbatch scripts/slurm/train_diffusion.sh
```

### Train Critic

**Local:**
```bash
python experiments/train_critic.py \
    --config configs/training/della_critic.yaml \
    --data_dir $SCRATCH/data/critic_data \
    --checkpoint_dir $SCRATCH/checkpoints/critic
```

**Cluster:**
```bash
bash scripts/submit_training.sh critic
```

## Monitoring

### Check Job Status

```bash
squeue -u $USER
```

### Monitor Training Progress

```bash
bash scripts/monitor_training.sh
```

Or manually:
```bash
# View latest training log
tail -f logs/train_diffusion_*.out

# View latest critic log
tail -f logs/train_critic_*.out
```

### Check GPU Usage

```bash
squeue -j <JOBID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
```

## Configuration

### Routing Diffusion Config

Key settings in `configs/training/della_diffusion.yaml`:

- **Model**: `num_timesteps=1000`, `hidden_dim=256`, `num_layers=6`
- **Training**: `batch_size=256` (for 8 GPUs), `num_epochs=200`
- **Distributed**: `backend=nccl`, `num_gpus=8`
- **Checkpointing**: Saves every 10 epochs to `$SCRATCH/checkpoints`

### Critic Config

Key settings in `configs/training/della_critic.yaml`:

- **Model**: `hidden_dim=128`, `num_layers=4`
- **Training**: `batch_size=64` (for 4 GPUs), `num_epochs=100`
- **Distributed**: `backend=nccl`, `num_gpus=4`

## Storage

- **$SCRATCH**: Fast storage for checkpoints and data
  - `$SCRATCH/checkpoints/`: Model checkpoints
  - `$SCRATCH/data/`: Training data
  - `$SCRATCH/logs/`: Training logs

- **Home directory**: Slower, use for code only

## Troubleshooting

### Job Fails to Start

- Check SLURM output: `cat logs/train_diffusion_*.err`
- Verify environment: `source venv/bin/activate`
- Check module loading: `module list`

### Out of Memory

- Reduce `batch_size` in config
- Reduce `num_layers` or `hidden_dim`
- Request more memory: `#SBATCH --mem=512G`

### CUDA Errors

- Verify CUDA module: `module load cuda/11.8`
- Check GPU availability: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### Data Not Found

- Verify data directory: `ls $SCRATCH/data/`
- Check data generation completed: `squeue -u $USER`
- Regenerate data if needed

## Resuming Training

To resume from a checkpoint:

```bash
python experiments/train_routing.py \
    --config configs/training/della_diffusion.yaml \
    --data_dir $SCRATCH/data/routing_states \
    --checkpoint_dir $SCRATCH/checkpoints \
    --resume $SCRATCH/checkpoints/checkpoint_epoch_50.pt
```

## Email Notifications

Update email in SLURM scripts:

```bash
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu
```

You'll receive emails when jobs:
- End (successfully)
- Fail
- Time out

## Best Practices

1. **Use $SCRATCH**: Faster I/O than home directory
2. **Monitor early**: Check logs after first few batches
3. **Save frequently**: Checkpoints every 10 epochs
4. **Test locally first**: Verify on small dataset before cluster
5. **Check GPU utilization**: Should be >90% during training

## Next Steps

After training:

1. **Evaluate model**: Run inference on test set
2. **Compare baselines**: Compare against DDIM and best-of-N
3. **Ablation studies**: Test different configurations
4. **Integration**: Use trained model in MCTS router

