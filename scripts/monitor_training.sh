#!/bin/bash
# Monitor training progress on Della

echo "=== Job Status ==="
squeue -u $USER

echo ""
echo "=== Latest Training Log ==="
LATEST_LOG=$(ls -t logs/train_diffusion_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "Latest log: $LATEST_LOG"
    echo "Last 20 lines:"
    tail -20 "$LATEST_LOG"
else
    echo "No training logs found"
fi

echo ""
echo "=== Latest Critic Log ==="
LATEST_CRITIC=$(ls -t logs/train_critic_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_CRITIC" ]; then
    echo "Latest log: $LATEST_CRITIC"
    echo "Last 20 lines:"
    tail -20 "$LATEST_CRITIC"
else
    echo "No critic logs found"
fi

echo ""
echo "=== GPU Usage ==="
if [ -n "$SLURM_JOB_ID" ]; then
    squeue -j $SLURM_JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
else
    echo "No active job ID found"
    echo "To monitor a specific job: squeue -j <JOBID> -o '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b'"
fi

echo ""
echo "=== Checkpoint Status ==="
if [ -n "$SCRATCH" ]; then
    CHECKPOINT_DIR="$SCRATCH/checkpoints"
else
    CHECKPOINT_DIR="checkpoints"
fi

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "Checkpoints in $CHECKPOINT_DIR:"
    ls -lh "$CHECKPOINT_DIR" | tail -10
else
    echo "No checkpoint directory found"
fi

