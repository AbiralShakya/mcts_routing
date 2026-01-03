#!/bin/bash
# Helper script to submit training jobs on Della

JOB_TYPE=$1
CONFIG=$2

if [ -z "$JOB_TYPE" ]; then
    echo "Usage: $0 <job_type> [config]"
    echo ""
    echo "Job types:"
    echo "  diffusion  - Train routing diffusion model"
    echo "  critic     - Train routing critic"
    echo "  data       - Generate training data"
    echo ""
    echo "Examples:"
    echo "  $0 diffusion configs/training/della_diffusion.yaml"
    echo "  $0 critic configs/training/della_critic.yaml"
    echo "  $0 data"
    exit 1
fi

case $JOB_TYPE in
    diffusion)
        if [ -z "$CONFIG" ]; then
            CONFIG="configs/training/della_diffusion.yaml"
        fi
        echo "Submitting diffusion training job with config: $CONFIG"
        sbatch scripts/slurm/train_diffusion.sh
        ;;
    critic)
        if [ -z "$CONFIG" ]; then
            CONFIG="configs/training/della_critic.yaml"
        fi
        echo "Submitting critic training job with config: $CONFIG"
        sbatch scripts/slurm/train_critic.sh
        ;;
    data)
        echo "Submitting data generation job"
        sbatch scripts/slurm/generate_data.sh
        ;;
    *)
        echo "Unknown job type: $JOB_TYPE"
        echo "Valid types: diffusion, critic, data"
        exit 1
        ;;
esac

echo "Job submitted. Check status with: squeue -u $USER"

