#!/bin/bash
# Quick script to run MCTS inference on della

# Default values
DIFFUSION_CHECKPOINT="checkpoints/checkpoint_epoch_200.pt"
CRITIC_CHECKPOINT="checkpoints/critic/critic_epoch_100.pt"
NETLIST_FILE="data/test_netlist.json"
OUTPUT_DIR="results"
SEED=42
CONFIG="configs/mcts/base.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --diffusion)
            DIFFUSION_CHECKPOINT="$2"
            shift 2
            ;;
        --critic)
            CRITIC_CHECKPOINT="$2"
            shift 2
            ;;
        --netlist)
            NETLIST_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --no-critic)
            CRITIC_CHECKPOINT=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--diffusion PATH] [--critic PATH] [--netlist PATH] [--output DIR] [--seed N] [--config PATH] [--no-critic]"
            exit 1
            ;;
    esac
done

# Set environment
export PYTHONPATH="/scratch/gpfs/MZHASAN/graph_vector_topological_insulator/mcts_routing:$PYTHONPATH"

# Build command
CMD="python experiments/run_mcts_inference.py \
    --config $CONFIG \
    --diffusion_checkpoint $DIFFUSION_CHECKPOINT \
    --netlist_file $NETLIST_FILE \
    --output_dir $OUTPUT_DIR \
    --seed $SEED"

# Add critic if provided
if [ -n "$CRITIC_CHECKPOINT" ]; then
    CMD="$CMD --critic_checkpoint $CRITIC_CHECKPOINT"
fi

echo "Running inference..."
echo "Command: $CMD"
echo ""

eval $CMD

