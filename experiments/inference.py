"""Inference entry point."""

import argparse
import yaml

from src.inference.mcts_diffusion import mcts_inference
from src.inference.ddim_inference import ddim_inference
from src.utils.seed import set_seed
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--method", type=str, choices=["ddpm", "ddim", "mcts"], default="mcts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run inference
    if args.method == "mcts":
        result = mcts_inference(config)
    elif args.method == "ddim":
        result = ddim_inference(config)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    print(f"Inference complete. Reward: {result['reward']}")


if __name__ == "__main__":
    main()

