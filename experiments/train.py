"""Training entry point."""

import argparse
import yaml
import torch

from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()

