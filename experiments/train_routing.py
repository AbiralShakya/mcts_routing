"""Training entry point for routing diffusion model."""

import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import os

from src.training.routing_trainer import RoutingDiffusionTrainer
from src.data.routing_dataset import RoutingStateDataset, collate_routing_states
from src.diffusion import RoutingDiffusion, create_routing_diffusion
from src.shared.encoders import create_shared_encoders
from src.utils.seed import set_seed
from src.utils.logging import setup_logging


def setup_logging_distributed():
    """Setup logging for distributed training."""
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        setup_logging()
    else:
        logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Train routing diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    setup_logging_distributed()
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint dir if provided
    if args.checkpoint_dir:
        config.setdefault('checkpointing', {})['checkpoint_dir'] = args.checkpoint_dir
    
    # Override data dir if provided
    if args.data_dir:
        config.setdefault('data', {})['data_dir'] = args.data_dir
    
    # Create model
    model_config = config.get('model', {})
    
    # Check if shared encoders should be used
    shared_encoders_config = config.get('shared_encoders', {})
    use_shared = shared_encoders_config.get('enabled', False)
    
    shared_net_encoder = None
    shared_congestion_encoder = None
    
    if use_shared:
        logger.info("Creating shared encoders for diffusion model")
        hidden_dim = model_config.get('hidden_dim', 256)
        net_feat_dim = model_config.get('net_feat_dim', 7)
        shared_net_encoder, shared_congestion_encoder = create_shared_encoders(
            hidden_dim=hidden_dim,
            net_feat_dim=net_feat_dim
        )
    
    model = create_routing_diffusion(
        model_config,
        shared_net_encoder=shared_net_encoder,
        shared_congestion_encoder=shared_congestion_encoder
    )
    
    # Create trainer
    trainer = RoutingDiffusionTrainer(model, config, device=None)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create datasets
    data_dir = config.get('data', {}).get('data_dir', args.data_dir)
    train_dataset = RoutingStateDataset(data_dir, split="train")
    val_dataset = RoutingStateDataset(data_dir, split="val")
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_routing_states,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_routing_states,
        pin_memory=True
    )
    
    if trainer.rank == 0:
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of epochs: {config.get('training', {}).get('num_epochs', 200)}")
    
    # Train
    num_epochs = config.get('training', {}).get('num_epochs', 200)
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    if trainer.rank == 0:
        logger.info("Training complete!")


if __name__ == "__main__":
    main()

