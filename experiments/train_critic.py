"""Training entry point for routing critic."""

import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import os

from src.critic.training import CriticTrainer, RoutingDataset, collate_routing_graphs
from src.critic.gnn import RoutingCritic
from src.critic.features import RoutingGraphBuilder
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
    parser = argparse.ArgumentParser(description="Train routing critic")
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
    
    # Create critic model
    model_config = config.get('model', {})
    critic = RoutingCritic(
        node_dim=model_config.get('node_dim', 64),
        edge_dim=model_config.get('edge_dim', 32),
        hidden_dim=model_config.get('hidden_dim', 128),
        num_layers=model_config.get('num_layers', 4),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Create trainer
    training_config = config.get('training', {})
    trainer = CriticTrainer(
        critic=critic,
        lr=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.critic.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Resumed from checkpoint {args.resume}")
    
    # Load dataset
    # This is simplified - would load actual RoutingExample data
    from src.critic.training import RoutingExample
    import pickle
    
    data_dir = Path(config.get('data', {}).get('data_dir', args.data_dir))
    data_file = data_dir / "critic_data.pkl"
    
    if not data_file.exists():
        logger.error(f"Critic data file not found: {data_file}")
        logger.error("Please run scripts/generate_critic_data.py first")
        return
    
    with open(data_file, 'rb') as f:
        examples = pickle.load(f)
    
    # Split data
    train_split = config.get('data', {}).get('train_split', 0.8)
    val_split = config.get('data', {}).get('val_split', 0.1)
    
    n_train = int(len(examples) * train_split)
    n_val = int(len(examples) * val_split)
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train+n_val]
    test_examples = examples[n_train+n_val:]
    
    # Create graph builder
    # This is simplified - would need actual grid
    from src.core.routing.grid import Grid
    grid = Grid(width=100, height=100)  # Placeholder
    graph_builder = RoutingGraphBuilder(grid)
    
    # Create datasets
    train_dataset = RoutingDataset(train_examples, graph_builder)
    val_dataset = RoutingDataset(val_examples, graph_builder)
    
    # Create data loaders
    batch_size = training_config.get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_routing_graphs,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_routing_graphs,
        pin_memory=True
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}")
    
    # Train
    num_epochs = training_config.get('num_epochs', 100)
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, log_interval=100)
        
        # Validation
        trainer.critic.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                graph, scores = batch
                graph = graph.to(trainer.device)
                scores = scores.to(trainer.device)
                
                pred = trainer.critic(graph)
                loss = trainer.criterion(pred, scores)
                val_loss += loss.item()
                val_count += 1
        
        val_metrics = {'loss': val_loss / max(val_count, 1)}
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )
        
        # Checkpointing
        if (epoch + 1) % config.get('checkpointing', {}).get('save_every_n_epochs', 10) == 0:
            checkpoint_dir = Path(config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints/critic'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'critic_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.critic.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Critic training complete!")


if __name__ == "__main__":
    main()

