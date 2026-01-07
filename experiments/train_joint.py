#!/usr/bin/env python3
"""Joint training pipeline with shared encoders.

This script implements the full training pipeline:
1. Create shared encoders (used by both diffusion and critic)
2. Train diffusion model
3. Generate critic data from diffusion trajectories
4. Train critic with same shared encoders
5. Save both models with shared encoder weights

Usage:
    python experiments/train_joint.py \
        --config configs/training/della_diffusion.yaml \
        --data_dir data/routing_states \
        --checkpoint_dir checkpoints \
        --phase diffusion  # or critic, or both
"""

import argparse
import yaml
import torch
import pickle
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.encoders import create_shared_encoders, SharedNetEncoder, SharedCongestionEncoder
from src.diffusion.model import create_routing_diffusion, RoutingState
from src.diffusion.schedule import DDPMSchedule
from src.training.routing_trainer import RoutingDiffusionTrainer
from src.data.routing_dataset import RoutingStateDataset, collate_routing_states
from src.critic.gnn import RoutingCritic
from src.critic.training import CriticTrainer, generate_synthetic_training_data
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_models_with_shared_encoders(config: dict, device: str = "cuda"):
    """Create diffusion and critic models with shared encoders.

    This ensures both models see routing states the same way.
    """
    model_config = config.get('model', {})
    hidden_dim = model_config.get('hidden_dim', 256)
    net_feat_dim = model_config.get('net_feat_dim', 7)

    # Create shared encoders ONCE
    logger.info(f"Creating shared encoders (hidden_dim={hidden_dim}, net_feat_dim={net_feat_dim})")
    net_encoder, cong_encoder = create_shared_encoders(
        hidden_dim=hidden_dim,
        net_feat_dim=net_feat_dim
    )
    net_encoder = net_encoder.to(device)
    cong_encoder = cong_encoder.to(device)

    # Create diffusion model with shared encoders
    diffusion = create_routing_diffusion(
        config=model_config,
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder
    ).to(device)

    logger.info(f"Diffusion model created: {sum(p.numel() for p in diffusion.parameters()):,} parameters")
    logger.info(f"  Using shared net encoder: {diffusion.using_shared_net_encoder}")
    logger.info(f"  Using shared congestion encoder: {diffusion.using_shared_congestion_encoder}")

    # Create critic model with SAME shared encoders
    critic_config = config.get('critic', model_config)
    critic = RoutingCritic(
        node_dim=critic_config.get('node_dim', 64),
        edge_dim=critic_config.get('edge_dim', 32),
        hidden_dim=hidden_dim,  # Must match for shared encoders
        num_layers=critic_config.get('num_layers', 4),
        dropout=critic_config.get('dropout', 0.1),
        shared_net_encoder=net_encoder,
        shared_congestion_encoder=cong_encoder,
        net_feat_dim=net_feat_dim
    ).to(device)

    logger.info(f"Critic model created: {sum(p.numel() for p in critic.parameters()):,} parameters")

    return diffusion, critic, net_encoder, cong_encoder


def train_diffusion(
    diffusion,
    config: dict,
    data_dir: str,
    checkpoint_dir: str,
    device: str = "cuda"
):
    """Train diffusion model."""
    logger.info("="*60)
    logger.info("PHASE 1: Training Diffusion Model")
    logger.info("="*60)

    data_path = Path(data_dir)
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = RoutingStateDataset(data_path)
    logger.info(f"Loaded {len(dataset)} samples from {data_path}")

    # Split into train/val
    train_split = config.get('data', {}).get('train_split', 0.9)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    # Create data loaders
    # Use num_workers=0 to avoid CUDA issues in forked subprocesses
    # Data may contain tensors that cause "Cannot re-initialize CUDA in forked subprocess"
    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_routing_states,
        num_workers=0,  # Must be 0 to avoid CUDA fork issues
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_routing_states,
        num_workers=0,  # Must be 0 to avoid CUDA fork issues
        pin_memory=False
    )

    logger.info(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")
    logger.info(f"Batch size: {batch_size}")

    # Create trainer
    trainer = RoutingDiffusionTrainer(
        model=diffusion,
        config=config,
        device=torch.device(device)
    )

    # Train
    num_epochs = config.get('training', {}).get('num_epochs', 200)
    logger.info(f"Training for {num_epochs} epochs...")
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)

    # Save final checkpoint
    final_path = checkpoint_path / "diffusion_final.pt"
    torch.save({
        'model_state_dict': diffusion.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"Saved final diffusion model to {final_path}")

    return diffusion


def generate_critic_data(
    diffusion,
    config: dict,
    output_dir: str,
    num_samples: int = 10000,
    device: str = "cuda"
):
    """Generate critic training data from diffusion trajectories."""
    logger.info("="*60)
    logger.info("PHASE 2: Generating Critic Training Data")
    logger.info("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    diffusion.eval()

    # Generate synthetic examples with varying difficulty
    logger.info(f"Generating {num_samples} critic training examples...")

    examples = generate_synthetic_training_data(
        num_examples=num_samples,
        grid_sizes=[(10, 10), (20, 20), (30, 30)],
        nets_range=(5, 30),
        device=device
    )

    # Save examples
    output_file = output_path / "critic_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(examples, f)

    logger.info(f"Saved {len(examples)} examples to {output_file}")

    # Also save as adversarial data (same format, different name for compatibility)
    adversarial_file = output_path / "adversarial_critic_data.pkl"
    with open(adversarial_file, 'wb') as f:
        pickle.dump(examples, f)

    logger.info(f"Also saved to {adversarial_file}")

    return examples


def train_critic(
    critic,
    config: dict,
    data_dir: str,
    checkpoint_dir: str,
    device: str = "cuda"
):
    """Train critic model."""
    logger.info("="*60)
    logger.info("PHASE 3: Training Critic Model")
    logger.info("="*60)

    data_path = Path(data_dir)
    checkpoint_path = Path(checkpoint_dir) / "critic"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Load critic data
    data_files = [
        data_path / "adversarial_critic_data.pkl",
        data_path / "critic_data.pkl"
    ]

    examples = None
    for data_file in data_files:
        if data_file.exists():
            logger.info(f"Loading critic data from {data_file}")
            with open(data_file, 'rb') as f:
                examples = pickle.load(f)
            break

    if examples is None:
        logger.error("No critic data found!")
        return critic

    logger.info(f"Loaded {len(examples)} critic training examples")

    # Create trainer
    training_config = config.get('training', {})
    trainer = CriticTrainer(
        critic=critic,
        lr=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5),
        device=device
    )

    # Create datasets
    from src.critic.training import RoutingDataset, collate_routing_graphs
    from src.critic.features import RoutingGraphBuilder
    from src.core.routing.grid import Grid

    # Split data
    train_split = config.get('data', {}).get('train_split', 0.8)
    n_train = int(len(examples) * train_split)

    train_examples = examples[:n_train]
    val_examples = examples[n_train:]

    # Create graph builder (use first example's grid or default)
    if examples and hasattr(examples[0], 'grid'):
        grid = examples[0].grid
    else:
        grid = Grid(width=20, height=20)

    graph_builder = RoutingGraphBuilder(grid)

    train_dataset = RoutingDataset(train_examples, graph_builder)
    val_dataset = RoutingDataset(val_examples, graph_builder)

    batch_size = training_config.get('batch_size', 16)
    # Use num_workers=0 because data may contain CUDA tensors (congestion_map)
    # CUDA tensors can't be accessed in worker subprocesses
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_routing_graphs,
        num_workers=0  # Must be 0 - data has CUDA tensors
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_routing_graphs,
        num_workers=0  # Must be 0 - data has CUDA tensors
    )

    logger.info(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Train
    num_epochs = training_config.get('num_epochs', 100)
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader, log_interval=50)
        val_metrics = trainer.evaluate(val_loader)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val MAE: {val_metrics['mae']:.4f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = checkpoint_path / f"critic_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': critic.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': config
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save final
    final_path = checkpoint_path / "critic_final.pt"
    torch.save({
        'model_state_dict': critic.state_dict(),
        'config': config
    }, final_path)
    logger.info(f"Saved final critic to {final_path}")

    return critic


def save_joint_checkpoint(
    diffusion,
    critic,
    net_encoder,
    cong_encoder,
    config: dict,
    checkpoint_dir: str
):
    """Save joint checkpoint with all models and shared encoders."""
    checkpoint_path = Path(checkpoint_dir)

    joint_checkpoint = {
        'diffusion_state_dict': diffusion.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'shared_net_encoder_state_dict': net_encoder.state_dict(),
        'shared_congestion_encoder_state_dict': cong_encoder.state_dict(),
        'config': config
    }

    joint_path = checkpoint_path / "joint_model.pt"
    torch.save(joint_checkpoint, joint_path)
    logger.info(f"Saved joint checkpoint to {joint_path}")

    return joint_path


def main():
    parser = argparse.ArgumentParser(description="Joint training with shared encoders")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--phase", type=str, choices=["diffusion", "critic", "both", "all"],
                        default="all", help="Training phase")
    parser.add_argument("--diffusion_checkpoint", type=str, default=None,
                        help="Load diffusion from checkpoint")
    parser.add_argument("--critic_samples", type=int, default=10000,
                        help="Number of critic training samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Phase: {args.phase}")

    # Create models with shared encoders
    diffusion, critic, net_encoder, cong_encoder = create_models_with_shared_encoders(
        config, args.device
    )

    # Load diffusion checkpoint if provided
    if args.diffusion_checkpoint:
        logger.info(f"Loading diffusion checkpoint from {args.diffusion_checkpoint}")
        checkpoint = torch.load(args.diffusion_checkpoint, map_location=args.device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])

    # Run training phases
    if args.phase in ["diffusion", "both", "all"]:
        diffusion = train_diffusion(
            diffusion, config, args.data_dir, args.checkpoint_dir, args.device
        )

    if args.phase in ["both", "all"]:
        # Generate critic data from diffusion
        generate_critic_data(
            diffusion, config, args.data_dir, args.critic_samples, args.device
        )

    if args.phase in ["critic", "both", "all"]:
        critic = train_critic(
            critic, config, args.data_dir, args.checkpoint_dir, args.device
        )

    # Save joint checkpoint
    if args.phase in ["both", "all"]:
        save_joint_checkpoint(
            diffusion, critic, net_encoder, cong_encoder, config, args.checkpoint_dir
        )

    logger.info("="*60)
    logger.info("Training pipeline complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
