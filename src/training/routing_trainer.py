"""Training pipeline for routing diffusion model (no decoder - direct RoutingState training)."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
import logging

from ..diffusion.model import RoutingDiffusion, RoutingState
from ..diffusion.schedule import DDPMSchedule


logger = logging.getLogger(__name__)


class RoutingDiffusionTrainer:
    """Train routing diffusion model on RoutingState data.
    
    Key difference from old approach: No decoder.
    Training directly on net latents (routing decisions).
    """
    
    def __init__(
        self,
        model: RoutingDiffusion,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.
        
        Args:
            model: RoutingDiffusion model
            config: Training configuration
            device: Device (if None, auto-detect)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        
        # DDP setup
        self.use_ddp = config.get('distributed', {}).get('enabled', False)
        if self.use_ddp:
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            dist.init_process_group(
                backend=config.get('distributed', {}).get('backend', 'nccl'),
                init_method=config.get('distributed', {}).get('init_method', 'env://')
            )
            self.model = DDP(self.model, device_ids=[self.rank])
        else:
            self.rank = 0
            self.world_size = 1
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('training', {}).get('learning_rate', 1e-4),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-5)
        )
        
        # Scheduler
        num_epochs = config.get('training', {}).get('num_epochs', 200)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs
        )
        
        # Training params
        self.T = model.num_timesteps
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        
        # Checkpointing
        checkpoint_dir = config.get('checkpointing', {}).get('checkpoint_dir', 'checkpoints')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Gradient clipping
        self.grad_clip_norm = config.get('training', {}).get('gradient_clip_norm', 1.0)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Data loader with RoutingState examples
            epoch: Current epoch
        
        Returns:
            Dict with loss values
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch components
            routing_states = batch['routing_state']  # List[RoutingState]
            net_features = batch['net_features'].to(self.device)  # [B, num_nets, F]
            net_positions = batch['net_positions'].to(self.device)  # [B, num_nets, 4]
            congestion_maps = batch.get('congestion_map')  # Optional[List[Tensor]]
            
            B = len(routing_states)
            
            # Sample timesteps
            t = torch.randint(0, self.T, (B,), device=self.device)
            
            # Forward diffusion: add noise to net latents
            net_latents_batch = []
            noise_batch = []
            
            for i, state in enumerate(routing_states):
                # Stack all net latents into single tensor
                net_ids = sorted(state.net_latents.keys())
                max_pips = max(len(state.net_latents[nid]) for nid in net_ids) if net_ids else 100
                
                latents = torch.zeros(len(net_ids), max_pips, device=self.device)
                noise = torch.zeros(len(net_ids), max_pips, device=self.device)
                
                for j, net_id in enumerate(net_ids):
                    latent = state.net_latents[net_id].to(self.device)
                    n = len(latent)
                    latents[j, :n] = latent
                    
                    # Sample noise
                    n_noise = torch.randn_like(latent)
                    noise[j, :n] = n_noise
                
                net_latents_batch.append(latents)
                noise_batch.append(noise)
            
            # Pad to same size for batching
            max_nets = max(lat.shape[0] for lat in net_latents_batch)
            max_pips = max(lat.shape[1] for lat in net_latents_batch)
            
            net_latents_padded = torch.zeros(B, max_nets, max_pips, device=self.device)
            noise_padded = torch.zeros(B, max_nets, max_pips, device=self.device)
            
            for i, (lat, n) in enumerate(zip(net_latents_batch, noise_batch)):
                n_nets, n_pips = lat.shape
                net_latents_padded[i, :n_nets, :n_pips] = lat
                noise_padded[i, :n_nets, :n_pips] = n
            
            # Forward diffusion: q(x_t | x_0)
            alpha_bar_t = self.model.schedule.get_alpha_bar(t)
            # Reshape for broadcasting
            alpha_bar_t = alpha_bar_t.view(B, 1, 1)
            
            x_t = torch.sqrt(alpha_bar_t) * net_latents_padded + torch.sqrt(1 - alpha_bar_t) * noise_padded
            
            # Predict noise
            predicted_noise = self.model(
                x_t,
                t,
                net_features,
                net_positions,
                congestion=None  # Could add congestion conditioning
            )
            
            # Compute loss (only on valid regions)
            # Mask out padding
            valid_mask = (net_latents_padded.abs().sum(dim=-1) > 0).unsqueeze(-1)  # [B, max_nets, 1]
            loss = ((predicted_noise - noise_padded) ** 2 * valid_mask).sum() / valid_mask.sum()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0 and self.rank == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
                )
        
        return {
            'loss': total_loss / max(num_batches, 1)
        }
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                routing_states = batch['routing_state']
                net_features = batch['net_features'].to(self.device)
                net_positions = batch['net_positions'].to(self.device)
                
                B = len(routing_states)
                t = torch.randint(0, self.T, (B,), device=self.device)
                
                # Forward diffusion (same as training)
                net_latents_batch = []
                noise_batch = []
                
                for state in routing_states:
                    net_ids = sorted(state.net_latents.keys())
                    max_pips = max(len(state.net_latents[nid]) for nid in net_ids) if net_ids else 100
                    
                    latents = torch.zeros(len(net_ids), max_pips, device=self.device)
                    noise = torch.zeros(len(net_ids), max_pips, device=self.device)
                    
                    for j, net_id in enumerate(net_ids):
                        latent = state.net_latents[net_id].to(self.device)
                        n = len(latent)
                        latents[j, :n] = latent
                        noise[j, :n] = torch.randn_like(latent)
                    
                    net_latents_batch.append(latents)
                    noise_batch.append(noise)
                
                # Pad and forward
                max_nets = max(lat.shape[0] for lat in net_latents_batch)
                max_pips = max(lat.shape[1] for lat in net_latents_batch)
                
                net_latents_padded = torch.zeros(B, max_nets, max_pips, device=self.device)
                noise_padded = torch.zeros(B, max_nets, max_pips, device=self.device)
                
                for i, (lat, n) in enumerate(zip(net_latents_batch, noise_batch)):
                    n_nets, n_pips = lat.shape
                    net_latents_padded[i, :n_nets, :n_pips] = lat
                    noise_padded[i, :n_nets, :n_pips] = n
                
                alpha_bar_t = self.model.schedule.get_alpha_bar(t).view(B, 1, 1)
                x_t = torch.sqrt(alpha_bar_t) * net_latents_padded + torch.sqrt(1 - alpha_bar_t) * noise_padded
                
                predicted_noise = self.model(x_t, t, net_features, net_positions)
                
                valid_mask = (net_latents_padded.abs().sum(dim=-1) > 0).unsqueeze(-1)
                loss = ((predicted_noise - noise_padded) ** 2 * valid_mask).sum() / valid_mask.sum()
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs (overrides config)
        """
        num_epochs = num_epochs or self.config.get('training', {}).get('num_epochs', 200)
        save_every = self.config.get('checkpointing', {}).get('save_every_n_epochs', 10)
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None and self.rank == 0:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {}
            
            # Learning rate step
            self.scheduler.step()
            
            # Logging (only on rank 0)
            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics.get('loss', 0):.4f}"
                )
            
            # Checkpointing
            if (epoch + 1) % save_every == 0 and self.rank == 0:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint.
        
        Args:
            path: Checkpoint file path
        
        Returns:
            Epoch number
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']

