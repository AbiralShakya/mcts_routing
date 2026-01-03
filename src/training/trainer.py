"""Training pipeline with DDP support for H100 cluster."""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional
import os
from pathlib import Path

from .loss import DiffusionLoss
from ..core.diffusion.forward_process import q_sample
from ..core.diffusion.schedule import DDPMSchedule


class RoutingDataset(Dataset):
    """Dataset for routing potential fields."""
    
    def __init__(self, data: list):
        """Initialize dataset.
        
        Args:
            data: List of (placement, netlist, x_0, metadata) tuples
        """
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        placement, netlist, x_0, metadata = self.data[idx]
        # Return x_0 and metadata (conditioning would be added separately)
        return {
            'x_0': x_0,
            'metadata': metadata
        }


class DiffusionTrainer:
    """Main trainer for diffusion model."""
    
    def __init__(
        self,
        model: nn.Module,
        decoder: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Diffusion model (UNet)
            decoder: Potential decoder
            config: Training configuration
            device: Device (if None, auto-detect)
        """
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.decoder = decoder.to(self.device)
        
        # DDP setup
        self.use_ddp = config.get('use_ddp', False)
        if self.use_ddp:
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            dist.init_process_group(backend='nccl')
            self.model = DDP(self.model, device_ids=[self.rank])
            self.decoder = DDP(self.decoder, device_ids=[self.rank])
        else:
            self.rank = 0
            self.world_size = 1
        
        # Loss function
        self.loss_fn = DiffusionLoss(
            lambda_lipschitz=config.get('lambda_lipschitz', 0.1),
            lambda_entropy=config.get('lambda_entropy', 0.01),
            lambda_smooth=config.get('lambda_smooth', 0.001)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.decoder.parameters()),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('num_epochs', 100)
        )
        
        # Diffusion schedule
        self.schedule = DDPMSchedule(num_timesteps=config.get('num_timesteps', 1000))
        self.T = config.get('num_timesteps', 1000)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Data loader
            epoch: Current epoch
        
        Returns:
            Dict with loss values
        """
        self.model.train()
        self.decoder.train()
        
        total_loss = 0.0
        total_diff_loss = 0.0
        total_lip_loss = 0.0
        total_ent_loss = 0.0
        total_smooth_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            x_0 = batch['x_0'].to(self.device)  # [B, H, W, C] or [B, C, H, W]
            
            # Ensure [B, C, H, W] format
            if x_0.dim() == 4 and x_0.shape[-1] < x_0.shape[1]:
                x_0 = x_0.permute(0, 3, 1, 2)
            
            B = x_0.shape[0]
            
            # Sample timestep
            t = torch.randint(0, self.T, (B,), device=self.device)
            
            # Forward diffusion
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, self.schedule)
            
            # Predict noise
            predicted_noise = self.model(x_t, t)
            
            # Compute loss
            loss = self.loss_fn(
                predicted_noise,
                noise,
                decoder=self.decoder,
                latent=x_t
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'diff_loss': total_diff_loss / max(num_batches, 1),
            'lip_loss': total_lip_loss / max(num_batches, 1),
            'ent_loss': total_ent_loss / max(num_batches, 1),
            'smooth_loss': total_smooth_loss / max(num_batches, 1)
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100
    ):
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs
        """
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
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
            
            # Checkpointing (every 10 epochs)
            if (epoch + 1) % 10 == 0 and self.rank == 0:
                self.save_checkpoint(epoch)
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate on validation set.
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        self.decoder.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x_0 = batch['x_0'].to(self.device)
                if x_0.dim() == 4 and x_0.shape[-1] < x_0.shape[1]:
                    x_0 = x_0.permute(0, 3, 1, 2)
                
                B = x_0.shape[0]
                t = torch.randint(0, self.T, (B,), device=self.device)
                noise = torch.randn_like(x_0)
                x_t = q_sample(x_0, t, self.schedule)
                
                predicted_noise = self.model(x_t, t)
                loss = self.loss_fn(predicted_noise, noise, self.decoder, x_t)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1)
        }
    
    def save_checkpoint(self, epoch: int):
        """Save checkpoint.
        
        Args:
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.use_ddp else self.model.state_dict(),
            'decoder_state_dict': self.decoder.module.state_dict() if self.use_ddp else self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
            self.decoder.module.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
